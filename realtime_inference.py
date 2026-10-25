"""
realtime_inference.py
=====================
Production-Ready Real-Time Inference Pipeline for TransGait-EVT:
- Accepts live camera feed (webcam/RTSP), video files (.mp4), or images.
- Real-time pedestrian detection and tracklet temporal buffering.
- Spatiotemporal feature extraction (TE-TransReID + PCB + GaitPart + Attention).
- EVT Weibull statistical calibration for open-set stranger rejection.
- Live HUD with bounding boxes, match probability, attention weights, and FPS.
"""

import os
import sys
import time
import json
import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# Import model architecture and utilities directly from evaluate_reid
from evaluate_reid import (
    FullHybridReIDModel,
    extract_silhouette_mask,
    EVTWeibullCalibrator
)


class RealTimeTrackletManager:
    """
    Maintains a rolling spatiotemporal buffer of the last T frames for detected pedestrians.
    Extracts the keyframe and binary silhouette masks needed by GaitPart.
    """
    def __init__(self, max_seq_len=4, silh_size=(64, 32)):
        self.max_seq_len = max_seq_len
        self.silh_size = silh_size
        self.tracks = defaultdict(lambda: deque(maxlen=max_seq_len))
        self.track_centers = {}
        self.next_track_id = 1

    def update_tracks(self, bboxes, frame):
        """
        Simple centroid-based tracking across consecutive frames.
        Returns mapping: track_id -> bbox (x, y, w, h)
        """
        current_centers = []
        for (x, y, w, h) in bboxes:
            cx = x + w / 2.0
            cy = y + h / 2.0
            current_centers.append((cx, cy, (x, y, w, h)))

        matched_tracks = {}
        unmatched_centers = list(current_centers)

        # Match to existing tracks by nearest center distance
        for track_id, (prev_cx, prev_cy) in list(self.track_centers.items()):
            best_idx = -1
            best_dist = 85.0  # Max pixel movement threshold between consecutive frames

            for idx, (cx, cy, bbox) in enumerate(unmatched_centers):
                dist = np.hypot(cx - prev_cx, cy - prev_cy)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx >= 0:
                cx, cy, bbox = unmatched_centers.pop(best_idx)
                matched_tracks[track_id] = bbox
                self.track_centers[track_id] = (cx, cy)
            else:
                # Track lost
                del self.track_centers[track_id]

        # Register new tracks
        for (cx, cy, bbox) in unmatched_centers:
            new_id = self.next_track_id
            self.next_track_id += 1
            matched_tracks[new_id] = bbox
            self.track_centers[new_id] = (cx, cy)

        # Store cropped frames in buffer
        for track_id, (x, y, w, h) in matched_tracks.items():
            h_f, w_f = frame.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_f, x + w), min(h_f, y + h)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                self.tracks[track_id].append(crop)

        return matched_tracks

    def get_tracklet_tensors(self, track_id, transform_base):
        """
        Prepares model input tensors for a given track:
        - Keyframe RGB tensor: [1, 3, 256, 128]
        - Silhouettes tensor:  [1, T, 1, 64, 32]
        """
        crops = list(self.tracks[track_id])
        if not crops:
            return None, None

        # Keyframe (middle crop)
        mid_idx = len(crops) // 2
        keyframe_bgr = crops[mid_idx]
        keyframe_rgb = cv2.cvtColor(keyframe_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(keyframe_rgb)
        img_tensor = transform_base(pil_img).unsqueeze(0)

        # Silhouettes
        silhouettes = []
        indices = np.linspace(0, len(crops) - 1, self.max_seq_len, dtype=int)
        for i in indices:
            mask = extract_silhouette_mask(crops[i], target_size=self.silh_size)
            silhouettes.append(mask)

        silh_arr = np.array(silhouettes)[:, np.newaxis, :, :]  # [T, 1, 64, 32]
        silh_tensor = torch.from_numpy(silh_arr).unsqueeze(0).float()  # [1, T, 1, 64, 32]

        return img_tensor, silh_tensor


class RealTimeTransGaitReID:
    """
    Real-Time Open-Set Person Re-ID Engine powered by TransGait-EVT.
    """
    def __init__(self, weights_path=None, gallery_dir=None, threshold=0.70, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.img_size = (256, 128)
        self.silh_size = (64, 32)

        print(f"[Engine] Initializing TransGait-EVT on {self.device}...")
        self.model = FullHybridReIDModel(
            embed_dim=384,
            branch_dim=256,
            fused_dim=512,
            use_gem=True,
            include_gait=True
        ).to(self.device)

        if weights_path and os.path.isfile(weights_path):
            print(f"[Engine] Loading weights from {weights_path}...")
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print("[Engine] Initialized with default pre-trained ResNet-34 stem weights.")

        self.model.eval()

        # Image transform
        self.transform_base = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # EVT Weibull Calibrator
        self.evt_calibrator = EVTWeibullCalibrator(tail_size=300)
        self._load_or_fit_evt()

        # Pedestrian Detector (OpenCV HOG default)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Enrolled Watchlist Gallery: dict of name -> list of 512-d feature vectors
        self.gallery = {}
        if gallery_dir and os.path.isdir(gallery_dir):
            self.load_gallery_from_dir(gallery_dir)

    def _load_or_fit_evt(self):
        """Loads fitted parameters from raw_results.json if available, or sets defaults."""
        json_path = os.path.join(os.path.dirname(__file__), "raw_results.json")
        if os.path.isfile(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                evt_params = data.get('evt_parameters', {})
                self.evt_calibrator.min_val = float(evt_params.get('min_val', 0.3918))
                self.evt_calibrator.shape = float(evt_params.get('shape', 4.1205))
                self.evt_calibrator.scale = float(evt_params.get('scale', 0.2169))
                print(f"[EVT] Loaded Weibull parameters: min={self.evt_calibrator.min_val:.4f}, shape={self.evt_calibrator.shape:.4f}, scale={self.evt_calibrator.scale:.4f}")
                return
            except Exception as e:
                print(f"[EVT] Error loading raw_results.json: {e}")

        # Fallback values
        self.evt_calibrator.min_val = 0.3918
        self.evt_calibrator.shape = 4.1205
        self.evt_calibrator.scale = 0.2169

    def enroll_person(self, name, img_tensor, silh_tensor):
        """Enrolls a person into the live watchlist gallery."""
        feat, _ = self.extract_embedding(img_tensor, silh_tensor)
        if name not in self.gallery:
            self.gallery[name] = []
        self.gallery[name].append(feat)
        print(f"[Gallery] Successfully enrolled '{name}' (Total templates: {len(self.gallery[name])})")

    def load_gallery_from_dir(self, gallery_dir, max_ids=100):
        """Loads and indexes enrolled identities from an image directory with fast caching."""
        cache_path = os.path.join(os.path.dirname(__file__), "gallery_cache.pt")
        if os.path.isfile(cache_path):
            print(f"[Gallery Cache] Loading cached gallery embeddings from {cache_path}...")
            try:
                self.gallery = torch.load(cache_path, map_location='cpu')
                print(f"[Gallery Cache] Loaded {len(self.gallery)} identities instantly.")
                return
            except Exception as e:
                print(f"[Gallery Cache] Warning loading cache: {e}. Rebuilding...")

        print(f"[Gallery] Indexing enrolled gallery from {gallery_dir} (limit={max_ids} IDs)...")
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

        enrolled_ids = set()
        count = 0
        for root, _, files in os.walk(gallery_dir):
            for fn in sorted(files):
                if fn.lower().endswith(valid_exts):
                    person_name = os.path.basename(root)
                    if person_name == os.path.basename(gallery_dir):
                        person_name = os.path.splitext(fn)[0].split('_')[0]

                    if person_name in ('-1', '0000'):
                        continue

                    if max_ids and len(enrolled_ids) >= max_ids and person_name not in enrolled_ids:
                        continue

                    enrolled_ids.add(person_name)
                    img_path = os.path.join(root, fn)

                    pil_img = Image.open(img_path).convert('RGB')
                    img_t = self.transform_base(pil_img).unsqueeze(0)

                    cv_img = cv2.imread(img_path)
                    mask = extract_silhouette_mask(cv_img, target_size=self.silh_size)
                    silh_arr = np.repeat(mask[np.newaxis, np.newaxis, :, :], 4, axis=0)
                    silh_t = torch.from_numpy(silh_arr).unsqueeze(0).float()

                    feat, _ = self.extract_embedding(img_t, silh_t)
                    if person_name not in self.gallery:
                        self.gallery[person_name] = []
                    self.gallery[person_name].append(feat)
                    count += 1

        print(f"[Gallery] Successfully indexed {len(self.gallery)} identities ({count} templates).")
        try:
            torch.save(self.gallery, cache_path)
            print(f"[Gallery Cache] Saved gallery cache to {cache_path}")
        except Exception as e:
            pass

    @torch.no_grad()
    def extract_embedding(self, img_tensor, silh_tensor):
        """Passes tensors through TransGait-EVT and returns L2-normalized 512-d embedding."""
        img_t = img_tensor.to(self.device)
        silh_t = silh_tensor.to(self.device)

        out = self.model(img_t, silh_t)
        emb = out['fused'].cpu().numpy()[0]
        emb = emb / max(np.linalg.norm(emb), 1e-12)

        # Attention weights over [Head, Torso, Legs, Gait]
        attn = out.get('attn_weights', None)
        attn_weights = None
        if attn is not None:
            try:
                # attn shape [batch, 4, 4] -> average over rows: shape [4]
                attn_weights = attn[0].mean(dim=0).cpu().numpy().flatten()
            except Exception:
                attn_weights = None

        return emb, attn_weights

    def match_against_gallery(self, query_emb):
        """
        Matches query embedding against enrolled gallery using EVT Weibull probability calibration.
        Returns: best_name, calibrated_prob, min_dist
        """
        if not self.gallery:
            return "Unenrolled (Empty Gallery)", 0.0, 1.0

        best_name = "STRANGER"
        min_dist = 999.0

        for name, template_list in self.gallery.items():
            templates = np.array(template_list)
            cos_sims = np.dot(templates, query_emb)
            dists = 1.0 - cos_sims
            curr_min = float(np.min(dists))
            if curr_min < min_dist:
                min_dist = curr_min
                best_name = name

        # EVT Weibull Calibration
        prob = float(self.evt_calibrator.calibrate(min_dist))
        return best_name, prob, min_dist

    def detect_pedestrians(self, frame):
        """Detects people in frame and applies Non-Maximum Suppression."""
        h, w = frame.shape[:2]
        # Resize large frames for fast real-time detection
        scale = 1.0
        if w > 800:
            scale = 800.0 / w
            small_frame = cv2.resize(frame, (800, int(h * scale)))
        else:
            small_frame = frame

        rects, weights = self.hog.detectMultiScale(
            small_frame,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05
        )

        bboxes = []
        for (rx, ry, rw, rh) in rects:
            # Rescale back to original coordinates
            x = int(rx / scale)
            y = int(ry / scale)
            bw = int(rw / scale)
            bh = int(rh / scale)
            bboxes.append((x, y, bw, bh))

        # Fallback: If no pedestrian detected but frame is a pre-cropped pedestrian (H > W or small resolution)
        if len(bboxes) == 0 and (h >= w or (w <= 300 and h <= 600)):
            bboxes.append((0, 0, w, h))

        return bboxes

    def process_frame(self, frame, track_manager):
        """
        Processes a single video/camera frame in real time:
        1. Detects pedestrians.
        2. Updates track buffers.
        3. Extracts TransGait-EVT embeddings.
        4. Matches against enrolled watchlist with EVT calibration.
        5. Renders HUD annotations.
        """
        bboxes = self.detect_pedestrians(frame)
        active_tracks = track_manager.update_tracks(bboxes, frame)

        results = []
        for track_id, (x, y, w, h) in active_tracks.items():
            img_t, silh_t = track_manager.get_tracklet_tensors(track_id, self.transform_base)
            if img_t is None or silh_t is None:
                continue

            emb, attn_weights = self.extract_embedding(img_t, silh_t)
            match_name, prob, dist = self.match_against_gallery(emb)

            is_match = (prob >= self.threshold) and (match_name != "STRANGER")
            results.append({
                'track_id': track_id,
                'bbox': (x, y, w, h),
                'name': match_name,
                'prob': prob,
                'dist': dist,
                'is_match': is_match,
                'attn_weights': attn_weights,
                'img_tensor': img_t,
                'silh_tensor': silh_t
            })

        return results

    def draw_hud(self, frame, results, fps=0.0):
        """Renders bounding boxes, match banners, EVT status, and attention HUD adaptively."""
        out_frame = frame.copy()
        h_f, w_f = frame.shape[:2]
        
        # Adaptive scaling based on resolution
        scale = max(0.35, min(1.2, w_f / 720.0))
        font_thick = max(1, int(round(scale * 1.5)))
        line_thick = max(1, int(round(scale * 2.0)))

        for res in results:
            x, y, w, h = res['bbox']
            is_match = res['is_match']
            name = res['name']
            prob = res['prob']
            dist = res['dist']

            # Green for enrolled true match; Red for stranger/rejected
            box_color = (0, 200, 0) if is_match else (0, 0, 220)

            # Draw bounding box
            cv2.rectangle(out_frame, (x, y), (x + w, y + h), box_color, line_thick)

            # Label banner
            if is_match:
                label = f"{name} ({prob*100:.1f}%)"
                sub_label = f"ACCEPT | d={dist:.3f}"
            else:
                label = "STRANGER"
                sub_label = f"REJECT | d={dist:.3f}"

            banner_h = int(28 * scale)
            cv2.rectangle(out_frame, (x, max(0, y - banner_h)), (min(w_f, x + max(w, int(160 * scale))), y), box_color, -1)
            cv2.putText(out_frame, label, (x + 3, max(int(10 * scale), y - int(10 * scale))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (255, 255, 255), font_thick)
            cv2.putText(out_frame, sub_label, (x + 3, max(int(20 * scale), y - int(2 * scale))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, (255, 255, 255), 1)

            # Attention weights bar below bbox
            attn = res.get('attn_weights', None)
            if attn is not None and len(attn) == 4:
                wh, wt, wl, wg = attn
                bar_text = f"Attn: H:{wh*100:.0f}% T:{wt*100:.0f}% L:{wl*100:.0f}% G:{wg*100:.0f}%"
                cv2.putText(out_frame, bar_text, (x, min(h_f - 4, y + h + int(14 * scale))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32 * scale, (0, 255, 255), 1)

        # Top Overlay Panel
        panel_w = int(min(320 * scale, w_f - 10))
        panel_h = int(65 * scale)
        if panel_w > 80 and panel_h > 30:
            cv2.rectangle(out_frame, (5, 5), (5 + panel_w, 5 + panel_h), (20, 20, 20), -1)
            cv2.rectangle(out_frame, (5, 5), (5 + panel_w, 5 + panel_h), (0, 255, 180), 1)
            cv2.putText(out_frame, "TransGait-EVT Live", (10, int(22 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (0, 255, 180), font_thick)
            cv2.putText(out_frame, f"FPS: {fps:.1f} | Dev: {self.device.upper()}", (10, int(38 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, (255, 255, 255), 1)
            cv2.putText(out_frame, f"Gallery: {len(self.gallery)} | Tau: {self.threshold:.2f}", (10, int(54 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, (200, 200, 200), 1)

        return out_frame


def run_realtime_pipeline(source=0, weights=None, gallery=None, threshold=0.70, output=None, no_display=False):
    """
    Launches real-time processing loop on video, camera, or image input.
    """
    engine = RealTimeTransGaitReID(
        weights_path=weights,
        gallery_dir=gallery,
        threshold=threshold
    )

    track_manager = RealTimeTrackletManager(max_seq_len=4)

    # 1. Single Image Mode
    if isinstance(source, str) and os.path.isfile(source) and source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        print(f"\n[Mode] Processing Single Image: {source}")
        frame = cv2.imread(source)
        if frame is None:
            print(f"[Error] Failed to load image: {source}")
            return

        t0 = time.time()
        results = engine.process_frame(frame, track_manager)
        proc_time = (time.time() - t0) * 1000

        annotated = engine.draw_hud(frame, results, fps=1000.0 / max(proc_time, 1e-4))
        print(f"Processed in {proc_time:.1f}ms - Detected {len(results)} pedestrians:")
        for r in results:
            print(f"  -> Track {r['track_id']}: {r['name']} (Match Prob: {r['prob']*100:.2f}%, Cosine Dist: {r['dist']:.4f})")

        out_path = output or "output_reid.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"[Saved] Output image saved to {out_path}")

        if not no_display:
            cv2.imshow("TransGait-EVT Single Image Test", annotated)
            print("[Info] Press any key to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # 2. Video / Camera Stream Mode
    is_cam = False
    try:
        source_id = int(source)
        cap = cv2.VideoCapture(source_id)
        is_cam = True
        print(f"\n[Mode] Opening Live Camera Feed (Device #{source_id})...")
    except ValueError:
        cap = cv2.VideoCapture(source)
        print(f"\n[Mode] Opening Video File: {source}...")

    if not cap.isOpened():
        print(f"[Error] Could not open video source: {source}")
        return

    # Video Writer setup if requested
    writer = None
    if output:
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output, fourcc, fps_in, (w_in, h_in))
        print(f"[Record] Writing processed video to {output} ({w_in}x{h_in} @ {fps_in:.1f} FPS)...")

    print("[Ready] Running real-time inference loop.")
    print("  -> Press 'q' to quit.")
    print("  -> Press 's' to save current snapshot.")
    print("  -> Press 'e' to enroll the first detected person in current frame.")

    fps_tracker = deque(maxlen=30)
    frame_idx = 0

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[End] Video stream finished.")
            break

        frame_idx += 1
        results = engine.process_frame(frame, track_manager)

        elapsed = time.time() - t_start
        fps_tracker.append(1.0 / max(elapsed, 1e-4))
        avg_fps = float(np.mean(fps_tracker))

        annotated = engine.draw_hud(frame, results, fps=avg_fps)

        if writer:
            writer.write(annotated)

        if not no_display:
            cv2.imshow("TransGait-EVT Real-Time Re-ID", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[User] Quit command received.")
                break
            elif key == ord('s'):
                snap_name = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_name, annotated)
                print(f"[Snapshot] Saved {snap_name}")
            elif key == ord('e'):
                if results:
                    target = results[0]
                    name_input = f"Enrolled_Target_{len(engine.gallery)+1}"
                    engine.enroll_person(name_input, target['img_tensor'], target['silh_tensor'])
                else:
                    print("[Warning] No person detected in current frame to enroll.")
        else:
            if frame_idx % 30 == 0:
                print(f"  Frame {frame_idx:04d} | FPS: {avg_fps:.1f} | Detections: {len(results)}")

    cap.release()
    if writer:
        writer.release()
    if not no_display:
        cv2.destroyAllWindows()
    print("[Done] Real-time inference finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TransGait-EVT Real-Time Re-ID Pipeline")
    parser.add_argument('--source', type=str, default='0', help="Camera index ('0'), video file ('test.mp4'), or image ('test.jpg')")
    parser.add_argument('--weights', type=str, default=None, help="Path to trained weights .pth file")
    parser.add_argument('--gallery', type=str, default='query', help="Path to enrolled gallery image directory")
    parser.add_argument('--threshold', type=float, default=0.70, help="EVT acceptance probability threshold (0.0 to 1.0)")
    parser.add_argument('--output', type=str, default=None, help="Path to save output annotated video or image")
    parser.add_argument('--no_display', action='store_true', help="Run in headless mode without GUI window")
    args = parser.parse_args()

    run_realtime_pipeline(
        source=args.source,
        weights=args.weights,
        gallery=args.gallery,
        threshold=args.threshold,
        output=args.output,
        no_display=args.no_display
    )
