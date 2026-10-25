/* ==============================================================================
   PARICHAY (परिचय) - AI Biometric Identity Discovery Platform Frontend
   ============================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // State variables
    let currentFile = null;
    let selectedSample = null;
    let currentThreshold = 0.70;
    let latestDetections = [];
    let currentFilter = 'all';
    let targetToRegister = null;

    // DOM Elements
    const dropzoneContainer = document.getElementById('dropzoneContainer');
    const mediaFileInput = document.getElementById('mediaFileInput');
    const dropzonePrompt = document.getElementById('dropzonePrompt');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const previewVideo = document.getElementById('previewVideo');
    const btnClearMedia = document.getElementById('btnClearMedia');
    
    const sampleGrid = document.getElementById('sampleGrid');
    const galleryList = document.getElementById('galleryList');
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdValue = document.getElementById('thresholdValue');
    
    const btnRunInference = document.getElementById('btnRunInference');
    const btnSpinner = document.getElementById('btnSpinner');
    const btnRunText = document.getElementById('btnRunText');

    const viewportPlaceholder = document.getElementById('viewportPlaceholder');
    const activeMediaWrapper = document.getElementById('activeMediaWrapper');
    const resultImage = document.getElementById('resultImage');
    const resultVideo = document.getElementById('resultVideo');
    const latencyBadge = document.getElementById('latencyBadge');

    const manifestCountBadge = document.getElementById('manifestCountBadge');
    const rosterContainer = document.getElementById('rosterContainer');
    const explanationText = document.getElementById('explanationText');
    const galleryCount = document.getElementById('galleryCount');
    const deviceText = document.getElementById('deviceText');

    const registrationModal = document.getElementById('registrationModal');
    const modalNameInput = document.getElementById('modalNameInput');
    const btnCancelModal = document.getElementById('btnCancelModal');
    const btnConfirmRegister = document.getElementById('btnConfirmRegister');

    // =========================================================================
    // 1. Initial Load (Stats, Samples, Watchlist)
    // =========================================================================
    fetchStats();
    fetchSamples();
    fetchGallery();

    async function fetchStats() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            if (data.device) deviceText.textContent = `${data.device} ACTIVE`;
            if (data.registered_count !== undefined) galleryCount.textContent = `${data.registered_count} IDs`;
        } catch (err) {
            console.error('Error fetching stats:', err);
        }
    }

    async function fetchSamples() {
        try {
            const res = await fetch('/api/samples');
            const data = await res.json();
            sampleGrid.innerHTML = '';

            if (!data.samples || data.samples.length === 0) {
                sampleGrid.innerHTML = '<div class="section-desc">No preset probes available.</div>';
                return;
            }

            data.samples.forEach((sample, idx) => {
                const card = document.createElement('div');
                card.className = `sample-card ${idx === 0 ? 'selected' : ''}`;
                card.dataset.filename = sample.filename;

                const iconHtml = sample.type === 'video'
                    ? `<div class="sample-type-icon">VIDEO</div>`
                    : `<img src="${sample.url}" alt="${sample.title}" class="sample-thumb">`;

                card.innerHTML = `
                    ${iconHtml}
                    <div class="sample-info">
                        <div class="sample-title">${sample.title}</div>
                        <div class="sample-tag">${sample.tag}</div>
                    </div>
                `;

                card.addEventListener('click', () => {
                    document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('selected'));
                    card.classList.add('selected');
                    selectedSample = sample;
                    currentFile = null;
                    showPreview(sample.url, sample.type === 'video');
                });

                sampleGrid.appendChild(card);

                if (idx === 0) {
                    selectedSample = sample;
                    showPreview(sample.url, sample.type === 'video');
                }
            });
        } catch (err) {
            console.error('Error fetching samples:', err);
        }
    }

    async function fetchGallery() {
        try {
            const res = await fetch('/api/gallery');
            const data = await res.json();
            galleryList.innerHTML = '';

            if (!data.identities || data.identities.length === 0) {
                galleryList.innerHTML = '<div class="section-desc">Watchlist is currently empty.</div>';
                return;
            }

            data.identities.forEach(item => {
                const el = document.createElement('div');
                el.className = 'sample-card';
                el.innerHTML = `
                    <div class="sample-type-icon" style="background: rgba(16, 185, 129, 0.15); color: #10b981;">ENR</div>
                    <div class="sample-info">
                        <div class="sample-title">${item.name}</div>
                        <div class="sample-tag">${item.template_count} template(s)</div>
                    </div>
                `;
                galleryList.appendChild(el);
            });
        } catch (err) {
            console.error('Error fetching gallery:', err);
        }
    }

    function showPreview(url, isVideo) {
        previewContainer.classList.remove('hidden');
        dropzonePrompt.classList.add('hidden');

        if (isVideo) {
            previewVideo.src = url;
            previewVideo.classList.remove('hidden');
            previewImage.classList.add('hidden');
        } else {
            previewImage.src = url;
            previewImage.classList.remove('hidden');
            previewVideo.classList.add('hidden');
        }
    }

    // =========================================================================
    // 2. Tab Navigation
    // =========================================================================
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetId = `content${btn.dataset.tab.charAt(0).toUpperCase() + btn.dataset.tab.slice(1)}`;
            const targetContent = document.getElementById(targetId);
            if (targetContent) targetContent.classList.add('active');
        });
    });

    // =========================================================================
    // 3. Dropzone Media Handling
    // =========================================================================
    dropzoneContainer.addEventListener('click', (e) => {
        if (e.target !== btnClearMedia) mediaFileInput.click();
    });

    mediaFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
    });

    dropzoneContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzoneContainer.classList.add('dragover');
    });

    dropzoneContainer.addEventListener('dragleave', () => dropzoneContainer.classList.remove('dragover'));

    dropzoneContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzoneContainer.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    });

    btnClearMedia.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFile = null;
        selectedSample = null;
        mediaFileInput.value = '';
        previewImage.src = '';
        previewVideo.src = '';
        previewContainer.classList.add('hidden');
        dropzonePrompt.classList.remove('hidden');
        document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('selected'));
    });

    function handleFile(file) {
        currentFile = file;
        selectedSample = null;
        document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('selected'));

        const isVideo = file.type.startsWith('video/') || file.name.match(/\.(mp4|avi|mov|mkv)$/i);
        const objUrl = URL.createObjectURL(file);
        showPreview(objUrl, isVideo);
    }

    // =========================================================================
    // 4. Threshold Slider
    // =========================================================================
    thresholdSlider.addEventListener('input', (e) => {
        currentThreshold = parseFloat(e.target.value);
        thresholdValue.textContent = currentThreshold.toFixed(2);
    });

    // =========================================================================
    // 5. Run PARICHAY Inference
    // =========================================================================
    btnRunInference.addEventListener('click', async () => {
        if (!currentFile && !selectedSample) {
            alert('Please drop an unknown media file or pick a preset probe first.');
            return;
        }

        const isVideo = (currentFile && (currentFile.type.startsWith('video/') || currentFile.name.match(/\.(mp4|avi|mov|mkv)$/i)))
            || (selectedSample && selectedSample.type === 'video');

        btnRunInference.disabled = true;
        btnSpinner.classList.remove('hidden');
        btnRunText.textContent = isVideo ? 'Analyzing Video Tracklets...' : 'Extracting Biometrics...';

        const formData = new FormData();
        formData.append('threshold', currentThreshold);

        let endpoint = '/api/predict_image';
        if (isVideo) {
            endpoint = '/api/predict_video';
            if (currentFile) formData.append('video', currentFile);
            else if (selectedSample) formData.append('sample_name', selectedSample.filename);
        } else {
            if (currentFile) formData.append('file', currentFile);
            else if (selectedSample) formData.append('sample_name', selectedSample.filename);
        }

        try {
            const res = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            const data = await res.json();
            if (!data.success) {
                alert('Analysis error: ' + (data.detail || 'Unknown error'));
                return;
            }

            renderResults(data, isVideo);
        } catch (err) {
            console.error('Error during analysis:', err);
            alert('Server connection error. Ensure PARICHAY backend is active.');
        } finally {
            btnRunInference.disabled = false;
            btnSpinner.classList.add('hidden');
            btnRunText.textContent = 'Analyze with PARICHAY';
        }
    });

    // =========================================================================
    // 6. Render Results & Who is Alike Manifest
    // =========================================================================
    function renderResults(data, isVideo) {
        viewportPlaceholder.classList.add('hidden');
        activeMediaWrapper.classList.remove('hidden');

        if (isVideo) {
            latencyBadge.textContent = `Video Analyzed: ${data.frames_analyzed} frames in ${(data.total_time_ms/1000).toFixed(1)}s`;
            resultVideo.src = data.processed_video_url;
            resultVideo.classList.remove('hidden');
            resultImage.classList.add('hidden');
        } else {
            latencyBadge.textContent = `Latency: ${data.latency_ms} ms (${data.device})`;
            resultImage.src = data.annotated_image;
            resultImage.classList.remove('hidden');
            resultVideo.classList.add('hidden');
        }

        latestDetections = data.detections || [];
        manifestCountBadge.textContent = `${latestDetections.length} Subjects`;
        renderRosterCards();

        if (latestDetections.length > 0) {
            const registeredCount = latestDetections.filter(d => d.is_registered).length;
            const unknownCount = latestDetections.length - registeredCount;
            explanationText.textContent = `PARICHAY processed the scene and discovered ${latestDetections.length} unique subject(s): ${registeredCount} Verified in Watchlist, ${unknownCount} Unregistered / Strangers. Look-alike similarity profiles have been mapped for all unknown subjects.`;
        } else {
            explanationText.textContent = `No pedestrian bounding boxes were resolved in this media stream. Try adjusting perspective or scale.`;
        }
    }

    function renderRosterCards() {
        rosterContainer.innerHTML = '';
        let list = latestDetections;

        if (currentFilter === 'registered') {
            list = list.filter(d => d.is_registered);
        } else if (currentFilter === 'unregistered') {
            list = list.filter(d => !d.is_registered);
        }

        if (list.length === 0) {
            rosterContainer.innerHTML = '<div class="roster-empty">No subjects match the selected filter.</div>';
            return;
        }

        list.forEach(det => {
            const card = document.createElement('div');
            card.className = `subject-card ${det.is_registered ? 'is-registered' : 'is-unregistered'}`;

            const topAlike = (det.alike_profiles && det.alike_profiles.length > 0) ? det.alike_profiles[0] : null;

            // Build 'Who is Alike' Section HTML
            let alikeHtml = '';
            if (topAlike) {
                alikeHtml = `
                    <div class="alike-section">
                        <div class="alike-header">
                            <span>WHO IS ALIKE</span>
                            <span>${topAlike.overall_similarity}% Overall Alignment</span>
                        </div>
                        <div class="alike-candidate">Most Resembles: <strong>${topAlike.name}</strong> (${topAlike.match_nature})</div>
                        <div class="alike-parts-grid">
                            <div class="alike-part-box">
                                <span>Hair/Head</span>
                                <span class="alike-part-val">${topAlike.parts.hair_head}%</span>
                            </div>
                            <div class="alike-part-box">
                                <span>Torso/Dress</span>
                                <span class="alike-part-val">${topAlike.parts.clothing_torso}%</span>
                            </div>
                            <div class="alike-part-box">
                                <span>Shoes/Legs</span>
                                <span class="alike-part-val">${topAlike.parts.footwear_legs}%</span>
                            </div>
                            <div class="alike-part-box">
                                <span>Gait Stride</span>
                                <span class="alike-part-val">${topAlike.parts.temporal_gait}%</span>
                            </div>
                        </div>
                    </div>
                `;
            }

            // Register Button for Unknowns
            const registerBtnHtml = (!det.is_registered)
                ? `<button type="button" class="btn-register-inline" data-id="${det.detection_id}">+ Register as New Identity</button>`
                : '';

            const avatarHtml = det.crop_image
                ? `<img src="${det.crop_image}" class="subject-avatar" alt="${det.identity}">`
                : `<div class="subject-avatar" style="display:flex;align-items:center;justify-content:center;color:#666;">Crop</div>`;

            card.innerHTML = `
                <div class="subject-main">
                    ${avatarHtml}
                    <div class="subject-info">
                        <div class="subject-name">${det.identity}</div>
                        <span class="subject-status-badge ${det.is_registered ? 'badge-reg' : 'badge-unreg'}">
                            ${det.is_registered ? 'REGISTERED' : 'UNREGISTERED / STRANGER'}
                        </span>
                        <div class="subject-scores">EVT Confidence: ${det.match_probability}% | Dist: ${det.cosine_distance.toFixed(4)}</div>
                    </div>
                </div>
                ${alikeHtml}
                ${registerBtnHtml}
            `;

            // Inline Register Click
            const regBtn = card.querySelector('.btn-register-inline');
            if (regBtn) {
                regBtn.addEventListener('click', () => {
                    targetToRegister = det.detection_id;
                    registrationModal.classList.remove('hidden');
                    modalNameInput.value = '';
                    modalNameInput.focus();
                });
            }

            rosterContainer.appendChild(card);
        });
    }

    // =========================================================================
    // 7. Filter Chips
    // =========================================================================
    document.querySelectorAll('.filter-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            currentFilter = chip.dataset.filter;
            renderRosterCards();
        });
    });

    // =========================================================================
    // 8. In-Session Identity Registration Modal
    // =========================================================================
    btnCancelModal.addEventListener('click', () => {
        registrationModal.classList.add('hidden');
        targetToRegister = null;
    });

    btnConfirmRegister.addEventListener('click', async () => {
        const name = modalNameInput.value.trim();
        if (!name || !targetToRegister) {
            alert('Please enter a name for this identity.');
            return;
        }

        const formData = new FormData();
        formData.append('detection_id', targetToRegister);
        formData.append('name', name);

        try {
            const res = await fetch('/api/register_detection', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();
            if (data.success) {
                alert(`Subject officially enrolled as '${name}' into Watchlist!`);
                registrationModal.classList.add('hidden');
                
                // Update local detection state
                const found = latestDetections.find(d => d.detection_id === targetToRegister);
                if (found) {
                    found.is_registered = true;
                    found.identity = name;
                    found.status_badge = 'REGISTERED IDENTITY';
                }
                galleryCount.textContent = `${data.new_gallery_count} IDs`;
                renderRosterCards();
                fetchGallery();
            } else {
                alert('Registration failed: ' + data.detail);
            }
        } catch (err) {
            console.error('Registration error:', err);
            alert('Error contacting server.');
        }
    });
});
