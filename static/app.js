/* ============================================================
   KLERT Autonomous Clinical AI Doctor Assistant Application Logic
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
    if (window.lucide) {
        window.lucide.createIcons();
    }

    // ============================================================
    // 0. DEMO AUTHENTICATION LOGIN SYSTEM
    // ============================================================
    const authOverlay = document.getElementById("authOverlay");
    const loginForm = document.getElementById("loginForm");
    const usernameInput = document.getElementById("usernameInput");
    const userRoleSelect = document.getElementById("userRoleSelect");
    const loginSubmitBtn = document.getElementById("loginSubmitBtn");
    const autoFillBtn = document.getElementById("autoFillBtn");
    const appHeader = document.getElementById("appHeader");
    const cockpitContainer = document.getElementById("cockpitContainer");
    const activeDoctorName = document.getElementById("activeDoctorName");
    const logoutBtn = document.getElementById("logoutBtn");

    // Check if session is already authenticated
    const savedUser = sessionStorage.getItem("klert_doctor_user");
    if (savedUser) {
        authenticateUser(savedUser);
    }

    if (autoFillBtn) {
        autoFillBtn.addEventListener("click", () => {
            userRoleSelect.value = "lead";
            usernameInput.value = "dr.anandita@deepcortex.ai";
            document.getElementById("passwordInput").value = "klert2026";
        });
    }

    if (loginForm) {
        loginForm.addEventListener("submit", (e) => {
            e.preventDefault();
            // Support both old btn-cyber-primary and new btn-login-submit
            loginSubmitBtn.disabled = true;
            const btnContent = loginSubmitBtn.querySelector(".btn-content");
            const btnLoader = loginSubmitBtn.querySelector(".btn-loader");
            if (btnContent) btnContent.style.display = "none";
            if (btnLoader) btnLoader.style.display = "inline-flex";

            setTimeout(() => {
                const selectedRoleText = userRoleSelect.options[userRoleSelect.selectedIndex].text.split("—")[0].trim();
                sessionStorage.setItem("klert_doctor_user", selectedRoleText);
                authenticateUser(selectedRoleText);
                btnSubmitReset();
            }, 700);
        });
    }

    function btnSubmitReset() {
        if (!loginSubmitBtn) return;
        loginSubmitBtn.disabled = false;
        const btnContent = loginSubmitBtn.querySelector(".btn-content");
        const btnLoader = loginSubmitBtn.querySelector(".btn-loader");
        if (btnContent) btnContent.style.display = "inline-flex";
        if (btnLoader) btnLoader.style.display = "none";
    }

    function authenticateUser(doctorName) {
        if (authOverlay) authOverlay.style.display = "none";
        if (appHeader) appHeader.style.display = "block";
        if (cockpitContainer) cockpitContainer.style.display = "block";
        if (activeDoctorName) activeDoctorName.textContent = doctorName;
        if (window.lucide) window.lucide.createIcons();
    }

    if (logoutBtn) {
        logoutBtn.addEventListener("click", () => {
            sessionStorage.removeItem("klert_doctor_user");
            if (authOverlay) authOverlay.style.display = "flex";
            if (appHeader) appHeader.style.display = "none";
            if (cockpitContainer) cockpitContainer.style.display = "none";
        });
    }

    // ============================================================
    // 1. NEURAL PARTICLE MATRIX CANVAS (Background Effect)
    // ============================================================
    const canvas = document.getElementById("neuralCanvas");
    if (canvas) {
        const ctx = canvas.getContext("2d");
        let width = canvas.width = window.innerWidth;
        let height = canvas.height = window.innerHeight;

        window.addEventListener("resize", () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        });

        const particles = [];
        const particleCount = Math.min(Math.floor(width / 18), 75);
        let mouse = { x: null, y: null, radius: 150 };

        window.addEventListener("mousemove", (e) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
        });

        class Particle {
            constructor() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.vx = (Math.random() - 0.5) * 0.8;
                this.vy = (Math.random() - 0.5) * 0.8;
                this.radius = Math.random() * 2 + 1;
                this.color = Math.random() > 0.4 ? "#38bdf8" : "#ec4899";
            }

            update() {
                this.x += this.vx;
                this.y += this.vy;

                if (this.x < 0 || this.x > width) this.vx *= -1;
                if (this.y < 0 || this.y > height) this.vy *= -1;

                if (mouse.x && mouse.y) {
                    let dx = mouse.x - this.x;
                    let dy = mouse.y - this.y;
                    let dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < mouse.radius) {
                        this.x -= (dx / dist) * 0.5;
                        this.y -= (dy / dist) * 0.5;
                    }
                }
            }

            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = this.color;
                ctx.shadowBlur = 8;
                ctx.shadowColor = this.color;
                ctx.fill();
                ctx.shadowBlur = 0;
            }
        }

        for (let i = 0; i < particleCount; i++) {
            particles.push(new Particle());
        }

        function animateParticles() {
            ctx.clearRect(0, 0, width, height);

            for (let i = 0; i < particles.length; i++) {
                particles[i].update();
                particles[i].draw();

                for (let j = i + 1; j < particles.length; j++) {
                    let dx = particles[i].x - particles[j].x;
                    let dy = particles[i].y - particles[j].y;
                    let dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist < 130) {
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        let alpha = (1 - dist / 130) * 0.25;
                        ctx.strokeStyle = i % 2 === 0 ? `rgba(56, 189, 248, ${alpha})` : `rgba(236, 72, 153, ${alpha})`;
                        ctx.lineWidth = 0.8;
                        ctx.stroke();
                    }
                }
            }
            requestAnimationFrame(animateParticles);
        }
        animateParticles();
    }

    // ============================================================
    // 2. 3D HOLOGRAM CARD TILT EFFECT
    // ============================================================
    const hologramCard = document.getElementById("hologramCard");
    if (hologramCard) {
        hologramCard.addEventListener("mousemove", (e) => {
            const rect = hologramCard.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = ((y - centerY) / centerY) * -14;
            const rotateY = ((x - centerX) / centerX) * 14;

            hologramCard.style.transform = `rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
        });

        hologramCard.addEventListener("mouseleave", () => {
            hologramCard.style.transform = `rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)`;
        });
    }

    // ============================================================
    // 3. WORKSTATION MULTI-TAB SWITCHING LOGIC
    // ============================================================
    const tabNavButtons = document.querySelectorAll(".tab-nav-btn");
    const cockpitPanes = document.querySelectorAll(".cockpit-pane");

    function switchCockpitTab(tabId) {
        tabNavButtons.forEach(btn => {
            if (btn.getAttribute("data-tab") === tabId) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });

        cockpitPanes.forEach(pane => {
            if (pane.id === tabId) {
                pane.classList.add("active");
            } else {
                pane.classList.remove("active");
            }
        });

        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    tabNavButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            switchCockpitTab(targetTab);
        });
    });

    const jumpToDiagBtn = document.getElementById("jumpToDiagBtn");
    const jumpToProgBtn = document.getElementById("jumpToProgBtn");
    const jumpToBenchBtn = document.getElementById("jumpToBenchBtn");
    const jumpToAssistBtn = document.getElementById("jumpToAssistBtn");

    if (jumpToDiagBtn) jumpToDiagBtn.addEventListener("click", () => switchCockpitTab("diagnosticTab"));
    if (jumpToProgBtn) jumpToProgBtn.addEventListener("click", () => switchCockpitTab("progressionTab"));
    if (jumpToBenchBtn) jumpToBenchBtn.addEventListener("click", () => switchCockpitTab("benchmarksTab"));
    if (jumpToAssistBtn) jumpToAssistBtn.addEventListener("click", () => switchCockpitTab("assistantTab"));

    // Quick Clinical Case Library Launcher from Overview
    document.querySelectorAll(".quick-case-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const sampleName = btn.getAttribute("data-sample");
            switchCockpitTab("diagnosticTab");
            setTimeout(() => {
                const sampleCard = document.querySelector(`.sample-card[data-sample="${sampleName}"]`);
                if (sampleCard) {
                    sampleCard.click();
                }
            }, 100);
        });
    });

    // ============================================================
    // 4. MODEL STATE & DIAGNOSTIC STUDIO LOGIC
    // ============================================================
    let selectedFile = null;
    let selectedSample = null;
    let selectedModel = "cnn";
    let currentResultData = null;

    const modelButtons = document.querySelectorAll(".model-btn");
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const browseBtn = document.getElementById("browseBtn");
    const uploadPreviewContainer = document.getElementById("uploadPreviewContainer");
    const uploadPreviewImg = document.getElementById("uploadPreviewImg");
    const previewFilename = document.getElementById("previewFilename");
    const removeFileBtn = document.getElementById("removeFileBtn");
    const samplesGrid = document.getElementById("samplesGrid");
    const analyzeBtn = document.getElementById("analyzeBtn");

    const emptyState = document.getElementById("emptyState");
    const resultsContent = document.getElementById("resultsContent");
    const diagnosisPill = document.getElementById("diagnosisPill");
    const diagnosisLabel = document.getElementById("diagnosisLabel");
    const confidenceNum = document.getElementById("confidenceNum");
    const activeModelTag = document.getElementById("activeModelTag");
    const targetLayerTag = document.getElementById("targetLayerTag");
    const probabilityList = document.getElementById("probabilityList");
    const devicePill = document.getElementById("devicePill");

    const imgCombined = document.getElementById("imgCombined");
    const imgOverlay = document.getElementById("imgOverlay");
    const imgHeatmap = document.getElementById("imgHeatmap");
    const imgOriginal = document.getElementById("imgOriginal");
    const blendBaseImg = document.getElementById("blendBaseImg");
    const blendHeatmapImg = document.getElementById("blendHeatmapImg");
    const blendSlider = document.getElementById("blendSlider");
    const blendValText = document.getElementById("blendValText");
    const tabButtons = document.querySelectorAll(".tab-btn");

    const imageModal = document.getElementById("imageModal");
    const modalImg = document.getElementById("modalImg");
    const modalTitle = document.getElementById("modalTitle");
    const closeModalBtn = document.getElementById("closeModalBtn");
    const fullscreenBtn = document.getElementById("fullscreenBtn");

    async function checkSystemHealth() {
        try {
            const res = await fetch("/api/health");
            if (res.ok) {
                const data = await res.json();
                if (devicePill) devicePill.textContent = data.device.toUpperCase();
            }
        } catch (e) { console.warn("Health check failed:", e); }
    }

    async function loadSamples() {
        try {
            const res = await fetch("/api/samples");
            if (res.ok) {
                const data = await res.json();
                if (data.samples && data.samples.length > 0) renderSamples(data.samples);
            }
        } catch (e) { console.error("Samples load error:", e); }
    }

    function renderSamples(samples) {
        if (!samplesGrid) return;
        samplesGrid.innerHTML = "";
        samples.forEach(sample => {
            const card = document.createElement("div");
            card.className = "sample-card";
            card.setAttribute("data-sample", sample.filename);

            let tagClass = "tag-general";
            const fname = sample.filename.toLowerCase();
            if (fname.includes("1")) tagClass = "tag-glioma";
            else if (fname.includes("2")) tagClass = "tag-meningioma";
            else if (fname.includes("3")) tagClass = "tag-notumor";
            else if (fname.includes("4")) tagClass = "tag-pituitary";

            card.innerHTML = `
                <div class="sample-img-thumb"><img src="${sample.url}" alt="${sample.label}"></div>
                <div class="sample-meta">
                    <span class="sample-title">${sample.label}</span>
                    <span class="sample-tag ${tagClass}">${sample.badge}</span>
                </div>
            `;

            card.addEventListener("click", () => selectSample(sample.filename, card));
            samplesGrid.appendChild(card);
        });
    }

    modelButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            modelButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            selectedModel = btn.getAttribute("data-model");
            if (selectedFile || selectedSample) runAnalysis();
        });
    });

    if (browseBtn) browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
    if (dropZone) {
        dropZone.addEventListener("click", () => { if (!selectedFile) fileInput.click(); });
        ["dragenter", "dragover"].forEach(evt => dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add("dragover"); }));
        ["dragleave", "drop"].forEach(evt => dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.remove("dragover"); }));
        dropZone.addEventListener("drop", (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) handleFileSelect(e.dataTransfer.files[0]);
        });
    }

    if (fileInput) fileInput.addEventListener("change", (e) => { if (e.target.files && e.target.files.length > 0) handleFileSelect(e.target.files[0]); });

    function handleFileSelect(file) {
        if (!file.type.match("image.*")) { alert("Select a valid JPG or PNG MRI image."); return; }
        selectedFile = file;
        selectedSample = null;
        clearActiveSampleCards();

        const reader = new FileReader();
        reader.onload = (e) => {
            uploadPreviewImg.src = e.target.result;
            previewFilename.textContent = file.name;
            uploadPreviewContainer.style.display = "flex";
            dropZone.querySelector(".drop-zone-content").style.display = "none";
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    if (removeFileBtn) removeFileBtn.addEventListener("click", (e) => { e.stopPropagation(); resetFileInput(); });

    function resetFileInput() {
        selectedFile = null;
        if (fileInput) fileInput.value = "";
        uploadPreviewContainer.style.display = "none";
        dropZone.querySelector(".drop-zone-content").style.display = "flex";
        if (!selectedSample) analyzeBtn.disabled = true;
    }

    function selectSample(filename, cardElement) {
        selectedSample = filename;
        selectedFile = null;
        resetFileInput();

        clearActiveSampleCards();
        cardElement.classList.add("active");
        analyzeBtn.disabled = false;
        runAnalysis();
    }

    function clearActiveSampleCards() {
        document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
    }

    if (analyzeBtn) analyzeBtn.addEventListener("click", runAnalysis);

    async function runAnalysis() {
        if (!selectedFile && !selectedSample) return;
        setLoading(true);

        const formData = new FormData();
        if (selectedFile) formData.append("file", selectedFile);
        else if (selectedSample) formData.append("sample_name", selectedSample);
        formData.append("model_type", selectedModel);

        try {
            const res = await fetch("/api/predict", { method: "POST", body: formData });
            if (!res.ok) { const err = await res.json(); throw new Error(err.detail || "Prediction error"); }
            const data = await res.json();
            currentResultData = data;
            renderResults(data);
        } catch (error) {
            console.error("Analysis Error:", error);
            alert(`Analysis Failed: ${error.message}`);
        } finally { setLoading(false); }
    }

    function setLoading(isLoading) {
        const btnContent = analyzeBtn.querySelector(".btn-content");
        const btnLoader = analyzeBtn.querySelector(".btn-loader");
        if (isLoading) {
            analyzeBtn.disabled = true;
            btnContent.style.display = "none";
            btnLoader.style.display = "inline-flex";
        } else {
            analyzeBtn.disabled = false;
            btnContent.style.display = "inline-flex";
            btnLoader.style.display = "none";
        }
    }

    function renderResults(data) {
        emptyState.style.display = "none";
        resultsContent.style.display = "flex";

        diagnosisLabel.textContent = data.label.toUpperCase();
        diagnosisPill.className = `diagnosis-type-pill ${data.prediction}`;
        confidenceNum.textContent = `${data.confidence.toFixed(1)}%`;

        activeModelTag.innerHTML = `<i data-lucide="cpu"></i> ${data.model_name}`;
        targetLayerTag.innerHTML = `<i data-lucide="layers"></i> ${data.metadata.target_layer}`;

        imgCombined.src = data.images.combined;
        imgOverlay.src = data.images.overlay;
        imgHeatmap.src = data.images.heatmap;
        imgOriginal.src = data.images.original;

        blendBaseImg.src = data.images.original;
        blendHeatmapImg.src = data.images.heatmap;
        updateBlendOpacity(blendSlider.value);

        renderProbabilityBars(data.probabilities, data.prediction);
        if (window.lucide) window.lucide.createIcons();
    }

    function renderProbabilityBars(probs, predictedClass) {
        probabilityList.innerHTML = "";
        const classNames = { glioma: "Glioma", meningioma: "Meningioma", notumor: "No Tumor (Healthy)", pituitary: "Pituitary Tumor" };

        for (const [cls, pct] of Object.entries(probs)) {
            const isWinner = cls === predictedClass;
            const row = document.createElement("div");
            row.className = "prob-row";
            row.innerHTML = `
                <div class="prob-header">
                    <span class="prob-name" style="${isWinner ? 'color: var(--text-primary); font-weight: 700;' : ''}">${classNames[cls] || cls} ${isWinner ? '✓' : ''}</span>
                    <span class="prob-pct">${pct.toFixed(1)}%</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill fill-${cls}" style="width: 0%;"></div>
                </div>
            `;
            probabilityList.appendChild(row);
            setTimeout(() => {
                const fill = row.querySelector(".prob-bar-fill");
                if (fill) fill.style.width = `${Math.max(pct, 1.5)}%`;
            }, 50);
        }
    }

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            tabButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            switchView(btn.getAttribute("data-view"));
        });
    });

    function switchView(viewName) {
        document.querySelectorAll(".view-pane").forEach(pane => pane.classList.remove("active"));
        const paneMap = { combined: "paneCombined", overlay: "paneOverlay", blend: "paneBlend", heatmap: "paneHeatmap", original: "paneOriginal" };
        const target = document.getElementById(paneMap[viewName]);
        if (target) target.classList.add("active");
    }

    if (blendSlider) blendSlider.addEventListener("input", (e) => updateBlendOpacity(e.target.value));

    function updateBlendOpacity(val) {
        if (blendHeatmapImg) blendHeatmapImg.style.opacity = val / 100;
        if (blendValText) blendValText.textContent = `${val}%`;
    }

    if (fullscreenBtn) {
        fullscreenBtn.addEventListener("click", () => {
            if (!currentResultData) return;
            modalImg.src = currentResultData.images.combined;
            modalTitle.textContent = "Detailed Diagnostic MRI Inspection";
            imageModal.style.display = "flex";
        });
    }

    if (closeModalBtn) closeModalBtn.addEventListener("click", () => { imageModal.style.display = "none"; });
    if (imageModal) imageModal.addEventListener("click", (e) => { if (e.target === imageModal) imageModal.style.display = "none"; });

    // ============================================================
    // 5. PROGRESSION STUDIO LOGIC (DICOM Dataset)
    // ============================================================
    const patientSelect = document.getElementById("patientSelect");
    const loadProgressionBtn = document.getElementById("loadProgressionBtn");
    const progEmptyState = document.getElementById("progEmptyState");
    const progressionDashboard = document.getElementById("progressionDashboard");
    const progSummaryRow = document.getElementById("progSummaryRow");
    const timepointsList = document.getElementById("timepointsList");
    let progressionChart = null;
    let currentProgData = null;

    async function loadPatients() {
        if (!patientSelect) return;
        try {
            const res = await fetch("/api/progression/patients");
            if (!res.ok) throw new Error("Patients endpoint failed");
            const data = await res.json();
            
            patientSelect.innerHTML = `<option value="">-- Select Patient (${data.count} available) --</option>`;
            data.patients.forEach(pid => {
                const opt = document.createElement("option");
                opt.value = pid;
                opt.textContent = `Patient ${pid}`;
                patientSelect.appendChild(opt);
            });
        } catch (e) {
            console.warn("Patients load error:", e);
            if (patientSelect) patientSelect.innerHTML = `<option value="">Dataset loading or unavailable</option>`;
        }
    }

    if (patientSelect) {
        patientSelect.addEventListener("change", () => {
            if (loadProgressionBtn) loadProgressionBtn.disabled = !patientSelect.value;
        });
    }

    if (loadProgressionBtn) loadProgressionBtn.addEventListener("click", runProgressionAnalysis);

    async function runProgressionAnalysis() {
        const patientId = patientSelect.value;
        if (!patientId) return;

        loadProgressionBtn.disabled = true;
        loadProgressionBtn.innerHTML = `<span class="spinner"></span> Analyzing DICOM Series...`;

        try {
            const res = await fetch(`/api/progression/${patientId}/analyze`);
            if (!res.ok) throw new Error("Progression analysis failed");
            const data = await res.json();
            currentProgData = data;
            renderProgressionDashboard(data);
        } catch (err) {
            alert(`Progression Analysis Error: ${err.message}`);
        } finally {
            loadProgressionBtn.disabled = false;
            loadProgressionBtn.innerHTML = `<i data-lucide="activity"></i> Analyze Longitudinal Progression`;
            if (window.lucide) window.lucide.createIcons();
        }
    }

    function renderProgressionDashboard(data) {
        if (progEmptyState) progEmptyState.style.display = "none";
        if (progressionDashboard) progressionDashboard.style.display = "flex";

        const summary = data.summary || {};
        const timepoints = data.timepoints || [];

        const trajColor = summary.trajectory_color || "secondary";
        const trajLabel = summary.trajectory || "N/A";
        const delta = summary.total_delta_cm3 !== undefined ? `${summary.total_delta_cm3 > 0 ? '+' : ''}${summary.total_delta_cm3} cm³` : "N/A";
        const pct = summary.total_pct_change !== undefined ? `${summary.total_pct_change > 0 ? '+' : ''}${summary.total_pct_change}%` : "N/A";

        if (progSummaryRow) {
            progSummaryRow.innerHTML = `
                <div class="prog-stat-card">
                    <span class="prog-stat-lbl">Trajectory Status</span>
                    <span class="badge-trajectory ${trajColor}">${trajLabel}</span>
                </div>
                <div class="prog-stat-card">
                    <span class="prog-stat-lbl">Baseline Volume</span>
                    <span class="prog-stat-val">${summary.baseline_volume_cm3 || 'N/A'} cm³</span>
                </div>
                <div class="prog-stat-card">
                    <span class="prog-stat-lbl">Latest Volume</span>
                    <span class="prog-stat-val">${summary.latest_volume_cm3 || 'N/A'} cm³</span>
                </div>
                <div class="prog-stat-card">
                    <span class="prog-stat-lbl">Volume Change (Δ)</span>
                    <span class="prog-stat-val ${summary.total_delta_cm3 > 0 ? 'text-pink' : 'text-cyan'}">${delta} (${pct})</span>
                </div>
            `;
        }

        if (timepointsList) {
            timepointsList.innerHTML = "";
            timepoints.forEach((tp, idx) => {
                const card = document.createElement("div");
                card.className = "tp-card";
                card.innerHTML = `
                    <div class="tp-header">
                        <span class="tp-date"><i data-lucide="calendar"></i> Session ${idx + 1}: ${tp.date || tp.session}</span>
                        <span class="tp-vol">${tp.volume_cm3 !== null ? tp.volume_cm3 + ' cm³' : 'N/A'}</span>
                    </div>
                    <div class="tp-details">
                        <span>Voxels: ${tp.tumor_voxels || 'N/A'}</span>
                        <span>Slices: ${tp.slices_with_tumor || 0}/${tp.total_slices || 0}</span>
                        <span>Spacing: ${tp.pixel_spacing_mm ? tp.pixel_spacing_mm.join('x') + 'mm' : 'N/A'}</span>
                    </div>
                `;
                timepointsList.appendChild(card);
            });
        }

        const progSlicesGrid = document.getElementById("progSlicesGrid");
        if (progSlicesGrid) {
            progSlicesGrid.innerHTML = "";
            timepoints.forEach((tp, idx) => {
                const isBaseline = idx === 0;
                const isLatest = idx === timepoints.length - 1;
                let tagText = `Session ${idx + 1}`;
                if (isBaseline) tagText = "Baseline Scan";
                else if (isLatest) tagText = "Latest Follow-up";

                const sliceBox = document.createElement("div");
                sliceBox.className = "slice-card-box";

                const imgHtml = tp.slice_image_b64
                    ? `<img src="${tp.slice_image_b64}" alt="MRI Slice ${tp.date}">`
                    : `<div style="color: var(--text-muted); font-size: 0.8rem; padding: 2rem;">No preview slice</div>`;

                sliceBox.innerHTML = `
                    <div class="slice-header-bar">
                        <span class="slice-title"><i data-lucide="calendar"></i> ${tagText} (${tp.date || 'N/A'})</span>
                        <span class="slice-vol-badge">${tp.volume_cm3 || 'N/A'} cm³</span>
                    </div>
                    <div class="slice-img-container">${imgHtml}</div>
                    <div class="slice-footer-info">
                        <span>Peak Tumor Slice: #${tp.peak_slice_index || 1}</span>
                        <span>Red Contour = Mask Area</span>
                    </div>
                `;
                progSlicesGrid.appendChild(sliceBox);
            });
        }

        renderProgressionChart(timepoints);
        if (window.lucide) window.lucide.createIcons();
    }

    function renderProgressionChart(timepoints) {
        const canvas = document.getElementById("progressionChartCanvas");
        if (!canvas || typeof Chart === "undefined") return;

        const validPoints = timepoints.filter(t => t.volume_cm3 !== null);
        const labels = validPoints.map(t => t.date || t.session);
        const volumes = validPoints.map(t => t.volume_cm3);

        if (progressionChart) progressionChart.destroy();

        const ctx = canvas.getContext("2d");
        progressionChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: labels,
                datasets: [{
                    label: "Tumor Volume (cm³)",
                    data: volumes,
                    borderColor: "#38bdf8",
                    backgroundColor: "rgba(56, 189, 248, 0.15)",
                    borderWidth: 3,
                    pointBackgroundColor: "#ec4899",
                    pointRadius: 6,
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: "#94a3b8", font: { family: "Outfit" } } } },
                scales: {
                    x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255, 255, 255, 0.05)" } },
                    y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255, 255, 255, 0.05)" }, title: { display: true, text: "Volume (cm³)", color: "#94a3b8" } }
                }
            }
        });
    }

    // ============================================================
    // 6. CLINICAL AI ASSISTANT STREAM GENERATOR
    // ============================================================
    const generateDiagnosticReportBtn = document.getElementById("generateDiagnosticReportBtn");
    const generateProgressionReportBtn = document.getElementById("generateProgressionReportBtn");
    const assistantOutputBox = document.getElementById("assistantOutputBox");

    if (generateDiagnosticReportBtn) {
        generateDiagnosticReportBtn.addEventListener("click", () => {
            if (!currentResultData) {
                alert("Please run a prediction in the Diagnostic Studio tab first.");
                switchCockpitTab("diagnosticTab");
                return;
            }
            switchCockpitTab("assistantTab");
            streamAIDoctorReport({
                tumor_type: currentResultData.label,
                confidence: currentResultData.confidence,
                probabilities: currentResultData.probabilities,
                model_used: currentResultData.model_used
            });
        });
    }

    if (generateProgressionReportBtn) {
        generateProgressionReportBtn.addEventListener("click", () => {
            if (!currentProgData) {
                alert("Please run a progression analysis in the Progression Studio tab first.");
                switchCockpitTab("progressionTab");
                return;
            }
            switchCockpitTab("assistantTab");
            const summary = currentProgData.summary || {};
            streamAIDoctorReport({
                tumor_type: `Longitudinal Progression File (${currentProgData.patient_id})`,
                confidence: 98.5,
                probabilities: {
                    trajectory: summary.trajectory || "STABLE",
                    baseline_vol: summary.baseline_volume_cm3 || 0,
                    latest_vol: summary.latest_volume_cm3 || 0,
                    pct_change: summary.total_pct_change || 0
                },
                model_used: "Volumetric 3D DICOM Segmentor"
            });
        });
    }

    async function streamAIDoctorReport(payload) {
        assistantOutputBox.innerHTML = `<p class="placeholder-text"><span class="spinner"></span> DeepCortex Clinical AI Doctor Assistant is compiling medical consultation report...</p>`;

        try {
            const res = await fetch("/api/discuss", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!res.ok) throw new Error("Consultation stream failed");

            const reader = res.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let markdownText = "";

            assistantOutputBox.innerHTML = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                markdownText += chunk;
                assistantOutputBox.innerHTML = renderMarkdown(markdownText);
            }
        } catch (err) {
            console.error("AI Doctor Consultation Error:", err);
            assistantOutputBox.innerHTML = `<p style="color: var(--accent);">Consultation Generation Error: ${err.message}</p>`;
        }
    }

    function renderMarkdown(txt) {
        let html = txt
            .replace(/### (.*)/g, '<h3>$1</h3>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/`([^`]+)`/g, '<code>$1</code>');

        html = html.replace(/```text\n([\s\S]*?)\n```/g, '<pre>$1</pre>');
        html = html.replace(/```\n([\s\S]*?)\n```/g, '<pre>$1</pre>');
        html = html.replace(/\n\n/g, '<br><br>');
        return html;
    }

    // Run Initial Loaders
    checkSystemHealth();
    loadSamples();
    loadPatients();
});
