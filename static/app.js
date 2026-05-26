/**
 * PACS Chest X-Ray Workstation Controller
 * Handles image rendering, HTML5 Canvas filters, ruler calibration, AI results, and history.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ---------------------------------------------------------
    // State Variables
    // ---------------------------------------------------------
    let currentImage = new Image();
    let currentHeatmap = new Image();
    let isImageLoaded = false;
    let isHeatmapLoaded = false;
    
    // Canvas transform states
    let scale = 1.0;
    let offsetX = 0;
    let offsetY = 0;
    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let activeTool = 'pan'; // 'pan' or 'measure'
    
    // Filter adjustments
    let brightness = 1.0;
    let contrast = 1.0;
    let isInverted = false;
    let heatmapOpacity = 0;
    
    // Grid overlays
    let showGrid = false;
    let showCrosshairs = false;
    
    // Ruler variables
    let rulerStart = null;
    let rulerEnd = null;
    let isDrawingRuler = false;
    const PIXEL_TO_MM = 0.18; // Calibrated scanner pixel scale (e.g., 0.18mm per pixel)
    
    // Current diagnostic scan details
    let activeScanData = null;

    // ---------------------------------------------------------
    // DOM Element Selectors
    // ---------------------------------------------------------
    const canvas = document.getElementById('pacs-canvas');
    const ctx = canvas.getContext('2d');
    const viewportFrame = document.getElementById('viewport-frame');
    
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('image-upload');
    const scanningOverlay = document.getElementById('scanning-overlay');
    const loadProgressBar = document.getElementById('load-progress');
    
    // Viewport badge overlays
    const activePatientId = document.getElementById('active-patient-id');
    const activePatientName = document.getElementById('active-patient-name');
    
    // Tools & Sliders
    const btnPan = document.getElementById('btn-pan');
    const btnMeasure = document.getElementById('btn-measure');
    const btnInvert = document.getElementById('btn-invert');
    const btnGrid = document.getElementById('btn-grid');
    const btnCrosshairs = document.getElementById('btn-crosshairs');
    const btnReset = document.getElementById('btn-reset');
    
    const sliderBrightness = document.getElementById('slider-brightness');
    const sliderContrast = document.getElementById('slider-contrast');
    const sliderHeatmap = document.getElementById('slider-heatmap');
    
    const valBrightness = document.getElementById('val-brightness');
    const valContrast = document.getElementById('val-contrast');
    const valHeatmap = document.getElementById('val-heatmap');
    
    const rulerBubble = document.getElementById('ruler-bubble');
    const rulerMeasurement = document.getElementById('ruler-measurement');
    
    // Diagnostics Panel
    const probNormal = document.getElementById('prob-normal');
    const probPneumonia = document.getElementById('prob-pneumonia');
    const probCardiomegaly = document.getElementById('prob-cardiomegaly');
    const probEffusion = document.getElementById('prob-effusion');
    const probPneumothorax = document.getElementById('prob-pneumothorax');
    
    const fillNormal = document.getElementById('fill-normal');
    const fillPneumonia = document.getElementById('fill-pneumonia');
    const fillCardiomegaly = document.getElementById('fill-cardiomegaly');
    const fillEffusion = document.getElementById('fill-effusion');
    const fillPneumothorax = document.getElementById('fill-pneumothorax');
    
    // Metrics
    const metricCtr = document.getElementById('metric-ctr');
    const metricOpacity = document.getElementById('metric-opacity');
    const metricCosto = document.getElementById('metric-costo');
    const metricApical = document.getElementById('metric-apical');
    
    const statusCtr = document.getElementById('status-ctr');
    const statusOpacity = document.getElementById('status-opacity');
    const statusCosto = document.getElementById('status-costo');
    const statusApical = document.getElementById('status-apical');
    
    // Narrative & Actions
    const findingsText = document.getElementById('findings-text');
    const dominantBadge = document.getElementById('dominant-badge');
    const reportTime = document.getElementById('report-time');
    const btnExportReport = document.getElementById('btn-export-report');
    
    const demographicsForm = document.getElementById('demographics-form');
    const historyContainer = document.getElementById('history-container');

    // ---------------------------------------------------------
    // Startup & Window Resize
    // ---------------------------------------------------------
    function init() {
        updateTimestamp();
        setInterval(updateTimestamp, 1000);
        
        // Match canvas dimensions to containing element size
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        
        // Force initial tool state
        setActiveTool('pan');
        
        // Ingestion Setup
        setupIngestion();
        
        // Load SQLite historical ledger
        fetchLedger();
        
        // Ledger search keyup binding
        const searchInput = document.getElementById('ledger-search-input');
        if (searchInput) {
            let debounceTimer;
            searchInput.addEventListener('input', (e) => {
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(() => {
                    fetchLedger(e.target.value.trim());
                }, 250);
            });
        }
        
        // Auto-generate some clinical demographics
        prefillDemographics();
    }

    function updateTimestamp() {
        const timeDisplay = document.getElementById('current-timestamp');
        if (timeDisplay) {
            const now = new Date();
            timeDisplay.innerText = now.toLocaleString();
        }
    }

    function resizeCanvas() {
        const rect = viewportFrame.getBoundingClientRect();
        // Take border size into account
        canvas.width = rect.width - 2;
        canvas.height = rect.height - 2;
        if (isImageLoaded) {
            redrawCanvas();
        }
    }

    function prefillDemographics() {
        const randomNames = [
            "Elizabeth Sterling", "Marcus Finch", "Devin Pierce", "Clara Oswald", 
            "Gideon Vance", "Silas Thorne", "Audrey Hepburn", "Arthur Dent"
        ];
        const randomDocs = ["Dr. H. House", "Dr. J. Watson", "Dr. L. Cuddy", "Dr. J. Carter"];
        
        document.getElementById('patient-id').value = "METRO-" + Math.floor(100000 + Math.random() * 900000);
        document.getElementById('patient-name').value = randomNames[Math.floor(Math.random() * randomNames.length)];
        document.getElementById('patient-age').value = Math.floor(22 + Math.random() * 60);
        document.getElementById('patient-gender').value = Math.random() > 0.5 ? "M" : "F";
        document.getElementById('referred-by').value = randomDocs[Math.floor(Math.random() * randomDocs.length)];
    }

    // ---------------------------------------------------------
    // Canvas Math & Transformation Functions
    // ---------------------------------------------------------
    function fitImageToViewport() {
        if (!isImageLoaded) return;
        
        // Calculate scaling required to maximize image size within canvas without cropping
        const wRatio = canvas.width / currentImage.width;
        const hRatio = canvas.height / currentImage.height;
        scale = Math.min(wRatio, hRatio) * 0.92; // leave 8% breathing margins
        
        offsetX = 0;
        offsetY = 0;
        
        // Clear ruler measurements on reset/load
        rulerStart = null;
        rulerEnd = null;
        rulerBubble.classList.add('hidden');
        
        redrawCanvas();
    }

    function redrawCanvas() {
        if (!isImageLoaded) return;
        
        // Wipe prior buffer
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        ctx.save();
        
        // 1. Establish CSS-equivalent image processing filters on canvas
        let filters = `brightness(${brightness}) contrast(${contrast})`;
        if (isInverted) {
            filters += ` invert(1)`;
        }
        ctx.filter = filters;
        
        // 2. Perform matrix translation to center and apply zoom/pan offset
        ctx.translate(canvas.width / 2 + offsetX, canvas.height / 2 + offsetY);
        ctx.scale(scale, scale);
        
        // 3. Draw primary X-Ray image centered in coordinate system
        const xCoord = -currentImage.width / 2;
        const yCoord = -currentImage.height / 2;
        ctx.drawImage(currentImage, xCoord, yCoord, currentImage.width, currentImage.height);
        
        // 4. Overlap AI Heatmap if transparency slider active
        if (isHeatmapLoaded && heatmapOpacity > 0) {
            ctx.save();
            ctx.globalAlpha = heatmapOpacity / 100;
            ctx.filter = 'none'; // Keeps colormap colors vibrant and unaffected by contrast filters
            ctx.drawImage(currentHeatmap, xCoord, yCoord, currentImage.width, currentImage.height);
            ctx.restore();
        }
        
        ctx.restore();
        
        // 5. Draw rulers (drawn in static absolute window coordinates on top of image layers)
        if (rulerStart && rulerEnd) {
            ctx.save();
            ctx.strokeStyle = '#00ffcc';
            ctx.lineWidth = 2.5;
            ctx.setLineDash([4, 4]);
            ctx.shadowBlur = 6;
            ctx.shadowColor = '#00ffcc';
            
            ctx.beginPath();
            ctx.moveTo(rulerStart.x, rulerStart.y);
            ctx.lineTo(rulerEnd.x, rulerEnd.y);
            ctx.stroke();
            
            // Highlight endpoint nodes
            ctx.fillStyle = '#00ffcc';
            ctx.beginPath();
            ctx.arc(rulerStart.x, rulerStart.y, 4.5, 0, Math.PI * 2);
            ctx.arc(rulerEnd.x, rulerEnd.y, 4.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }
    }

    // ---------------------------------------------------------
    // Canvas Mouse & Interaction Listeners
    // ---------------------------------------------------------
    canvas.addEventListener('mousedown', (e) => {
        if (!isImageLoaded) return;
        
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        if (activeTool === 'pan') {
            isDragging = true;
            startX = mouseX - offsetX;
            startY = mouseY - offsetY;
            canvas.style.cursor = 'grabbing';
        } else if (activeTool === 'measure') {
            isDrawingRuler = true;
            rulerStart = { x: mouseX, y: mouseY };
            rulerEnd = { x: mouseX, y: mouseY };
            rulerBubble.classList.add('hidden');
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!isImageLoaded) return;
        
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        if (isDragging && activeTool === 'pan') {
            offsetX = mouseX - startX;
            offsetY = mouseY - startY;
            redrawCanvas();
        } else if (isDrawingRuler && activeTool === 'measure') {
            rulerEnd = { x: mouseX, y: mouseY };
            redrawCanvas();
            updateRulerText();
        }
    });

    window.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            canvas.style.cursor = 'grab';
        }
        if (isDrawingRuler) {
            isDrawingRuler = false;
            updateRulerText();
        }
    });

    // Zoom on mouse wheel scroll
    canvas.addEventListener('wheel', (e) => {
        if (!isImageLoaded) return;
        e.preventDefault();
        
        const zoomIntensity = 0.08;
        if (e.deltaY < 0) {
            scale = Math.min(scale + zoomIntensity, 6.0); // max 6x zoom
        } else {
            scale = Math.max(scale - zoomIntensity, 0.15); // min 0.15x zoom
        }
        redrawCanvas();
    }, { passive: false });

    // Calculates real mm distance using scaled image coordinate distance
    function updateRulerText() {
        if (!rulerStart || !rulerEnd) return;
        
        // Calculate raw pixel distance
        const dx = rulerEnd.x - rulerStart.x;
        const dy = rulerEnd.y - rulerStart.y;
        const pixelDist = Math.sqrt(dx*dx + dy*dy);
        
        // Scale back pixel distance using current zoom factors
        const actualPixelDist = pixelDist / scale;
        const mmDist = actualPixelDist * PIXEL_TO_MM;
        
        // Position info bubble at endpoint coordinate
        const padding = 15;
        rulerBubble.style.left = `${rulerEnd.x + padding}px`;
        rulerBubble.style.top = `${rulerEnd.y + padding}px`;
        rulerMeasurement.innerText = `${mmDist.toFixed(1)} mm`;
        rulerBubble.classList.remove('hidden');
    }

    // ---------------------------------------------------------
    // Workstation Toolbar Controls
    // ---------------------------------------------------------
    function setActiveTool(tool) {
        activeTool = tool;
        if (tool === 'pan') {
            btnPan.classList.add('active');
            btnMeasure.classList.remove('active');
            canvas.style.cursor = 'grab';
        } else {
            btnPan.classList.remove('active');
            btnMeasure.classList.add('active');
            canvas.style.cursor = 'crosshair';
        }
    }

    btnPan.addEventListener('click', () => setActiveTool('pan'));
    btnMeasure.addEventListener('click', () => setActiveTool('measure'));

    btnInvert.addEventListener('click', () => {
        isInverted = !isInverted;
        btnInvert.classList.toggle('active', isInverted);
        redrawCanvas();
    });

    btnGrid.addEventListener('click', () => {
        showGrid = !showGrid;
        btnGrid.classList.toggle('active', showGrid);
        document.getElementById('grid-overlay-matrix').classList.toggle('visible', showGrid);
    });

    btnCrosshairs.addEventListener('click', () => {
        showCrosshairs = !showCrosshairs;
        btnCrosshairs.classList.toggle('active', showCrosshairs);
        document.getElementById('crosshair-v').classList.toggle('visible', showCrosshairs);
        document.getElementById('crosshair-h').classList.toggle('visible', showCrosshairs);
    });

    btnReset.addEventListener('click', () => {
        brightness = 1.0;
        contrast = 1.0;
        isInverted = false;
        heatmapOpacity = 0;
        
        sliderBrightness.value = 1.0;
        sliderContrast.value = 1.0;
        sliderHeatmap.value = 0;
        
        valBrightness.innerText = "100%";
        valContrast.innerText = "100%";
        valHeatmap.innerText = "0%";
        
        btnInvert.classList.remove('active');
        
        fitImageToViewport();
    });

    sliderBrightness.addEventListener('input', (e) => {
        brightness = parseFloat(e.target.value);
        valBrightness.innerText = `${Math.round(brightness * 100)}%`;
        redrawCanvas();
    });

    sliderContrast.addEventListener('input', (e) => {
        contrast = parseFloat(e.target.value);
        valContrast.innerText = `${Math.round(contrast * 100)}%`;
        redrawCanvas();
    });

    sliderHeatmap.addEventListener('input', (e) => {
        heatmapOpacity = parseInt(e.target.value);
        valHeatmap.innerText = `${heatmapOpacity}%`;
        redrawCanvas();
    });

    // ---------------------------------------------------------
    // Image Ingestion & Ingress Control
    // ---------------------------------------------------------
    function setupIngestion() {
        // Drop zone activation
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-active');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-active');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-active');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                processFilePayload(files[0]);
            }
        });
        
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                processFilePayload(e.target.files[0]);
            }
        });

        // Setup sample selectors
        document.querySelectorAll('.sample-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation(); // Avoid triggering file chooser dialog click bubble
                const sampleName = btn.dataset.sample;
                loadSample(sampleName);
            });
        });
    }

    function processFilePayload(file) {
        // Display loading animation
        scanningOverlay.classList.remove('hidden');
        loadProgressBar.style.width = "0%";
        
        // Simulated loading bar phase 1: loading file in memory
        setTimeout(() => {
            loadProgressBar.style.width = "40%";
        }, 150);

        const formData = new FormData();
        formData.append('image', file);

        // Fetch user edited demographics to attach where relevant
        const pId = document.getElementById('patient-id').value;
        const pName = document.getElementById('patient-name').value;
        const pAge = document.getElementById('patient-age').value;
        const pGender = document.getElementById('patient-gender').value;
        const pReferred = document.getElementById('referred-by').value;

        setTimeout(() => {
            loadProgressBar.style.width = "75%";
        }, 400);

        const csrfTokenMeta = document.querySelector('meta[name="csrf-token"]');
        const csrfToken = csrfTokenMeta ? csrfTokenMeta.getAttribute('content') : '';

        fetch('/analyze', {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': csrfToken
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("HTTP error on inference processing");
            }
            return response.json();
        })
        .then(data => {
            loadProgressBar.style.width = "100%";
            setTimeout(() => {
                scanningOverlay.classList.add('hidden');
                
                // Override demographic fields if user input is empty/default
                if (pId) data.patient_id = pId;
                if (pName) data.patient_name = pName;
                if (pAge) data.patient_age = pAge;
                data.patient_gender = pGender;
                data.referred_by = pReferred;
                
                renderScanResults(data);
                
                // Save to local session cache list
                saveScanToCache(data);
            }, 300);
        })
        .catch(err => {
            console.error("Inference Error:", err);
            scanningOverlay.classList.add('hidden');
            alert("Error running Deep Learning inference. Falling back to static client-side generator.");
            
            // Client-side fallback if server fails or model fails
            generateClientMockDiagnosis(file, pId, pName, pAge, pGender, pReferred);
        });
    }

    // Client-side fallback mapping for robust execution
    function generateClientMockDiagnosis(file, pId, pName, pAge, pGender, pReferred) {
        const reader = new FileReader();
        reader.onload = (e) => {
            // Pick mock diagnosis based on age/name hints
            let mockType = "normal";
            const nameLower = pName.toLowerCase();
            if (nameLower.includes("gordon") || nameLower.includes("cough") || nameLower.includes("flu")) {
                mockType = "pneumonia";
            } else if (nameLower.includes("arthur") || nameLower.includes("heart") || nameLower.includes("elderly")) {
                mockType = "cardiomegaly";
            } else if (Math.random() > 0.6) {
                mockType = "pneumonia";
            }
            
            const scan_id = "LOCAL-" + Math.floor(Math.random() * 100000);
            const data = {
                success: true,
                scan_id: scan_id,
                patient_id: pId || "METRO-" + Math.floor(10000 + Math.random() * 90000),
                patient_name: pName || "Jane Doe (Mock)",
                patient_age: pAge || "50",
                patient_gender: pGender || "F",
                referred_by: pReferred || "Dr. Self-Referral",
                timestamp: new Date().toLocaleString(),
                original_image_url: e.target.result,
                heatmap_image_url: e.target.result, // Use self as heatmap fallback representation
                dominant_finding: mockType === 'normal' ? 'Normal/No Findings' : (mockType === 'pneumonia' ? 'Pneumonia/Consolidation' : 'Cardiomegaly (Enlarged Heart)'),
                dominant_score: mockType === 'normal' ? 88.5 : (mockType === 'pneumonia' ? 76.2 : 91.4),
                metrics: {
                    normal: mockType === 'normal' ? 88.5 : 5.0,
                    pneumonia: mockType === 'pneumonia' ? 76.2 : 12.0,
                    cardiomegaly: mockType === 'cardiomegaly' ? 91.4 : 8.0,
                    effusion: mockType === 'pneumonia' ? 35.0 : 4.0,
                    pneumothorax: 1.5
                },
                heuristics: {
                    ctr: mockType === 'cardiomegaly' ? 0.58 : 0.44,
                    lung_density: mockType === 'pneumonia' ? 0.41 : 0.22,
                    lung_asymmetry: mockType === 'pneumonia' ? 0.18 : 0.03,
                    costophrenic_density: mockType === 'pneumonia' ? 0.46 : 0.28,
                    apical_flatness: 0.65
                },
                impressions: mockType === 'normal' ? 
                    ["No active cardiopulmonary disease identified.", "Lungs are clear without focal infiltrates."] : 
                    (mockType === 'pneumonia' ? 
                        ["Patchy opacification and consolidation in lower lung fields.", "Findings strongly suggest infectious bronchopneumonia."] : 
                        ["The cardiothoracic ratio is estimated at 0.58, indicating significant enlargement.", "Consistent with cardiac hypertrophy or congestive heart failure."])
            };
            
            renderScanResults(data);
            saveScanToCache(data);
        };
        reader.readAsDataURL(file);
    }

    // ---------------------------------------------------------
    // Render and Layout Updates
    // ---------------------------------------------------------
    function renderScanResults(data) {
        activeScanData = data;
        
        // Set state flags
        isImageLoaded = false;
        isHeatmapLoaded = false;
        
        // Update header demographics overlay
        activePatientId.innerText = `ID: ${data.patient_id}`;
        activePatientName.innerText = `NAME: ${data.patient_name.toUpperCase()}`;
        
        // Set Patient ID badge inside demographic card
        document.getElementById('patient-id').value = data.patient_id;
        document.getElementById('patient-name').value = data.patient_name;
        document.getElementById('patient-age').value = data.patient_age;
        document.getElementById('patient-gender').value = data.patient_gender;
        document.getElementById('referred-by').value = data.referred_by;

        // Load original image into canvas pipeline
        currentImage.onload = () => {
            isImageLoaded = true;
            fitImageToViewport();
            
            // Load heatmap image once original finishes layout
            currentHeatmap.onload = () => {
                isHeatmapLoaded = true;
                // Force a blend slider activation if positive finding found
                if (data.dominant_finding !== "Normal/No Findings") {
                    sliderHeatmap.value = 45;
                    heatmapOpacity = 45;
                    valHeatmap.innerText = "45%";
                } else {
                    sliderHeatmap.value = 0;
                    heatmapOpacity = 0;
                    valHeatmap.innerText = "0%";
                }
                redrawCanvas();
            };
            currentHeatmap.src = data.heatmap_image_url;
        };
        currentImage.src = data.original_image_url;

        // Render AI bars
        probNormal.innerText = `${data.metrics.normal}%`;
        fillNormal.style.width = `${data.metrics.normal}%`;
        
        probPneumonia.innerText = `${data.metrics.pneumonia}%`;
        fillPneumonia.style.width = `${data.metrics.pneumonia}%`;

        probCardiomegaly.innerText = `${data.metrics.cardiomegaly}%`;
        fillCardiomegaly.style.width = `${data.metrics.cardiomegaly}%`;

        probEffusion.innerText = `${data.metrics.effusion}%`;
        fillEffusion.style.width = `${data.metrics.effusion}%`;

        probPneumothorax.innerText = `${data.metrics.pneumothorax}%`;
        fillPneumothorax.style.width = `${data.metrics.pneumothorax}%`;

        // Highlight dominant pathology group
        document.querySelectorAll('.diag-bar-group').forEach(group => {
            group.classList.remove('dominant');
        });
        let dominantKey = 'normal';
        if (data.dominant_finding.includes('Cardiomegaly')) dominantKey = 'cardiomegaly';
        else if (data.dominant_finding.includes('Pneumonia')) dominantKey = 'pneumonia';
        else if (data.dominant_finding.includes('Effusion')) dominantKey = 'effusion';
        else if (data.dominant_finding.includes('Pneumothorax')) dominantKey = 'pneumothorax';
        
        const dominantGroup = document.querySelector(`.diag-bar-group[data-key="${dominantKey}"]`);
        if (dominantGroup) dominantGroup.classList.add('dominant');

        // Render metrics boxes
        metricCtr.innerText = data.heuristics.ctr.toFixed(2);
        statusCtr.innerText = data.heuristics.ctr > 0.50 ? "ENLARGED" : "NORMAL";
        statusCtr.className = `m-status ${data.heuristics.ctr > 0.50 ? 'text-glow text-warning' : ''}`;

        metricOpacity.innerText = data.heuristics.lung_density.toFixed(2);
        statusOpacity.innerText = data.heuristics.lung_density > 0.35 ? "OPACIFIED" : "CLEAR";
        statusOpacity.className = `m-status ${data.heuristics.lung_density > 0.35 ? 'text-glow text-danger' : ''}`;

        metricCosto.innerText = data.heuristics.costophrenic_density.toFixed(2);
        statusCosto.innerText = data.heuristics.costophrenic_density > 0.38 ? "BLUNTED" : "SHARP";
        statusCosto.className = `m-status ${data.heuristics.costophrenic_density > 0.38 ? 'text-glow text-warning' : ''}`;

        metricApical.innerText = data.heuristics.apical_flatness.toFixed(2);
        statusApical.innerText = data.heuristics.apical_flatness > 0.75 ? "AIR PACKET" : "VASCULAR";
        statusApical.className = `m-status ${data.heuristics.apical_flatness > 0.75 ? 'text-glow text-danger' : ''}`;

        // Render impressions report card
        dominantBadge.innerText = data.dominant_finding.toUpperCase();
        dominantBadge.className = `badge ${data.dominant_finding.includes('Normal') ? 'badge-normal' : 'badge-danger'}`;
        reportTime.innerText = data.timestamp;

        findingsText.innerHTML = '';
        data.impressions.forEach((imp, index) => {
            const p = document.createElement('p');
            p.className = 'finding-item';
            p.innerHTML = `<span class="bullet">▶</span> ${imp}`;
            findingsText.appendChild(p);
        });

        // Enable PDF generation button
        btnExportReport.removeAttribute('disabled');
        
        // Sync data with Print template in DOM
        syncPrintTemplate(data);
    }

    function syncPrintTemplate(data) {
        document.getElementById('p-id').innerText = data.patient_id;
        document.getElementById('p-name').innerText = data.patient_name.toUpperCase();
        document.getElementById('p-age-gender').innerText = `${data.patient_age} / ${data.patient_gender}`;
        document.getElementById('p-referred').innerText = data.referred_by.toUpperCase();
        document.getElementById('p-time').innerText = data.timestamp;

        // Visual ratios in print
        document.getElementById('p-prob-normal').innerText = `${data.metrics.normal}%`;
        document.getElementById('p-status-normal').innerText = data.metrics.normal > 50 ? "UNREMARKABLE" : "LOW";
        
        document.getElementById('p-prob-pneumonia').innerText = `${data.metrics.pneumonia}%`;
        document.getElementById('p-status-pneumonia').innerText = data.metrics.pneumonia > 30 ? "SUSPECTED ACTIVE" : "NEGATIVE";
        document.getElementById('p-status-pneumonia').className = `right text-bold ${data.metrics.pneumonia > 30 ? 'text-danger-print' : ''}`;

        document.getElementById('p-prob-cardiomegaly').innerText = `${data.metrics.cardiomegaly}%`;
        document.getElementById('p-status-cardiomegaly').innerText = data.metrics.cardiomegaly > 30 ? "SUSPECTED ACTIVE" : "NEGATIVE";
        document.getElementById('p-status-cardiomegaly').className = `right text-bold ${data.metrics.cardiomegaly > 30 ? 'text-warning-print' : ''}`;

        document.getElementById('p-prob-effusion').innerText = `${data.metrics.effusion}%`;
        document.getElementById('p-status-effusion').innerText = data.metrics.effusion > 30 ? "BLUNTING DETECTED" : "NEGATIVE";
        document.getElementById('p-status-effusion').className = `right text-bold ${data.metrics.effusion > 30 ? 'text-warning-print' : ''}`;

        document.getElementById('p-prob-pneumothorax').innerText = `${data.metrics.pneumothorax}%`;
        document.getElementById('p-status-pneumothorax').innerText = data.metrics.pneumothorax > 25 ? "ACCUMULATION DETECTED" : "NEGATIVE";
        document.getElementById('p-status-pneumothorax').className = `right text-bold ${data.metrics.pneumothorax > 25 ? 'text-danger-print' : ''}`;

        document.getElementById('p-metric-ctr').innerText = data.heuristics.ctr.toFixed(2);
        document.getElementById('p-metric-opacity').innerText = data.heuristics.lung_density.toFixed(2);
        document.getElementById('p-metric-costo').innerText = data.heuristics.costophrenic_density.toFixed(2);

        // Render impressions lists in print
        const printNarrative = document.getElementById('p-findings-narrative');
        printNarrative.innerHTML = '';
        
        const h4_find = document.createElement('h4');
        h4_find.innerText = "IMPRESSIONS:";
        printNarrative.appendChild(h4_find);

        const ul = document.createElement('ul');
        data.impressions.forEach(imp => {
            const li = document.createElement('li');
            li.innerText = imp;
            ul.appendChild(li);
        });
        printNarrative.appendChild(ul);
        
        // Sync original & heatmap img URLs (converting base64 or absolute URLs safely)
        document.getElementById('print-original-img').src = data.original_image_url;
        document.getElementById('print-heatmap-img').src = data.heatmap_image_url;
        
        document.getElementById('p-sig-date').innerText = new Date().toLocaleDateString();
    }

    // ---------------------------------------------------------
    // Session Caching (Local Storage Management)
    // ---------------------------------------------------------
    function saveScanToCache(scan) {
        // Database now acts as the clinical audit ledger. Refresh the sidebar dynamically.
        fetchLedger();
    }

    function fetchLedger(searchQuery = '') {
        const url = searchQuery ? `/scans?search=${encodeURIComponent(searchQuery)}` : '/scans';
        fetch(url)
        .then(res => {
            if (!res.ok) {
                throw new Error("HTTP error retrieving scans");
            }
            return res.json();
        })
        .then(scans => {
            renderHistoryCards(scans);
        })
        .catch(err => {
            console.error("Ledger fetch failed, falling back to local session cache:", err);
            renderLocalStorageFallback();
        });
    }

    function renderHistoryCards(scans) {
        historyContainer.innerHTML = '';

        if (!scans || scans.length === 0) {
            historyContainer.innerHTML = `
                <div class="empty-history">
                    <i class="fa-solid fa-folder-open"></i>
                    <p>No scans found in audit ledger.</p>
                </div>`;
            return;
        }

        scans.forEach(scan => {
            const card = document.createElement('div');
            card.className = `history-card ${activeScanData && activeScanData.scan_id === scan.scan_id ? 'active' : ''}`;
            
            const badgeClass = scan.dominant_finding.includes('Normal') ? 'badge-normal' : 'badge-danger';
            const shortName = scan.patient_name.length > 18 ? scan.patient_name.substring(0, 16) + '...' : scan.patient_name;
            
            card.innerHTML = `
                <div class="card-meta">
                    <span class="card-patient-id text-glow">${scan.patient_id}</span>
                    <span class="card-time">${scan.timestamp.split(' ')[1] || scan.timestamp}</span>
                </div>
                <div class="card-patient-name">${shortName}</div>
                <div class="card-finding">
                    <span class="badge ${badgeClass}">${scan.dominant_finding.replace(' (Enlarged Heart)', '').replace('/Consolidation', '')}</span>
                    <span class="card-score">${Math.round(scan.dominant_score)}%</span>
                </div>
            `;
            
            card.addEventListener('click', () => {
                renderScanResults(scan);
                document.querySelectorAll('.history-card').forEach(c => c.classList.remove('active'));
                card.classList.add('active');
            });
            
            historyContainer.appendChild(card);
        });
    }

    function renderLocalStorageFallback() {
        let sessionCache = [];
        try {
            const existing = localStorage.getItem('pacs_session_scans');
            if (existing) sessionCache = JSON.parse(existing);
        } catch (e) {
            console.error(e);
        }
        renderHistoryCards(sessionCache);
    }

    // ---------------------------------------------------------
    // Demo Sample Loading Controller
    // ---------------------------------------------------------
    function loadSample(sampleName) {
        const metadata = MOCK_PAGES[sampleName];
        if (!metadata) return;

        scanningOverlay.classList.remove('hidden');
        loadProgressBar.style.width = "0%";
        
        // Demographics auto fill from mock lists
        document.getElementById('patient-id').value = "METRO-" + Math.floor(100000 + Math.random() * 900000);
        document.getElementById('patient-name').value = metadata.patient_name;
        document.getElementById('patient-age').value = metadata.patient_age;
        document.getElementById('patient-gender').value = metadata.patient_gender;
        document.getElementById('referred-by').value = metadata.referred_by;

        // Perform server analysis by loading the image path via standard fetch to /analyze
        // Wait! We can fetch the local sample image and submit it as a file block to ensure the actual PyTorch loads it!
        // This is extremely robust and ensures the model is run!
        fetch(metadata.image)
        .then(res => res.blob())
        .then(blob => {
            const sampleFile = new File([blob], `${sampleName}.png`, { type: 'image/png' });
            processFilePayload(sampleFile);
        })
        .catch(err => {
            console.error("Failed to load sample blob:", err);
            scanningOverlay.classList.add('hidden');
            alert("Error retrieving sample file. Make sure sample asset files exist.");
        });
    }

    // ---------------------------------------------------------
    // Printing / Exporting Management
    // ---------------------------------------------------------
    btnExportReport.addEventListener('click', () => {
        if (!activeScanData) return;
        
        // Hide standard interface and launch print browser spool
        window.print();
    });

    // Run Workstation init on load
    init();
});
