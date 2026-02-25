/* =========================================
   Emotion AI Dashboard — Frontend Logic
   ========================================= */

// --- Init Lucide Icons ---
lucide.createIcons();

// --- Constants ---
const API_BASE = `${window.location.protocol}//${window.location.host}/v1/emotion_classification`;

// Emotion classes matching backend EmotionDataConfig.CLASSES order
const EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'];
const EMOTION_COLORS = {
    'Happy': '#22c55e',
    'Neutral': '#3b82f6',
    'Sad': '#6366f1',
    'Angry': '#ef4444',
    'Disgust': '#f59e0b',
    'Fear': '#8b5cf6',
    'Surprise': '#f97316',
};

// --- DOM Elements ---
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const btnBrowse = document.getElementById('btn-browse');
const previewContainer = document.getElementById('preview-container');
const uploadedImg = document.getElementById('uploaded-image');
const canvas = document.getElementById('detection-canvas');
const btnAnalyze = document.getElementById('btn-analyze');
const emotionBars = document.getElementById('emotion-bars');
const procTimeEl = document.getElementById('proc-time');
const modelNameEl = document.getElementById('model-name');

const streamImg = document.getElementById('webcam-stream');
const btnToggle = document.getElementById('btn-toggle');
const btnToggleText = document.getElementById('btn-toggle-text');
const emotionOverlay = document.getElementById('emotion-overlay');
const overlayLabel = document.getElementById('overlay-label');
const activityLog = document.getElementById('activity-log');

let socket = null;
let currentFile = null;

/* ===========================================================
   1. IMAGE UPLOAD & CLASSIFICATION
   =========================================================== */

// --- Drop zone interactions ---
dropZone.addEventListener('click', () => fileInput.click());
btnBrowse.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileSelected(e.target.files[0]);
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFileSelected(file);
});

// Prevent default browser drag behaviour
window.addEventListener('dragover', e => e.preventDefault(), false);
window.addEventListener('drop', e => e.preventDefault(), false);

// --- Handle file selection (preview only) ---
function handleFileSelected(file) {
    currentFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        uploadedImg.src = e.target.result;
        previewContainer.style.display = 'block';
        dropZone.style.display = 'none';

        // Clear previous detections
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    };
    reader.readAsDataURL(file);

    // Enable analyze button
    btnAnalyze.disabled = false;
    clearResults();
}

// --- Analyze button ---
btnAnalyze.addEventListener('click', () => {
    if (currentFile) analyzeImage(currentFile);
});

// --- Call backend APIs ---
async function analyzeImage(file) {
    btnAnalyze.disabled = true;
    btnAnalyze.querySelector('span')?.remove();
    const origHTML = btnAnalyze.innerHTML;
    btnAnalyze.innerHTML = '<i data-lucide="loader-2" class="spin"></i> Đang phân tích...';
    lucide.createIcons();

    const startTime = performance.now();

    const formData = new FormData();
    formData.append('file_upload', file);

    // We need a second formData for predict since each can only be consumed once
    const formData2 = new FormData();
    formData2.append('file_upload', file);

    try {
        const [detectRes, predictRes] = await Promise.all([
            fetch(`${API_BASE}/detect`, { method: 'POST', body: formData }),
            fetch(`${API_BASE}/predict`, { method: 'POST', body: formData2 })
        ]);

        const detectData = await detectRes.json();
        const predictData = await predictRes.json();

        const elapsed = Math.round(performance.now() - startTime);

        // Draw bounding boxes from detect
        drawDetections(detectData.results || []);

        // Show emotion probabilities from predict
        renderEmotionBars(predictData.probs || [], predictData.predicted_class);

        // Footer info
        procTimeEl.textContent = `${elapsed}ms`;
        modelNameEl.textContent = predictData.predictor_name || 'ResNet18';

    } catch (err) {
        console.error(err);
        emotionBars.innerHTML = '<p style="color:#ef4444;">Lỗi kết nối Server!</p>';
    }

    btnAnalyze.innerHTML = '<i data-lucide="scan-face"></i> Phân tích';
    lucide.createIcons();
    btnAnalyze.disabled = false;
}

// --- Draw face bounding boxes on canvas ---
function drawDetections(faces) {
    // Wait for image to be rendered
    requestAnimationFrame(() => {
        canvas.width = uploadedImg.clientWidth;
        canvas.height = uploadedImg.clientHeight;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const scaleX = canvas.width / uploadedImg.naturalWidth;
        const scaleY = canvas.height / uploadedImg.naturalHeight;

        faces.forEach(face => {
            // face is a tuple: [x1, y1, x2, y2, conf]
            const [x1, y1, x2, y2, conf] = face;
            const rx = x1 * scaleX;
            const ry = y1 * scaleY;
            const rw = (x2 - x1) * scaleX;
            const rh = (y2 - y1) * scaleY;

            // Draw rectangle
            ctx.strokeStyle = '#6366f1';
            ctx.lineWidth = 2.5;
            ctx.strokeRect(rx, ry, rw, rh);

            // Label badge
            const label = 'FACE DETECTED';
            ctx.font = '600 11px Inter, sans-serif';
            const textW = ctx.measureText(label).width + 12;
            ctx.fillStyle = '#6366f1';
            ctx.beginPath();
            ctx.roundRect(rx, ry - 22, textW, 20, 4);
            ctx.fill();
            ctx.fillStyle = '#fff';
            ctx.fillText(label, rx + 6, ry - 7);
        });
    });
}

// --- Render emotion bars ---
function renderEmotionBars(probs, predictedClass) {
    if (!probs || probs.length === 0) return;

    // probs order: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise (matches EmotionDataConfig.CLASSES)
    let html = '';
    probs.forEach((p, i) => {
        const name = EMOTION_CLASSES[i] || `Class ${i}`;
        const color = EMOTION_COLORS[name] || '#6366f1';
        const pct = Math.round(p * 100);
        html += `
            <div class="emotion-item">
                <div class="emotion-label">
                    <span class="name">
                        <span class="dot-sm" style="background:${color}"></span>
                        ${name}
                    </span>
                    <span class="pct">${pct}%</span>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width:${pct}%;background:${color};"></div>
                </div>
            </div>
        `;
    });
    emotionBars.innerHTML = html;
}

function clearResults() {
    emotionBars.innerHTML = '';
    procTimeEl.textContent = '—';
    modelNameEl.textContent = '—';
}


/* ===========================================================
   2. REALTIME WEBCAM STREAM (WebSocket)
   =========================================================== */

btnToggle.addEventListener('click', toggleStream);

function toggleStream() {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
    } else {
        const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        socket = new WebSocket(`${wsProto}//${window.location.host}/v1/emotion_classification/ws`);
        socket.binaryType = 'blob';

        socket.onopen = () => {
            btnToggleText.textContent = 'Dừng Camera';
            btnToggle.classList.add('streaming');
            updateOverlay('Đang kết nối...');
            clearLog();
        };

        socket.onmessage = (event) => {
            const url = URL.createObjectURL(event.data);
            streamImg.src = url;
            streamImg.onload = () => URL.revokeObjectURL(url);
        };

        socket.onclose = () => {
            btnToggleText.textContent = 'Bắt đầu Camera';
            btnToggle.classList.remove('streaming');
            hideOverlay();
            streamImg.src = '';
        };

        socket.onerror = () => {
            addLogEntry('Lỗi kết nối WebSocket', '#ef4444');
        };
    }
}

// --- Overlay on video ---
function updateOverlay(text) {
    emotionOverlay.style.display = 'flex';
    overlayLabel.textContent = text;
}
function hideOverlay() {
    emotionOverlay.style.display = 'none';
}

// --- Activity Log ---
function addLogEntry(message, color = '#22c55e') {
    const now = new Date();
    const time = now.toLocaleTimeString('vi-VN', { hour12: false });

    // Remove empty state
    const emptyMsg = activityLog.querySelector('.log-empty');
    if (emptyMsg) emptyMsg.remove();

    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-info" style="color:${color}">${message}</span>
    `;
    activityLog.prepend(entry);

    // Keep only last 50 entries
    while (activityLog.children.length > 50) {
        activityLog.removeChild(activityLog.lastChild);
    }
}

function clearLog() {
    activityLog.innerHTML = '<p class="log-empty">Đang chờ dữ liệu...</p>';
}


/* ===========================================================
   3. MISC
   =========================================================== */

// Fullscreen toggle for video
document.getElementById('btn-fullscreen')?.addEventListener('click', () => {
    const wrapper = document.getElementById('video-wrapper');
    if (!document.fullscreenElement) {
        wrapper.requestFullscreen?.();
    } else {
        document.exitFullscreen?.();
    }
});

// Spin animation for loader
const style = document.createElement('style');
style.textContent = `
    @keyframes spin { to { transform: rotate(360deg); } }
    .spin svg, svg.spin { animation: spin 1s linear infinite; }
`;
document.head.appendChild(style);