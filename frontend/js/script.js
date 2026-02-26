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
const btnReupload = document.getElementById('btn-reupload');
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

    // Enable analyze button & show reupload button
    btnAnalyze.disabled = false;
    btnReupload.style.display = 'flex';
    clearResults();
}

// --- Reupload button: reset to initial state ---
btnReupload.addEventListener('click', resetUpload);

function resetUpload() {
    currentFile = null;
    fileInput.value = '';
    uploadedImg.src = '';
    previewContainer.style.display = 'none';
    dropZone.style.display = '';
    btnAnalyze.disabled = true;
    btnReupload.style.display = 'none';
    clearResults();

    // Clear canvas
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// --- Analyze button ---
btnAnalyze.addEventListener('click', () => {
    if (currentFile) analyzeImage(currentFile);
});

// --- Call backend APIs ---
async function analyzeImage(file) {
    btnAnalyze.disabled = true;
    btnAnalyze.querySelector('span')?.remove();
    btnAnalyze.innerHTML = '<i data-lucide="loader-2" class="spin"></i> Đang phân tích...';
    lucide.createIcons();

    const startTime = performance.now();

    try {
        const formData = new FormData();
        formData.append('file_upload', file);

        const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: formData });
        const data = await res.json();

        const elapsed = Math.round(performance.now() - startTime);

        // Nếu không phát hiện khuôn mặt
        if (!data.faces || data.faces.length === 0) {
            emotionBars.innerHTML = `
                <div style="text-align:center; padding:20px 0;">
                    <div style="font-size:2.5rem; margin-bottom:8px;">😶</div>
                    <p style="color:#f59e0b; font-weight:600; font-size:.95rem;">Không phát hiện khuôn mặt</p>
                    <p style="color:#94a3b8; font-size:.8rem; margin-top:4px;">Vui lòng tải lên ảnh có khuôn mặt rõ ràng</p>
                </div>
            `;
            procTimeEl.textContent = `${elapsed}ms`;
            modelNameEl.textContent = '—';

            btnAnalyze.innerHTML = '<i data-lucide="scan-face"></i> Phân tích';
            lucide.createIcons();
            btnAnalyze.disabled = false;
            return;
        }

        // Vẽ bounding boxes với emotion labels
        drawDetections(data.faces);

        // Hiển thị kết quả per-face
        renderMultiFaceResults(data.faces);

        // Footer info
        procTimeEl.textContent = `${elapsed}ms`;
        modelNameEl.textContent = data.predictor_name || 'ResNet18';

    } catch (err) {
        console.error(err);
        emotionBars.innerHTML = '<p style="color:#ef4444;">Lỗi kết nối Server!</p>';
    }

    btnAnalyze.innerHTML = '<i data-lucide="scan-face"></i> Phân tích';
    lucide.createIcons();
    btnAnalyze.disabled = false;
}

// --- Draw face bounding boxes with emotion labels on canvas ---
function drawDetections(faces) {
    requestAnimationFrame(() => {
        canvas.width = uploadedImg.clientWidth;
        canvas.height = uploadedImg.clientHeight;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const scaleX = canvas.width / uploadedImg.naturalWidth;
        const scaleY = canvas.height / uploadedImg.naturalHeight;

        faces.forEach(face => {
            const [x1, y1, x2, y2] = face.box;
            const rx = x1 * scaleX;
            const ry = y1 * scaleY;
            const rw = (x2 - x1) * scaleX;
            const rh = (y2 - y1) * scaleY;

            const emotionColor = EMOTION_COLORS[face.predicted_class] || '#6366f1';

            // Draw rectangle
            ctx.strokeStyle = emotionColor;
            ctx.lineWidth = 2.5;
            ctx.strokeRect(rx, ry, rw, rh);

            // Label badge with emotion
            const label = `#${face.face_id} ${face.predicted_class} ${Math.round(face.best_prob * 100)}%`;
            ctx.font = '600 11px Inter, sans-serif';
            const textW = ctx.measureText(label).width + 12;
            ctx.fillStyle = emotionColor;
            ctx.beginPath();
            ctx.roundRect(rx, ry - 22, textW, 20, 4);
            ctx.fill();
            ctx.fillStyle = '#fff';
            ctx.fillText(label, rx + 6, ry - 7);
        });
    });
}

// --- Render per-face emotion results ---
function renderMultiFaceResults(faces) {
    if (!faces || faces.length === 0) return;

    let html = '';

    faces.forEach(face => {
        const emotionColor = EMOTION_COLORS[face.predicted_class] || '#6366f1';
        const pctBest = Math.round(face.best_prob * 100);

        html += `
            <div class="face-result-section">
                <div class="face-result-header">
                    <span class="face-badge" style="background:${emotionColor}">
                        Khuôn mặt #${face.face_id}
                    </span>
                    <span class="face-emotion" style="color:${emotionColor}">
                        ${face.predicted_class} — ${pctBest}%
                    </span>
                </div>
        `;

        face.probs.forEach((p, i) => {
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

        html += '</div>';
    });

    emotionBars.innerHTML = html;
}

function clearResults() {
    emotionBars.innerHTML = '';
    procTimeEl.textContent = '—';
    modelNameEl.textContent = '—';
}


/* ===========================================================
   2. REALTIME WEBCAM STREAM (WebSocket) — Server & Client
   =========================================================== */

// Camera source: 'server' or 'client'
let cameraSource = 'server';
let clientStream = null;   // MediaStream from getUserMedia
let captureInterval = null; // setInterval ID for frame capture

const btnSourceServer = document.getElementById('btn-source-server');
const btnSourceClient = document.getElementById('btn-source-client');
const clientVideo = document.getElementById('client-video');
const clientCanvas = document.getElementById('client-canvas');

// --- Camera source toggle ---
btnSourceServer.addEventListener('click', () => switchSource('server'));
btnSourceClient.addEventListener('click', () => switchSource('client'));

function switchSource(source) {
    if (source === cameraSource) return;

    // Stop current stream if running
    if (socket && socket.readyState === WebSocket.OPEN) {
        stopStream();
    }

    cameraSource = source;

    // Update active button styling
    btnSourceServer.classList.toggle('active', source === 'server');
    btnSourceClient.classList.toggle('active', source === 'client');

    addLogEntry(`Chuyển sang camera: ${source === 'server' ? 'Server' : 'Client'}`, '#818cf8');
}

// --- Toggle stream button ---
btnToggle.addEventListener('click', toggleStream);

function toggleStream() {
    if (socket && socket.readyState === WebSocket.OPEN) {
        stopStream();
    } else {
        if (cameraSource === 'server') {
            startServerStream();
        } else {
            startClientStream();
        }
    }
}

// --- Server camera mode ---
function startServerStream() {
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${wsProto}//${window.location.host}/v1/emotion_classification/ws`);
    socket.binaryType = 'blob';

    socket.onopen = () => {
        btnToggleText.textContent = 'Dừng Camera';
        btnToggle.classList.add('streaming');
        updateOverlay('📡 Server Camera');
        clearLog();
        addLogEntry('Kết nối Server Camera thành công', '#22c55e');
    };

    socket.onmessage = (event) => {
        const url = URL.createObjectURL(event.data);
        streamImg.src = url;
        streamImg.onload = () => URL.revokeObjectURL(url);
    };

    socket.onclose = () => {
        onStreamStopped();
    };

    socket.onerror = () => {
        addLogEntry('Lỗi kết nối WebSocket (Server)', '#ef4444');
    };
}

// --- Client camera mode ---
async function startClientStream() {
    try {
        // Request browser camera access
        clientStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
        });
        clientVideo.srcObject = clientStream;
        await clientVideo.play();

        addLogEntry('Camera Client đã mở', '#22c55e');
    } catch (err) {
        addLogEntry(`Không thể truy cập camera: ${err.message}`, '#ef4444');
        return;
    }

    // Connect to client WebSocket endpoint
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${wsProto}//${window.location.host}/v1/emotion_classification/ws-client`);
    socket.binaryType = 'blob';

    socket.onopen = () => {
        btnToggleText.textContent = 'Dừng Camera';
        btnToggle.classList.add('streaming');
        updateOverlay('📱 Client Camera');
        clearLog();
        addLogEntry('Kết nối Client Camera thành công', '#22c55e');

        // Start capturing frames and sending them
        startFrameCapture();
    };

    socket.onmessage = (event) => {
        // Receive annotated frame from server
        const url = URL.createObjectURL(event.data);
        streamImg.src = url;
        streamImg.onload = () => URL.revokeObjectURL(url);
    };

    socket.onclose = () => {
        onStreamStopped();
    };

    socket.onerror = () => {
        addLogEntry('Lỗi kết nối WebSocket (Client)', '#ef4444');
    };
}

function startFrameCapture() {
    const ctx = clientCanvas.getContext('2d');

    captureInterval = setInterval(() => {
        if (!clientVideo.videoWidth || !socket || socket.readyState !== WebSocket.OPEN) return;

        clientCanvas.width = clientVideo.videoWidth;
        clientCanvas.height = clientVideo.videoHeight;
        ctx.drawImage(clientVideo, 0, 0);

        const dataUrl = clientCanvas.toDataURL('image/jpeg', 0.7);
        socket.send(dataUrl);
    }, 100); // ~10 FPS
}

// --- Stop everything ---
function stopStream() {
    // Close WebSocket
    if (socket) {
        socket.close();
        socket = null;
    }

    // Stop client camera if active
    stopClientCamera();

    onStreamStopped();
}

function stopClientCamera() {
    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }
    if (clientStream) {
        clientStream.getTracks().forEach(track => track.stop());
        clientStream = null;
    }
    clientVideo.srcObject = null;
}

function onStreamStopped() {
    btnToggleText.textContent = 'Bắt đầu Camera';
    btnToggle.classList.remove('streaming');
    hideOverlay();
    streamImg.src = '';
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