/* =============================================
   Emotion Express — Game Engine (Vanilla JS)
   ============================================= */

// ==========================================
// 1. DOM & CANVAS SETUP
// ==========================================
const video = document.getElementById('webcam');
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const hudGate = document.getElementById('hud-gate');
const hudEmotion = document.getElementById('hud-emotion');
const hudRound = document.getElementById('hud-round');
const startScreen = document.getElementById('start-screen');
const winScreen = document.getElementById('win-screen');
const roundCompleteScreen = document.getElementById('round-complete-screen');
const roundTitleEl = document.getElementById('round-title');
const roundInfoEl = document.getElementById('round-info');
const btnStart = document.getElementById('btn-start');
const btnRestart = document.getElementById('btn-restart');
const btnContinue = document.getElementById('btn-continue');
const btnQuit = document.getElementById('btn-quit');
const btnStop = document.getElementById('btn-stop');

let W, H;

function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);
resize();


// ==========================================
// 2. WEBCAM INIT
// ==========================================
async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: 1280, height: 720 }
        });
        video.srcObject = stream;
    } catch (err) {
        console.error('Camera access denied:', err);
    }
}


// ==========================================
// 3. REALTIME EMOTION DETECTION via WebSocket
// ==========================================
const _wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL = `${_wsProto}//${window.location.host}/v1/emotion_classification/game-ws`;
const FRAME_SEND_INTERVAL = 500; // ms giữa mỗi lần gửi frame

let emotionWS = null;
let frameSendTimer = null;
let faceDetected = false;

// Canvas ẩn để chụp frame từ webcam
const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

function connectEmotionWS() {
    if (emotionWS && emotionWS.readyState === WebSocket.OPEN) return;

    emotionWS = new WebSocket(WS_URL);

    emotionWS.onopen = () => {
        console.log('[EmotionWS] Connected');
        startSendingFrames();
    };

    emotionWS.onmessage = (event) => {
        try {
            const result = JSON.parse(event.data);
            faceDetected = result.face_detected;

            if (result.face_detected && result.emotion) {
                lastDetectedEmotion = result.emotion;
                const emojiMap = { sad: '😢', happy: '😄', surprised: '😲' };
                hudEmotion.textContent = `${emojiMap[result.emotion] || '😐'} ${result.raw_label} (${(result.confidence * 100).toFixed(0)}%)`;
            } else if (result.face_detected && !result.emotion) {
                lastDetectedEmotion = null;
                hudEmotion.textContent = `😐 ${result.raw_label || '---'}`;
            } else {
                lastDetectedEmotion = null;
                faceDetected = false;
                hudEmotion.textContent = '⚠️ Không thấy mặt';
            }
        } catch (e) {
            console.warn('[EmotionWS] Parse error:', e);
        }
    };

    emotionWS.onclose = () => {
        console.log('[EmotionWS] Disconnected');
        stopSendingFrames();
        if (gameRunning) setTimeout(connectEmotionWS, 2000);
    };

    emotionWS.onerror = (err) => {
        console.warn('[EmotionWS] Error:', err);
    };
}

function disconnectEmotionWS() {
    stopSendingFrames();
    if (emotionWS) {
        emotionWS.onclose = null;
        emotionWS.close();
        emotionWS = null;
    }
}

function startSendingFrames() {
    stopSendingFrames();
    frameSendTimer = setInterval(sendFrame, FRAME_SEND_INTERVAL);
}

function stopSendingFrames() {
    if (frameSendTimer) {
        clearInterval(frameSendTimer);
        frameSendTimer = null;
    }
}

function sendFrame() {
    if (!emotionWS || emotionWS.readyState !== WebSocket.OPEN) return;
    if (!video.videoWidth || !video.videoHeight) return;

    captureCanvas.width = 320;
    captureCanvas.height = 240;
    captureCtx.drawImage(video, 0, 0, 320, 240);

    const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.6);
    emotionWS.send(dataUrl);
}


// ==========================================
// 4. GAME STATE
// ==========================================
const BASE_GATE_EMOTIONS = ['sad', 'happy', 'surprised'];
const EMOTION_EMOJI = { sad: '😢', happy: '😄', surprised: '😲' };
const EMOTION_COLOR = {
    sad: { main: '#3b82f6', glow: 'rgba(59,130,246,.35)' },
    happy: { main: '#22c55e', glow: 'rgba(34,197,94,.35)' },
    surprised: { main: '#f97316', glow: 'rgba(249,115,22,.35)' },
};

let gameRunning = false;
let animFrameId = null;
let gatesPassed = 0;
let currentGateIndex = 0;

// Round system
let currentRound = 1;
let totalGates = 3;
let gateEmotions = [];

// Track objects
let trackOffset = 0;
const TRACK_SPEED = 3;

// Gate (obstacle)
let gate = null;

// Train position
const TRAIN = {
    w: 100,
    h: 130,
    get x() { return W / 2 - this.w / 2; },
    get y() { return H - this.h - 40; },
};

// Detection state
let lastDetectedEmotion = null;


// ==========================================
// 5. GATE GENERATION
// ==========================================

/**
 * Generate a list of gate emotions for a round.
 * Randomly picks from the base emotions pool.
 */
function generateGateEmotions(count) {
    const list = [];
    for (let i = 0; i < count; i++) {
        list.push(BASE_GATE_EMOTIONS[Math.floor(Math.random() * BASE_GATE_EMOTIONS.length)]);
    }
    return list;
}


// ==========================================
// 6. GATE FACTORY
// ==========================================
function createGate(emotionKey) {
    const gateW = 220;
    const gateH = 100;
    return {
        emotion: emotionKey,
        emoji: EMOTION_EMOJI[emotionKey],
        x: W / 2 - gateW / 2,
        y: -gateH - 40,                   // start above screen
        w: gateW,
        h: gateH,
        speed: 2.2,
        passed: false,
        matched: false,
        flash: 0, // for success flash effect
    };
}

function spawnNextGate() {
    if (currentGateIndex < gateEmotions.length) {
        gate = createGate(gateEmotions[currentGateIndex]);
    } else {
        gate = null;
    }
}


// ==========================================
// 7. DRAWING FUNCTIONS
// ==========================================

// --- Train tracks ---
function drawTracks() {
    const trackW = 70;
    const railW = 6;
    const tieSpacing = 40;
    const cx = W / 2;

    ctx.save();

    // Rails
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = railW;
    ctx.shadowColor = 'rgba(148,163,184,.3)';
    ctx.shadowBlur = 8;

    ctx.beginPath();
    ctx.moveTo(cx - trackW / 2, 0);
    ctx.lineTo(cx - trackW / 2, H);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(cx + trackW / 2, 0);
    ctx.lineTo(cx + trackW / 2, H);
    ctx.stroke();

    // Ties (cross-bars) — scrolling
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 4;
    ctx.shadowBlur = 0;

    const startY = (trackOffset % tieSpacing) - tieSpacing;
    for (let y = startY; y < H + tieSpacing; y += tieSpacing) {
        ctx.beginPath();
        ctx.moveTo(cx - trackW / 2 - 10, y);
        ctx.lineTo(cx + trackW / 2 + 10, y);
        ctx.stroke();
    }

    ctx.restore();
}

// --- Train ---
function drawTrain() {
    const { x, y, w, h } = TRAIN;

    ctx.save();

    // Body
    const grad = ctx.createLinearGradient(x, y, x, y + h);
    grad.addColorStop(0, '#6366f1');
    grad.addColorStop(1, '#4338ca');
    ctx.fillStyle = grad;
    roundRect(x, y, w, h, 14);
    ctx.fill();

    // Glow
    ctx.shadowColor = 'rgba(99,102,241,.5)';
    ctx.shadowBlur = 20;
    roundRect(x, y, w, h, 14);
    ctx.fill();
    ctx.shadowBlur = 0;

    // Window
    ctx.fillStyle = '#a5b4fc';
    roundRect(x + 20, y + 15, w - 40, 30, 6);
    ctx.fill();

    // Light
    ctx.fillStyle = '#fbbf24';
    ctx.beginPath();
    ctx.arc(x + w / 2, y + 8, 6, 0, Math.PI * 2);
    ctx.fill();

    // Wheels
    ctx.fillStyle = '#1e1b4b';
    ctx.beginPath();
    ctx.arc(x + 20, y + h, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x + w - 20, y + h, 10, 0, Math.PI * 2);
    ctx.fill();

    // Emoji label
    ctx.font = '28px serif';
    ctx.textAlign = 'center';
    ctx.fillText('🚂', x + w / 2, y + h - 20);

    ctx.restore();
}

// --- Gate (obstacle) ---
function drawGate(g) {
    if (!g) return;

    const { x, y, w, h, emoji, emotion, matched, flash } = g;
    const colors = EMOTION_COLOR[emotion];

    ctx.save();

    // Flash effect on match
    if (flash > 0) {
        ctx.globalAlpha = flash;
        ctx.fillStyle = colors.main;
        ctx.fillRect(0, 0, W, H);
        ctx.globalAlpha = 1;
    }

    // Gate pillars
    const pillarW = 14;
    ctx.fillStyle = colors.main;
    ctx.shadowColor = colors.glow;
    ctx.shadowBlur = 16;
    roundRect(x - pillarW, y - 10, pillarW, h + 20, 6);
    ctx.fill();
    roundRect(x + w, y - 10, pillarW, h + 20, 6);
    ctx.fill();
    ctx.shadowBlur = 0;

    // Gate bar (top)
    const barGrad = ctx.createLinearGradient(x, y, x + w, y);
    barGrad.addColorStop(0, colors.main);
    barGrad.addColorStop(1, matched ? '#22c55e' : colors.main);
    ctx.fillStyle = barGrad;
    roundRect(x, y, w, 10, 4);
    ctx.fill();

    // Background plate
    ctx.fillStyle = 'rgba(0,0,0,.55)';
    ctx.strokeStyle = colors.main;
    ctx.lineWidth = 2;
    roundRect(x, y + 14, w, h - 14, 12);
    ctx.fill();
    ctx.stroke();

    // Emoji
    ctx.font = '44px serif';
    ctx.textAlign = 'center';
    ctx.fillText(emoji, x + w / 2, y + h / 2 + 22);

    // Label
    ctx.font = '600 13px Inter, sans-serif';
    ctx.fillStyle = colors.main;
    ctx.fillText(emotion.toUpperCase(), x + w / 2, y + h - 4);

    // Matched badge
    if (matched) {
        ctx.font = '700 16px Inter, sans-serif';
        ctx.fillStyle = '#22c55e';
        ctx.fillText('✓ MATCHED!', x + w / 2, y - 16);
    }

    ctx.restore();
}

// --- Utility: rounded rect ---
function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

// --- Scanline / vignette overlay ---
function drawVignette() {
    const grad = ctx.createRadialGradient(W / 2, H / 2, H * .3, W / 2, H / 2, H * .9);
    grad.addColorStop(0, 'transparent');
    grad.addColorStop(1, 'rgba(0,0,0,.45)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);
}


// ==========================================
// 8. GAME LOOP
// ==========================================
let lastTime = 0;

function gameLoop(timestamp) {
    if (!gameRunning) return;

    const dt = timestamp - lastTime;
    lastTime = timestamp;

    // --- Clear ---
    ctx.clearRect(0, 0, W, H);

    // --- Update track scroll ---
    trackOffset += TRACK_SPEED;

    // --- Emotion is updated via WebSocket onmessage (no polling needed) ---

    // --- Update gate ---
    if (gate && !gate.passed) {
        gate.y += gate.speed;

        // Re-center horizontally on resize
        gate.x = W / 2 - gate.w / 2;

        // Check if gate reached the train zone
        const trainTop = TRAIN.y;
        const gateBottom = gate.y + gate.h;

        if (gateBottom >= trainTop && !gate.matched) {
            // Check emotion match — chỉ khi đã detect được mặt
            if (faceDetected && lastDetectedEmotion === gate.emotion) {
                gate.matched = true;
                gate.flash = 0.4;
            }
        }

        // Fade flash
        if (gate.flash > 0) gate.flash -= 0.01;

        // Gate has passed below train — resolve
        if (gate.y > TRAIN.y + TRAIN.h / 2) {
            if (gate.matched) {
                // ✅ Cảm xúc khớp — vượt qua thành công
                gate.passed = true;
                gatesPassed++;
                currentGateIndex++;
                hudGate.textContent = `Cổng: ${gatesPassed} / ${totalGates}`;

                // Check round complete
                if (gatesPassed >= totalGates) {
                    setTimeout(showRoundComplete, 400);
                    gameRunning = false;
                } else {
                    // Spawn next gate after short delay
                    setTimeout(spawnNextGate, 900);
                }
            } else {
                // ❌ Không khớp — reset gate về đầu, thử lại
                gate.y = -gate.h - 40;
                gate.passed = false;
                gate.matched = false;
                gate.flash = 0;
            }
        }
    }

    // --- Draw ---
    drawVignette();
    drawTracks();
    drawGate(gate);
    drawTrain();

    animFrameId = requestAnimationFrame(gameLoop);
}


// ==========================================
// 9. START / STOP / ROUND COMPLETE / QUIT
// ==========================================
function startGame() {
    startScreen.style.display = 'none';
    winScreen.style.display = 'none';
    roundCompleteScreen.style.display = 'none';

    // Reset state
    currentRound = 1;
    totalGates = 3;
    gatesPassed = 0;
    currentGateIndex = 0;
    gate = null;
    trackOffset = 0;
    lastDetectedEmotion = null;
    faceDetected = false;
    lastTime = performance.now();
    gameRunning = true;

    // Generate gates for round 1
    gateEmotions = generateGateEmotions(totalGates);

    hudGate.textContent = `Cổng: 0 / ${totalGates}`;
    hudRound.textContent = `Vòng ${currentRound}`;
    hudEmotion.textContent = '🔄 Đang kết nối...';

    // Kết nối WebSocket emotion detection
    connectEmotionWS();

    // Spawn first gate after 1s
    setTimeout(spawnNextGate, 1000);

    animFrameId = requestAnimationFrame(gameLoop);
}

function startNextRound() {
    roundCompleteScreen.style.display = 'none';

    // Advance round
    currentRound++;
    totalGates += 2;
    gatesPassed = 0;
    currentGateIndex = 0;
    gate = null;
    trackOffset = 0;
    lastDetectedEmotion = null;
    faceDetected = false;
    lastTime = performance.now();
    gameRunning = true;

    // Generate gates for new round
    gateEmotions = generateGateEmotions(totalGates);

    hudGate.textContent = `Cổng: 0 / ${totalGates}`;
    hudRound.textContent = `Vòng ${currentRound}`;
    hudEmotion.textContent = '😐 Đang nhận diện...';

    // Kết nối WebSocket nếu chưa kết nối
    connectEmotionWS();

    // Spawn first gate after 1s
    setTimeout(spawnNextGate, 1000);

    animFrameId = requestAnimationFrame(gameLoop);
}

function showRoundComplete() {
    gameRunning = false;
    if (animFrameId) cancelAnimationFrame(animFrameId);

    // Clear canvas
    ctx.clearRect(0, 0, W, H);
    gate = null;

    // Update overlay text
    roundTitleEl.textContent = `VÒNG ${currentRound} HOÀN THÀNH!`;
    roundInfoEl.textContent = `Vòng tiếp theo: ${totalGates + 2} chướng ngại vật`;

    roundCompleteScreen.style.display = 'flex';
}

function quitToHome() {
    gameRunning = false;
    disconnectEmotionWS();
    if (animFrameId) cancelAnimationFrame(animFrameId);
    window.location.href = 'index.html';
}

function showWin() {
    gameRunning = false;
    disconnectEmotionWS();
    if (animFrameId) cancelAnimationFrame(animFrameId);

    // Clear canvas
    ctx.clearRect(0, 0, W, H);
    gate = null;

    winScreen.style.display = 'flex';
}

// Listeners
btnStart.addEventListener('click', () => {
    initCamera();
    startGame();
});

btnRestart.addEventListener('click', () => {
    startGame();
});

btnContinue.addEventListener('click', () => {
    startNextRound();
});

btnQuit.addEventListener('click', () => {
    quitToHome();
});

btnStop.addEventListener('click', () => {
    quitToHome();
});


// ==========================================
// 10. INIT
// ==========================================
// Pre-init camera so it's ready when user clicks start
initCamera();
