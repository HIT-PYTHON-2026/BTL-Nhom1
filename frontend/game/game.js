// ============================================================
// EMOTION EXPRESS - Game Engine
// ============================================================

// --- DOM ---
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const video = document.getElementById('webcam');
const startOverlay = document.getElementById('startOverlay');
const victoryOverlay = document.getElementById('victoryOverlay');
const hudEl = document.getElementById('hud');
const scoreDisplay = document.getElementById('scoreDisplay');
const emotionDisplay = document.getElementById('emotionDisplay');

// ============================================================
// MOCK API — Replace with real backend call in production
// ============================================================
/**
 * Placeholder for emotion detection.
 * Future: capture video frame → POST /v1/emotion_classification/predict
 *         → map predicted_class to one of the 3 emotions.
 * @returns {string} 'sad' | 'happy' | 'surprised'
 */
function getCurrentEmotion() {
    const emotions = ['sad', 'happy', 'surprised'];
    return emotions[Math.floor(Math.random() * emotions.length)];
}

// ============================================================
// CONFIG
// ============================================================
const CFG = {
    trackSpeed: 3,
    gateSpeed: 1.4,
    tieSpacing: 42,
    railWidth: 6,
    trainYRatio: 0.72,
    checkStart: 0.55,
    checkEnd: 0.87,
    checkInterval: 50,   // frames between emotion checks
    spawnDelay: 1800,     // ms before next gate appears
};

const GATES = [
    { emotion: 'sad',       emoji: '😢', label: 'SAD',       color: '#42A5F5' },
    { emotion: 'happy',     emoji: '😃', label: 'HAPPY',     color: '#FFCA28' },
    { emotion: 'surprised', emoji: '😮', label: 'SURPRISED', color: '#AB47BC' },
];

// ============================================================
// STATE
// ============================================================
let S = {};

function resetState() {
    S = {
        running: false, over: false,
        trackOff: 0, gateIdx: 0, passed: 0,
        detected: '---',
        gate: null, checkTick: 0,
        particles: [], smoke: [],
        shake: 0, frame: 0,
        // computed on resize:
        tw: 0, cx: 0, trainW: 0, trainY: 0, trainH: 0,
        gateW: 0, gateH: 66, zoneTop: 0, zoneBot: 0,
    };
}
resetState();

// ============================================================
// SIZING
// ============================================================
function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const w = canvas.width, h = canvas.height;
    S.tw = Math.min(Math.max(w * 0.12, 100), 180);
    S.cx = w / 2;
    S.trainW = S.tw * 1.4;
    S.trainY = h * CFG.trainYRatio;
    S.trainH = h * 0.14;
    S.gateW = S.tw * 2.6;
    S.zoneTop = h * CFG.checkStart;
    S.zoneBot = h * CFG.checkEnd;
}
window.addEventListener('resize', resize);
resize();

// ============================================================
// CAMERA
// ============================================================
async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (e) {
        console.warn('Camera unavailable:', e);
    }
}

// ============================================================
// HELPERS
// ============================================================
function rrect(x, y, w, h, r) {
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

// ============================================================
// PARTICLES
// ============================================================
function burst(x, y, color, n = 35) {
    for (let i = 0; i < n; i++) {
        const a = Math.random() * Math.PI * 2;
        const sp = 2 + Math.random() * 5;
        S.particles.push({
            x, y,
            vx: Math.cos(a) * sp, vy: Math.sin(a) * sp,
            r: 2 + Math.random() * 4,
            color, alpha: 1, decay: 0.012 + Math.random() * 0.015,
        });
    }
}
function updateParticles() {
    for (let i = S.particles.length - 1; i >= 0; i--) {
        const p = S.particles[i];
        p.x += p.vx; p.y += p.vy;
        p.vy += 0.08;
        p.alpha -= p.decay;
        if (p.alpha <= 0) S.particles.splice(i, 1);
    }
}
function drawParticles() {
    for (const p of S.particles) {
        ctx.globalAlpha = p.alpha;
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
    }
    ctx.globalAlpha = 1;
}

// ============================================================
// SMOKE
// ============================================================
function updateSmoke() {
    if (S.running && Math.random() < 0.3) {
        S.smoke.push({
            x: S.cx + S.trainW * 0.12 + (Math.random() - 0.5) * 6,
            y: S.trainY - 18,
            r: 2 + Math.random() * 2, alpha: 0.5,
            vy: -(0.8 + Math.random() * 0.5),
            vx: (Math.random() - 0.5) * 0.4,
            g: 0.06 + Math.random() * 0.04,
        });
    }
    for (let i = S.smoke.length - 1; i >= 0; i--) {
        const s = S.smoke[i];
        s.x += s.vx; s.y += s.vy;
        s.r += s.g; s.alpha -= 0.007;
        if (s.alpha <= 0) S.smoke.splice(i, 1);
    }
}
function drawSmoke() {
    for (const s of S.smoke) {
        ctx.globalAlpha = s.alpha;
        ctx.fillStyle = '#90A4AE';
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fill();
    }
    ctx.globalAlpha = 1;
}

// ============================================================
// DRAW: TRACKS
// ============================================================
function drawTracks() {
    const h = canvas.height;
    const lx = S.cx - S.tw / 2, rx = S.cx + S.tw / 2;

    // corridor bg
    ctx.fillStyle = 'rgba(10,10,25,0.35)';
    ctx.fillRect(lx - 22, 0, S.tw + 44, h);

    // ties
    S.trackOff = (S.trackOff + CFG.trackSpeed) % CFG.tieSpacing;
    ctx.strokeStyle = '#6D4C41'; ctx.lineWidth = 5;
    for (let y = -CFG.tieSpacing + S.trackOff; y < h; y += CFG.tieSpacing) {
        ctx.beginPath(); ctx.moveTo(lx - 10, y); ctx.lineTo(rx + 10, y); ctx.stroke();
    }

    // rails
    ctx.fillStyle = '#B0BEC5';
    ctx.fillRect(lx - CFG.railWidth / 2, 0, CFG.railWidth, h);
    ctx.fillRect(rx - CFG.railWidth / 2, 0, CFG.railWidth, h);
    ctx.fillStyle = 'rgba(236,239,241,0.35)';
    ctx.fillRect(lx - CFG.railWidth / 2 + 1, 0, 2, h);
    ctx.fillRect(rx - CFG.railWidth / 2 + 1, 0, 2, h);
}

// ============================================================
// DRAW: CHECK ZONE
// ============================================================
function drawCheckZone() {
    if (!S.gate || S.gate.done) return;
    const pulse = 0.12 + Math.sin(Date.now() * 0.003) * 0.06;
    ctx.fillStyle = `rgba(76,175,80,${pulse})`;
    ctx.fillRect(S.cx - S.gateW / 2, S.zoneTop, S.gateW, S.zoneBot - S.zoneTop);

    ctx.setLineDash([8, 5]);
    ctx.strokeStyle = 'rgba(76,175,80,0.45)'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(S.cx - S.gateW / 2, S.zoneTop); ctx.lineTo(S.cx + S.gateW / 2, S.zoneTop); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(S.cx - S.gateW / 2, S.zoneBot); ctx.lineTo(S.cx + S.gateW / 2, S.zoneBot); ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = 'rgba(76,175,80,0.65)';
    ctx.font = 'bold 11px "Outfit",sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('▼ MATCH ZONE ▼', S.cx, S.zoneTop - 5);
}

// ============================================================
// DRAW: TRAIN
// ============================================================
function drawTrain() {
    const { cx, trainY: ty, trainW: tw, trainH: th } = S;
    ctx.save();
    if (S.shake > 0) {
        ctx.translate((Math.random() - 0.5) * S.shake, (Math.random() - 0.5) * S.shake);
        S.shake *= 0.88; if (S.shake < 0.4) S.shake = 0;
    }

    // shadow
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    ctx.beginPath(); ctx.ellipse(cx, ty + th + 6, tw * 0.42, 5, 0, 0, Math.PI * 2); ctx.fill();

    // body
    const bg = ctx.createLinearGradient(cx - tw / 2, 0, cx + tw / 2, 0);
    bg.addColorStop(0, '#C62828'); bg.addColorStop(0.4, '#EF5350'); bg.addColorStop(1, '#B71C1C');
    ctx.fillStyle = bg;
    rrect(cx - tw / 2, ty, tw, th, 10); ctx.fill();

    // glow outline
    ctx.shadowColor = '#FF5252'; ctx.shadowBlur = 18;
    ctx.strokeStyle = 'rgba(255,82,82,0.45)'; ctx.lineWidth = 2;
    rrect(cx - tw / 2, ty, tw, th, 10); ctx.stroke();
    ctx.shadowBlur = 0;

    // nose
    ctx.fillStyle = '#D32F2F';
    ctx.beginPath(); ctx.moveTo(cx - tw * 0.28, ty); ctx.lineTo(cx, ty - 18); ctx.lineTo(cx + tw * 0.28, ty); ctx.closePath(); ctx.fill();

    // headlight
    const hl = ctx.createRadialGradient(cx, ty - 13, 2, cx, ty - 13, 14);
    hl.addColorStop(0, 'rgba(255,255,200,0.85)'); hl.addColorStop(1, 'rgba(255,255,200,0)');
    ctx.fillStyle = hl; ctx.beginPath(); ctx.arc(cx, ty - 13, 14, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#FFF9C4'; ctx.beginPath(); ctx.arc(cx, ty - 13, 3.5, 0, Math.PI * 2); ctx.fill();

    // window
    const wg = ctx.createLinearGradient(0, ty + 12, 0, ty + 12 + th * 0.28);
    wg.addColorStop(0, '#BBDEFB'); wg.addColorStop(1, '#64B5F6');
    ctx.fillStyle = wg; rrect(cx - tw * 0.22, ty + 12, tw * 0.44, th * 0.28, 4); ctx.fill();

    // stripe
    ctx.fillStyle = '#FFCA28'; ctx.fillRect(cx - tw / 2 + 6, ty + th * 0.55, tw - 12, 3);

    // wheels
    for (const wx of [cx - tw * 0.28, cx + tw * 0.28]) {
        ctx.fillStyle = '#37474F'; ctx.beginPath(); ctx.arc(wx, ty + th, 9, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#78909C'; ctx.beginPath(); ctx.arc(wx, ty + th, 3.5, 0, Math.PI * 2); ctx.fill();
    }

    // chimney
    ctx.fillStyle = '#546E7A'; rrect(cx + tw * 0.12, ty - 4, 12, 8, 2); ctx.fill();
    ctx.fillStyle = '#37474F'; rrect(cx + tw * 0.12 - 2, ty - 7, 16, 4, 2); ctx.fill();

    ctx.restore();
}

// ============================================================
// DRAW: GATE
// ============================================================
function drawGate() {
    const g = S.gate;
    if (!g || g.done) return;
    const gx = S.cx - S.gateW / 2, gy = g.y, gh = S.gateH;

    // bar gradient
    const bg = ctx.createLinearGradient(gx, gy, gx + S.gateW, gy);
    bg.addColorStop(0, 'rgba(0,0,0,0)'); bg.addColorStop(0.15, g.def.color);
    bg.addColorStop(0.85, g.def.color); bg.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.globalAlpha = 0.82; ctx.fillStyle = bg;
    rrect(gx, gy, S.gateW, gh, 8); ctx.fill(); ctx.globalAlpha = 1;

    // neon border
    ctx.shadowColor = g.def.color; ctx.shadowBlur = 14;
    ctx.strokeStyle = g.def.color; ctx.lineWidth = 2;
    rrect(gx, gy, S.gateW, gh, 8); ctx.stroke(); ctx.shadowBlur = 0;

    // pulse in check zone
    if (g.inZone) {
        const p = 0.4 + Math.sin(Date.now() * 0.01) * 0.4;
        ctx.strokeStyle = `rgba(255,255,255,${p * 0.6})`; ctx.lineWidth = 3;
        rrect(gx - 3, gy - 3, S.gateW + 6, gh + 6, 10); ctx.stroke();
    }

    // pillars
    ctx.fillStyle = g.def.color; ctx.globalAlpha = 0.55;
    rrect(gx, gy - 8, 10, gh + 16, 3); ctx.fill();
    rrect(gx + S.gateW - 10, gy - 8, 10, gh + 16, 3); ctx.fill();
    ctx.globalAlpha = 1;

    // emoji circle
    const cr = gh * 0.52;
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.beginPath(); ctx.arc(S.cx, gy + gh / 2, cr, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = g.def.color; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(S.cx, gy + gh / 2, cr, 0, Math.PI * 2); ctx.stroke();

    ctx.font = `${cr * 1.3}px serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(g.def.emoji, S.cx, gy + gh / 2 + 2);

    // label below
    ctx.font = 'bold 13px "Outfit",sans-serif';
    ctx.fillStyle = '#fff'; ctx.textBaseline = 'top';
    ctx.fillText(g.def.label, S.cx, gy + gh + 8);
}

// ============================================================
// GATE LOGIC
// ============================================================
function spawnGate() {
    if (S.gateIdx >= GATES.length) return;
    S.gate = { def: GATES[S.gateIdx], y: -80, inZone: false, done: false };
    S.checkTick = 0;
}

function updateGate() {
    const g = S.gate;
    if (!g || g.done) return;

    g.y += CFG.gateSpeed;
    const mid = g.y + S.gateH / 2;

    // in check zone?
    if (mid >= S.zoneTop && mid <= S.zoneBot) {
        g.inZone = true;
        S.checkTick++;
        if (S.checkTick % CFG.checkInterval === 1) {
            const em = getCurrentEmotion();
            S.detected = em;
            emotionDisplay.textContent = em.toUpperCase();

            if (em === g.def.emotion) {
                // MATCHED
                g.done = true;
                S.passed++;
                S.shake = 10;
                scoreDisplay.textContent = `${S.passed} / 3`;
                burst(S.cx, g.y + S.gateH / 2, g.def.color, 45);

                if (S.passed >= 3) {
                    setTimeout(victory, 1200);
                } else {
                    S.gateIdx++;
                    setTimeout(spawnGate, CFG.spawnDelay);
                }
            }
        }
    } else {
        g.inZone = false;
    }

    // gate scrolled past → reset same gate
    if (g.y > canvas.height + 60 && !g.done) {
        g.y = -80;
        g.inZone = false;
        S.checkTick = 0;
    }
}

// ============================================================
// VICTORY & CONFETTI
// ============================================================
let confetti = [];
let confettiAnim = null;

function victory() {
    S.running = false; S.over = true;
    hudEl.classList.add('hidden');
    victoryOverlay.classList.remove('hidden');
    spawnConfetti();
}

function spawnConfetti() {
    const cc = document.getElementById('confettiCanvas');
    cc.width = window.innerWidth; cc.height = window.innerHeight;
    const cctx = cc.getContext('2d');
    const colors = ['#FF6B6B', '#FFE66D', '#4ECDC4', '#A78BFA', '#F472B6', '#42A5F5'];
    confetti = [];
    for (let i = 0; i < 180; i++) {
        confetti.push({
            x: Math.random() * cc.width,
            y: Math.random() * -cc.height,
            w: 6 + Math.random() * 8,
            h: 4 + Math.random() * 5,
            color: colors[Math.floor(Math.random() * colors.length)],
            vy: 2 + Math.random() * 3,
            vx: (Math.random() - 0.5) * 2,
            rot: Math.random() * 360,
            rv: (Math.random() - 0.5) * 8,
        });
    }
    function animConfetti() {
        cctx.clearRect(0, 0, cc.width, cc.height);
        for (const c of confetti) {
            c.x += c.vx; c.y += c.vy; c.rot += c.rv;
            if (c.y > cc.height + 20) { c.y = -20; c.x = Math.random() * cc.width; }
            cctx.save();
            cctx.translate(c.x, c.y);
            cctx.rotate(c.rot * Math.PI / 180);
            cctx.fillStyle = c.color;
            cctx.fillRect(-c.w / 2, -c.h / 2, c.w, c.h);
            cctx.restore();
        }
        confettiAnim = requestAnimationFrame(animConfetti);
    }
    animConfetti();
}

// ============================================================
// GAME LOOP
// ============================================================
let animId = null;

function gameLoop() {
    if (!S.running) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // dark vignette
    const vg = ctx.createRadialGradient(S.cx, canvas.height / 2, canvas.height * 0.3, S.cx, canvas.height / 2, canvas.height * 0.8);
    vg.addColorStop(0, 'rgba(0,0,0,0)'); vg.addColorStop(1, 'rgba(0,0,0,0.35)');
    ctx.fillStyle = vg; ctx.fillRect(0, 0, canvas.width, canvas.height);

    drawTracks();
    drawCheckZone();
    updateSmoke(); drawSmoke();
    drawTrain();
    updateGate(); drawGate();
    updateParticles(); drawParticles();

    S.frame++;
    animId = requestAnimationFrame(gameLoop);
}

// ============================================================
// START / RESTART
// ============================================================
function startGame() {
    resetState();
    resize();
    startOverlay.classList.add('hidden');
    victoryOverlay.classList.add('hidden');
    hudEl.classList.remove('hidden');
    scoreDisplay.textContent = '0 / 3';
    emotionDisplay.textContent = '---';
    if (confettiAnim) cancelAnimationFrame(confettiAnim);

    S.running = true;
    setTimeout(spawnGate, 800);
    gameLoop();
}

document.getElementById('startBtn').addEventListener('click', startGame);
document.getElementById('restartBtn').addEventListener('click', startGame);

// ============================================================
// INIT
// ============================================================
initCamera();
