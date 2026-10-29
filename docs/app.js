import { Solver2D } from "./solver.js";

const L = 40;            // box side (same units as the Python scenes)
const DT = 0.006;
const USER_WALL = 200;   // height of hand-drawn walls

// ---------------------------------------------------------------- scenes
// Each scene returns a "shape" (0..1 per cell, scaled by the height slider),
// an optional fixed potential, and the packet to launch.
const SCENES = {
  "double-slit": {
    label: "Double slit", heightLabel: "Wall height", height: 200, max: 300,
    shape: (X, Y) => slitShape(X, Y, [-2, 2], 1.0),
    packet: { x: -12, y: 0, kx: 5, ky: 0, sx: 1.5, sy: 5 },
    note: "Each slit alone would give a broad blob. Together they interfere: fringes at sin θ = nλ/d.",
  },
  "single-slit": {
    label: "Single slit", heightLabel: "Wall height", height: 200, max: 300,
    shape: (X, Y) => slitShape(X, Y, [0], 1.6),
    packet: { x: -12, y: 0, kx: 5, ky: 0, sx: 1.5, sy: 5 },
    note: "Squeezing the wave through a gap spreads it out: Heisenberg's Δy·Δp ≥ ħ/2 at work.",
  },
  tunnelling: {
    label: "Tunnelling", heightLabel: "Barrier height", height: 8, max: 20,
    shape: (X) => (Math.abs(X) < 0.25 ? 1 : 0), blur: false,
    packet: { x: -9, y: 0, kx: 4, ky: 0, sx: 1.8, sy: 2.5 },
    note: "Mean energy E = 8. At barrier height 8, about half the packet gets through. Raise it and watch the transmitted part fade away exponentially.",
  },
  stadium: {
    label: "Stadium billiard", heightLabel: "Wall height", height: 200, max: 300,
    shape: (X, Y) => {
      const xc = Math.max(-6, Math.min(6, X));
      return (X - xc) ** 2 + Y ** 2 <= 100 ? 0 : 1;
    },
    packet: { x: -6, y: -2, kx: 4, ky: 2.5, sx: 1.2, sy: 1.2 },
    note: "Classically chaotic: a ray bounces ergodically. The wave scrambles into a speckle of random-looking interference.",
  },
  crystal: {
    label: "Crystal", heightLabel: "Atom height", height: 40, max: 120,
    shape: (X, Y) => {
      let v = 0;
      const a = 2.4;
      for (let i = 0; i < 4; i++)
        for (let j = -8; j <= 8; j++) {
          const cx = 0.5 + i * a, cy = j * a + (i % 2) * a / 2;
          const r2 = (X - cx) ** 2 + (Y - cy) ** 2;
          if (r2 < 2) v += Math.exp(-r2 / (2 * 0.09));
        }
      return v;
    }, blur: false,
    packet: { x: -10, y: 0, kx: 5, ky: 0, sx: 1.8, sy: 4.5 },
    note: "A lattice of atoms acts as a diffraction grating. Bragg reflection sends the wave out in sharp beams.",
  },
  trap: {
    label: "Harmonic trap", heightLabel: "Trap strength (100 ω²)", height: 100, max: 300,
    shape: (X, Y) => 0.005 * (X * X + Y * Y), blur: false,
    packet: { x: 5, y: 0, kx: 0, ky: 5, sx: 0.35, sy: 1.2 },
    note: "A squeezed coherent state orbits like a classical particle while its width breathes twice per orbit.",
  },
  empty: {
    label: "Empty box", heightLabel: "—", height: 0, max: 1,
    shape: () => 0,
    packet: { x: -8, y: 0, kx: 4, ky: 0, sx: 1.2, sy: 1.2 },
    note: "A free Gaussian packet drifts and spreads: σ(t) = σ₀√(1 + (t/2σ₀²)²). Draw your own walls!",
  },
};

function slitShape(X, Y, centres, width) {
  if (Math.abs(X + 5) >= 0.25) return 0;
  for (const c of centres) if (Math.abs(Y - c) < width / 2) return 0;
  return 1;
}

// ---------------------------------------------------------------- state
const $ = (id) => document.getElementById(id);
const canvas = $("view"), overlay = $("overlay");
const ctx = canvas.getContext("2d");
const octx = overlay.getContext("2d");

let sim, n, img, shape, userMask, outlineMask, alphaMask;
let sceneKey = "double-slit";
let running = true, tool = "launch", style = "phase";
let ref = null;
let drag = null;

function buildSolver(res) {
  n = res;
  sim = new Solver2D(n, L, DT);
  canvas.width = canvas.height = n;
  img = ctx.createImageData(n, n);
  userMask = new Float64Array(n * n);
  loadScene(sceneKey);
}

function loadScene(key) {
  sceneKey = key;
  const sc = SCENES[key];
  userMask.fill(0); // drawn walls belong to the scene they were drawn in
  shape = new Float64Array(n * n);
  for (let iy = 0; iy < n; iy++)
    for (let ix = 0; ix < n; ix++) shape[iy * n + ix] = sc.shape(sim.x[ix], sim.x[iy]);
  if (sc.blur !== false) shape = sim.blur(shape, 1.5 * sim.dx);
  const h = $("height");
  h.max = sc.max; h.value = sc.height;
  $("heightRow").hidden = key === "empty";
  $("heightLabel").textContent = sc.heightLabel;
  $("note").textContent = sc.note;
  rebuildPotential();
  resetPacket();
}

function rebuildPotential() {
  const H = +$("height").value;
  $("heightVal").textContent = H.toFixed(H < 20 ? 1 : 0);
  const V = new Float64Array(n * n);
  const user = userMask.some((v) => v > 0) ? sim.blur(userMask, 1.0 * sim.dx) : null;
  for (let i = 0; i < V.length; i++) V[i] = H * shape[i] + (user ? USER_WALL * user[i] : 0);
  sim.setPotential(V);
  // overlay masks (shade + outline at half of the "reference" height)
  const vref = Math.max(H, 1e-9);
  alphaMask = new Float32Array(n * n);
  outlineMask = new Uint8Array(n * n);
  const thr = 0.5;
  const high = new Uint8Array(n * n);
  for (let i = 0; i < V.length; i++) {
    const a = Math.min(1, (H * shape[i]) / vref) + (user ? Math.min(1, user[i]) : 0);
    alphaMask[i] = 0.9 * Math.min(1, a);
    high[i] = (H > 0 && shape[i] >= thr) || (user && user[i] >= 0.5) ? 1 : 0;
  }
  for (let iy = 0; iy < n; iy++)
    for (let ix = 0; ix < n; ix++) {
      const i = iy * n + ix;
      if (!high[i]) continue;
      const nb = [ix > 0 ? i - 1 : i, ix < n - 1 ? i + 1 : i, iy > 0 ? i - n : i, iy < n - 1 ? i + n : i];
      if (nb.some((j) => !high[j])) outlineMask[i] = 1;
    }
}

function resetPacket() {
  const p = SCENES[sceneKey].packet;
  sim.clear();
  sim.addPacket(p.x, p.y, p.kx, p.ky, p.sx * widthScale(), p.sy * widthScale());
  sim.normalize();
  ref = null;
}

const widthScale = () => +$("width").value;

// ---------------------------------------------------------------- render
function hsv(h, s, v, out, o) {
  const i = Math.floor(h * 6), f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  let r, g, b;
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    default: r = v; g = p; b = q;
  }
  out[o] = r * 255; out[o + 1] = g * 255; out[o + 2] = b * 255;
}

const INFERNO = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66],
  [245, 125, 21], [250, 193, 39], [252, 255, 164]];
function inferno(v, out, o) {
  const x = v * (INFERNO.length - 1), i = Math.min(INFERNO.length - 2, Math.floor(x)), f = x - i;
  for (let c = 0; c < 3; c++) out[o + c] = INFERNO[i][c] * (1 - f) + INFERNO[i + 1][c] * f;
}

function draw() {
  const { re, im } = sim;
  let maxA = 0;
  for (let i = 0; i < re.length; i++) {
    const a = re[i] * re[i] + im[i] * im[i];
    if (a > maxA) maxA = a;
  }
  const target = Math.sqrt(maxA) * 0.8 + 1e-12;
  ref = ref === null ? target : 0.85 * ref + 0.15 * target;
  const d = img.data;
  const fill = [33, 38, 51], edge = [204, 212, 230];
  const rgb = [0, 0, 0];
  for (let iy = 0; iy < n; iy++) {
    const row = n - 1 - iy; // +y up
    for (let ix = 0; ix < n; ix++) {
      const i = iy * n + ix, o = (row * n + ix) * 4;
      const amp = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
      if (style === "phase") {
        const v = Math.min(1, Math.pow(amp / ref, 0.6));
        let h = Math.atan2(im[i], re[i]) / (2 * Math.PI);
        if (h < 0) h += 1;
        hsv(h, 0.9, v, rgb, 0);
      } else {
        inferno(Math.min(1, (amp / ref) ** 1.0), rgb, 0);
      }
      const a = alphaMask[i];
      let r = rgb[0] * (1 - a) + fill[0] * a, g = rgb[1] * (1 - a) + fill[1] * a, b = rgb[2] * (1 - a) + fill[2] * a;
      if (outlineMask[i]) { r = 0.25 * r + 0.75 * edge[0]; g = 0.25 * g + 0.75 * edge[1]; b = 0.25 * b + 0.75 * edge[2]; }
      d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function drawOverlay() {
  const w = overlay.width, h = overlay.height;
  octx.clearRect(0, 0, w, h);
  if (!drag || tool !== "launch") return;
  const [x0, y0] = toPx(drag.x0, drag.y0), [x1, y1] = toPx(drag.x1, drag.y1);
  octx.strokeStyle = "rgba(255,255,255,0.9)";
  octx.fillStyle = "rgba(255,255,255,0.9)";
  octx.lineWidth = 2;
  octx.beginPath(); octx.arc(x0, y0, 6, 0, 2 * Math.PI); octx.stroke();
  octx.beginPath(); octx.moveTo(x0, y0); octx.lineTo(x1, y1); octx.stroke();
  const ang = Math.atan2(y1 - y0, x1 - x0);
  octx.beginPath();
  octx.moveTo(x1, y1);
  octx.lineTo(x1 - 10 * Math.cos(ang - 0.4), y1 - 10 * Math.sin(ang - 0.4));
  octx.lineTo(x1 - 10 * Math.cos(ang + 0.4), y1 - 10 * Math.sin(ang + 0.4));
  octx.fill();
  const k = launchK();
  octx.font = "13px ui-monospace, monospace";
  octx.fillText(`|k| = ${Math.hypot(k[0], k[1]).toFixed(1)}`, x1 + 8, y1 - 8);
}

// ---------------------------------------------------------------- input
function toWorld(ev) {
  const r = overlay.getBoundingClientRect();
  const u = (ev.clientX - r.left) / r.width, v = (ev.clientY - r.top) / r.height;
  return [-L / 2 + u * L, L / 2 - v * L];
}
function toPx(x, y) {
  return [((x + L / 2) / L) * overlay.width, ((L / 2 - y) / L) * overlay.height];
}
function launchK() {
  const dx = drag.x1 - drag.x0, dy = drag.y1 - drag.y0;
  const len = Math.hypot(dx, dy);
  const kmax = 0.45 * Math.PI / sim.dx;
  const k = Math.min(kmax, 0.6 * len);
  return len < 1e-6 ? [0, 0] : [(k * dx) / len, (k * dy) / len];
}
function paint(x, y, erase) {
  const r = +$("brush").value;
  for (let iy = 0; iy < n; iy++) {
    const dy = sim.x[iy] - y;
    if (Math.abs(dy) > r) continue;
    for (let ix = 0; ix < n; ix++) {
      const dX = sim.x[ix] - x;
      if (dX * dX + dy * dy <= r * r) userMask[iy * n + ix] = erase ? 0 : 1;
    }
  }
}

overlay.addEventListener("pointerdown", (ev) => {
  overlay.setPointerCapture(ev.pointerId);
  const [x, y] = toWorld(ev);
  drag = { x0: x, y0: y, x1: x, y1: y };
  if (tool !== "launch") { paint(x, y, tool === "erase"); rebuildPotential(); }
});
overlay.addEventListener("pointermove", (ev) => {
  if (!drag) return;
  const [x, y] = toWorld(ev);
  if (tool === "launch") { drag.x1 = x; drag.y1 = y; }
  else {
    // interpolate so fast strokes stay connected
    const steps = Math.ceil(Math.hypot(x - drag.x1, y - drag.y1) / 0.2);
    for (let s = 1; s <= steps; s++)
      paint(drag.x1 + ((x - drag.x1) * s) / steps, drag.y1 + ((y - drag.y1) * s) / steps, tool === "erase");
    drag.x1 = x; drag.y1 = y;
    rebuildPotential();
  }
});
overlay.addEventListener("pointerup", () => {
  if (drag && tool === "launch") {
    const [kx, ky] = launchK();
    const s = widthScale();
    if (!$("superpose").checked) sim.clear();
    sim.addPacket(drag.x0, drag.y0, kx, ky, 1.2 * s, 1.2 * s);
    sim.normalize();
    ref = null;
  }
  drag = null;
});

for (const btn of document.querySelectorAll("[data-tool]"))
  btn.addEventListener("click", () => {
    tool = btn.dataset.tool;
    document.querySelectorAll("[data-tool]").forEach((b) => b.setAttribute("aria-pressed", b === btn));
  });
for (const btn of document.querySelectorAll("[data-style]"))
  btn.addEventListener("click", () => {
    style = btn.dataset.style;
    document.querySelectorAll("[data-style]").forEach((b) => b.setAttribute("aria-pressed", b === btn));
  });

const sceneSel = $("scene");
for (const [k, sc] of Object.entries(SCENES)) sceneSel.add(new Option(sc.label, k));
sceneSel.addEventListener("change", () => loadScene(sceneSel.value));
$("height").addEventListener("input", rebuildPotential);
$("reset").addEventListener("click", resetPacket);
$("clearWalls").addEventListener("click", () => { userMask.fill(0); rebuildPotential(); });
$("play").addEventListener("click", () => {
  running = !running;
  $("play").textContent = running ? "Pause" : "Play";
});
$("res").addEventListener("change", () => buildSolver(+$("res").value));
$("speed").addEventListener("input", () => ($("speedVal").textContent = $("speed").value));
$("width").addEventListener("input", () => ($("widthVal").textContent = (+$("width").value).toFixed(1) + "×"));

function resizeOverlay() {
  const r = canvas.getBoundingClientRect();
  overlay.width = Math.round(r.width * devicePixelRatio);
  overlay.height = Math.round(r.height * devicePixelRatio);
  octx.setTransform(1, 0, 0, 1, 0, 0);
}
window.addEventListener("resize", resizeOverlay);

// ---------------------------------------------------------------- loop
let last = performance.now(), fpsAvg = 60;
function frame(now) {
  if (running) sim.step(+$("speed").value);
  draw();
  drawOverlay();
  const dtms = now - last; last = now;
  fpsAvg = 0.9 * fpsAvg + 0.1 * (1000 / Math.max(dtms, 1));
  $("stats").textContent =
    `t = ${sim.t.toFixed(2)}   ∫|ψ|² = ${sim.norm().toFixed(3)}   ${fpsAvg.toFixed(0)} fps`;
  requestAnimationFrame(frame);
}

buildSolver(+$("res").value);
resizeOverlay();
requestAnimationFrame(frame);

// exposed for console tinkering and automated checks
window.wavepacket = { get sim() { return sim; }, SCENES, loadScene };
