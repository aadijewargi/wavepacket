// Validates the browser solver (docs/solver.js) against the same analytic
// results as the Python suite. Run with:  node --test tests/js/
import { test } from "node:test";
import assert from "node:assert/strict";
import { FFT, Solver2D } from "../../docs/solver.js";

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b}`);

test("FFT round trip and delta transform", () => {
  const n = 64, f = new FFT(n);
  const re = new Float64Array(n), im = new Float64Array(n);
  re[0] = 1;
  f.transform(re, im, -1);
  for (let i = 0; i < n; i++) { close(re[i], 1, 1e-12, "delta re"); close(im[i], 0, 1e-12, "delta im"); }
  const x = Float64Array.from({ length: n }, (_, i) => Math.sin(3 * i) + 0.1 * i);
  const xr = Float64Array.from(x), xi = new Float64Array(n);
  f.transform(xr, xi, -1);
  f.transform(xr, xi, +1);
  for (let i = 0; i < n; i++) close(xr[i] / n, x[i], 1e-12, "round trip");
});

test("free Gaussian drifts at k and spreads as sigma0*sqrt(1+(t/2sigma0^2)^2)", () => {
  const s = new Solver2D(128, 40, 0.01);
  s.setAbsorber(1, 0); // switch absorption off
  s.addPacket(0, 0, 3, -1, 1.5, 1.0);
  close(s.norm(), 1, 1e-12, "initial norm");
  s.step(300);
  const m = s.moments();
  const w = (s0, t) => s0 * Math.sqrt(1 + (t / (2 * s0 * s0)) ** 2);
  close(m.x, 9, 1e-6, "<x>");
  close(m.y, -3, 1e-6, "<y>");
  close(m.sx, w(1.5, 3), 1e-6, "sigma_x");
  close(m.sy, w(1.0, 3), 1e-6, "sigma_y");
  close(s.norm(), 1, 1e-12, "norm conserved");
});

test("coherent state in a harmonic trap follows the classical orbit", () => {
  const s = new Solver2D(128, 24, 2 * Math.PI / 800);
  s.setAbsorber(1, 0);
  const V = new Float64Array(s.N);
  for (let iy = 0; iy < s.n; iy++)
    for (let ix = 0; ix < s.n; ix++) V[iy * s.n + ix] = 0.5 * (s.x[ix] ** 2 + s.x[iy] ** 2);
  s.setPotential(V);
  const sig = Math.sqrt(0.5);
  s.addPacket(3, 0, 0, 3, sig, sig);
  s.step(200); // quarter period
  const m = s.moments();
  close(m.x, 0, 1e-3, "<x> at T/4");
  close(m.y, 3, 1e-3, "<y> at T/4");
  close(m.sx, sig, 1e-3, "width stays constant");
});

test("absorbing boundary removes a wave that leaves the box", () => {
  const s = new Solver2D(128, 40, 0.01);
  s.addPacket(10, 0, 4, 0, 1.5, 1.5);
  s.step(800);
  assert.ok(s.norm() < 1e-2, `norm after leaving: ${s.norm()}`);
});
