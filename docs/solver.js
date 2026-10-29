// Split-step Fourier solver for the 2D time-dependent Schrödinger equation.
// Units: hbar = m = 1. Same algorithm as the Python package (Strang splitting,
// complex absorbing boundary), written for speed in plain JavaScript.

export class FFT {
  constructor(n) {
    if (n & (n - 1)) throw new Error("FFT size must be a power of two");
    this.n = n;
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / n);
      this.sin[i] = Math.sin((2 * Math.PI * i) / n);
    }
  }

  // In-place unnormalised transform. sign = -1 forward, +1 inverse.
  transform(re, im, sign) {
    const n = this.n, rev = this.rev, C = this.cos, S = this.sin;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = 0, k = 0; j < half; j++, k += step) {
          const a = i + j, b = a + half;
          const wr = C[k], wi = sign * S[k];
          const tr = re[b] * wr - im[b] * wi;
          const ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr; im[a] += ti;
        }
      }
    }
  }
}

export class Solver2D {
  constructor(n, L, dt) {
    this.n = n; this.L = L; this.dx = L / n; this.N = n * n;
    this.fft = new FFT(n);
    this.re = new Float64Array(this.N);
    this.im = new Float64Array(this.N);
    this.V = new Float64Array(this.N);
    this.W = new Float64Array(this.N);
    this.colRe = new Float64Array(n);
    this.colIm = new Float64Array(n);
    this.x = new Float64Array(n);
    this.k = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      this.x[i] = -L / 2 + i * this.dx;
      this.k[i] = ((2 * Math.PI) / L) * (i < n / 2 ? i : i - n);
    }
    this.t = 0;
    this.setAbsorber(4.0, 6.0);
    this.setDt(dt);
  }

  setAbsorber(width, strength) {
    const { n, x, L } = this;
    for (let iy = 0; iy < n; iy++)
      for (let ix = 0; ix < n; ix++) {
        const d = Math.min(L / 2 - Math.abs(x[ix]), L / 2 - Math.abs(x[iy]));
        const s = Math.min(1, Math.max(0, (width - d) / width));
        this.W[iy * n + ix] = strength * s * s;
      }
    if (this.dt) this.setDt(this.dt);
  }

  setPotential(V) {
    this.V.set(V);
    this.setDt(this.dt);
  }

  setDt(dt) {
    this.dt = dt;
    const { n, N, k } = this;
    // kinetic factor exp(-i k^2 dt / 2), with the 1/N inverse-FFT normalisation folded in
    this.kinRe = new Float64Array(N);
    this.kinIm = new Float64Array(N);
    for (let iy = 0; iy < n; iy++)
      for (let ix = 0; ix < n; ix++) {
        const ph = -0.5 * (k[ix] * k[ix] + k[iy] * k[iy]) * dt;
        this.kinRe[iy * n + ix] = Math.cos(ph) / N;
        this.kinIm[iy * n + ix] = Math.sin(ph) / N;
      }
    // half-step potential factor exp(-i (V - iW) dt / 2)
    this.potRe = new Float64Array(N);
    this.potIm = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const decay = Math.exp(-0.5 * this.W[i] * dt);
      const ph = -0.5 * this.V[i] * dt;
      this.potRe[i] = decay * Math.cos(ph);
      this.potIm[i] = decay * Math.sin(ph);
    }
  }

  fft2(re, im, sign) {
    const { n, fft, colRe, colIm } = this;
    for (let r = 0; r < n; r++)
      fft.transform(re.subarray(r * n, r * n + n), im.subarray(r * n, r * n + n), sign);
    for (let c = 0; c < n; c++) {
      for (let r = 0; r < n; r++) { colRe[r] = re[r * n + c]; colIm[r] = im[r * n + c]; }
      fft.transform(colRe, colIm, sign);
      for (let r = 0; r < n; r++) { re[r * n + c] = colRe[r]; im[r * n + c] = colIm[r]; }
    }
  }

  static mulInPlace(re, im, fr, fi, power2) {
    for (let i = 0; i < re.length; i++) {
      let a = fr[i], b = fi[i];
      if (power2) { const t = a * a - b * b; b = 2 * a * b; a = t; }
      const r = re[i], m = im[i];
      re[i] = r * a - m * b;
      im[i] = r * b + m * a;
    }
  }

  step(nsteps) {
    const { re, im } = this;
    Solver2D.mulInPlace(re, im, this.potRe, this.potIm, false);
    for (let s = 0; s < nsteps; s++) {
      this.fft2(re, im, -1);
      Solver2D.mulInPlace(re, im, this.kinRe, this.kinIm, false);
      this.fft2(re, im, +1);
      Solver2D.mulInPlace(re, im, this.potRe, this.potIm, s < nsteps - 1);
    }
    this.t += nsteps * this.dt;
  }

  norm() {
    let s = 0;
    for (let i = 0; i < this.N; i++) s += this.re[i] ** 2 + this.im[i] ** 2;
    return s * this.dx * this.dx;
  }

  // Add a Gaussian packet (amplitude-normalised to 1) to the current state.
  addPacket(x0, y0, kx, ky, sx, sy) {
    const { n, x, re, im } = this;
    const amp = 1 / Math.sqrt(2 * Math.PI * sx * sy);
    for (let iy = 0; iy < n; iy++) {
      const dy = x[iy] - y0;
      const gy = Math.exp(-(dy * dy) / (4 * sy * sy));
      if (gy < 1e-12) continue;
      for (let ix = 0; ix < n; ix++) {
        const dX = x[ix] - x0;
        const g = amp * gy * Math.exp(-(dX * dX) / (4 * sx * sx));
        if (g < 1e-14) continue;
        const ph = kx * x[ix] + ky * x[iy];
        re[iy * n + ix] += g * Math.cos(ph);
        im[iy * n + ix] += g * Math.sin(ph);
      }
    }
  }

  normalize() {
    const s = Math.sqrt(this.norm());
    if (s > 0) for (let i = 0; i < this.N; i++) { this.re[i] /= s; this.im[i] /= s; }
  }

  clear() { this.re.fill(0); this.im.fill(0); this.t = 0; }

  // Spatial moments of |psi|^2 (used by the self-test).
  moments() {
    const { n, x, re, im } = this;
    let s = 0, mx = 0, my = 0, mxx = 0, myy = 0;
    for (let iy = 0; iy < n; iy++)
      for (let ix = 0; ix < n; ix++) {
        const p = re[iy * n + ix] ** 2 + im[iy * n + ix] ** 2;
        s += p; mx += p * x[ix]; my += p * x[iy];
        mxx += p * x[ix] * x[ix]; myy += p * x[iy] * x[iy];
      }
    mx /= s; my /= s;
    return { x: mx, y: my, sx: Math.sqrt(mxx / s - mx * mx), sy: Math.sqrt(myy / s - my * my) };
  }

  // Gaussian blur (via FFT) used to soften sharp walls.
  blur(field, width) {
    const { n, N, k } = this;
    const re = Float64Array.from(field), im = new Float64Array(N);
    this.fft2(re, im, -1);
    for (let iy = 0; iy < n; iy++)
      for (let ix = 0; ix < n; ix++) {
        const g = Math.exp(-0.5 * width * width * (k[ix] * k[ix] + k[iy] * k[iy])) / N;
        re[iy * n + ix] *= g; im[iy * n + ix] *= g;
      }
    this.fft2(re, im, +1);
    return re;
  }
}
