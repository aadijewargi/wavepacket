"""Regenerate every image and animation used in the README.

    python examples/make_media.py            # everything (a few minutes)
    python examples/make_media.py --only plots
"""

from __future__ import annotations

import argparse
import io
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import wavepacket as wp  # noqa: E402
from wavepacket import analytic as an  # noqa: E402
from wavepacket import potentials as P  # noqa: E402
from wavepacket import render, scenes  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "media")

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117", "savefig.facecolor": "#0d1117",
    "text.color": "#e6edf3", "axes.labelcolor": "#e6edf3", "axes.edgecolor": "#8b949e",
    "xtick.color": "#8b949e", "ytick.color": "#8b949e", "axes.grid": True,
    "grid.color": "#30363d", "grid.linewidth": 0.6, "font.size": 11,
    "legend.facecolor": "#161b22", "legend.edgecolor": "#30363d",
})
C_SIM, C_TH, C_ACC = "#58a6ff", "#f0883e", "#3fb950"


def scene_gif(name, path, frames=120, size=None, fps=25):
    sc = scenes.get(name)
    s = sc.solver()
    ref = render.SmoothedReference()
    imgs = [render.frame(sc.psi0, sc.V, "phase", ref(sc.psi0), sc.vmax)]
    for psi in s.evolve(sc.psi0, sc.t_final, frames):
        imgs.append(render.frame(psi, sc.V, "phase", ref(psi), sc.vmax))
    imgs += [imgs[-1]] * 12  # brief hold on the last frame
    render.save_gif(imgs, path, fps=fps, size=size)
    print(f"  {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


def tunnelling_1d_gif(path, frames=110):
    g = wp.Grid.line(4096, 204.8)
    k0, sigma, a = 1.5, 5.0, 1.0
    V0 = 0.5 * k0 ** 2  # barrier exactly as high as the mean energy
    V = P.rectangular_barrier(g, V0, 0.0, a)
    s = wp.SplitStepSolver(g, V, dt=0.01, absorber=P.absorbing_boundary(g, 15.0, 1.0))
    psi = wp.gaussian_packet(g, -35.0, k0, sigma)
    x = g.axes[0]
    win = (x > -60) & (x < 60)
    imgs = []
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=80)
    T_th = an.packet_transmission(k0, sigma, V0, a)
    for i in range(frames):
        if i:
            psi = s.step(psi, 45)
        ax.clear()
        ax.fill_between(x[win], 0, np.where(V[win] > 0, 0.25, 0), color="#8b949e", alpha=0.35,
                        step="mid", label=r"barrier $V_0=E$")
        ax.plot(x[win], psi.real[win], color=C_SIM, lw=0.9, alpha=0.8, label=r"Re $\psi$")
        ax.plot(x[win], psi.imag[win], color="#d2a8ff", lw=0.9, alpha=0.6, label=r"Im $\psi$")
        ax.plot(x[win], np.abs(psi[win]) ** 2 * 4, color=C_TH, lw=2, label=r"$4|\psi|^2$")
        T = np.sum(np.abs(psi[x > a]) ** 2) * g.dV
        ax.set_title(f"t = {s.t:5.1f}    P(transmitted) = {T:.3f}    theory = {T_th:.3f}",
                     fontsize=11, family="monospace")
        ax.set_ylim(-0.6, 0.75)
        ax.set_xlim(-60, 60)
        ax.set_xlabel("x")
        ax.legend(loc="upper left", fontsize=8, ncol=4)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        imgs.append(np.array(Image.open(buf).convert("RGB")))
    plt.close(fig)
    render.save_gif(imgs + [imgs[-1]] * 15, path, fps=20)
    print(f"  {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


def plot_tunnelling(path):
    """T(E) through a barrier: analytic plane-wave curve, packet average, simulation."""
    g = wp.Grid.line(8192, 409.6)
    V0, a, sigma = 1.0, 3.0, 10.0
    k_sim = np.sqrt(2 * np.linspace(0.6, 4.0, 18))
    T_sim = []
    for k0 in k_sim:
        s = wp.SplitStepSolver(g, P.rectangular_barrier(g, V0, 0.0, a), dt=0.005)
        t = 160.0 / k0  # long enough for the packets to separate
        psi = s.step(wp.gaussian_packet(g, -80.0, k0, sigma), int(round(t / s.dt)))
        T_sim.append(np.sum(np.abs(psi[g.axes[0] > a]) ** 2) * g.dV)
    E = np.linspace(0.05, 4.2, 800)
    k_line = np.sqrt(2 * E)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=110)
    ax.plot(E, an.barrier_transmission(E, V0, a), color="#8b949e", lw=1.2, ls="--",
            label="plane wave, analytic")
    ax.plot(E, [an.packet_transmission(k, sigma, V0, a) for k in k_line], color=C_TH, lw=2,
            label=r"Gaussian packet, analytic $\int T\,|\phi(k)|^2dk$")
    ax.plot(0.5 * k_sim ** 2, T_sim, "o", color=C_SIM, ms=7, label="split-step simulation")
    ax.axvline(V0, color=C_ACC, lw=1, alpha=0.7)
    ax.text(V0 + 0.03, 0.05, r"$E=V_0$", color=C_ACC)
    ax.set_xlabel(r"mean energy $E=\hbar^2k_0^2/2m$")
    ax.set_ylabel("transmission probability")
    ax.set_title(r"Rectangular barrier, $V_0=1$, $a=3$ ($\hbar=m=1$): tunnelling and resonances")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}  max |sim - theory| = "
          f"{max(abs(np.array(T_sim) - [an.packet_transmission(k, sigma, V0, a) for k in k_sim])):.1e}")


def plot_double_slit(path):
    d, w = 3.0, 0.6
    g = wp.Grid.square(512, 60.0)
    V = P.slits(g, 2, separation=d, slit_width=w, wall_x=-5.0, thickness=0.4, height=200.0)
    s = wp.SplitStepSolver(g, V, dt=0.005, absorber=P.absorbing_boundary(g, 4.0, 5.0))
    psi = s.step(wp.gaussian_packet(g, (-14.0, 0.0), (8.0, 0.0), (2.0, 8.0)), 520)
    X, _ = g.mesh()
    tr = np.where(X > -4.0, psi, 0.0)
    big = np.zeros((8 * g.n[1], g.n[0]), complex)
    big[: g.n[1]] = tr
    spec = (np.abs(np.fft.fft(big, axis=0)) ** 2).sum(axis=1)
    ky = 2 * np.pi * np.fft.fftfreq(big.shape[0], g.spacing[1])
    o = np.argsort(ky)
    ky, spec = ky[o], spec[o] / spec.max()
    theory = np.cos(ky * d / 2) ** 2 * np.sinc(ky * w / (2 * np.pi)) ** 2
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=110)
    ax.plot(ky, theory, color=C_TH, lw=2, label=r"Fraunhofer: $\cos^2(k_yd/2)\,\mathrm{sinc}^2(k_yw/2)$")
    ax.plot(ky, spec, color=C_SIM, lw=1.5, label="simulation (transmitted wave)")
    for n in (1, 2, 3):
        ax.axvline(2 * np.pi * n / d, color=C_ACC, lw=0.8, alpha=0.6)
        ax.axvline(-2 * np.pi * n / d, color=C_ACC, lw=0.8, alpha=0.6)
    ax.set_xlim(-9, 9)
    ax.set_xlabel(r"transverse wavenumber $k_y$   (green: $k_y = 2\pi n/d$)")
    ax.set_ylabel("intensity (normalised)")
    ax.set_title(f"Double slit far field, d = {d}, w = {w}, k0 = 8")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}  corr = {np.corrcoef(spec[np.abs(ky) < 7], theory[np.abs(ky) < 7])[0, 1]:.4f}")


def plot_convergence(path):
    g = wp.Grid.line(512, 30.0)
    V = 0.1 * g.axes[0] ** 4
    psi0 = wp.gaussian_packet(g, -2.0, 1.0, 0.7)
    T = 2.0

    def run(dt):
        return wp.SplitStepSolver(g, V, dt=dt).step(psi0, int(round(T / dt)))

    ref = run(T / 16384)
    dts = T / 2 ** np.arange(4, 11)
    errs = [np.sqrt(np.sum(np.abs(run(dt) - ref) ** 2) * g.dV) for dt in dts]

    # energy drift over a long run, and spreading of a free packet
    gl = wp.Grid.line(2048, 200.0)
    s = wp.SplitStepSolver(gl, dt=0.05)
    psi = wp.gaussian_packet(gl, 0.0, 0.0, 1.5)
    ts, widths = [0.0], [wp.position_spread(gl, psi)[0]]
    for _ in range(40):
        psi = s.step(psi, 10)
        ts.append(s.t)
        widths.append(wp.position_spread(gl, psi)[0])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), dpi=110)
    a1.loglog(dts, errs, "o-", color=C_SIM, label="split-step error")
    a1.loglog(dts, errs[0] * (dts / dts[0]) ** 2, "--", color=C_TH, label=r"$\propto\Delta t^2$")
    a1.set_xlabel(r"time step $\Delta t$")
    a1.set_ylabel(r"$\|\psi_{\Delta t}-\psi_{\rm ref}\|_2$ at $t=2$")
    a1.set_title("Strang splitting is 2nd order")
    a1.legend(fontsize=9)
    tt = np.linspace(0, ts[-1], 200)
    a2.plot(tt, an.free_gaussian_width(tt, 1.5), color=C_TH, lw=2,
            label=r"$\sigma_0\sqrt{1+(\hbar t/2m\sigma_0^2)^2}$")
    a2.plot(ts[::2], widths[::2], "o", color=C_SIM, ms=5, label="simulation")
    a2.set_xlabel("t")
    a2.set_ylabel(r"$\sigma(t)$")
    a2.set_title("Free Gaussian spreading")
    a2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def gallery(path, names=("double-slit", "crystal", "stadium", "tunnelling"), size=300):
    tiles = []
    times = {"double-slit": 4.6, "crystal": 4.0, "stadium": 9.0, "tunnelling": 4.5}
    for name in names:
        sc = scenes.get(name)
        s = sc.solver()
        psi = s.step(sc.psi0, int(round(times[name] / sc.dt)))
        tiles.append(render.to_image(render.frame(psi, sc.V, "phase", None, sc.vmax), size))
    gap = 6
    img = Image.new("RGB", (len(tiles) * size + (len(tiles) - 1) * gap, size), (13, 17, 23))
    for i, t in enumerate(tiles):
        img.paste(t, (i * (size + gap), 0))
    img.save(path)
    print(f"  {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["gifs", "plots"], default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if args.only in (None, "gifs"):
        scene_gif("double-slit", os.path.join(OUT, "double-slit.gif"), frames=90, size=400, fps=20)
        for name in ("stadium", "crystal", "tunnelling", "oscillator"):
            scene_gif(name, os.path.join(OUT, f"{name}.gif"), frames=72, size=260, fps=18)
        tunnelling_1d_gif(os.path.join(OUT, "tunnelling-1d.gif"))
    if args.only in (None, "plots"):
        gallery(os.path.join(OUT, "gallery.png"))
        plot_tunnelling(os.path.join(OUT, "tunnelling-vs-theory.png"))
        plot_double_slit(os.path.join(OUT, "double-slit-vs-fraunhofer.png"))
        plot_convergence(os.path.join(OUT, "convergence.png"))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
