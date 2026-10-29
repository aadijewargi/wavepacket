"""Scattering: barrier tunnelling and double-slit diffraction."""

import numpy as np
import pytest

import wavepacket as wp
from wavepacket import analytic as an
from wavepacket import potentials as P


@pytest.mark.parametrize(
    "V0, label",
    [(1.30, "tunnelling, E < V0"), (1.125, "E = V0"), (0.80, "over the barrier, E > V0")],
)
def test_barrier_transmission_matches_analytic(V0, label):
    # dx = 0.05, barrier width 1.0 = 20 cells, packet energy E = k0^2/2 = 1.125
    g = wp.Grid.line(8192, 409.6)
    k0, sigma, a = 1.5, 10.0, 1.0
    V = P.rectangular_barrier(g, V0, 0.0, a)
    s = wp.SplitStepSolver(g, V, dt=0.005)
    psi = s.step(wp.gaussian_packet(g, -80.0, k0, sigma), 22000)  # t = 110
    x = g.axes[0]
    T_num = np.sum(np.abs(psi[x > a]) ** 2) * g.dV
    R_num = np.sum(np.abs(psi[x < 0]) ** 2) * g.dV
    T_exact = an.packet_transmission(k0, sigma, V0, a)
    assert T_num == pytest.approx(T_exact, abs=2e-3), label
    assert T_num + R_num == pytest.approx(1.0, abs=1e-6)


def test_deep_tunnelling_is_exponentially_small():
    g = wp.Grid.line(8192, 409.6)
    k0, sigma, a, V0 = 1.0, 12.0, 3.0, 2.0
    s = wp.SplitStepSolver(g, P.rectangular_barrier(g, V0, 0.0, a), dt=0.005)
    psi = s.step(wp.gaussian_packet(g, -80.0, k0, sigma), 32000)  # t = 160
    T_num = np.sum(np.abs(psi[g.axes[0] > a]) ** 2) * g.dV
    T_exact = an.packet_transmission(k0, sigma, V0, a)
    assert T_exact < 1e-3
    assert T_num == pytest.approx(T_exact, rel=0.05)


def _fraunhofer(ky, d, w):
    return np.cos(ky * d / 2) ** 2 * np.sinc(ky * w / (2 * np.pi)) ** 2


def test_double_slit_fringes_match_fraunhofer_theory():
    """The transverse momentum spectrum behind the slits is the far-field pattern."""
    d, w = 3.0, 0.6
    g = wp.Grid.square(512, 60.0)
    V = P.slits(g, 2, separation=d, slit_width=w, wall_x=-5.0, thickness=0.4, height=200.0)
    W = P.absorbing_boundary(g, 4.0, 5.0)
    psi = wp.gaussian_packet(g, (-14.0, 0.0), (8.0, 0.0), (2.0, 8.0))
    s = wp.SplitStepSolver(g, V, dt=0.005, absorber=W)
    psi = s.step(psi, 520)

    X, _ = g.mesh()
    transmitted = np.where(X > -4.0, psi, 0.0)
    pad = 8  # zero-pad in y for a finer k_y sampling
    big = np.zeros((pad * g.n[1], g.n[0]), complex)
    big[: g.n[1]] = transmitted
    spec = (np.abs(np.fft.fft(big, axis=0)) ** 2).sum(axis=1)
    ky = 2 * np.pi * np.fft.fftfreq(big.shape[0], g.spacing[1])
    order = np.argsort(ky)
    ky, spec = ky[order], spec[order] / spec.max()

    theory = _fraunhofer(ky, d, w)
    window = np.abs(ky) < 7.0
    corr = np.corrcoef(spec[window], theory[window])[0, 1]
    assert corr > 0.98

    # positions of the first and second order maxima on the +k_y side
    for n in (1, 2):
        guess = an.double_slit_peaks(d, (n,))[0]
        sel = np.abs(ky - guess) < 0.8
        k_num = ky[sel][np.argmax(spec[sel])]
        k_th = ky[sel][np.argmax(theory[sel])]
        assert k_num == pytest.approx(k_th, rel=0.03)
        assert k_num == pytest.approx(guess, rel=0.06)
