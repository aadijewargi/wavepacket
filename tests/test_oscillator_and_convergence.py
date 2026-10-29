"""Harmonic oscillator coherent states and the order of the Strang splitting."""

import numpy as np
import pytest

import wavepacket as wp
from wavepacket import analytic as an
from wavepacket import potentials as P


def test_coherent_state_follows_classical_orbit_and_keeps_its_shape():
    g = wp.Grid.line(512, 40.0)
    omega = 1.0
    V = P.harmonic(g, omega)
    x0 = 4.0
    psi0 = wp.harmonic_ground_state(g, omega, center=x0)
    s = wp.SplitStepSolver(g, V, dt=2 * np.pi / 2000)
    psi = psi0
    sigma = np.sqrt(0.5 / omega)
    for _ in range(8):
        psi = s.step(psi, 250)
        assert wp.expectation_position(g, psi)[0] == pytest.approx(
            an.oscillator_trajectory(s.t, x0, 0.0, omega), abs=1e-4
        )
        assert wp.position_spread(g, psi)[0] == pytest.approx(sigma, rel=1e-4)
    # after one full period the state returns to itself up to a global phase
    overlap = abs(np.sum(np.conj(psi0) * psi) * g.dV)
    assert overlap == pytest.approx(1.0, abs=1e-6)


def test_2d_coherent_state_traces_a_circle():
    g = wp.Grid.square(128, 24.0)
    V = P.harmonic(g, 1.0)
    psi = wp.harmonic_ground_state(g, 1.0, center=(3.0, 0.0), k0=(0.0, 3.0))
    s = wp.SplitStepSolver(g, V, dt=2 * np.pi / 800)
    for _ in range(4):
        psi = s.step(psi, 100)
        r = wp.expectation_position(g, psi)
        np.testing.assert_allclose(r, [3 * np.cos(s.t), 3 * np.sin(s.t)], atol=1e-4)


def test_strang_splitting_is_second_order():
    """Halving dt should cut the error by ~4x in an anharmonic potential."""
    g = wp.Grid.line(512, 30.0)
    V = 0.1 * g.axes[0] ** 4
    psi0 = wp.gaussian_packet(g, -2.0, 1.0, 0.7)
    T = 2.0

    def run(dt):
        s = wp.SplitStepSolver(g, V, dt=dt)
        return s.step(psi0, int(round(T / dt)))

    ref = run(T / 8192)
    dts = [T / 64, T / 128, T / 256]
    errs = [np.sqrt(np.sum(np.abs(run(dt) - ref) ** 2) * g.dV) for dt in dts]
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > 1.9) and np.all(orders < 2.1), orders
