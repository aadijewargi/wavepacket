"""Exact solutions and conservation laws."""

import numpy as np
import pytest

import wavepacket as wp
from wavepacket import analytic as an
from wavepacket import potentials as P


@pytest.fixture
def line():
    return wp.Grid.line(2048, 200.0)


def test_initial_packet_is_normalised_with_correct_moments(line):
    psi = wp.gaussian_packet(line, -20.0, 1.5, 3.0)
    s = wp.SplitStepSolver(line)
    assert s.norm(psi) == pytest.approx(1.0, abs=1e-12)
    assert wp.expectation_position(line, psi)[0] == pytest.approx(-20.0, abs=1e-10)
    assert wp.position_spread(line, psi)[0] == pytest.approx(3.0, rel=1e-10)
    assert wp.expectation_momentum(line, psi)[0] == pytest.approx(1.5, abs=1e-10)
    # E = <p^2>/2m = (k0^2 + sigma_k^2)/2 with sigma_k = 1/(2 sigma)
    assert s.energy(psi) == pytest.approx(0.5 * (1.5 ** 2 + 1 / 36), rel=1e-10)


def test_free_packet_matches_exact_solution(line):
    """With V = 0 the kinetic step is exact, so the error is round-off only."""
    x0, k0, s0 = -30.0, 2.0, 2.0
    psi = wp.gaussian_packet(line, x0, k0, s0)
    s = wp.SplitStepSolver(line, dt=0.01)
    psi = s.step(psi, 1000)
    exact = an.free_gaussian(line.axes[0], s.t, x0, k0, s0)
    assert np.max(np.abs(psi - exact)) < 1e-10


@pytest.mark.parametrize("t", [0.0, 5.0, 10.0, 20.0])
def test_free_packet_spreading_law(line, t):
    s0 = 1.5
    psi = wp.gaussian_packet(line, 0.0, 0.0, s0)
    s = wp.SplitStepSolver(line, dt=0.05)
    psi = s.step(psi, int(round(t / 0.05)))
    assert wp.position_spread(line, psi)[0] == pytest.approx(
        an.free_gaussian_width(t, s0), rel=1e-8
    )


def test_group_velocity_is_hbar_k_over_m():
    g = wp.Grid.line(2048, 200.0)
    mass, k0 = 2.5, 3.0
    psi = wp.gaussian_packet(g, -40.0, k0, 3.0)
    s = wp.SplitStepSolver(g, dt=0.01, mass=mass)
    psi = s.step(psi, 2000)
    assert wp.expectation_position(g, psi)[0] == pytest.approx(-40.0 + k0 / mass * s.t, abs=1e-8)


def test_norm_is_conserved_to_roundoff_in_2d_billiard():
    g = wp.Grid.square(128, 24.0)
    V = P.stadium(g, half_length=3.0, radius=5.0, height=500.0)
    psi = wp.gaussian_packet(g, (-2.0, 1.0), (6.0, 4.0), 1.0)
    s = wp.SplitStepSolver(g, V, dt=0.002)
    psi = s.step(psi, 1500)
    assert s.norm(psi) == pytest.approx(1.0, abs=1e-11)


def test_energy_is_conserved_in_anharmonic_well():
    g = wp.Grid.line(1024, 40.0)
    V = 0.25 * g.axes[0] ** 4 - 2.0 * g.axes[0] ** 2  # double well
    s = wp.SplitStepSolver(g, V, dt=1e-3)
    psi = wp.gaussian_packet(g, -2.0, 1.0, 0.5)
    E0 = s.energy(psi)
    drift = []
    for _ in range(20):
        psi = s.step(psi, 250)
        drift.append(abs(s.energy(psi) - E0) / abs(E0))
    assert max(drift) < 1e-4
    assert s.norm(psi) == pytest.approx(1.0, abs=1e-11)


def test_ehrenfest_theorem_in_double_well():
    r"""d<p>/dt = -<dV/dx>, checked with a centred finite difference."""
    g = wp.Grid.line(1024, 40.0)
    x = g.axes[0]
    V = 0.25 * x ** 4 - 2.0 * x ** 2
    dV = x ** 3 - 4.0 * x
    dt = 1e-3
    s = wp.SplitStepSolver(g, V, dt=dt)
    psi = s.step(wp.gaussian_packet(g, -2.0, 1.0, 0.5), 700)
    h = 20
    p_minus = wp.expectation_momentum(g, psi)[0]
    mid = s.step(psi, h)
    force = -np.sum(dV * np.abs(mid) ** 2) * g.dV
    p_plus = wp.expectation_momentum(g, s.step(mid, h))[0]
    assert (p_plus - p_minus) / (2 * h * dt) == pytest.approx(force, rel=2e-3)


def test_absorbing_boundary_removes_outgoing_wave(line):
    W = P.absorbing_boundary(line, width=20.0, strength=2.0)
    psi = wp.gaussian_packet(line, 40.0, 3.0, 3.0)
    s = wp.SplitStepSolver(line, dt=0.01, absorber=W)
    assert s.norm(s.step(psi, 100)) == pytest.approx(1.0, abs=1e-6)  # not there yet
    psi = s.step(psi, 6000)
    assert s.norm(psi) < 1e-3


def test_rejects_bad_inputs(line):
    with pytest.raises(ValueError):
        wp.SplitStepSolver(line, potential=np.zeros(10))
    with pytest.raises(ValueError):
        wp.SplitStepSolver(line, absorber=-np.ones(line.shape))
    with pytest.raises(ValueError):
        wp.Grid((8, 8, 8), (1, 1, 1))
