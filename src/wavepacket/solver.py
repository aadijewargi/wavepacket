r"""Split-step Fourier integrator for the time-dependent Schrödinger equation.

We solve

.. math::

    i\hbar\,\partial_t \psi = \Big[-\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf r)
    - i W(\mathbf r)\Big]\psi

on a periodic box, where :math:`W\ge 0` is an optional complex absorbing
potential (CAP) that soaks up outgoing waves near the edges.

The propagator over one step is approximated by the symmetric (Strang)
splitting

.. math::

    e^{-i(\hat T+\hat V)\Delta t/\hbar} =
    e^{-i\hat V\Delta t/2\hbar}\,e^{-i\hat T\Delta t/\hbar}\,
    e^{-i\hat V\Delta t/2\hbar} + \mathcal O(\Delta t^3),

where the kinetic factor is diagonal in momentum space (applied with an FFT)
and the potential factor is diagonal in position space.  Each factor is
exactly unitary when :math:`W=0`, so the norm is conserved to round-off, and
the global error is second order in :math:`\Delta t`.
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional

import numpy as np

from .grid import Grid, integrate


class SplitStepSolver:
    """Propagate a wavefunction on a :class:`~wavepacket.grid.Grid`.

    Parameters
    ----------
    grid : Grid
        Spatial grid (1D or 2D).
    potential : ndarray, optional
        Real potential :math:`V` sampled on the grid. Defaults to zero.
    dt : float
        Time step.
    absorber : ndarray, optional
        Non-negative absorbing potential :math:`W` (see
        :func:`wavepacket.potentials.absorbing_boundary`).
    mass, hbar : float
        Particle mass and reduced Planck constant (natural units by default).
    """

    def __init__(
        self,
        grid: Grid,
        potential: Optional[np.ndarray] = None,
        dt: float = 1e-3,
        absorber: Optional[np.ndarray] = None,
        mass: float = 1.0,
        hbar: float = 1.0,
    ) -> None:
        self.grid = grid
        self.mass = float(mass)
        self.hbar = float(hbar)
        self.V = np.zeros(grid.shape) if potential is None else np.asarray(potential, float)
        self.W = None if absorber is None else np.asarray(absorber, float)
        if self.V.shape != grid.shape:
            raise ValueError(f"potential shape {self.V.shape} != grid shape {grid.shape}")
        if self.W is not None and (self.W.shape != grid.shape or np.any(self.W < 0)):
            raise ValueError("absorber must be non-negative with the grid's shape")
        self.t = 0.0
        self._k2 = grid.k2()
        self.set_dt(dt)

    # ------------------------------------------------------------- set-up
    def set_dt(self, dt: float) -> None:
        """Change the time step and rebuild the cached propagator factors."""
        self.dt = float(dt)
        h, m = self.hbar, self.mass
        self._kin = np.exp(-1j * h * self._k2 * self.dt / (2 * m))
        half = self.V.astype(complex)
        if self.W is not None:
            half = half - 1j * self.W
        self._pot_half = np.exp(-1j * half * self.dt / (2 * h))

    def set_potential(self, V: np.ndarray) -> None:
        self.V = np.asarray(V, float)
        self.set_dt(self.dt)

    # ------------------------------------------------------------ stepping
    def step(self, psi: np.ndarray, nsteps: int = 1) -> np.ndarray:
        """Advance ``psi`` by ``nsteps`` time steps and return the new array.

        Consecutive half-steps of the potential are fused, so ``n`` steps cost
        ``n`` forward/inverse FFT pairs and ``n + 1`` potential multiplies.
        """
        if nsteps <= 0:
            return psi
        fft, ifft = np.fft.fftn, np.fft.ifftn
        psi = psi * self._pot_half
        full = self._pot_half ** 2
        for i in range(nsteps):
            psi = ifft(self._kin * fft(psi))
            psi = psi * (full if i < nsteps - 1 else self._pot_half)
        self.t += nsteps * self.dt
        return psi

    def evolve(
        self,
        psi: np.ndarray,
        t_final: float,
        n_frames: int,
        callback: Optional[Callable[[float, np.ndarray], None]] = None,
    ) -> Iterator[np.ndarray]:
        """Yield ``n_frames`` snapshots evenly spaced in ``(t, t + t_final]``."""
        steps_total = int(round(t_final / self.dt))
        per = max(1, steps_total // max(1, n_frames))
        for _ in range(n_frames):
            psi = self.step(psi, per)
            if callback is not None:
                callback(self.t, psi)
            yield psi

    # ---------------------------------------------------------- observables
    def norm(self, psi: np.ndarray) -> float:
        return integrate(self.grid, np.abs(psi) ** 2)

    def kinetic_energy(self, psi: np.ndarray) -> float:
        phik = np.fft.fftn(psi)
        # Parseval: sum|psi|^2 dV = sum|phi_k|^2 dV / N
        N = psi.size
        return float(
            self.hbar ** 2 / (2 * self.mass)
            * np.sum(self._k2 * np.abs(phik) ** 2) * self.grid.dV / N
        )

    def potential_energy(self, psi: np.ndarray) -> float:
        return integrate(self.grid, self.V * np.abs(psi) ** 2)

    def energy(self, psi: np.ndarray) -> float:
        """Total energy :math:`\\langle\\psi|H|\\psi\\rangle` (unnormalised)."""
        return self.kinetic_energy(psi) + self.potential_energy(psi)


# ------------------------------------------------------------------ moments
def expectation_position(grid: Grid, psi: np.ndarray) -> np.ndarray:
    """Return :math:`\\langle \\mathbf r\\rangle` (normalised by the norm)."""
    rho = np.abs(psi) ** 2
    n = rho.sum()
    return np.array([float((X * rho).sum() / n) for X in grid.mesh()])


def position_spread(grid: Grid, psi: np.ndarray) -> np.ndarray:
    """Standard deviation of each position coordinate."""
    rho = np.abs(psi) ** 2
    n = rho.sum()
    out = []
    for X in grid.mesh():
        m1 = (X * rho).sum() / n
        m2 = (X ** 2 * rho).sum() / n
        out.append(np.sqrt(max(m2 - m1 ** 2, 0.0)))
    return np.array(out)


def expectation_momentum(grid: Grid, psi: np.ndarray, hbar: float = 1.0) -> np.ndarray:
    """Return :math:`\\langle \\mathbf p\\rangle` computed in Fourier space."""
    phik = np.abs(np.fft.fftn(psi)) ** 2
    n = phik.sum()
    return np.array([float(hbar * (K * phik).sum() / n) for K in grid.kmesh()])
