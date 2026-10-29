"""Potentials used by the example scenes.

All functions take a :class:`~wavepacket.grid.Grid` and return a real array of
the grid's shape. Hard walls are modelled as finite but tall barriers
(``height`` well above the packet energy), which is what the split-step method
handles gracefully.
"""

from __future__ import annotations

import numpy as np

from .grid import Grid


def free(grid: Grid) -> np.ndarray:
    return np.zeros(grid.shape)


def harmonic(grid: Grid, omega: float = 1.0, mass: float = 1.0, center=None) -> np.ndarray:
    r"""Isotropic oscillator :math:`V = \tfrac12 m\omega^2 |\mathbf r-\mathbf r_0|^2`."""
    center = np.zeros(grid.ndim) if center is None else np.atleast_1d(center)
    r2 = sum((X - c) ** 2 for X, c in zip(grid.mesh(), center))
    return 0.5 * mass * omega ** 2 * r2


def rectangular_barrier(grid: Grid, height: float, x0: float, width: float) -> np.ndarray:
    r"""Barrier :math:`V=V_0` for :math:`x_0 \le x < x_0 + a` (any dimension, along x).

    Grid points are included when their *cell* lies inside the barrier, so the
    effective width is an integer number of cells; choose ``width`` as a
    multiple of ``dx`` for exact comparison with analytic results.
    """
    X = grid.mesh()[0]
    dx = grid.spacing[0]
    inside = (X >= x0 - 1e-9 * dx) & (X < x0 + width - 1e-9 * dx)
    return np.where(inside, float(height), 0.0)


def slits(
    grid: Grid,
    n_slits: int = 2,
    separation: float = 2.0,
    slit_width: float = 0.5,
    wall_x: float = 0.0,
    thickness: float = 0.3,
    height: float = 1e3,
) -> np.ndarray:
    """A wall at ``x = wall_x`` (perpendicular to x) with ``n_slits`` openings.

    The slits are centred symmetrically about ``y = 0`` with centre-to-centre
    distance ``separation``.
    """
    if grid.ndim != 2:
        raise ValueError("slits need a 2D grid")
    X, Y = grid.mesh()
    wall = np.abs(X - wall_x) < thickness / 2
    centres = (np.arange(n_slits) - (n_slits - 1) / 2) * separation
    opening = np.zeros_like(wall)
    for c in centres:
        opening |= np.abs(Y - c) < slit_width / 2
    return np.where(wall & ~opening, float(height), 0.0)


def stadium(
    grid: Grid,
    half_length: float = 4.0,
    radius: float = 6.0,
    height: float = 1e3,
) -> np.ndarray:
    """Bunimovich stadium billiard: a rectangle capped by two semicircles.

    The straight section spans ``|x| <= half_length`` with half-width
    ``radius``; the potential is zero inside and ``height`` outside. Classical
    motion in the stadium is fully chaotic.
    """
    if grid.ndim != 2:
        raise ValueError("stadium needs a 2D grid")
    X, Y = grid.mesh()
    xc = np.clip(X, -half_length, half_length)
    inside = (X - xc) ** 2 + Y ** 2 <= radius ** 2
    return np.where(inside, 0.0, float(height))


def gaussian_bump(grid: Grid, height: float, center, width: float) -> np.ndarray:
    """A smooth Gaussian hill (or well, if ``height < 0``)."""
    r2 = sum((X - c) ** 2 for X, c in zip(grid.mesh(), np.atleast_1d(center)))
    return height * np.exp(-r2 / (2 * width ** 2))


def absorbing_boundary(grid: Grid, width: float, strength: float = 5.0) -> np.ndarray:
    r"""Quadratic complex absorbing potential in a layer of thickness ``width``.

    Returns :math:`W(\mathbf r) = W_0 \max_d s_d^2` where :math:`s_d\in[0,1]`
    is the depth into the layer along axis ``d``. Use it as the ``absorber``
    argument of :class:`~wavepacket.solver.SplitStepSolver`; it makes the
    periodic box behave like an open domain for outgoing waves.
    """
    s = np.zeros(grid.shape)
    for X, L in zip(grid.mesh(), grid.length):
        d = L / 2 - np.abs(X)  # distance to the nearest edge along this axis
        s = np.maximum(s, np.clip((width - d) / width, 0.0, 1.0))
    return strength * s ** 2
