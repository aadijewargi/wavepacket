r"""Initial wavefunctions.

A Gaussian wavepacket with position spread :math:`\sigma` (the standard
deviation of :math:`|\psi|^2`) and mean wavenumber :math:`k_0` is

.. math::

    \psi(x,0) = (2\pi\sigma^2)^{-1/4}
    \exp\!\Big[-\frac{(x-x_0)^2}{4\sigma^2} + i k_0 x\Big],

and in 2D the product of two such factors.
"""

from __future__ import annotations

import numpy as np

from .grid import Grid, integrate


def gaussian_packet(grid: Grid, center, k0, sigma) -> np.ndarray:
    """Normalised Gaussian wavepacket.

    ``center``, ``k0`` and ``sigma`` may be scalars (1D) or length-``ndim``
    sequences ``(x, y)``.
    """
    center = np.broadcast_to(np.atleast_1d(center).astype(float), (grid.ndim,))
    k0 = np.broadcast_to(np.atleast_1d(k0).astype(float), (grid.ndim,))
    sigma = np.broadcast_to(np.atleast_1d(sigma).astype(float), (grid.ndim,))
    psi = np.ones(grid.shape, dtype=complex)
    for X, c, k, s in zip(grid.mesh(), center, k0, sigma):
        psi = psi * (2 * np.pi * s ** 2) ** -0.25 * np.exp(
            -((X - c) ** 2) / (4 * s ** 2) + 1j * k * X
        )
    return psi


def normalize(grid: Grid, psi: np.ndarray) -> np.ndarray:
    return psi / np.sqrt(integrate(grid, np.abs(psi) ** 2))


def harmonic_ground_state(grid: Grid, omega: float = 1.0, mass: float = 1.0,
                          hbar: float = 1.0, center=0.0, k0=0.0) -> np.ndarray:
    r"""Displaced/boosted oscillator ground state (a coherent state).

    Its width :math:`\sigma = \sqrt{\hbar/(2m\omega)}` stays constant while the
    centre follows the classical trajectory.
    """
    sigma = np.sqrt(hbar / (2 * mass * omega))
    return gaussian_packet(grid, center, k0, sigma)
