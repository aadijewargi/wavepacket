"""Uniform periodic grids in position space and their reciprocal (momentum) grids."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np


@dataclass
class Grid:
    """An N-dimensional uniform grid on a periodic box.

    Parameters
    ----------
    n : sequence of int
        Number of points along each axis (powers of two are fastest for FFTs).
    length : sequence of float
        Physical side length of the box along each axis. The box is centred on
        the origin, i.e. axis ``d`` spans ``[-L_d/2, L_d/2)``.

    Notes
    -----
    Arrays are indexed ``[i_x]`` in 1D and ``[i_y, i_x]`` in 2D (image order),
    so a 2D array can be passed straight to ``imshow``.
    """

    n: Tuple[int, ...]
    length: Tuple[float, ...]
    axes: Tuple[np.ndarray, ...] = field(init=False, repr=False)
    kaxes: Tuple[np.ndarray, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.n = tuple(int(v) for v in np.atleast_1d(self.n))
        self.length = tuple(float(v) for v in np.atleast_1d(self.length))
        if len(self.n) != len(self.length):
            raise ValueError("n and length must have the same number of dimensions")
        if len(self.n) not in (1, 2):
            raise ValueError("only 1D and 2D grids are supported")
        self.axes = tuple(
            -L / 2 + L * np.arange(N) / N for N, L in zip(self.n, self.length)
        )
        self.kaxes = tuple(
            2 * np.pi * np.fft.fftfreq(N, d=L / N) for N, L in zip(self.n, self.length)
        )

    # ------------------------------------------------------------------ basics
    @property
    def ndim(self) -> int:
        return len(self.n)

    @property
    def shape(self) -> Tuple[int, ...]:
        # image order: (ny, nx) in 2D
        return tuple(reversed(self.n))

    @property
    def spacing(self) -> Tuple[float, ...]:
        return tuple(L / N for N, L in zip(self.n, self.length))

    @property
    def dV(self) -> float:
        """Volume element (dx in 1D, dx*dy in 2D)."""
        return float(np.prod(self.spacing))

    @property
    def extent(self) -> Tuple[float, ...]:
        """``imshow`` extent ``(xmin, xmax, ymin, ymax)`` for a 2D grid."""
        x, y = self.axes[0], self.axes[1]
        dx, dy = self.spacing
        return (x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2)

    # ---------------------------------------------------------------- meshes
    def mesh(self) -> Tuple[np.ndarray, ...]:
        """Position coordinates broadcast to the full grid: ``(X,)`` or ``(X, Y)``."""
        if self.ndim == 1:
            return (self.axes[0],)
        X, Y = np.meshgrid(self.axes[0], self.axes[1], indexing="xy")
        return X, Y

    def kmesh(self) -> Tuple[np.ndarray, ...]:
        """Wavenumber coordinates in FFT order: ``(KX,)`` or ``(KX, KY)``."""
        if self.ndim == 1:
            return (self.kaxes[0],)
        KX, KY = np.meshgrid(self.kaxes[0], self.kaxes[1], indexing="xy")
        return KX, KY

    def k2(self) -> np.ndarray:
        """|k|^2 on the reciprocal grid (FFT order)."""
        return sum(K ** 2 for K in self.kmesh())

    @property
    def k_max(self) -> float:
        """Largest resolvable wavenumber (Nyquist) along the coarsest axis."""
        return min(np.pi / d for d in self.spacing)

    @classmethod
    def square(cls, n: int, length: float) -> "Grid":
        return cls((n, n), (length, length))

    @classmethod
    def line(cls, n: int, length: float) -> "Grid":
        return cls((n,), (length,))


def integrate(grid: Grid, f: np.ndarray) -> float:
    """Riemann sum of ``f`` over the grid (spectrally accurate for smooth periodic f)."""
    return float(np.sum(f).real * grid.dV)


__all__: Sequence[str] = ["Grid", "integrate"]
