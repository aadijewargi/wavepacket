"""Ready-made 2D experiments (natural units: hbar = m = 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from . import potentials as P
from .grid import Grid
from .states import gaussian_packet


@dataclass
class Scene:
    name: str
    title: str
    grid: Grid
    psi0: np.ndarray
    V: np.ndarray
    W: Optional[np.ndarray]
    dt: float
    t_final: float
    vmax: float  # value of V drawn fully opaque

    def solver(self):
        from .solver import SplitStepSolver
        return SplitStepSolver(self.grid, self.V, self.dt, absorber=self.W)


def double_slit(n: int = 512) -> Scene:
    g = Grid.square(n, 40.0)
    V = P.slits(g, 2, separation=4.0, slit_width=1.0, wall_x=-5.0, thickness=0.5, height=200.0)
    V = P.smooth(g, V, 1.5 * g.spacing[0])
    psi = gaussian_packet(g, (-12.0, 0.0), (5.0, 0.0), (1.5, 6.0))
    return Scene("double-slit", "Double slit", g, psi, V, P.absorbing_boundary(g, 4.0, 6.0),
                 0.005, 5.5, 200.0)


def single_slit(n: int = 512) -> Scene:
    g = Grid.square(n, 40.0)
    V = P.slits(g, 1, slit_width=1.6, wall_x=-5.0, thickness=0.5, height=200.0)
    V = P.smooth(g, V, 1.5 * g.spacing[0])
    psi = gaussian_packet(g, (-12.0, 0.0), (5.0, 0.0), (1.5, 6.0))
    return Scene("single-slit", "Single slit", g, psi, V, P.absorbing_boundary(g, 4.0, 6.0),
                 0.005, 5.5, 200.0)


def tunnelling(n: int = 512) -> Scene:
    """A packet hits a barrier exactly as tall as its mean energy: T is about 1/2."""
    g = Grid.square(n, 40.0)
    k0 = 4.0
    E = 0.5 * k0 ** 2
    V = P.rectangular_barrier(g, E, -0.25, 0.5)
    psi = gaussian_packet(g, (-9.0, 0.0), (k0, 0.0), (1.8, 2.5))
    return Scene("tunnelling", "Tunnelling through a barrier", g, psi, V,
                 P.absorbing_boundary(g, 4.0, 6.0), 0.005, 4.5, E)


def stadium(n: int = 512) -> Scene:
    g = Grid.square(n, 32.0)
    V = P.smooth(g, P.stadium(g, half_length=5.0, radius=8.5, height=400.0), 1.5 * g.spacing[0])
    psi = gaussian_packet(g, (-6.0, -2.0), (5.0, 3.0), 1.2)
    return Scene("stadium", "Quantum stadium billiard", g, psi, V, None, 0.005, 15.0, 400.0)


def crystal(n: int = 512) -> Scene:
    g = Grid.square(n, 40.0)
    V = np.zeros(g.shape)
    a = 2.4  # lattice constant
    for i in range(4):
        for j in range(-8, 9):
            V += P.gaussian_bump(g, 40.0, (0.5 + i * a, j * a + (a / 2) * (i % 2)), 0.3)
    psi = gaussian_packet(g, (-10.0, 0.0), (5.0, 0.0), (1.8, 4.5))
    return Scene("crystal", "Scattering off a crystal", g, psi, V,
                 P.absorbing_boundary(g, 4.0, 6.0), 0.005, 5.0, 40.0)


def oscillator(n: int = 256) -> Scene:
    g = Grid.square(n, 24.0)
    V = P.harmonic(g, 1.0)
    # a squeezed, displaced packet: its width breathes twice per orbit
    psi = gaussian_packet(g, (5.0, 0.0), (0.0, 5.0), (0.35, 1.2))
    # outline drawn at V = 50, i.e. r = 10
    return Scene("oscillator", "Squeezed state in a harmonic trap", g, psi, V, None,
                 0.005, 2 * np.pi, 100.0)


SCENES: Dict[str, Callable[..., Scene]] = {
    "double-slit": double_slit,
    "single-slit": single_slit,
    "tunnelling": tunnelling,
    "stadium": stadium,
    "crystal": crystal,
    "oscillator": oscillator,
}


def get(name: str, **kw) -> Scene:
    try:
        return SCENES[name](**kw)
    except KeyError:
        raise KeyError(f"unknown scene {name!r}; choose from {sorted(SCENES)}") from None
