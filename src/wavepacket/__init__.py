"""wavepacket: the time-dependent Schrödinger equation, solved with split-step Fourier."""

from .grid import Grid, integrate
from .solver import (
    SplitStepSolver,
    expectation_momentum,
    expectation_position,
    position_spread,
)
from .states import gaussian_packet, harmonic_ground_state, normalize
from . import analytic, potentials

__version__ = "0.1.0"

__all__ = [
    "Grid",
    "integrate",
    "SplitStepSolver",
    "expectation_position",
    "expectation_momentum",
    "position_spread",
    "gaussian_packet",
    "harmonic_ground_state",
    "normalize",
    "analytic",
    "potentials",
]
