r"""Closed-form results used to validate the solver."""

from __future__ import annotations

import numpy as np


def free_gaussian(x, t, x0=0.0, k0=0.0, sigma0=1.0, mass=1.0, hbar=1.0):
    r"""Exact free-particle evolution of a 1D Gaussian packet.

    .. math::

        \psi(x,t) = \frac{(2\pi\sigma_0^2)^{-1/4}}{\sqrt{1+i\tau}}
        \exp\!\Big[-\frac{(x-x_0-v t)^2}{4\sigma_0^2(1+i\tau)}
        + i k_0 x - i\frac{\hbar k_0^2 t}{2m}\Big],
        \qquad \tau = \frac{\hbar t}{2 m\sigma_0^2},\; v = \frac{\hbar k_0}{m}.
    """
    x = np.asarray(x, float)
    tau = hbar * t / (2 * mass * sigma0 ** 2)
    v = hbar * k0 / mass
    a = 1 + 1j * tau
    return (
        (2 * np.pi * sigma0 ** 2) ** -0.25 / np.sqrt(a)
        * np.exp(-((x - x0 - v * t) ** 2) / (4 * sigma0 ** 2 * a)
                 + 1j * k0 * x - 1j * hbar * k0 ** 2 * t / (2 * mass))
    )


def free_gaussian_width(t, sigma0=1.0, mass=1.0, hbar=1.0):
    r""":math:`\sigma(t) = \sigma_0\sqrt{1 + (\hbar t / 2m\sigma_0^2)^2}`."""
    return sigma0 * np.sqrt(1 + (hbar * np.asarray(t) / (2 * mass * sigma0 ** 2)) ** 2)


def barrier_transmission(E, V0, a, mass=1.0, hbar=1.0):
    r"""Plane-wave transmission probability through a rectangular barrier.

    For :math:`E<V_0`, with :math:`\kappa=\sqrt{2m(V_0-E)}/\hbar`,

    .. math:: T = \Big[1 + \frac{V_0^2\sinh^2(\kappa a)}{4E(V_0-E)}\Big]^{-1},

    and for :math:`E>V_0` the :math:`\sinh` becomes :math:`\sin` of
    :math:`q a` with :math:`q=\sqrt{2m(E-V_0)}/\hbar` and :math:`V_0-E\to E-V_0`.
    """
    E = np.asarray(E, float)
    out = np.empty_like(E)
    below = E < V0
    above = E > V0
    equal = ~(below | above)
    kap = np.sqrt(2 * mass * np.clip(V0 - E, 0, None)) / hbar
    q = np.sqrt(2 * mass * np.clip(E - V0, 0, None)) / hbar
    with np.errstate(divide="ignore", invalid="ignore"):
        out[below] = 1 / (1 + V0 ** 2 * np.sinh(kap[below] * a) ** 2
                          / (4 * E[below] * (V0 - E[below])))
        out[above] = 1 / (1 + V0 ** 2 * np.sin(q[above] * a) ** 2
                          / (4 * E[above] * (E[above] - V0)))
    out[equal] = 1 / (1 + mass * a ** 2 * V0 / (2 * hbar ** 2))
    return out


def packet_transmission(k0, sigma, V0, a, mass=1.0, hbar=1.0, n=4001):
    r"""Transmission of a Gaussian packet: :math:`\int T(E(k))\,|\phi(k)|^2 dk`.

    The momentum density of the packet is Gaussian with mean :math:`k_0` and
    standard deviation :math:`\sigma_k = 1/(2\sigma)`.
    """
    sk = 1 / (2 * sigma)
    k = np.linspace(max(k0 - 8 * sk, 1e-6), k0 + 8 * sk, n)
    w = np.exp(-((k - k0) ** 2) / (2 * sk ** 2))
    E = (hbar * k) ** 2 / (2 * mass)
    T = barrier_transmission(E, V0, a, mass, hbar)
    return float(_trapz(T * w, k) / _trapz(w, k))


def _trapz(y, x):
    # np.trapz was renamed np.trapezoid in NumPy 2.0
    f = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return f(y, x)


def double_slit_peaks(separation, orders=(1, 2, 3)):
    r"""Transverse wavenumbers of the Fraunhofer interference maxima,
    :math:`k_y = 2\pi n/d`. Equivalent to :math:`\sin\theta_n = n\lambda/d`."""
    return np.array([2 * np.pi * n / separation for n in orders])


def oscillator_trajectory(t, x0, p0, omega=1.0, mass=1.0):
    r"""Classical (and Ehrenfest) motion in a harmonic well."""
    t = np.asarray(t, float)
    return x0 * np.cos(omega * t) + p0 / (mass * omega) * np.sin(omega * t)
