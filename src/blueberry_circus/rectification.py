"""Nieuwenhuizen's near-ionization energy-space rectification threshold.

The result implemented here is conditional drift for an almost parabolic
Kepler orbit, averaged over one revolution. It is neither a stationary-state
claim nor a mechanism for extracting net work from an equilibrium vacuum.

Reference: T. M. Nieuwenhuizen, arXiv:1611.10200, Eqs. (2.30)--(2.37).
"""
from __future__ import annotations

import math

import numpy as np

from .oracles import beta_coefficient


def critical_angular_momentum() -> float:
    """Return ``Lc = f(0) = 16 / (5 pi sqrt(3))`` in units of ``hbar``."""
    return 16.0 / (5.0 * math.pi * math.sqrt(3.0))


def threshold_quadrature(order: int = 96) -> float:
    """Independently evaluate Nieuwenhuizen's improper integral for ``f(0)``.

    The nested domain ``-inf < u < inf``, ``-inf < v < u`` is mapped to a
    square and integrated by tensor-product Gauss--Legendre quadrature. No
    closed-form value enters the numerical calculation.
    """
    if (isinstance(order, bool) or not isinstance(order, (int, np.integer)) or
            order < 2):
        raise ValueError("quadrature order must be an integer >= 2")

    nodes, weights = np.polynomial.legendre.leggauss(int(order))

    # u = tan(pi x/2), x in (-1, 1), maps the outer real line.
    u = np.tan(0.5 * math.pi * nodes)
    du_dx = 0.5 * math.pi / np.cos(0.5 * math.pi * nodes) ** 2

    # v = u - t, t=(1+y)/(1-y), maps v in (-inf, u) to y in (-1,1).
    t = (1.0 + nodes) / (1.0 - nodes)
    dt_dy = 2.0 / (1.0 - nodes) ** 2
    uu = u[:, None]
    vv = uu - t[None, :]

    numerator = (5.0 + 3.0 * uu**2 + 8.0 * uu * vv - vv**2 +
                 4.0 * uu**3 * vv + uu**2 * vv**2)
    denominator = ((1.0 + uu**2) ** 2 *
                   (3.0 + uu**2 + uu * vv + vv**2) ** 4)
    jacobian = du_dx[:, None] * dt_dy[None, :]
    integral = np.sum(weights[:, None] * weights[None, :] * jacobian *
                      numerator / denominator)
    prefactor = 2.0**3 * 3.0**4 / (5.0 * math.pi**2)
    return float(prefactor * integral)


def near_ionization_drift(L: float, Z: float = 1.0) -> float:
    """Return per-revolution drift ``3 pi beta^2 (Lc-L) / L^6``.

    ``L`` is dimensionless angular momentum in units of ``hbar``. Positive
    drift below ``Lc`` means energy gain toward ionization; negative drift above
    ``Lc`` means energy loss. This is the near-zero-energy asymptote only.
    """
    L = float(L)
    Z = float(Z)
    if not math.isfinite(L) or L <= 0.0:
        raise ValueError("L must be finite and positive")
    if not math.isfinite(Z) or Z <= 0.0:
        raise ValueError("Z must be finite and positive")
    beta = beta_coefficient(Z)
    return 3.0 * math.pi * beta**2 * (critical_angular_momentum() - L) / L**6


__all__ = [
    "critical_angular_momentum",
    "threshold_quadrature",
    "near_ionization_drift",
]
