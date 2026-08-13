"""Binding potentials and their force Jacobians.

Each potential exposes ``force(x)`` and ``force_jacobian(x) = d force / d x``.
The Jacobian is required by the Landau--Lifshitz radiation-reaction reduction
(:mod:`dynamics`), which needs ``d F_ext / dt = J . v``.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .constants import Units, SI


@dataclass
class Harmonic:
    """Isotropic harmonic binding ``U = 1/2 m omega0^2 r^2``."""
    omega0: float
    mass: float = 1.0

    def potential(self, x):
        x = np.asarray(x, float)
        return 0.5 * self.mass * self.omega0**2 * np.dot(x, x)

    def force(self, x):
        x = np.asarray(x, float)
        return -self.mass * self.omega0**2 * x

    def force_jacobian(self, x):
        return -self.mass * self.omega0**2 * np.eye(len(np.atleast_1d(x)))


@dataclass
class Coulomb:
    """Attractive Coulomb binding ``U = -Z k_e q^2 / r`` (electron-nucleus).

    A Plummer softening length ``softening`` (default 0) regularizes the r->0
    singularity for finite-step integration; set it well below the orbit scale.
    """
    Z: float = 1.0
    units: Units = SI
    softening: float = 0.0
    charge: float = None      # defaults to units.charge
    mass: float = None        # defaults to units.mass

    def __post_init__(self):
        if self.charge is None:
            self.charge = self.units.charge
        if self.mass is None:
            self.mass = self.units.mass
        self._coef = self.Z * self.units.k_e * self.charge**2

    def _rs(self, x):
        x = np.asarray(x, float)
        return np.sqrt(np.dot(x, x) + self.softening**2)

    def potential(self, x):
        return -self._coef / self._rs(x)

    def force(self, x):
        x = np.asarray(x, float)
        rs = self._rs(x)
        return -self._coef * x / rs**3

    def force_jacobian(self, x):
        x = np.asarray(x, float)
        rs = self._rs(x)
        I = np.eye(len(x))
        return -self._coef * (I / rs**3 - 3.0 * np.outer(x, x) / rs**5)
