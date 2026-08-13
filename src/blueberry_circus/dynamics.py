"""Equations of motion: Lorentz force, radiation reaction, RK4 integrator.

The electron obeys

    m a = F_pot(x) + q (E + v x B) + F_rad,

with the radiation reaction handled by the **Landau--Lifshitz reduction of
order** -- the standard non-runaway form of Abraham--Lorentz valid when
``tau omega << 1`` (always true for bound states):

    F_rad = tau * d/dt F_ext ≈ tau * ( J_pot(x) . v + q dE/dt ),

where ``J_pot = d F_pot / d x``. The magnetic and ``(v.grad)E`` contributions to
``dF_ext/dt`` are O(v/c) smaller and are omitted from ``F_rad`` (they remain in
``F_ext`` itself unless the dipole approximation is used). For the *linear*
oscillator this reproduces exactly :func:`oracles.transfer_landau_lifshitz`,
which is the analytic oracle the integrator is checked against.

The literal Abraham--Lorentz third-derivative form ``m tau xdddot`` admits
runaway / pre-accelerating solutions and is therefore **not** integrated
directly; see ``docs/theory.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .constants import Units, SI


@dataclass
class Particle:
    charge: float
    mass: float

    @classmethod
    def electron(cls, units: Units = SI):
        return cls(charge=units.charge, mass=units.mass)


def _accel(x, v, t, *, particle, potential, field, units, rr, dipole):
    q, m = particle.charge, particle.mass
    r_eval = np.zeros(3) if dipole else x
    F = potential.force(x) if potential is not None else np.zeros(3)
    if field is not None:
        E = field.E(r_eval, t)
        F = F + q * E
        if not dipole:
            B = field.B(r_eval, t)
            F = F + q * np.cross(v, B)
    if rr == "landau_lifshitz":
        dFdt = np.zeros(3)
        if potential is not None:
            dFdt = dFdt + potential.force_jacobian(x) @ v
        if field is not None:
            dFdt = dFdt + q * field.dEdt(r_eval, t)
        F = F + units.tau * dFdt
    elif rr not in ("none", None):
        raise ValueError(f"unknown radiation-reaction model {rr!r}")
    return F / m


@dataclass
class Trajectory:
    t: np.ndarray
    x: np.ndarray            # (T,3)
    v: np.ndarray            # (T,3)
    units: Units = SI

    @property
    def r(self):
        return np.linalg.norm(self.x, axis=1)


def integrate(*, field, potential, particle: Particle, t_grid, x0, v0,
              rr: str = "landau_lifshitz", units: Units = SI,
              dipole: bool = True) -> Trajectory:
    """Fixed-step RK4 integration of the SED equation of motion.

    Parameters
    ----------
    field : ZPFBackground or None
    potential : Harmonic/Coulomb or None
    rr : {"landau_lifshitz","none"}
    dipole : evaluate the field at the origin (atom << wavelength).
    """
    t_grid = np.asarray(t_grid, float)
    n = len(t_grid)
    X = np.empty((n, 3)); V = np.empty((n, 3))
    x = np.array(x0, float); v = np.array(v0, float)
    X[0] = x; V[0] = v
    kw = dict(particle=particle, potential=potential, field=field,
              units=units, rr=rr, dipole=dipole)
    for i in range(n - 1):
        t = t_grid[i]; h = t_grid[i + 1] - t
        a1 = _accel(x, v, t, **kw)
        k1x, k1v = v, a1
        a2 = _accel(x + 0.5 * h * k1x, v + 0.5 * h * k1v, t + 0.5 * h, **kw)
        k2x, k2v = v + 0.5 * h * k1v, a2
        a3 = _accel(x + 0.5 * h * k2x, v + 0.5 * h * k2v, t + 0.5 * h, **kw)
        k3x, k3v = v + 0.5 * h * k2v, a3
        a4 = _accel(x + h * k3x, v + h * k3v, t + h, **kw)
        k4x, k4v = v + h * k3v, a4
        x = x + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        X[i + 1] = x; V[i + 1] = v
    return Trajectory(t_grid, X, V, units)
