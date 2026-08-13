"""Observables extracted from trajectories.

"""
from __future__ import annotations

import numpy as np

from .constants import Units, SI
from .dynamics import Trajectory, Particle
from .oracles import larmor_power


def position_variance(traj: Trajectory, burn_in: float = 0.0):
    """Per-component variance (Var(x),Var(y),Var(z)) after a burn-in fraction."""
    i0 = int(burn_in * len(traj.t))
    x = traj.x[i0:]
    return x.var(axis=0)


def radial_distribution(traj: Trajectory, bins=80, rmax=None, burn_in: float = 0.0):
    """Normalized radial probability density P(r) (integral P dr = 1)."""
    i0 = int(burn_in * len(traj.t))
    r = traj.r[i0:]
    rmax = rmax if rmax is not None else float(r.max())
    hist, edges = np.histogram(r, bins=bins, range=(0.0, rmax), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def total_energy(traj: Trajectory, potential, particle: Particle):
    """Mechanical energy E(t) = 1/2 m v^2 + U(x) along the trajectory."""
    ke = 0.5 * particle.mass * np.sum(traj.v**2, axis=1)
    pe = np.array([potential.potential(xi) for xi in traj.x]) if potential else 0.0
    return ke + pe


def angular_momentum(traj: Trajectory, particle: Particle):
    """L(t) = m x x v (vector)."""
    return particle.mass * np.cross(traj.x, traj.v)


def mean_radiated_power(traj: Trajectory, units: Units = SI, burn_in: float = 0.0):
    """Mean Larmor power <m tau a^2> from finite-differenced acceleration."""
    i0 = int(burn_in * len(traj.t))
    t = traj.t[i0:]; v = traj.v[i0:]
    a = np.gradient(v, t, axis=0)
    P = larmor_power(np.linalg.norm(a, axis=1), units)
    return float(P.mean())


def mean_absorbed_power(traj: Trajectory, field, particle: Particle,
                        burn_in: float = 0.0, dipole: bool = True):
    """Mean power delivered by the field, <q E . v>."""
    i0 = int(burn_in * len(traj.t))
    t = traj.t[i0:]; x = traj.x[i0:]; v = traj.v[i0:]
    P = np.empty(len(t))
    for k in range(len(t)):
        r_eval = np.zeros(3) if dipole else x[k]
        P[k] = particle.charge * np.dot(field.E(r_eval, t[k]), v[k])
    return float(P.mean())
