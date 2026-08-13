"""O5 -- the self-ionization watchdog (the honest headline result).

The reproducible verdict of full-3-D SED hydrogen (Nieuwenhuizen & Liska 2015) is
that the atom does NOT settle into a stable ground state -- run long enough it
self-ionizes. This module instruments that: it reports the *ionization time*
``t_ion`` -- the first time the mechanical energy crosses above a (near-zero)
unbinding threshold and *stays* above it -- or ``None`` if the orbit remains bound
over the window (an honest NULL on stability, never a silent "it's stable").
"""
from __future__ import annotations

import numpy as np

from .constants import Units, SI
from .dynamics import Trajectory, Particle
from .observables import total_energy


def ionization_time(traj: Trajectory, potential, particle: Particle, *,
                    e_threshold: float = 0.0, sustain_frac: float = 0.05):
    """First time ``E(t) > e_threshold`` sustained for ``sustain_frac`` of the run.

    Returns ``(t_ion, detail)`` where ``t_ion`` is ``None`` if the orbit stays
    bound over the window (NULL-first on stability). The default
    ``e_threshold = 0.0`` is the unit-agnostic unbinding criterion: a positive
    total mechanical energy means the orbit is unbound and will escape. ``detail``
    records the diagnostics.
    """
    E = total_energy(traj, potential, particle)
    n = len(E)
    sustain = max(1, int(sustain_frac * n))
    above = E > e_threshold
    t_ion = None
    for i in range(n - sustain + 1):
        if above[i] and np.all(above[i:i + sustain]):
            t_ion = float(traj.t[i])
            break
    detail = {
        "e_threshold": e_threshold,
        "E_initial": float(E[0]),
        "E_final": float(E[-1]),
        "E_max": float(E.max()),
        "r_final": float(traj.r[-1]),
        "r_max": float(traj.r.max()),
        "sustain_steps": sustain,
        "ionized": t_ion is not None,
    }
    return t_ion, detail


def is_bound(traj: Trajectory, potential, particle: Particle, **kw) -> bool:
    """True iff the orbit never sustainedly unbinds over the window."""
    t_ion, _ = ionization_time(traj, potential, particle, **kw)
    return t_ion is None
