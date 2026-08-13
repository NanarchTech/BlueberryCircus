"""Ensemble runners: fluctuation--dissipation balance and radial statistics.

Single ZPF realizations have large power-balance scatter (~20%); the
fluctuation--dissipation theorem (``<P_rad> = <P_abs>`` at stationarity) is a
statement about the *ensemble* mean. :func:`fdt_balance_ensemble` averages the
radiated and absorbed power over many seeded realizations to tighten the balance.
Any backend may be used; NumPy is the default reference.
"""
from __future__ import annotations

import numpy as np

from .constants import Units, SI
from .dynamics import Particle
from .potentials import Harmonic
from .zpf import ZPFBackground
from . import observables as obs


def fdt_balance_ensemble(units: Units, *, omega0: float = 1.0,
                         band=(0.3, 3.0), n_modes: int = 120, n_seeds: int = 24,
                         t_max: float = 500.0, dt: float = 0.03,
                         burn_in: float = 0.4, backend=None) -> dict:
    """Ensemble-mean radiated vs absorbed power for the SHO-in-ZPF (1-D field).

    Returns a dict with ``P_rad``, ``P_abs``, ``ratio``, ``rel_imbalance``
    (``|P_rad-P_abs|/P_abs``), and per-seed arrays. The ensemble mean drives
    ``rel_imbalance`` below the single-realization ~20% scatter.
    """
    if backend is None:
        from .backends import NumpyBackend
        backend = NumpyBackend()
    particle = Particle(units.charge, units.mass)
    pot = Harmonic(omega0, mass=units.mass)
    t = np.arange(0.0, t_max, dt)
    rad = np.empty(n_seeds)
    absb = np.empty(n_seeds)
    for s in range(n_seeds):
        field = ZPFBackground.one_dimensional(band[0], band[1], n_modes,
                                              seed=s, units=units)
        tr = backend.integrate(field=field, potential=pot, particle=particle,
                               t_grid=t, x0=[0, 0, 0], v0=[0, 0, 0],
                               rr="landau_lifshitz", units=units, dipole=True)
        rad[s] = obs.mean_radiated_power(tr, units, burn_in)
        absb[s] = obs.mean_absorbed_power(tr, field, particle, burn_in, dipole=True)
    P_rad = float(rad.mean())
    P_abs = float(absb.mean())
    return {
        "P_rad": P_rad, "P_abs": P_abs,
        "ratio": P_rad / P_abs if P_abs else float("inf"),
        "rel_imbalance": abs(P_rad - P_abs) / abs(P_abs) if P_abs else float("inf"),
        "n_seeds": n_seeds, "rad": rad, "abs": absb,
    }
