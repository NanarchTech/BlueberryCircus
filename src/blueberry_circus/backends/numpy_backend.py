"""The NumPy reference backend -- the deterministic trust root.

Wraps the hand-rolled RK4 + Landau-Lifshitz integrator in
:mod:`blueberry_circus.dynamics`. It is bit-reproducible (seeded RNG, no network)
and is the oracle the Rust backend's cross-language enclosure test is checked
against.
"""
from __future__ import annotations

from .base import Backend
from ..dynamics import integrate as _integrate


class NumpyBackend(Backend):
    name = "numpy"

    def integrate(self, *, field, potential, particle, t_grid, x0, v0,
                  rr: str, units, dipole: bool):
        return _integrate(field=field, potential=potential, particle=particle,
                          t_grid=t_grid, x0=x0, v0=v0, rr=rr, units=units,
                          dipole=dipole)
