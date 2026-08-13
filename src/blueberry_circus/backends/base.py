"""The backend abstraction (Strawberry-Fields-style three-method contract).

A :class:`Backend` is the numeric kernel that integrates a *compiled* program's
equation of motion. The Python layer (Program / Engine / compile passes) is the
Strawberry-Fields analogue; a Backend is the Walrus analogue -- it does the
number crunching and nothing else. All backends consume the SAME pure-data inputs
(SI numbers + a frozen ZPF realization) and return a :class:`~blueberry_circus.dynamics.Trajectory`,
so the NumPy reference backend is the oracle every other backend is diffed against.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Backend(ABC):
    """Integrate one compiled single-particle SED program over a time grid."""

    name: str = "abstract"

    @abstractmethod
    def integrate(self, *, field, potential, particle, t_grid, x0, v0,
                  rr: str, units, dipole: bool):
        """Return a :class:`Trajectory`. Inputs are pure data; see :mod:`dynamics`."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Backend {self.name!r}>"
