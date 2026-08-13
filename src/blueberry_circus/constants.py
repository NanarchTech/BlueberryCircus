"""Physical constants and unit systems for BlueberryCircus.

All SED dynamics in this package are expressed with an *explicit* vacuum
permittivity ``eps0``. The capacitive storage of the fluctuating electric
field energy,

    u_E = 1/2 * eps0 * <E^2>,

is the term that sets the absolute scale of the zero-point background (Puthoff,
Phys. Rev. D 35, 3266 (1987)). We therefore never absorb ``eps0`` into an
effective coupling; it is carried through every formula and is a free field of
the :class:`Units` dataclass so that scaled test systems remain dimensionally
self-consistent.

2018 CODATA SI values are used for the physical system.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

# --- CODATA 2018 SI constants -------------------------------------------------
EPS0 = 8.8541878128e-12      # vacuum permittivity        [F/m]
MU0 = 1.25663706212e-6       # vacuum permeability        [H/m]
C = 2.99792458e8             # speed of light             [m/s]
HBAR = 1.054571817e-34       # reduced Planck constant    [J s]
E_CHARGE = 1.602176634e-19   # elementary charge          [C]
M_E = 9.1093837015e-31       # electron mass              [kg]
K_E = 1.0 / (4.0 * math.pi * EPS0)            # Coulomb constant [N m^2/C^2]
ALPHA = E_CHARGE**2 / (4.0 * math.pi * EPS0 * HBAR * C)   # fine-structure
A0 = HBAR / (M_E * C * ALPHA)                 # Bohr radius [m]


def radiation_reaction_time(charge: float = E_CHARGE,
                            mass: float = M_E,
                            eps0: float = EPS0,
                            c: float = C) -> float:
    """Abraham--Lorentz characteristic time ``tau = q^2 / (6 pi eps0 m c^3)``.

    This is the single parameter controlling radiative damping. For the electron
    ``tau ~ 6.26e-24 s`` (``tau * omega << 1`` for all bound-state frequencies,
    i.e. the radiation reaction is a tiny perturbation -- the regime in which the
    Landau--Lifshitz reduction of order is exact to leading order).
    """
    return charge**2 / (6.0 * math.pi * eps0 * mass * c**3)


@dataclass(frozen=True)
class Units:
    """A closed, dimensionally consistent constant set.

    Two factory methods are provided:

    * :meth:`si` -- real electron-in-vacuum constants (the physical system).
    * :meth:`scaled` -- an abstract system (``m = omega0 = 1``) in which the
      radiation-reaction time ``tau`` is chosen so that the damping
      ``gamma = tau * omega0^2`` is *numerically* moderate. This lets the
      time-domain integrator reach the stationary state in a short run while the
      analytic transfer-function oracle (which is unit-agnostic) still applies.
      The scaled system is used only for integrator-fidelity and
      fluctuation--dissipation tests; the physical ground-state result is
      certified in SI.
    """
    eps0: float
    c: float
    hbar: float
    charge: float
    mass: float

    @property
    def tau(self) -> float:
        return radiation_reaction_time(self.charge, self.mass, self.eps0, self.c)

    @property
    def k_e(self) -> float:
        return 1.0 / (4.0 * math.pi * self.eps0)

    @classmethod
    def si(cls) -> "Units":
        return cls(eps0=EPS0, c=C, hbar=HBAR, charge=E_CHARGE, mass=M_E)

    @classmethod
    def scaled(cls, gamma_over_omega0: float = 0.05, omega0: float = 1.0,
               mass: float = 1.0, hbar: float = 1.0) -> "Units":
        """Abstract units with target damping ``gamma = gamma_over_omega0 * omega0``.

        We fix ``mass``, ``hbar``, ``c = 1`` and ``omega0`` and then solve for the
        charge that yields the requested ``tau = gamma / omega0^2``:
        ``tau = q^2 / (6 pi eps0 m c^3)`` with ``eps0 = 1``, ``c = 1``  =>
        ``q = sqrt(6 pi m tau)``.
        """
        c = 1.0
        eps0 = 1.0
        tau = gamma_over_omega0 * omega0 / omega0**2
        charge = math.sqrt(6.0 * math.pi * mass * tau)  # eps0=c=1
        return cls(eps0=eps0, c=c, hbar=hbar, charge=charge, mass=mass)


SI = Units.si()
