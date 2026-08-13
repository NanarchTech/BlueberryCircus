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

    Three factory methods are provided:

    * :meth:`si` -- real electron-in-vacuum constants (the physical system).
    * :meth:`bohr` -- physical hydrogen in atomic units, with the fine-structure
      constant retained in ``c`` and radiation reaction.
    * :meth:`scaled` -- an abstract system (``m = omega0 = 1``) in which the
      radiation-reaction time ``tau`` is chosen so that the damping
      ``gamma = tau * omega0^2`` is *numerically* moderate. This lets the
      time-domain integrator reach the stationary state in a short run while the
      analytic transfer-function oracle (which is unit-agnostic) still applies.
      The scaled system is an accelerated oscillator and numerical-stress
      system only. It is used for integrator-fidelity and
      fluctuation--dissipation tests; it does not reproduce physical hydrogen
      coupling or timescales.
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
    def bohr(cls) -> "Units":
        """Physical hydrogen normalization in Bohr (atomic) units.

        The definitions ``m = hbar = e = a0 = omega_B = 1`` require
        ``4 pi eps0 = 1`` and ``c = 1/alpha``. Consequently the Abraham--Lorentz
        time in units of the Bohr time is
        ``tau = (2/3) alpha^3 = beta^2``. Unlike :meth:`scaled`, no coupling is
        accelerated for numerical convenience.
        """
        return cls(
            eps0=1.0 / (4.0 * math.pi),
            c=1.0 / ALPHA,
            hbar=1.0,
            charge=1.0,
            mass=1.0,
        )

    @classmethod
    def scaled(cls, gamma_over_omega0: float = 0.05, omega0: float = 1.0,
               mass: float = 1.0, hbar: float = 1.0) -> "Units":
        """Accelerated oscillator/stress units with target numerical damping.

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
BOHR = Units.bohr()


def setterfield_rescale(units: Units, U: float) -> Units:
    """Apply Setterfield's speculative *static* cosmological co-scaling profile.

    For a positive scale ``U``, the proposal is represented exactly as
    ``hbar, eps0 -> U``, ``c -> U^-1``, ``e -> U^1/2``, and ``m -> U^2``.
    This function implements a hypothesis map, not an empirical endorsement.
    Its principal use is testing whether the resulting constant set is merely a
    time-reparameterization of dimensionless hydrogen dynamics.
    """
    U = float(U)
    if not math.isfinite(U) or U <= 0.0:
        raise ValueError("Setterfield scale U must be finite and positive")
    return Units(
        eps0=units.eps0 * U,
        c=units.c / U,
        hbar=units.hbar * U,
        charge=units.charge * math.sqrt(U),
        mass=units.mass * U**2,
    )
