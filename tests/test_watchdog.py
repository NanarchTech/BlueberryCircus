"""O5 self-ionization watchdog (the headline result)."""
import numpy as np

from blueberry_circus.constants import Units
from blueberry_circus.dynamics import Particle, integrate
from blueberry_circus.potentials import Coulomb
from blueberry_circus.zpf import ZPFBackground
from blueberry_circus.watchdog import ionization_time, is_bound


def _coulomb(U):
    return Coulomb(Z=1.0, units=U, charge=U.charge, mass=U.mass, softening=2e-2)


def test_bound_kepler_orbit_is_null_on_ionization():
    U = Units.scaled(0.02, 1.0)
    P = Particle(U.charge, U.mass)
    coul = _coulomb(U)
    r0 = 1.0
    vc = np.sqrt(np.linalg.norm(coul.force([r0, 0, 0])) * r0 / U.mass)
    t = np.arange(0, 120, 0.004)
    tr = integrate(field=None, potential=coul, particle=P, t_grid=t,
                   x0=[r0, 0, 0], v0=[0, vc, 0], rr="none", units=U, dipole=False)
    t_ion, detail = ionization_time(tr, coul, P)
    assert t_ion is None              # honest NULL on stability, not a fake "stable"
    assert is_bound(tr, coul, P)
    assert detail["E_final"] < 0.0    # stays bound


def test_sed_hydrogen_self_ionizes():
    # The headline: 3-D SED hydrogen self-ionizes (Nieuwenhuizen-Liska 2015).
    # SCALED units: gamma/omega0 = 0.02, five-plus orders of magnitude MORE
    # damping than the physical electron (~1e-7). This reproduces the effect
    # in reachable wall-clock; it is not the physical-hydrogen timescale.
    U = Units.scaled(0.02, 1.0)
    P = Particle(U.charge, U.mass)
    coul = _coulomb(U)
    r0 = 1.0
    vc = np.sqrt(np.linalg.norm(coul.force([r0, 0, 0])) * r0 / U.mass)
    field = ZPFBackground.isotropic_3d(0.3, 4.0, 150, seed=7, units=U)
    t = np.arange(0, 80, 0.004)
    tr = integrate(field=field, potential=coul, particle=P, t_grid=t,
                   x0=[r0, 0, 0], v0=[0, vc, 0], rr="landau_lifshitz", units=U,
                   dipole=False)
    t_ion, detail = ionization_time(tr, coul, P)
    assert t_ion is not None and detail["ionized"]
    assert detail["E_final"] > detail["E_initial"]   # energy rose: unbinding
    assert detail["r_max"] > 5.0                      # orbit wandered far out
