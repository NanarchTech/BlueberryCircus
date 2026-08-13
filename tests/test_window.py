"""The Cole-Zou moving spectral window."""
import numpy as np

from blueberry_circus.constants import Units
from blueberry_circus.dynamics import Particle, integrate
from blueberry_circus.potentials import Coulomb
from blueberry_circus.zpf import ZPFBackground
from blueberry_circus.observables import total_energy
from blueberry_circus.window import (orbital_frequency, taper_weights, _band,
                                     coulomb_coef, WindowedField)


def test_orbital_frequency_matches_circular_orbit():
    # omega(r) = sqrt(coef/(m r^3)); a circular orbit has v=omega*r.
    U = Units.scaled(0.02, 1.0)
    coul = Coulomb(Z=1.0, units=U, charge=U.charge, mass=U.mass)
    coef = coulomb_coef(coul, U)
    r = 1.3
    w = float(orbital_frequency(r, coef, U.mass))
    v_circ = np.sqrt(np.linalg.norm(coul.force([r, 0, 0])) * r / U.mass)
    assert np.isclose(w, v_circ / r, rtol=1e-12)


def test_taper_weights_are_continuous_and_bounded():
    # The window weight is a continuous (C0) function of omega in [0,1]: 1 in the
    # band, ramping to 0 over the taper margin. (This is the no-discontinuous-
    # switching property; finite mode count is a separate discretization limit.)
    omega_c = 1.0
    lo, hi = _band(omega_c, 0.03)
    w = np.linspace(lo * 0.3, hi * 1.8, 5000)
    weights = taper_weights(w, omega_c, 0.03, 0.5)
    assert weights.min() >= 0.0 and weights.max() <= 1.0
    assert np.max(np.abs(np.diff(weights))) < 1e-2          # continuous on a fine grid
    # weight is 1 at band center, 0 well outside
    assert np.isclose(float(taper_weights([omega_c], omega_c, 0.03, 0.5)[0]), 1.0)
    assert float(taper_weights([omega_c * 5], omega_c, 0.03, 0.5)[0]) == 0.0


def test_window_injects_no_energy_into_kepler_when_zpf_off():
    # The gate: with the ZPF amplitude zeroed, the *sliding window machinery* must
    # not corrupt a pure Kepler orbit -> energy conserved across the slides.
    U = Units.scaled(0.02, 1.0)
    P = Particle(U.charge, U.mass)
    coul = Coulomb(Z=1.0, units=U, charge=U.charge, mass=U.mass, softening=2e-2)
    base = ZPFBackground.isotropic_3d(0.3, 4.0, 80, seed=7, units=U)
    base_off = ZPFBackground(base.omegas, base.kvecs, base.evecs,
                             np.zeros_like(base.amps), base.phases, U)
    wf = WindowedField(base_off, coef=coulomb_coef(coul, U), mass=U.mass,
                       f_band=0.03, taper=0.5, units=U)
    r0 = 1.0
    vc = np.sqrt(np.linalg.norm(coul.force([r0, 0, 0])) * r0 / U.mass)
    t = np.arange(0, 120, 0.004)
    tr = integrate(field=wf, potential=coul, particle=P, t_grid=t,
                   x0=[r0, 0, 0], v0=[0, vc, 0], rr="none", units=U, dipole=False)
    E = total_energy(tr, coul, P)
    assert (E.max() - E.min()) / abs(E.mean()) < 1e-8


def test_windowed_field_has_active_modes_and_runs():
    # Choose a band that actually CONTAINS the orbital frequency, so the window
    # has active modes along the orbit (a band above/below the modes would make
    # the window trivially empty and the test vacuous).
    U = Units.scaled(0.02, 1.0)
    P = Particle(U.charge, U.mass)
    coul = Coulomb(Z=1.0, units=U, charge=U.charge, mass=U.mass, softening=2e-2)
    base = ZPFBackground.isotropic_3d(0.1, 0.35, 200, seed=3, units=U)
    # units omitted on purpose -> must be derived from the base field, not SI
    wf = WindowedField(base, coef=coulomb_coef(coul, U), mass=U.mass, f_band=0.1)
    assert wf.units is base.units
    r0 = 1.0
    assert (wf._weights(r0) > 0).sum() > 0          # the window is genuinely active
    vc = np.sqrt(np.linalg.norm(coul.force([r0, 0, 0])) * r0 / U.mass)
    t = np.arange(0, 30, 0.004)
    tr = integrate(field=wf, potential=coul, particle=P, t_grid=t,
                   x0=[r0, 0, 0], v0=[0, vc, 0], rr="landau_lifshitz", units=U,
                   dipole=False)
    assert np.all(np.isfinite(tr.x))
