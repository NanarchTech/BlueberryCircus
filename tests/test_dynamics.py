import numpy as np
from blueberry_circus.constants import Units
from blueberry_circus.dynamics import Particle, integrate
from blueberry_circus.potentials import Harmonic, Coulomb
from blueberry_circus.observables import total_energy, angular_momentum

U = Units.scaled(gamma_over_omega0=0.05, omega0=1.0)
P = Particle(U.charge, U.mass)


def test_closed_sho_conserves_energy():
    pot = Harmonic(1.0, mass=U.mass)
    t = np.arange(0, 200, 0.01)
    tr = integrate(field=None, potential=pot, particle=P, t_grid=t,
                   x0=[1.0, 0, 0], v0=[0, 0, 0], rr="none", units=U)
    E = total_energy(tr, pot, P)
    assert (E.max() - E.min()) / E.mean() < 1e-6


def test_landau_lifshitz_free_decay_rate_matches_tau_omega2():
    pot = Harmonic(1.0, mass=U.mass)
    t = np.arange(0, 400, 0.02)
    tr = integrate(field=None, potential=pot, particle=P, t_grid=t,
                   x0=[1.0, 0, 0], v0=[0, 0, 0], rr="landau_lifshitz", units=U)
    E = 0.5 * U.mass * tr.v[:, 0]**2 + 0.5 * U.mass * tr.x[:, 0]**2
    slope, _ = np.polyfit(tr.t, np.log(E), 1)
    assert abs(-slope - U.tau * 1.0**2) / (U.tau * 1.0**2) < 1e-2


def test_kepler_conserves_energy_and_angular_momentum():
    coul = Coulomb(Z=1.0, units=U, charge=U.charge, mass=U.mass)
    r0 = 1.0
    Fmag = np.linalg.norm(coul.force([r0, 0, 0]))
    v = np.sqrt(Fmag * r0 / U.mass)
    t = np.arange(0, 120, 0.01)
    tr = integrate(field=None, potential=coul, particle=P, t_grid=t,
                   x0=[r0, 0, 0], v0=[0, v, 0], rr="none", units=U, dipole=False)
    E = total_energy(tr, coul, P); L = angular_momentum(tr, P)[:, 2]
    assert (E.max() - E.min()) / abs(E.mean()) < 1e-8
    assert (L.max() - L.min()) / abs(L.mean()) < 1e-8


def test_radiation_reaction_makes_orbit_decay():
    # gentle damping + Plummer softening so the inspiral stays numerically
    # resolved (an unregularized r->0 core would eject the particle).
    Ud = Units.scaled(gamma_over_omega0=0.02, omega0=1.0)
    Pd = Particle(Ud.charge, Ud.mass)
    coul = Coulomb(Z=1.0, units=Ud, charge=Ud.charge, mass=Ud.mass, softening=2e-2)
    r0 = 1.0
    Fmag = np.linalg.norm(coul.force([r0, 0, 0]))
    v = np.sqrt(Fmag * r0 / Ud.mass)
    t = np.arange(0, 120, 0.004)
    tr = integrate(field=None, potential=coul, particle=Pd, t_grid=t,
                   x0=[r0, 0, 0], v0=[0, v, 0], rr="landau_lifshitz", units=Ud,
                   dipole=False)
    E = total_energy(tr, coul, Pd)
    assert np.all(np.isfinite(tr.x))
    assert E[-1] < E[0]                      # binds tighter: radiative collapse
    assert tr.r[-1] < tr.r[0]
