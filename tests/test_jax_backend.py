"""The JAX backend vs the NumPy oracle (enclosure), + ensemble vmap.

Self-skips when JAX is not installed, so the suite stays green without it.
"""
import numpy as np
import pytest

from blueberry_circus.constants import Units
from blueberry_circus.dynamics import Particle
from blueberry_circus.potentials import Harmonic, Coulomb
from blueberry_circus.zpf import ZPFBackground
from blueberry_circus import oracles
from blueberry_circus.observables import total_energy
from blueberry_circus.backends import NumpyBackend
from blueberry_circus.backends.jax_backend import JaxBackend, is_available

pytestmark = [
    pytest.mark.jax,
    pytest.mark.skipif(not is_available(), reason="JAX not installed"),
]


def test_jax_matches_numpy_sho_under_enclosure_and_passes_O2():
    U = Units.scaled(0.05, 1.0); w0 = 1.0
    P = Particle(U.charge, U.mass)
    pot = Harmonic(w0, mass=U.mass)
    field = ZPFBackground.one_dimensional(0.75, 0.85, 1, seed=0, units=U)
    t = np.arange(0, 400, 0.04)
    common = dict(field=field, potential=pot, particle=P, t_grid=t,
                  x0=[0, 0, 0], v0=[0, 0, 0], rr="landau_lifshitz", units=U,
                  dipole=True)
    np_tr = NumpyBackend().integrate(**common)
    jx_tr = JaxBackend().integrate(**common)
    scale = 1.0 + np.max(np.abs(np_tr.x))
    assert np.max(np.abs(jx_tr.x - np_tr.x)) <= 1e-6 * scale     # enclosure
    exact = oracles.phase_averaged_variance(
        field.amps, field.omegas,
        lambda w: oracles.transfer_landau_lifshitz(w, w0, U))
    meas = jx_tr.x[len(t) // 3:, 0].var()
    assert abs(meas - exact) / exact < 1.5e-2


def test_jax_matches_numpy_kepler():
    U = Units.scaled(0.02, 1.0)
    P = Particle(U.charge, U.mass)
    coul = Coulomb(Z=1.0, units=U, charge=U.charge, mass=U.mass, softening=2e-2)
    r0 = 1.0
    vc = np.sqrt(np.linalg.norm(coul.force([r0, 0, 0])) * r0 / U.mass)
    t = np.arange(0, 120, 0.004)
    common = dict(field=None, potential=coul, particle=P, t_grid=t,
                  x0=[r0, 0, 0], v0=[0, vc, 0], rr="none", units=U, dipole=False)
    np_tr = NumpyBackend().integrate(**common)
    jx_tr = JaxBackend().integrate(**common)
    assert np.max(np.abs(jx_tr.x - np_tr.x)) <= 1e-6 * (1 + np.max(np.abs(np_tr.x)))
    E = total_energy(jx_tr, coul, P)
    assert (E.max() - E.min()) / abs(E.mean()) < 1e-5


def test_jax_matches_numpy_with_mismatched_potential_mass():
    # Harmonic mass != particle mass: the JAX harmonic force must use the
    # potential's own mass (not the particle mass) to match the numpy oracle.
    U = Units.scaled(0.05, 1.0)
    P = Particle(U.charge, U.mass)
    pot = Harmonic(1.0, mass=3.0)
    t = np.arange(0, 50, 0.02)
    common = dict(field=None, potential=pot, particle=P, t_grid=t,
                  x0=[1, 0, 0], v0=[0, 0, 0], rr="none", units=U, dipole=True)
    a = NumpyBackend().integrate(**common)
    b = JaxBackend().integrate(**common)
    assert np.max(np.abs(a.x - b.x)) <= 1e-6 * (1 + np.max(np.abs(a.x)))


def test_engine_runs_on_jax_backend():
    import blueberry_circus as bc
    U = Units.scaled(0.05, 1.0)
    prog = bc.Program(n_particles=1, units=U)
    with prog.context as q:
        bc.Harmonic(1.0) | q[0]
        bc.ZPF(band=(0.5, 2.0), n_modes=20, seed=1, mode="one_dimensional") | q[0]
        bc.RadiationReaction("landau_lifshitz") | q[0]
    res = bc.Engine(backend="jax", dt=0.05, t_max=60).run(prog, x0=[0, 0, 0], v0=[0, 0, 0])
    assert res.observables["trajectory_finite"]
