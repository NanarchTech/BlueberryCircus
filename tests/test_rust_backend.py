"""The Rust LL-RK4 backend vs the NumPy oracle (cross-language gate).

Agreement is asserted under ENCLOSURE tolerance (1-ULP CPython/Rust divergence is
expected), never bit-equality, and both backends must anchor to the closed-form
O2 oracle. The whole module self-skips when the cdylib is not built, so the suite
stays green on machines without the Rust toolchain.
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
from blueberry_circus.backends.rust_backend import RustBackend, is_available

pytestmark = [
    pytest.mark.rust,
    pytest.mark.skipif(
        not is_available(),
        reason="rust backend cdylib not built (run scripts/build_rust.sh)"),
]


def test_rust_matches_numpy_sho_under_enclosure_and_passes_O2():
    U = Units.scaled(0.05, 1.0); w0 = 1.0
    P = Particle(U.charge, U.mass)
    pot = Harmonic(w0, mass=U.mass)
    field = ZPFBackground.one_dimensional(0.75, 0.85, 1, seed=0, units=U)
    t = np.arange(0, 400, 0.04)
    common = dict(field=field, potential=pot, particle=P, t_grid=t,
                  x0=[0, 0, 0], v0=[0, 0, 0], rr="landau_lifshitz", units=U,
                  dipole=True)
    np_tr = NumpyBackend().integrate(**common)
    rs_tr = RustBackend().integrate(**common)
    # 1. Agreement under enclosure tolerance (NOT bit-equality).
    scale = 1.0 + np.max(np.abs(np_tr.x))
    assert np.max(np.abs(rs_tr.x - np_tr.x)) <= 1e-7 * scale
    # 2. Both anchor to the closed-form O2 oracle.
    exact = oracles.phase_averaged_variance(
        field.amps, field.omegas,
        lambda w: oracles.transfer_landau_lifshitz(w, w0, U))
    for tr in (np_tr, rs_tr):
        meas = tr.x[len(t) // 3:, 0].var()
        assert abs(meas - exact) / exact < 1.5e-2


def test_rust_matches_numpy_kepler_and_conserves_energy():
    U = Units.scaled(0.02, 1.0)
    P = Particle(U.charge, U.mass)
    coul = Coulomb(Z=1.0, units=U, charge=U.charge, mass=U.mass, softening=2e-2)
    r0 = 1.0
    vc = np.sqrt(np.linalg.norm(coul.force([r0, 0, 0])) * r0 / U.mass)
    t = np.arange(0, 120, 0.004)
    common = dict(field=None, potential=coul, particle=P, t_grid=t,
                  x0=[r0, 0, 0], v0=[0, vc, 0], rr="none", units=U, dipole=False)
    np_tr = NumpyBackend().integrate(**common)
    rs_tr = RustBackend().integrate(**common)
    scale = 1.0 + np.max(np.abs(np_tr.x))
    assert np.max(np.abs(rs_tr.x - np_tr.x)) <= 1e-6 * scale
    E = total_energy(rs_tr, coul, P)
    assert (E.max() - E.min()) / abs(E.mean()) < 1e-5


def test_rust_matches_numpy_with_mismatched_mass_and_units():
    # potential mass != particle mass (harmonic), and Coulomb built with units
    # different from the integrate-time units -- both backends must still agree,
    # because they read the frozen potential params, not the integrate units.
    U = Units.scaled(0.05, 1.0)
    Uother = Units.scaled(0.02, 1.0)
    P = Particle(U.charge, U.mass)
    pot = Harmonic(1.0, mass=3.0)                    # mass != particle mass
    t = np.arange(0, 50, 0.02)
    common = dict(field=None, potential=pot, particle=P, t_grid=t,
                  x0=[1, 0, 0], v0=[0, 0, 0], rr="none", units=U, dipole=True)
    a = NumpyBackend().integrate(**common)
    b = RustBackend().integrate(**common)
    assert np.max(np.abs(a.x - b.x)) <= 1e-7 * (1 + np.max(np.abs(a.x)))
    # Coulomb built with Uother, integrated with U
    coul = Coulomb(Z=1.0, units=Uother, charge=Uother.charge, mass=Uother.mass,
                   softening=2e-2)
    r0 = 1.0
    vc = np.sqrt(np.linalg.norm(coul.force([r0, 0, 0])) * r0 / Uother.mass)
    t2 = np.arange(0, 60, 0.004)
    c2 = dict(field=None, potential=coul,
              particle=Particle(Uother.charge, Uother.mass), t_grid=t2,
              x0=[r0, 0, 0], v0=[0, vc, 0], rr="none", units=U, dipole=False)
    a2 = NumpyBackend().integrate(**c2)
    b2 = RustBackend().integrate(**c2)
    assert np.max(np.abs(a2.x - b2.x)) <= 1e-6 * (1 + np.max(np.abs(a2.x)))


def test_engine_can_run_on_rust_backend():
    import blueberry_circus as bc
    U = Units.scaled(0.05, 1.0)
    prog = bc.Program(n_particles=1, units=U)
    with prog.context as q:
        bc.Harmonic(1.0) | q[0]
        bc.ZPF(band=(0.5, 2.0), n_modes=30, seed=1, mode="one_dimensional") | q[0]
        bc.RadiationReaction("landau_lifshitz") | q[0]
    res = bc.Engine(backend="rust", dt=0.05, t_max=60).run(prog, x0=[0, 0, 0], v0=[0, 0, 0])
    assert res.observables["trajectory_finite"]
    assert all(c.recheck() == bc.PASS for c in res.certificates)
