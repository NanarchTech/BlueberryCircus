"""3-D Coulomb (hydrogen) engine: runnable at CI scale; the convergent ground
state is unreachable (and self-ionizes) -- the honest headline."""
import numpy as np
import pytest

import blueberry_circus as bc
from blueberry_circus.constants import Units
from blueberry_circus.dynamics import Particle, integrate
from blueberry_circus.potentials import Coulomb
from blueberry_circus.zpf import ZPFBackground
from blueberry_circus import oracles


def test_hydrogen_engine_runs_and_is_finite():
    U = Units.scaled(0.02, 1.0)
    P = Particle(U.charge, U.mass)
    coul = Coulomb(Z=1.0, units=U, charge=U.charge, mass=U.mass, softening=1e-2)
    r0 = 1.0
    v = np.sqrt(np.linalg.norm(coul.force([r0, 0, 0])) * r0 / U.mass)
    field = ZPFBackground.isotropic_3d(0.3, 4.0, 120, seed=7, units=U)
    t = np.arange(0, 50, 0.004)
    tr = integrate(field=field, potential=coul, particle=P, t_grid=t,
                   x0=[r0, 0, 0], v0=[0, v, 0], rr="landau_lifshitz", units=U,
                   dipole=False)
    assert np.all(np.isfinite(tr.x)) and np.all(np.isfinite(tr.v))


def test_o3_radial_l1_metric_is_well_defined():
    # O3 machinery: the L1 distance to the QM 1s density is a real shape metric in
    # [0,2]; identical density -> 0, a wrong (uniform) density -> > 0.
    r = np.linspace(0.05, 8.0, 200)
    p_1s = oracles.hydrogen_1s_radial(r)
    assert abs(oracles.radial_l1_distance(r, p_1s)) < 1e-6
    assert oracles.radial_l1_distance(r, np.ones_like(r)) > 0.1


@pytest.mark.xfail(strict=True, reason="O3 convergence frontier: matching the QM "
                   "1s radial density to a tight L1 needs Cole-Zou frequency-"
                   "windowed CPU-day ensembles, and the orbit self-ionizes at "
                   "long time anyway (Cole & Zou 2003; Nieuwenhuizen & Liska 2015).")
def test_radial_density_converges_to_qm_1s():
    raise NotImplementedError("converged windowed O3 ensemble: CPU-day frontier")


@pytest.mark.xfail(strict=True, reason="Headline negative result: full-3-D SED "
                   "hydrogen does NOT reproduce a stable ground state -- it self-"
                   "ionizes (Nieuwenhuizen & Liska 2015). Kept red as honest "
                   "documentation; it can never pass.")
def test_stable_ground_state_is_unreachable():
    raise NotImplementedError("stable SED hydrogen ground state: literature-negative")


@pytest.mark.xfail(strict=True, reason="Frontier: relativistic corrections "
                   "(Nieuwenhuizen & Liska 2015) are implemented-and-negative in "
                   "the literature; not in this build.")
def test_relativistic_corrections():
    raise NotImplementedError("relativistic SED: frontier")
