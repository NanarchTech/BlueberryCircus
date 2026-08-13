"""Analytic oracles: ground-state polarizability and the Coulomb-LL
damping coefficient (the hydrogen-physics guard).
"""
import numpy as np

from blueberry_circus import oracles
from blueberry_circus.constants import Units, SI, ALPHA
from blueberry_circus.potentials import Coulomb


def test_static_polarizability_matches_closed_form():
    U = Units.scaled(0.05, 1.0)
    w0 = 1.0
    a0 = oracles.polarizability(0.0, w0, U)
    assert np.isclose(a0.imag, 0.0)
    assert np.isclose(a0.real, oracles.static_polarizability_target(w0, U), rtol=1e-12)
    # polarizability is q * H_AL by construction
    H = oracles.transfer_abraham_lorentz(np.array([0.3, 0.7]), w0, U)
    al = oracles.polarizability(np.array([0.3, 0.7]), w0, U)
    assert np.allclose(al, U.charge * H, rtol=1e-12)


def test_beta_coefficient_identity():
    # beta = sqrt(2/3) Z alpha^{3/2} = Z / 1964.71 (Nieuwenhuizen-Liska Eq. 9)
    assert np.isclose(oracles.beta_coefficient(1.0), 1.0 / 1964.71, rtol=2e-4)
    assert np.isclose(oracles.beta_coefficient(3.0), 3.0 * oracles.beta_coefficient(1.0))
    # sanity: it is built from the fine-structure constant, not free
    assert np.isclose(oracles.beta_coefficient(1.0), (2.0 / 3.0) ** 0.5 * ALPHA ** 1.5)


def test_coulomb_ll_damping_closed_form_equals_jacobian_path():
    # The hydrogen damping term, computed via the generic force-Jacobian path the
    # integrator uses, must equal the hand-derived Coulomb closed form -- the one
    # place the Coulomb physics could be silently wrong while the SHO oracle passes.
    U = Units.scaled(0.02, 1.0)
    coul = Coulomb(Z=1.0, units=U, charge=U.charge, mass=U.mass, softening=2e-2)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.normal(size=3)
        v = rng.normal(size=3)
        a_jac = oracles.coulomb_ll_damping_accel(x, v, coul, U)
        a_cf = oracles.coulomb_ll_damping_closed_form(x, v, coul, U)
        assert np.allclose(a_jac, a_cf, rtol=1e-9, atol=1e-12)
