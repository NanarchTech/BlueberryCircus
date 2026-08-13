"""Static Setterfield co-scaling: invariants and dynamical conjugacy."""
import math

import numpy as np
import pytest

from blueberry_circus import constants
from blueberry_circus.constants import ALPHA
from blueberry_circus.dynamics import Particle, integrate
from blueberry_circus.observables import angular_momentum, total_energy
from blueberry_circus.oracles import bohr_radius
from blueberry_circus.potentials import Coulomb
from blueberry_circus.zpf import ZPFBackground


def _dimensionless_invariants(units):
    alpha = (units.charge**2 /
             (4.0 * math.pi * units.eps0 * units.hbar * units.c))
    a0 = bohr_radius(units)
    hartree = units.k_e * units.charge**2 / a0
    omega_b = hartree / units.hbar
    beta = math.sqrt(2.0 / 3.0) * alpha**1.5
    return np.array([alpha, a0, hartree, beta, units.tau * omega_b])


def test_setterfield_profile_preserves_dimensionless_hydrogen_data():
    scaled = constants.setterfield_rescale(constants.BOHR, 4.0)
    assert scaled.hbar == 4.0 * constants.BOHR.hbar
    assert scaled.eps0 == 4.0 * constants.BOHR.eps0
    assert scaled.c == constants.BOHR.c / 4.0
    assert scaled.charge == 2.0 * constants.BOHR.charge
    assert scaled.mass == 16.0 * constants.BOHR.mass
    assert np.allclose(_dimensionless_invariants(scaled),
                       _dimensionless_invariants(constants.BOHR), rtol=1e-12, atol=0.0)
    assert math.isclose(_dimensionless_invariants(scaled)[0], ALPHA,
                        rel_tol=1e-12)


@pytest.mark.parametrize("factor", [0.0, -1.0, math.inf, math.nan])
def test_setterfield_profile_rejects_nonphysical_scale_factors(factor):
    with pytest.raises(ValueError, match="finite and positive"):
        constants.setterfield_rescale(constants.BOHR, factor)


def test_static_coscaling_is_dynamically_conjugate_with_mapped_zpf_band():
    factor = 4.0
    u1 = constants.BOHR
    u4 = constants.setterfield_rescale(u1, factor)
    field1 = ZPFBackground.isotropic_3d(0.5, 2.0, 24, seed=2026, units=u1)
    field4 = ZPFBackground.isotropic_3d(0.5 / factor, 2.0 / factor, 24,
                                       seed=2026, units=u4)
    assert np.array_equal(field4.phases, field1.phases)
    assert np.allclose(field4.omegas * factor, field1.omegas, rtol=1e-15)
    assert np.allclose(field4.kvecs, field1.kvecs, rtol=1e-15, atol=1e-15)

    particle1 = Particle(u1.charge, u1.mass)
    particle4 = Particle(u4.charge, u4.mass)
    coulomb1 = Coulomb(units=u1, charge=u1.charge, mass=u1.mass)
    coulomb4 = Coulomb(units=u4, charge=u4.charge, mass=u4.mass)
    t1 = np.arange(0.0, 2.0, 0.01)
    t4 = factor * t1
    common = dict(x0=[1.0, 0.0, 0.0], rr="landau_lifshitz", dipole=False)
    tr1 = integrate(field=field1, potential=coulomb1, particle=particle1,
                    t_grid=t1, v0=[0.0, 1.0, 0.0], units=u1, **common)
    tr4 = integrate(field=field4, potential=coulomb4, particle=particle4,
                    t_grid=t4, v0=[0.0, 1.0 / factor, 0.0], units=u4, **common)

    assert np.max(np.abs(tr4.x - tr1.x)) < 1e-9
    assert np.max(np.abs(factor * tr4.v - tr1.v)) < 1e-9
    assert np.max(np.abs(total_energy(tr4, coulomb4, particle4) -
                         total_energy(tr1, coulomb1, particle1))) < 1e-9
    l1 = angular_momentum(tr1, particle1) / u1.hbar
    l4 = angular_momentum(tr4, particle4) / u4.hbar
    assert np.max(np.abs(l4 - l1)) < 1e-9
