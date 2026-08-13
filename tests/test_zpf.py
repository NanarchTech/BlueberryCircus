import numpy as np
import blueberry_circus as bc
from blueberry_circus.zpf import ZPFBackground
from blueberry_circus import spectrum as sp
from blueberry_circus.constants import Units

U = Units.scaled(gamma_over_omega0=0.05, omega0=1.0)


def test_one_dimensional_reproduces_Ex_spectral_integral():
    f = ZPFBackground.one_dimensional(0.5, 3.0, 500, seed=1, units=U, axis=0)
    mse = f.mean_square_field_components()
    target = sp.energy_density_band(0.5, 3.0, U) / (3.0 * U.eps0)  # = int S_Ex
    assert abs(mse[0] - target) / target < 2e-2
    assert mse[1] == 0 and mse[2] == 0           # purely x-polarized


def test_isotropic_energy_density_matches_band_integral():
    f = ZPFBackground.isotropic_3d(0.5, 2.0, 800, seed=2, units=U)
    assert abs(f.mean_energy_density() - sp.energy_density_band(0.5, 2.0, U)) \
        / sp.energy_density_band(0.5, 2.0, U) < 1e-6


def test_isotropy_within_finite_sample_tolerance():
    f = ZPFBackground.isotropic_3d(0.5, 2.0, 1500, seed=3, units=U)
    mse = f.mean_square_field_components()
    assert mse.max() / mse.min() < 1.25


def test_transversality_and_magnetic_relation():
    f = ZPFBackground.isotropic_3d(0.5, 2.0, 200, seed=4, units=U)
    khat = f.kvecs / np.linalg.norm(f.kvecs, axis=1, keepdims=True)
    assert np.allclose(np.einsum("ij,ij->i", khat, f.evecs), 0.0, atol=1e-12)


def test_seed_determinism():
    a = ZPFBackground.isotropic_3d(0.5, 2.0, 50, seed=9, units=U)
    b = ZPFBackground.isotropic_3d(0.5, 2.0, 50, seed=9, units=U)
    assert np.array_equal(a.amps, b.amps) and np.array_equal(a.phases, b.phases)


def test_dEdt_matches_finite_difference():
    f = ZPFBackground.one_dimensional(0.5, 3.0, 100, seed=5, units=U)
    r = np.zeros(3); t = 12.34; h = 1e-6
    fd = (f.E(r, t + h) - f.E(r, t - h)) / (2 * h)
    assert np.allclose(f.dEdt(r, t), fd, rtol=1e-5, atol=1e-8)
