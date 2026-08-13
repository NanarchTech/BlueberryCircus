import numpy as np
import blueberry_circus as bc
from blueberry_circus import spectrum as sp
from blueberry_circus.constants import SI


def test_mode_energy_is_half_hbar_omega():
    w = np.array([1e15, 4e16, 1e17])
    assert np.allclose(sp.mode_energy(w), 0.5 * SI.hbar * w)


def test_rho_equals_modedensity_times_modeenergy():
    w = np.linspace(1e15, 1e17, 50)
    assert np.allclose(sp.rho(w), sp.mode_density(w) * sp.mode_energy(w))


def test_Ex_is_third_of_total_over_eps0():
    # <E^2> = (1/eps0) rho ; per component = 1/3 of that
    w = np.linspace(1e15, 1e17, 50)
    assert np.allclose(sp.spectral_density_Ex(w), sp.rho(w) / (3.0 * SI.eps0))


def test_energy_density_band_matches_quadrature():
    val = sp.energy_density_band(1e15, 5e16)
    w = np.linspace(1e15, 5e16, 200001)
    ref = np.trapezoid(sp.rho(w), w)
    assert abs(val - ref) / ref < 1e-4
    assert val > 0
