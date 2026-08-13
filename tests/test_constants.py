import math
import blueberry_circus as bc
from blueberry_circus.constants import Units, radiation_reaction_time


def test_fine_structure_and_bohr_radius():
    assert abs(1.0 / bc.ALPHA - 137.035999) < 1e-2
    assert abs(bc.A0 - 5.29177210903e-11) / 5.29177210903e-11 < 1e-6


def test_electron_radiation_reaction_time():
    tau = radiation_reaction_time()
    assert abs(tau - 6.2667e-24) / 6.2667e-24 < 1e-3   # ~6.26e-24 s


def test_scaled_units_target_damping():
    for g in (0.01, 0.05, 0.2):
        u = Units.scaled(gamma_over_omega0=g, omega0=1.0)
        # gamma = tau * omega0^2 must equal requested g
        assert abs(u.tau * 1.0**2 - g) / g < 1e-12


def test_eps0_is_explicit_in_units():
    u = bc.SI
    assert u.eps0 == bc.EPS0
    assert abs(u.k_e - 1.0 / (4 * math.pi * bc.EPS0)) / u.k_e < 1e-12
