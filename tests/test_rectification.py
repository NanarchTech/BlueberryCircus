"""O5: Nieuwenhuizen's conditional near-ionization drift threshold."""
import math
import importlib

import pytest

from blueberry_circus.oracles import beta_coefficient


def _rectification():
    return importlib.import_module("blueberry_circus.rectification")


def test_threshold_quadrature_independently_recovers_closed_form():
    """Catches a missing Jacobian, wrong nested limit, or copied closed form."""
    rectification = _rectification()
    numerical = rectification.threshold_quadrature(order=96)
    exact = 16.0 / (5.0 * math.pi * math.sqrt(3.0))
    assert abs(numerical - exact) < 1e-8
    assert abs(numerical - rectification.critical_angular_momentum()) < 1e-8


def test_near_ionization_drift_changes_sign_only_at_Lc():
    rectification = _rectification()
    lc = rectification.critical_angular_momentum()
    below = rectification.near_ionization_drift(lc - 0.01)
    at = rectification.near_ionization_drift(lc)
    above = rectification.near_ionization_drift(lc + 0.01)
    assert below > 0.0
    assert abs(at) < 1e-20
    assert above < 0.0


def test_near_ionization_drift_matches_eq_2_37_and_Z_scaling():
    rectification = _rectification()
    L = 0.55
    lc = 16.0 / (5.0 * math.pi * math.sqrt(3.0))
    expected = 3.0 * math.pi * beta_coefficient(1.0)**2 * (lc - L) / L**6
    assert math.isclose(rectification.near_ionization_drift(L), expected,
                        rel_tol=1e-14)
    assert math.isclose(rectification.near_ionization_drift(L, Z=2.0),
                        4.0 * expected, rel_tol=1e-14)


def test_critical_perihelion_is_0172921_bohr_radii():
    rectification = _rectification()
    lc = rectification.critical_angular_momentum()
    assert math.isclose(lc**2 / 2.0, 0.172921, abs_tol=1e-6)


@pytest.mark.parametrize("L", [0.0, -1.0, math.inf, math.nan])
def test_near_ionization_drift_rejects_invalid_angular_momentum(L):
    with pytest.raises(ValueError, match="L must be finite and positive"):
        _rectification().near_ionization_drift(L)


@pytest.mark.parametrize("Z", [0.0, -1.0, math.inf, math.nan])
def test_near_ionization_drift_rejects_invalid_nuclear_charge(Z):
    with pytest.raises(ValueError, match="Z must be finite and positive"):
        _rectification().near_ionization_drift(0.55, Z=Z)


def test_threshold_quadrature_accepts_odd_gauss_order():
    numerical = _rectification().threshold_quadrature(order=15)
    exact = 16.0 / (5.0 * math.pi * math.sqrt(3.0))
    assert abs(numerical - exact) < 1e-3


@pytest.mark.parametrize("order", [0, 1, 3.5])
def test_threshold_quadrature_rejects_invalid_orders(order):
    rectification = _rectification()
    with pytest.raises(ValueError, match="integer >= 2"):
        rectification.threshold_quadrature(order=order)
