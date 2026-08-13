"""O1 -- Puthoff (1987) ground state: measure our SED constants against CODATA.

Puthoff's power balance reproduces the Bohr ground state (a0, -13.6 eV, L=hbar).
These tests check that BlueberryCircus's SI constants + spectrum reproduce those
analytic targets. (The *dynamical* SED run self-ionizes -- see test_watchdog;
O1 is the static target Puthoff's balance lands on, not a stability claim.)
"""
import numpy as np

from blueberry_circus import oracles
from blueberry_circus.constants import SI, A0, ALPHA
from blueberry_circus.certify import rel_error_certificate, PASS

# CODATA 2018 reference values
A0_CODATA = 5.29177210903e-11          # m
E1_CODATA_J = -2.1798723611035e-18     # J  (= -13.605693 eV)
EV = 1.602176634e-19                    # J


def test_bohr_radius_matches_codata():
    a0 = oracles.bohr_radius(SI)
    assert np.isclose(a0, A0_CODATA, rtol=1e-6)
    assert np.isclose(a0, A0, rtol=1e-12)       # consistent with constants.A0


def test_ground_state_energy_is_minus_13_6_eV():
    E1 = oracles.hydrogen_ground_state_energy(SI)
    assert np.isclose(E1, E1_CODATA_J, rtol=1e-5)
    assert np.isclose(E1 / EV, -13.605693, rtol=1e-5)


def test_ground_state_angular_momentum_is_hbar():
    # Algebraic identity, not a measurement: L = m * v(a0) * a0 reduces to hbar
    # exactly, so this locks the unit bookkeeping, not a physical prediction.
    assert np.isclose(oracles.bohr_angular_momentum(SI), SI.hbar, rtol=1e-12)


def test_puthoff_balance_diagnostics():
    b = oracles.puthoff_power_balance(SI)
    assert b["larmor_power"] > 0 and b["rho_zpf_at_omega0"] > 0
    assert np.isclose(b["angular_momentum_over_hbar"], 1.0, rtol=1e-12)
    # orbital frequency at a0 is the Bohr value omega0 = alpha c / a0 = E_h/hbar
    assert np.isclose(b["omega0"], ALPHA * SI.c / oracles.bohr_radius(SI), rtol=1e-6)


def test_puthoff_ground_state_certificate():
    a0 = oracles.bohr_radius(SI)
    c = rel_error_certificate(
        "puthoff_bohr_radius",
        "closed-form Bohr radius from SI constants equals CODATA a0 (the static target Puthoff 1987 power balance lands on)",
        value=a0, reference=A0_CODATA, tolerance=1e-6,
        method="closed form a0 = 4 pi eps0 hbar^2 / (m e^2) evaluated in SI")
    assert c.recheck() == PASS
