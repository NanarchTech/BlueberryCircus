"""O0 -- the field two-point function: discrete ZPF realization -> continuum.

The exact phase-averaged autocorrelation of the discrete random-phase field must
converge to the band-limited continuum correlator C(t) = integral S_Ex cos(wt) dw.
This is the acceptance oracle that the ZPF field carries the right spectrum in the
time domain; without it every downstream SED number is unanchored.
"""
import numpy as np

from blueberry_circus import oracles
from blueberry_circus.constants import Units
from blueberry_circus.zpf import ZPFBackground


def test_discrete_autocorrelation_converges_to_continuum():
    U = Units.scaled(0.05, 1.0)
    w_lo, w_hi, N = 0.5, 5.0, 6000
    field = ZPFBackground.one_dimensional(w_lo, w_hi, N, seed=3, units=U, axis=0)
    t = np.linspace(0.0, 3.0, 400)
    c_disc = oracles.field_autocorrelation(field, t, component=0)
    c_cont = oracles.field_two_point_continuum(t, w_lo, w_hi, U)
    scale = np.max(np.abs(c_cont))
    rel = np.max(np.abs(c_disc - c_cont)) / scale
    assert rel < 0.05, f"O0 discrete-vs-continuum rel-err {rel:.3e} exceeds 5%"


def test_autocorrelation_at_zero_is_mean_square_field():
    U = Units.scaled(0.05, 1.0)
    field = ZPFBackground.one_dimensional(0.5, 5.0, 4000, seed=1, units=U, axis=0)
    c0 = oracles.field_autocorrelation(field, 0.0, component=0)[0]
    msf = field.mean_square_field_components()[0]       # <E_x^2>
    assert np.isclose(c0, msf, rtol=1e-12)
    # transverse component carries no power for the axis-aligned 1-D field
    assert np.isclose(oracles.field_autocorrelation(field, 0.0, component=1)[0], 0.0)
