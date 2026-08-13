import numpy as np
import pytest
from blueberry_circus import oracles
from blueberry_circus.constants import Units, SI
from blueberry_circus.dynamics import Particle, integrate
from blueberry_circus.zpf import ZPFBackground
from blueberry_circus.potentials import Harmonic


@pytest.mark.parametrize("w0", [1e16, 4e16, 1e17])
def test_sed_ground_state_normalization_is_hbar_over_2mw(w0):
    I = oracles.sed_ground_state_integral(w0, SI)
    T = oracles.ground_state_variance_target(w0, SI)
    assert abs(I - T) / T < 5e-3


def test_phase_averaged_variance_single_mode_closed_form():
    U = Units.scaled(0.05, 1.0)
    a = np.array([0.3]); w = np.array([0.8])
    H = oracles.transfer_landau_lifshitz(w, 1.0, U)
    expect = 0.5 * a[0]**2 * abs(H[0])**2
    got = oracles.phase_averaged_variance(a, w, lambda x: oracles.transfer_landau_lifshitz(x, 1.0, U))
    assert abs(got - expect) < 1e-15


def test_integrator_reproduces_single_mode_steady_state_variance():
    """Rigorous integrator oracle: one driven mode -> exact (1/2)a^2|H|^2."""
    U = Units.scaled(0.05, 1.0); w0 = 1.0
    P = Particle(U.charge, U.mass); pot = Harmonic(w0, mass=U.mass)
    field = ZPFBackground.one_dimensional(0.7, 0.9, 1, seed=0, units=U, axis=0)
    t = np.arange(0, 400, 0.04)
    tr = integrate(field=field, potential=pot, particle=P, t_grid=t,
                   x0=[0, 0, 0], v0=[0, 0, 0], rr="landau_lifshitz", units=U)
    meas = tr.x[len(t)//3:, 0].var()
    exact = oracles.phase_averaged_variance(
        field.amps, field.omegas,
        lambda x: oracles.transfer_landau_lifshitz(x, w0, U))
    assert abs(meas - exact) / exact < 1.5e-2
