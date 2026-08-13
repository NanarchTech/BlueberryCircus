"""Headline: the exactly-solvable SHO-in-ZPF, certified two ways."""
import numpy as np
from blueberry_circus import oracles
from blueberry_circus.constants import Units, SI
from blueberry_circus.dynamics import Particle, integrate
from blueberry_circus.zpf import ZPFBackground
from blueberry_circus.potentials import Harmonic
from blueberry_circus.certify import rel_error_certificate, PASS


def test_certified_sed_ground_state_si():
    """Physics oracle: integral S_Ex|H_AL|^2 = hbar/(2 m omega0) (Boyer 1975)."""
    w0 = 2.5e16
    val = oracles.sed_ground_state_integral(w0, SI)
    ref = oracles.ground_state_variance_target(w0, SI)
    cert = rel_error_certificate(
        kind="sed_ground_state_variance",
        claim="stationary <x^2> of AL oscillator in ZPF equals hbar/(2 m omega0)",
        value=val, reference=ref, tolerance=5e-3,
        method="numerical integral S_Ex |H_AL|^2 dω, Lorentzian substitution (Boyer 1975)")
    assert cert.recheck() == PASS
    assert cert.rule == "residual_le_tol"        # re-encoded, not rel_error_le_tol


def test_integrator_certificate_single_mode():
    """Integrator oracle: time-domain variance = exact |H|^2 (no sampling noise)."""
    U = Units.scaled(0.05, 1.0); w0 = 1.0
    P = Particle(U.charge, U.mass); pot = Harmonic(w0, mass=U.mass)
    field = ZPFBackground.one_dimensional(0.75, 0.85, 1, seed=0, units=U)
    t = np.arange(0, 400, 0.04)
    tr = integrate(field=field, potential=pot, particle=P, t_grid=t,
                   x0=[0, 0, 0], v0=[0, 0, 0], rr="landau_lifshitz", units=U)
    meas = tr.x[len(t)//3:, 0].var()
    exact = oracles.phase_averaged_variance(
        field.amps, field.omegas,
        lambda x: oracles.transfer_landau_lifshitz(x, w0, U))
    cert = rel_error_certificate(kind="integrator_fidelity",
                       claim="RK4+LL steady-state variance matches transfer fn",
                       value=meas, reference=exact, tolerance=1.5e-2,
                       method="single-mode RK4+LL time-domain run vs transfer function")
    assert cert.recheck() == PASS


def test_few_well_separated_modes():
    U = Units.scaled(0.05, 1.0); w0 = 1.0
    P = Particle(U.charge, U.mass); pot = Harmonic(w0, mass=U.mass)
    # three modes all detuned >= 0.4 from resonance: fast beats, small scatter
    field = ZPFBackground.one_dimensional(1.4, 2.6, 3, seed=2, units=U)
    t = np.arange(0, 500, 0.03)
    tr = integrate(field=field, potential=pot, particle=P, t_grid=t,
                   x0=[0, 0, 0], v0=[0, 0, 0], rr="landau_lifshitz", units=U)
    meas = tr.x[len(t)//3:, 0].var()
    exact = oracles.phase_averaged_variance(
        field.amps, field.omegas,
        lambda x: oracles.transfer_landau_lifshitz(x, w0, U))
    assert abs(meas - exact) / exact < 5e-2
