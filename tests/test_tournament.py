"""Acceptance tests for the energy-audited hypothesis tournament."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from blueberry_circus.constants import ALPHA
from blueberry_circus.rectification import (
    critical_angular_momentum,
    near_ionization_drift,
)
from blueberry_circus.tournament import (
    ACTIVE_CONTROL,
    CHANNEL_SUPPRESSED,
    NO_EFFECT,
    NULL,
    EnergyLedger,
    FiniteShellResponse,
    HypothesisResult,
    InverseSquareControl,
    inverse_square_gain_function,
    MultipoleStorage,
    nieuwenhuizen_gain_function,
    OrbitState,
    SetterfieldDrive,
    TournamentConfig,
    classify_hypothesis,
    run_hypothesis_arm,
    point_charge_drift,
    point_charge_channels,
    point_charge_drift_surface,
    stochastic_cell,
)


def test_orbit_state_uses_physical_bohr_geometry():
    state = OrbitState(energy=-0.01, angular_momentum=0.45)
    k = math.sqrt(0.02)
    eccentricity = math.sqrt(1.0 - (k * 0.45) ** 2)
    assert state.k == pytest.approx(k)
    assert state.eccentricity == pytest.approx(eccentricity)
    assert state.perihelion == pytest.approx(0.45**2 / (1.0 + eccentricity))
    x, p = state.perihelion_phase_point()
    assert np.cross(x, p)[2] == pytest.approx(0.45)
    assert 0.5 * np.dot(p, p) - 1.0 / np.linalg.norm(x) == pytest.approx(-0.01)


@pytest.mark.parametrize("energy,L", [(0.0, 0.5), (-0.1, 0.0), (-0.5, 2.0)])
def test_orbit_state_rejects_nonelliptic_cells(energy, L):
    with pytest.raises(ValueError):
        OrbitState(energy=energy, angular_momentum=L)


def test_energy_ledger_sign_convention_and_closure():
    ledger = EnergyLedger.from_channels(
        mechanical_energy_change=0.7,
        zpf_work=1.2,
        radiative_loss=0.4,
        schott_boundary_energy=0.1,
        external_parameter_work=0.2,
        internal_mode_exchange=-0.2,
    )
    # dE = Wzpf - Erad - dEschott + Wext + Wint + residual.
    assert ledger.numerical_closure_residual == pytest.approx(0.0)
    assert ledger.relative_closure <= 1e-15


def test_default_preregistration_is_exact_and_json_roundtrips():
    cfg = TournamentConfig()
    lc = critical_angular_momentum()
    assert cfg.energies == (-0.05, -0.02, -0.01)
    assert cfg.angular_momenta == pytest.approx((0.45, 0.55, lc - 0.01,
                                                 lc + 0.01, 0.65, 0.8))
    assert cfg.coupling_scales == (1.0, 4.0, 8.0, 16.0)
    assert len(cfg.seeds) == 32 and len(set(cfg.seeds)) == 32
    assert cfg.n_modes == 2048
    assert TournamentConfig.from_json(cfg.to_json()) == cfg


def test_complete_point_charge_surface_and_near_ionization_limit():
    lc = critical_angular_momentum()
    energies = (-0.02, -0.01)
    momenta = (lc - 0.01, lc + 0.01)
    surface = point_charge_drift_surface(energies, momenta, order=64,
                                         per_orbit=True)
    assert surface.shape == (2, 2)
    assert np.all(np.isfinite(surface))
    assert surface[1, 0] > 0.0
    assert surface[1, 1] < 0.0
    # The finite-E result approaches the independently certified PR1 asymptote.
    for L in momenta:
        asymptote = near_ionization_drift(L)
        finite = point_charge_drift(-1e-3, L, order=128, per_orbit=True)
        assert finite == pytest.approx(asymptote, rel=0.02)


def test_complete_gain_quadrature_has_both_endpoints_and_internal_convergence():
    assert nieuwenhuizen_gain_function(0.0) == pytest.approx(
        critical_angular_momentum(), abs=0.0,
    )
    assert nieuwenhuizen_gain_function(1.0) == 0.5
    assert nieuwenhuizen_gain_function(0.1, order=48) == pytest.approx(
        nieuwenhuizen_gain_function(0.1, order=64), rel=1e-4,
    )


def test_point_charge_circular_endpoint_is_puthoff_balance_curve():
    # kappa=1 means L=1/k. Eq. (2.35): D=beta^2*k^8*(k/2-1).
    k = 0.8
    E = -0.5 * k**2
    L = 1.0 / k
    from blueberry_circus.oracles import beta_coefficient
    expected = beta_coefficient() ** 2 * k**8 * (k / 2.0 - 1.0)
    assert point_charge_drift(E, L, per_orbit=False) == pytest.approx(expected)


def test_setterfield_drive_is_canonical_and_records_parameter_work():
    drive = SetterfieldDrive(amplitude=0.05, omega_ratio=1.0, phase=0.0)
    x = np.array([1.0, 0.0, 0.0])
    p = np.array([0.0, 1.0, 0.0])
    xdot, pdot = drive.canonical_rhs(0.0, x, p)
    assert xdot == pytest.approx(p / drive.mass(0.0))
    assert pdot == pytest.approx(-x)
    assert drive.external_power(0.0, p) == pytest.approx(
        -drive.mass_rate(0.0) * np.dot(p, p) / (2.0 * drive.mass(0.0) ** 2)
    )
    assert SetterfieldDrive(0.0, 1.0, 0.0).external_power(0.3, p) == 0.0


def test_setterfield_driven_hamiltonian_ledger_closes():
    drive = SetterfieldDrive(amplitude=0.01, omega_ratio=0.1, phase=0.0)
    result = drive.integrate_canonical(
        OrbitState(-0.5, 1.0), periods=0.02, steps=2000,
    )
    assert result.ledger.zpf_work == 0.0
    assert result.ledger.radiative_loss == 0.0
    assert result.ledger.internal_mode_exchange == 0.0
    assert result.ledger.relative_closure < 1e-8


def test_finite_shell_uses_one_reciprocal_form_factor_for_both_channels():
    shell = FiniteShellResponse(radius=0.1)
    omega = np.array([0.0, 1.0, 10.0])
    f = shell.form_factor(omega)
    absorbed, radiated = shell.filter_channel_amplitudes(omega,
                                                         np.ones(3),
                                                         np.ones(3))
    assert absorbed == pytest.approx(f)
    assert radiated == pytest.approx(f)
    point = FiniteShellResponse(radius=0.0)
    assert point.form_factor(omega) == pytest.approx(np.ones(3))


def test_finite_shell_point_limit_recovers_stochastic_baseline_exactly():
    cfg = TournamentConfig(
        energies=(-0.02,), angular_momenta=(0.55,), coupling_scales=(4.0,),
        seeds=(3, 5, 7, 11), n_modes=64, max_resolution_levels=2,
        convergence_rtol=0.5,
    )
    state = OrbitState(-0.02, 0.55)
    baseline = stochastic_cell(state, coupling_scale=4.0, config=cfg)
    point = stochastic_cell(
        state, coupling_scale=4.0, config=cfg,
        response=FiniteShellResponse(radius=0.0),
    )
    assert point.mean_drift == pytest.approx(baseline.mean_drift, abs=1e-30)
    assert point.ledger.radiative_loss == pytest.approx(
        baseline.ledger.radiative_loss, abs=1e-30,
    )


def test_inverse_square_zero_limit_and_calculated_critical_value():
    control = InverseSquareControl(d=0.0)
    x = np.array([1.2, -0.4, 0.1])
    assert control.potential(x) == pytest.approx(-1.0 / np.linalg.norm(x))
    assert control.force(x) == pytest.approx(-x / np.linalg.norm(x) ** 3)
    dc, mu_at_max, hmax = InverseSquareControl.critical_d(order=64)
    assert dc == pytest.approx(-hmax**2)
    # Direct evaluation of the published Eqs. (56)--(64) exposes a stronger
    # contradiction than the rounded prose value -35.8: H peaks near 0.59,
    # not at the quoted H(0)=5.99 endpoint.
    assert -55.0 < dc < -52.0
    assert 0.55 < mu_at_max < 0.65
    # At d=0, mu=1 and Eq. (56) must reduce to the independently certified
    # point-charge radial limit: G(1)=3 Lc/2.
    assert inverse_square_gain_function(1.0, order=48) == pytest.approx(
        1.5 * critical_angular_momentum(), rel=1e-4,
    )
    state = OrbitState(-0.01, 0.55)
    assert control.drift_channels(state, order=64) == pytest.approx(
        point_charge_channels(state, order=64, per_orbit=True),
        rel=0.0, abs=0.0,
    )


def test_multipole_surrogate_is_conservative_without_zpf_or_damping():
    model = MultipoleStorage(omega_ratio=2.0, eta=1e-4)
    state = OrbitState(-0.2, 0.8)
    result = model.integrate_conservative(state, periods=0.1, steps=4000)
    assert result.ledger.zpf_work == 0.0
    assert result.ledger.radiative_loss == 0.0
    assert result.ledger.external_parameter_work == 0.0
    assert result.relative_total_energy_error < 1e-6
    assert result.ledger.relative_closure < 1e-6


def test_stochastic_estimator_reproduces_from_stored_seed_and_resolution():
    cfg = TournamentConfig(
        energies=(-0.02,), angular_momenta=(0.55,),
        coupling_scales=(4.0,), seeds=(17, 29, 41, 53),
        n_modes=64, max_resolution_levels=2, convergence_rtol=0.5,
    )
    state = OrbitState(-0.02, 0.55)
    a = stochastic_cell(state, coupling_scale=4.0, config=cfg)
    b = stochastic_cell(state, coupling_scale=4.0, config=cfg)
    assert a == b
    assert a.seeds == cfg.seeds
    assert a.coupling_scale == 4.0
    assert a.resolutions == ((64, cfg.timestep),
                             (128, cfg.timestep / 2.0))
    assert a.ledger.relative_closure < 0.01
    changed_dt = TournamentConfig(
        energies=(-0.02,), angular_momenta=(0.55,),
        coupling_scales=(4.0,), seeds=cfg.seeds,
        n_modes=64, timestep=cfg.timestep * 8.0,
        max_resolution_levels=2, convergence_rtol=0.5,
    )
    assert stochastic_cell(state, coupling_scale=4.0,
                           config=changed_dt).mean_drift != a.mean_drift


def test_preregistered_32_seed_2048_mode_resolution_gate_executes():
    cfg = TournamentConfig(
        energies=(-0.02,), angular_momenta=(0.55,), coupling_scales=(4.0,),
    )
    result = stochastic_cell(OrbitState(-0.02, 0.55), coupling_scale=4.0,
                             config=cfg)
    assert len(result.seeds) == 32
    assert result.resolutions == ((2048, 0.002), (4096, 0.001))
    assert result.converged
    assert abs(result.resolution_drifts[1] - result.resolution_drifts[0]) < (
        cfg.convergence_rtol
        * max(abs(result.resolution_drifts[0]),
              abs(result.resolution_drifts[1]))
    )


def test_classification_is_closed_vocabulary_and_suppression_is_strict():
    low_l = (0.45, 0.55)
    suppressed = [
        HypothesisResult.cell("FiniteShellResponse", -0.01, L, -2.0, -1.0,
                              EnergyLedger.from_channels(-1.5, 0.5, 2.0),
                              converged=True)
        for L in low_l
    ]
    assert classify_hypothesis(suppressed, low_l_cells=low_l,
                               coupling_converged=True) == CHANNEL_SUPPRESSED
    assert classify_hypothesis(suppressed, low_l_cells=low_l,
                               coupling_converged=False) == NULL

    active = [HypothesisResult.cell(
        "SetterfieldDrive", -0.01, 0.45, -1.0, -0.1,
        EnergyLedger.from_channels(-0.5, 0.0, 0.0,
                                   external_parameter_work=-0.5),
        converged=True,
    )]
    assert classify_hypothesis(active, low_l_cells=(0.45,),
                               coupling_converged=True) == ACTIVE_CONTROL
    assert classify_hypothesis([], low_l_cells=low_l,
                               coupling_converged=True) == NULL
    assert "STABLE_GROUND_STATE" not in {
        ACTIVE_CONTROL, CHANNEL_SUPPRESSED, NO_EFFECT, NULL
    }


def test_result_json_is_machine_reproducible():
    result = HypothesisResult.cell(
        "PointCharge", -0.02, 0.55, 1.0, 2.0,
        EnergyLedger.from_channels(1.5, 2.0, 0.5), converged=True,
        seeds=(1, 2), resolutions=((32, 0.01), (64, 0.005)),
    )
    assert HypothesisResult.from_json(result.to_json()) == result
    assert json.loads(result.to_json())["classification"] in {
        "CHANNEL_SUPPRESSED", "ACTIVE_CONTROL", "NO_EFFECT",
        "DESTABILIZED", "NULL",
    }
    tampered = json.loads(result.to_json())
    tampered["ledger"]["numerical_closure_residual"] = 1.0
    assert HypothesisResult.from_json(json.dumps(tampered)).ledger.relative_closure > 0.1


def test_preregistered_arm_parameter_sets():
    cfg = TournamentConfig()
    assert cfg.setterfield_amplitudes == (0.01, 0.05, 0.1)
    assert cfg.setterfield_omega_ratios == (0.1, 1.0, 10.0)
    assert len(cfg.setterfield_phases) == 4
    assert cfg.shell_radii == pytest.approx((ALPHA**2, 1e-3, 1e-2, 0.1, 0.3))
    assert cfg.inverse_square_d == pytest.approx((ALPHA**2, 0.0, -10.0,
                                                  -35.8, -40.0))
    assert cfg.multipole_omega_ratios == (0.5, 1.0, 2.0, 10.0)
    assert cfg.multipole_eta == (1e-6, 1e-4, 1e-2, 1e-1)


def test_reduced_four_arm_runner_is_machine_readable_and_closed_vocabulary():
    cfg = TournamentConfig(
        energies=(-0.02,), angular_momenta=(0.55,), coupling_scales=(4.0,),
        seeds=(13, 17, 19, 23), n_modes=32, max_resolution_levels=2,
        convergence_rtol=0.8,
    )
    cases = {
        "setterfield": {"amplitude": 0.01, "omega_ratio": 0.1, "phase": 0.0,
                         "integration_steps": 256, "integration_periods": 0.001},
        "finite_shell": {"radius": 0.0},
        "inverse_square": {"d": 0.0},
        "multipole": {"omega_ratio": 1.0, "eta": 0.0,
                       "integration_steps": 256, "integration_periods": 0.001},
    }
    for arm, parameters in cases.items():
        report = run_hypothesis_arm(arm, parameters, config=cfg,
                                    quadrature_order=32)
        assert report["classification"] in {
            "CHANNEL_SUPPRESSED", "ACTIVE_CONTROL", "NO_EFFECT",
            "DESTABILIZED", "NULL",
        }
        assert report["arm"] == arm
        assert report["config"] == json.loads(cfg.to_json())
        assert report["cells"]
        assert report["cells"][0]["coupling_scale"] == 4.0
        json.dumps(report, sort_keys=True, allow_nan=False)


def test_all_four_zero_parameter_limits_recover_the_same_baseline_cell():
    cfg = TournamentConfig(
        energies=(-0.02,), angular_momenta=(0.55,), coupling_scales=(4.0,),
        seeds=(31, 37, 41, 43), n_modes=32, max_resolution_levels=2,
        convergence_rtol=0.8,
    )
    common = {"config": cfg, "quadrature_order": 32}
    reports = [
        run_hypothesis_arm("setterfield", {
            "amplitude": 0.0, "omega_ratio": 1.0, "phase": 0.0,
            "integration_steps": 32, "integration_periods": 0.001,
        }, **common),
        run_hypothesis_arm("finite_shell", {"radius": 0.0}, **common),
        run_hypothesis_arm("inverse_square", {"d": 0.0}, **common),
        run_hypothesis_arm("multipole", {
            "omega_ratio": 1.0, "eta": 0.0,
            "integration_steps": 32, "integration_periods": 0.001,
        }, **common),
    ]
    cells = [report["cells"][0] for report in reports]
    reference = cells[0]
    for report, cell in zip(reports, cells):
        assert report["classification"] == NO_EFFECT
        assert cell["mean_drift"] == reference["mean_drift"]
        assert cell["confidence_low"] == reference["confidence_low"]
        assert cell["confidence_high"] == reference["confidence_high"]
        assert cell["ledger"] == reference["ledger"]
