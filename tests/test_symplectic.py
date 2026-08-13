"""Symplectic / Gaussian-state phase-space certificates.

The headline synergy, in its honest, verified form: the linear
(Boyer) SED sector reproduces the *full* quantum-vacuum covariance, certified as
an equality-to-tolerance claim (PASS-capable), and strictly stronger than the
position-only O2 oracle. Physicality (nu >= 1/2) is kept as a separate, honest
diagnostic whose vacuum verdict is NULL (the pure-state boundary) -- and which,
crucially, does NOT flag ionization (that is the vacuum cert's job).
"""
import numpy as np
import pytest

import blueberry_circus as bc
from blueberry_circus import symplectic as sp
from blueberry_circus.constants import SI, Units
from blueberry_circus.certify import PASS, FAIL, NULL, save_bundle, load_bundle, \
    audit_overclaim, build_chain, verify_chain


# --- nondimensionalization: the units adapter --------------------------------
@pytest.mark.parametrize("units,w0", [
    (SI, 2.5e16),
    (Units.scaled(0.05, 1.0), 1.0),
    (Units.scaled(0.1, 3.0, mass=2.0, hbar=0.7), 3.0),   # non-trivial m, hbar
])
def test_to_quadrature_maps_vacuum_to_half_identity(units, w0):
    C = sp.vacuum_target_covariance_xv(w0, units)
    sigma = sp.to_quadrature_covariance(C, mass=units.mass, omega0=w0, hbar=units.hbar)
    assert np.allclose(sigma, 0.5 * np.eye(2), atol=1e-12)


def test_to_quadrature_validation():
    with pytest.raises(ValueError):
        sp.to_quadrature_covariance(np.eye(3), mass=1, omega0=1, hbar=1)
    with pytest.raises(ValueError):
        sp.to_quadrature_covariance(np.eye(2), mass=-1, omega0=1, hbar=1)


# --- the symplectic eigenvalue is an exact closed form for one mode ----------
def test_symplectic_eigenvalue_is_sqrt_det_2x2():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.normal(size=(2, 2))
        sigma = a @ a.T + 0.1 * np.eye(2)          # SPD
        assert sp.symplectic_eigenvalue(sigma) == pytest.approx(np.sqrt(np.linalg.det(sigma)))


def test_symplectic_eigenvalue_multimode_matches_blockwise():
    # Two independent vacuum modes (q1,q2,p1,p2 ordering) -> nu_min = 1/2.
    sigma = 0.5 * np.eye(4)
    assert sp.symplectic_eigenvalue(sigma) == pytest.approx(0.5, abs=1e-10)


# --- mode extraction from the 6x6 Result.covariance() readout ----------------
def test_mode_covariance_extraction():
    cov6 = np.arange(36, dtype=float).reshape(6, 6)
    cov6 = 0.5 * (cov6 + cov6.T)                    # symmetric like a real cov
    for axis in (0, 1, 2):
        m = sp.mode_covariance_xv(cov6, axis)
        i, j = axis, axis + 3
        assert m[0, 0] == cov6[i, i] and m[1, 1] == cov6[j, j]
        assert m[0, 1] == cov6[i, j] and m[1, 0] == cov6[j, i]
    with pytest.raises(ValueError):
        sp.mode_covariance_xv(np.eye(4), 0)


# --- vacuum-correspondence certificate ---------------------------------------
def test_vacuum_cert_pass_on_exact_vacuum():
    C = sp.vacuum_target_covariance_xv(2.5e16, SI)
    c = sp.vacuum_covariance_certificate(C, mass=SI.mass, omega0=2.5e16,
                                         hbar=SI.hbar, tolerance=1e-9)
    assert c.status == PASS and c.recheck() == PASS
    assert c.residual < 1e-12
    assert c.value == pytest.approx(0.5, abs=1e-9)   # nu = 1/2


def test_vacuum_cert_fails_when_off_vacuum():
    C = 2.0 * sp.vacuum_target_covariance_xv(2.5e16, SI)   # twice the vacuum spread
    c = sp.vacuum_covariance_certificate(C, mass=SI.mass, omega0=2.5e16,
                                         hbar=SI.hbar, tolerance=1e-2)
    assert c.recheck() == FAIL


def test_vacuum_cert_rejects_squeezed_state_o2_would_pass():
    # The genuine 'pins more than O2' property: a squeezed Gaussian state with the
    # vacuum's <x^2> (so a position-only O2 check PASSes) but a wrong <v^2> is
    # REJECTED by the full-covariance vacuum cert -- a discrimination O2 cannot make.
    w0 = 2.5e16
    Csq = sp.vacuum_target_covariance_xv(w0, SI).copy()
    Csq[1, 1] *= 4.0                                        # 4x <v^2>, <x^2> untouched
    c = sp.vacuum_covariance_certificate(Csq, mass=SI.mass, omega0=w0,
                                         hbar=SI.hbar, tolerance=1e-2)
    assert c.provenance["sigma_qq"] == pytest.approx(0.5, abs=1e-9)   # O2 would PASS
    assert c.recheck() == FAIL                                        # vacuum cert FAILs


def test_diverged_covariance_still_serializes(tmp_path):
    # A diverged/overflowing covariance (the ionization regime) must still produce
    # serializable, re-checkable certs -- FAIL vacuum, NULL physicality -- never
    # inf/NaN on the canonical hash surface. 
    w0 = 2.5e16
    for C in [1e160 * sp.vacuum_target_covariance_xv(w0, SI),
              np.full((2, 2), np.inf)]:
        with np.errstate(all="ignore"):
            cv = sp.vacuum_covariance_certificate(C, mass=SI.mass, omega0=w0,
                                                  hbar=SI.hbar, tolerance=1e-2)
            cp = sp.physicality_certificate(C, mass=SI.mass, omega0=w0,
                                            hbar=SI.hbar, nu_uncertainty=1e-3)
        assert cv.recheck() == FAIL and cp.recheck() == NULL
        save_bundle([cv, cp], str(tmp_path / "div.json"))   # must not raise
        assert verify_chain(build_chain([cv, cp]))["verified"] is True  # hash surface OK


# --- the headline: certify the SED vacuum from the band-limited spectrum ------
def test_certify_sed_vacuum_si_pass():
    w0 = 2.5e16
    c = sp.certify_sed_vacuum(w0, w0 / 10, w0 * 10, SI, tolerance=1e-4)
    assert c.status == PASS and c.recheck() == PASS
    assert c.residual < 1e-4
    # PASS pins the FULL (q,p) covariance, not just <x^2> (see the squeezed-state
    # test for the genuine 'pins more than O2' property; numerically it tracks O2):
    assert c.provenance["sigma_qq"] == pytest.approx(0.5, abs=1e-3)   # <x^2>
    assert c.provenance["sigma_pp"] == pytest.approx(0.5, abs=1e-3)   # <v^2> (kinematically tied)
    assert c.provenance["sigma_qp"] == 0.0                            # <x v> = 0 (stationarity)


def test_certify_sed_vacuum_carries_conjugate_oracle():
    # Documentation-as-test: the cert genuinely consumes <v^2> (the conjugate
    # quadrature), not just the position oracle O2 re-labelled.
    w0 = 2.5e16
    c = sp.certify_sed_vacuum(w0, w0 / 10, w0 * 10, SI, tolerance=1e-4)
    assert c.provenance["vv_integral"] == pytest.approx(c.provenance["vv_target"], rel=1e-3)
    assert c.provenance["xx_integral"] == pytest.approx(c.provenance["xx_target"], rel=1e-3)
    assert "uv_note" in c.provenance        # UV-sensitivity disclosed on the cert


def test_certify_sed_vacuum_band_validation():
    with pytest.raises(ValueError):
        sp.certify_sed_vacuum(1.0, 2.0, 3.0, SI, tolerance=1e-3)   # omega_lo > omega0


# --- UV honesty: <x^2> is UV-benign, <v^2> is quadratically UV-divergent ------
def test_uv_xx_stable_vv_diverges():
    # The honest UV story (this test previously passed for the WRONG reason: the
    # tangent grid never sampled the UV). <x^2> is flat across bands; <v^2> blows
    # up on a wide band -- the divergence the relativistic cutoff regularizes.
    from blueberry_circus.oracles import sed_band_covariance_xv, \
        ground_state_variance_target, ground_state_momentum_target
    w0 = 2.5e16
    xt = ground_state_variance_target(w0, SI)
    vt = ground_state_momentum_target(w0, SI) / SI.mass**2
    Cmod = sed_band_covariance_xv(w0, w0 / 10, w0 * 10, SI)
    Cwide = sed_band_covariance_xv(w0, w0 / 1e6, w0 * 1e6, SI)
    # <x^2>: UV-benign (flat) on both bands.
    assert abs(Cmod[0, 0] - xt) / xt < 1e-3
    assert abs(Cwide[0, 0] - xt) / xt < 1e-3
    # <v^2>: vacuum on the modest band, but DIVERGES (>1000x) on the wide band.
    assert abs(Cmod[1, 1] - vt) / vt < 1e-3
    assert Cwide[1, 1] / vt > 1e3


# --- physicality certificate: NULL at the vacuum boundary --------------------
def test_physicality_is_null_at_vacuum_boundary():
    C = sp.vacuum_target_covariance_xv(2.5e16, SI)
    c = sp.physicality_certificate(C, mass=SI.mass, omega0=2.5e16, hbar=SI.hbar,
                                   nu_uncertainty=1e-6)
    assert c.value == pytest.approx(0.5, abs=1e-9)
    assert c.recheck() == NULL                # straddles 1/2 -> honest NULL, not PASS


def test_physicality_pass_above_vacuum():
    C = 4.0 * sp.vacuum_target_covariance_xv(2.5e16, SI)   # nu = 2 (thermal-like)
    c = sp.physicality_certificate(C, mass=SI.mass, omega0=2.5e16, hbar=SI.hbar,
                                   nu_uncertainty=1e-3)
    assert c.value == pytest.approx(2.0, abs=1e-9)
    assert c.recheck() == PASS


def test_physicality_fail_subheisenberg():
    # A classical SED distribution below the uncertainty bound has no quantum
    # counterpart -> FAIL (this is what symplectic_physical actually detects).
    C = 0.25 * sp.vacuum_target_covariance_xv(2.5e16, SI)  # nu = 0.125
    c = sp.physicality_certificate(C, mass=SI.mass, omega0=2.5e16, hbar=SI.hbar,
                                   nu_uncertainty=1e-3)
    assert c.recheck() == FAIL


def test_physicality_does_NOT_detect_ionization_vacuum_cert_does():
    # The honest correction to the synergy brief: a self-ionizing trajectory has
    # HUGE variance -> nu >> 1/2 -> physicality reads PASS (a valid high-entropy
    # state). The ionization detector is the VACUUM cert, whose residual blows up.
    w0 = 2.5e16
    Cion = 1e6 * sp.vacuum_target_covariance_xv(w0, SI)
    phys = sp.physicality_certificate(Cion, mass=SI.mass, omega0=w0, hbar=SI.hbar,
                                      nu_uncertainty=1e-3)
    vac = sp.vacuum_covariance_certificate(Cion, mass=SI.mass, omega0=w0,
                                           hbar=SI.hbar, tolerance=1e-2)
    assert phys.recheck() == PASS             # NOT a fail -- physicality misses it
    assert vac.recheck() == FAIL              # the vacuum cert catches it


# --- end-to-end: Engine -> covariance -> certificate plumbing ----------------
def test_end_to_end_covariance_certificate_plumbing():
    U = Units.scaled(0.05, 1.0)
    prog = bc.Program(n_particles=1, units=U)
    with prog.context as q:
        bc.Harmonic(1.0) | q[0]
        bc.ZPF(band=(0.5, 2.0), n_modes=40, seed=1, mode="one_dimensional") | q[0]
        bc.RadiationReaction("landau_lifshitz") | q[0]
    res = bc.Engine(dt=0.05, t_max=120).run(prog, x0=[0, 0, 0], v0=[0, 0, 0])
    cov6 = res.covariance()
    assert cov6.shape == (6, 6)
    C = sp.mode_covariance_xv(cov6, axis=0)            # the driven oscillator mode
    phys = sp.physicality_certificate(C, mass=U.mass, omega0=1.0, hbar=U.hbar,
                                      nu_uncertainty=0.05)
    # We assert the PLUMBING (a well-formed, re-checkable cert with nu > 0), NOT a
    # vacuum PASS -- a short single-realization run is equilibration-limited.
    assert phys.value > 0.0
    assert phys.recheck() in (PASS, FAIL, NULL)


# --- serialization: certs stay on the canonical (finite) hash surface --------
def test_symplectic_certs_serialize_and_chain(tmp_path):
    w0 = 2.5e16
    certs = [
        sp.certify_sed_vacuum(w0, w0 / 10, w0 * 10, SI, tolerance=1e-4),
        sp.physicality_certificate(sp.vacuum_target_covariance_xv(w0, SI),
                                   mass=SI.mass, omega0=w0, hbar=SI.hbar,
                                   nu_uncertainty=1e-6),
    ]
    assert not any(audit_overclaim(c) for c in certs)
    path = tmp_path / "bundle.json"
    save_bundle(certs, str(path))
    back = load_bundle(str(path))
    assert [c.recheck() for c in back] == [c.recheck() for c in certs]
    envs = build_chain(certs)                          # hash-chain must not choke
    assert verify_chain(envs)["verified"] is True
