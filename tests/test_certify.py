"""Assurance layer: BlueberryCircus over the canonical nanarch_certify bridge.

The home-grown rule registry is gone; these tests pin the bridge to the canonical
five-rule registry and exercise the rel-error re-encode and the envelope chain.
"""
import math
import pytest

from blueberry_circus.certify import (Certificate, RULES, PASS, FAIL, NULL,
                                       finalize, audit_overclaim,
                                       rel_error_certificate, save_bundle,
                                       load_bundle, build_chain, verify_chain,
                                       recheck_envelope)


def _cert(rule, **kw):
    return Certificate(kind="k", claim="c", method="test", rule=rule, **kw)


def test_single_canonical_rule_registry():
    # One source of truth: the canonical five rules. No rel_error_le_tol.
    assert set(RULES) == {"residual_le_tol", "enclosure_pos", "enclosure_contains",
                          "chern_licensed", "symplectic_physical"}
    with pytest.raises(ValueError):
        _cert("rel_error_le_tol")            # canonical raises on unknown rule


def test_residual_le_tol():
    assert _cert("residual_le_tol", residual=1e-9, tolerance=1e-6).recheck() == PASS
    assert _cert("residual_le_tol", residual=1e-3, tolerance=1e-6).recheck() == FAIL
    assert _cert("residual_le_tol").recheck() == NULL


def test_enclosure_rules_canonical_semantics():
    assert _cert("enclosure_contains", value=0.5, enclosure=(0.0, 1.0)).recheck() == PASS
    assert _cert("enclosure_contains", value=2.0, enclosure=(0.0, 1.0)).recheck() == FAIL
    assert _cert("enclosure_pos", enclosure=(0.1, 1.0)).recheck() == PASS
    assert _cert("enclosure_pos", enclosure=(-1.0, -0.1)).recheck() == FAIL
    # straddling zero is an honest NULL under the canonical rule (not FAIL)
    assert _cert("enclosure_pos", enclosure=(-0.1, 1.0)).recheck() == NULL


def test_rel_error_certificate_reencodes_as_residual_le_tol():
    c = rel_error_certificate("boyer", "x^2", value=1.001, reference=1.0, tolerance=1e-2)
    assert c.rule == "residual_le_tol"          # NOT rel_error_le_tol
    assert c.recheck() == PASS and c.status == PASS
    assert math.isclose(c.residual, 0.001, rel_tol=1e-9)
    bad = rel_error_certificate("boyer", "x^2", value=1.5, reference=1.0, tolerance=1e-2)
    assert bad.recheck() == FAIL
    # non-finite value -> residual = +inf -> FAIL, and value is kept off the hash surface
    nanc = rel_error_certificate("k", "c", value=float("nan"), reference=1.0, tolerance=1e-2)
    assert nanc.recheck() == FAIL and nanc.value is None


def test_recheck_is_independent_of_stored_status():
    c = _cert("residual_le_tol", status=PASS, residual=10.0, tolerance=1e-6)
    assert c.recheck() == FAIL
    assert audit_overclaim(c) is True


def test_finalize_sets_true_status():
    c = finalize(_cert("enclosure_pos", status=PASS, enclosure=(-1.0, -0.1)))
    assert c.status == FAIL
    assert audit_overclaim(c) is False           # status now honest


def test_bundle_roundtrip(tmp_path):
    c = finalize(_cert("enclosure_contains", value=0.5, enclosure=(0.0, 1.0)))
    p = tmp_path / "b.json"
    save_bundle([c], str(p))
    (c2,) = load_bundle(str(p))
    assert c2.recheck() == PASS and c2.kind == "k"


def test_envelope_chain_verifies_and_polyglot_recheck():
    # emit -> build_chain -> verify_chain green; recheck_envelope agrees with status.
    c = rel_error_certificate("sed_ground_state_variance",
                              "x^2 = hbar/2mw0", value=2.3165e-21,
                              reference=2.3154e-21, tolerance=5e-3)
    envs = build_chain([c])
    rep = verify_chain(envs)
    assert rep["verified"] is True
    assert recheck_envelope(envs[0]) == c.status == PASS


def test_tampered_envelope_body_is_caught():
    # A forged PASS in the body must be flipped by the polyglot recheck.
    c = finalize(_cert("residual_le_tol", residual=1e-9, tolerance=1e-6))
    envs = build_chain([c])
    envs[0]["body"]["residual"] = 10.0          # tamper: exceeds tolerance
    assert recheck_envelope(envs[0]) == FAIL    # recomputed from numbers, not trusted


def test_fail_cert_from_zero_reference_stays_serializable():
    # reference==0 cannot form a relative error: the cert must FAIL AND still be
    # canonicalizable (finite residual sentinel, never inf on the hash surface).
    c = rel_error_certificate("k", "c", value=1.0, reference=0.0, tolerance=1e-3)
    assert c.recheck() == FAIL and math.isfinite(c.residual)
    envs = build_chain([c])                      # would raise if residual were inf
    assert verify_chain(envs)["verified"] is True
    assert recheck_envelope(envs[0]) == FAIL
