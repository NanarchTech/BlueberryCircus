"""Cross-language certificate recheck via nanarch-verify.

The strongest independence property BlueberryCircus advertises is that its
verdicts are re-derivable by a *separate language and numeric stack that trusts
no emitter*. This test wires an external Rust verifier (`nanarch-verify`) to
re-check a real emitted bundle:

  * a genuine bundle (vacuum PASS + physicality NULL) must re-derive consistently
    -> the verifier exits 0 (RESULT PASS);
  * a tampered bundle (stored PASS, but the numbers re-derive FAIL) must be
    REJECTED -> the verifier exits 1 (RESULT FAIL).

Asserting BOTH directions is deliberate: a verifier that always said PASS would
pass the first check and fail the second, so the test cannot pass trivially.

The verifier is an OPTIONAL native build (like the Rust/JAX backends). Point
`BLUEBERRY_VERIFY_BIN` at the `verify` binary; the tests are behind the
`verify` marker (`pytest -m verify`) and self-skip when the binary is absent.
"""
import os
import subprocess

import pytest

from blueberry_circus.constants import SI
from blueberry_circus import symplectic as sp
from blueberry_circus.certify import Certificate, PASS, save_bundle


def _verify_bin():
    """Resolve the nanarch-verify binary from BLUEBERRY_VERIFY_BIN, or None."""
    env = os.environ.get("BLUEBERRY_VERIFY_BIN")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    return None


pytestmark = pytest.mark.verify

_BIN = _verify_bin()
_SKIP = pytest.mark.skipif(
    _BIN is None,
    reason="nanarch-verify binary not found; point BLUEBERRY_VERIFY_BIN at it")


def _run(path):
    return subprocess.run([_BIN, path], capture_output=True, text=True)


@_SKIP
def test_rust_verifier_rederives_pass(tmp_path):
    w0 = 2.5e16
    certs = [
        sp.certify_sed_vacuum(w0, w0 / 10, w0 * 10, SI, tolerance=1e-4),
        sp.physicality_certificate(sp.vacuum_target_covariance_xv(w0, SI),
                                   mass=SI.mass, omega0=w0, hbar=SI.hbar,
                                   nu_uncertainty=1e-6),
    ]
    path = str(tmp_path / "good.json")
    save_bundle(certs, path)
    r = _run(path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "nanarch-verify" in r.stdout            # it really is the verifier
    # the Rust stack independently re-derived the SAME verdicts:
    assert "stored=PASS recomputed=PASS" in r.stdout
    assert "stored=NULL recomputed=NULL" in r.stdout
    assert "RESULT: PASS" in r.stdout


@_SKIP
def test_rust_verifier_rederives_symplectic_physical_paths(tmp_path):
    # symplectic_physical is the newest rule in the verifier's coverage; cross-check its PASS
    # (thermal, nu>=1/2) and FAIL (sub-Heisenberg) paths, not just the NULL boundary
    # (else an inverted Rust boundary comparison would pass the suite).
    w0 = 2.5e16
    Cvac = sp.vacuum_target_covariance_xv(w0, SI)
    thermal = sp.physicality_certificate(4.0 * Cvac, mass=SI.mass, omega0=w0,
                                         hbar=SI.hbar, nu_uncertainty=1e-3)   # nu=2 -> PASS
    subh = sp.physicality_certificate(0.04 * Cvac, mass=SI.mass, omega0=w0,
                                      hbar=SI.hbar, nu_uncertainty=1e-3)       # nu=0.02 -> FAIL
    assert thermal.recheck() == "PASS" and subh.recheck() == "FAIL"
    path = str(tmp_path / "phys.json")
    save_bundle([thermal, subh], path)
    r = _run(path)
    # Consistency is the parity claim; the exit code is 1 only because a genuine
    # FAIL verdict is present (not an inconsistency).
    assert "all_consistent: true" in r.stdout, r.stdout + r.stderr
    assert "stored=PASS recomputed=PASS" in r.stdout
    assert "stored=FAIL recomputed=FAIL" in r.stdout


@_SKIP
def test_rust_verifier_rejects_tampered_bundle(tmp_path):
    # stored PASS, but residual (1.0) >> tolerance (1e-4): the numbers re-derive
    # FAIL. A trusting reader would accept it; the independent verifier must not.
    tampered = Certificate(
        kind="vacuum_covariance_correspondence", claim="tampered residual",
        method="adversarial", rule="residual_le_tol", status=PASS,
        residual=1.0, tolerance=1e-4)
    path = str(tmp_path / "bad.json")
    save_bundle([tampered], path)
    r = _run(path)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "stored=PASS recomputed=FAIL" in r.stdout
    assert "RESULT: FAIL" in r.stdout
