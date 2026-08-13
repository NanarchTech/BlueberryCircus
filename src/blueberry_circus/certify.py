"""BlueberryCircus assurance bridge over the canonical ``nanarch_certify``.

This module USED to carry a home-grown re-implementation of the Nanarch
certificate rule registry. That is gone. The single source of truth for the
recheck rules is the canonical :mod:`nanarch_certify` package — the
same object the Rust verifier and the cross-language envelope are defined
against. Keeping a second registry here would let the two drift; the bridge
removes that risk by *importing* the canonical :class:`Certificate` and re-using
its rules verbatim.

What the bridge adds on top of the canonical object (ergonomics, not rules):

* :func:`rel_error_certificate` — the canonical registry has **no**
  ``rel_error_le_tol`` rule (its five are ``residual_le_tol``,
  ``enclosure_pos``, ``enclosure_contains``, ``chern_licensed``,
  ``symplectic_physical``). BlueberryCircus's relative-error claims (the Boyer
  oscillator oracle) are therefore *re-encoded* as ``residual_le_tol`` with
  ``residual = |value - reference| / |reference|`` and ``tolerance = tol``,
  made a first-class constructor.
* :func:`finalize` — set ``status`` to the independently re-derived verdict
  (canonical ``Certificate`` is an immutable-by-convention dataclass with no
  ``finalize`` method of its own).
* :func:`audit_overclaim` — stored ``PASS`` while the rule re-derives non-``PASS``.
* :func:`save_bundle` / :func:`load_bundle` — JSON bundle I/O over canonical certs.

The envelope helpers (:func:`to_envelope`, :func:`build_chain`,
:func:`verify_chain`, :func:`recheck_envelope`, :func:`canonical_hash`) are
re-exported so the assurance pipeline imports them from one place.

Resolution policy (import-or-raise; never silently degrade — a missing assurance
layer is a FAIL, not a skipped NULL): first a real installed
``nanarch_certify`` (its rule-carrying submodules must actually import, so an
empty namespace directory on ``sys.path`` cannot shadow it), then a
``BLUEBERRY_CERTIFY_SRC`` override, then the vendored mirror shipped inside
this package (``_vendor/``).
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Optional


def _usable_canonical() -> bool:
    """True only if a REAL ``nanarch_certify`` is importable.

    A bare ``import nanarch_certify`` is not sufficient evidence: any directory
    named ``nanarch_certify`` on ``sys.path`` (including the caller's cwd)
    resolves as an empty PEP-420 namespace package. Requiring the submodules
    that carry the rule registry means an incomplete or shadowing copy falls
    through to the vendored mirror instead of bricking the import.
    """
    try:
        import nanarch_certify.certificate  # noqa: F401
        import nanarch_certify.canonical    # noqa: F401
        import nanarch_certify.envelope     # noqa: F401
        return True
    except ImportError:
        for mod in [m for m in sys.modules if m.split(".")[0] == "nanarch_certify"]:
            del sys.modules[mod]
        return False


def _ensure_canonical() -> None:
    """Make ``nanarch_certify`` importable, or raise with a clear remediation."""
    if _usable_canonical():
        return
    here = os.path.dirname(os.path.abspath(__file__))          # .../blueberry_circus
    candidates = [
        os.environ.get("BLUEBERRY_CERTIFY_SRC"),
        # Shipped vendored verbatim mirror -- INSIDE the package, so it survives
        # `pip install` and `import blueberry_circus` works from site-packages.
        # Installed / source / CI all resolve to the same code.
        os.path.join(here, "_vendor"),
    ]
    for cand in candidates:
        if cand and os.path.isdir(os.path.join(cand, "nanarch_certify")):
            sys.path.insert(0, os.path.abspath(cand))
            if _usable_canonical():
                return
            sys.path.pop(0)
    raise ImportError(
        "BlueberryCircus requires the canonical 'nanarch_certify' package (the "
        "assurance layer). It was not importable and none of the candidate "
        "paths contained it. Set BLUEBERRY_CERTIFY_SRC to a directory "
        "containing the 'nanarch_certify' package (5 rules including "
        "symplectic_physical)."
    )


_ensure_canonical()

# Single source of truth: the canonical certificate object + rule registry.
from nanarch_certify.certificate import (  # noqa: E402
    Certificate, RULES, PASS, FAIL, NULL,
)
from nanarch_certify.canonical import canonical_hash, float_hex_token  # noqa: E402
from nanarch_certify.envelope import (  # noqa: E402
    to_envelope, build_chain, verify_chain, verify_envelope, recheck_envelope,
    GENESIS,
)

# Default emitter method label for BlueberryCircus-emitted certificates.
_DEFAULT_METHOD = "blueberry_circus SED simulation"


# --- ergonomics on top of the canonical object (NOT new rules) ---------------
def finalize(cert: Certificate) -> Certificate:
    """Set ``status`` to the independently re-derived verdict and return the cert.

    Canonical ``Certificate`` has no ``finalize`` method; emitters that want the
    stored ``status`` to match the rule call this. ``recheck`` itself never trusts
    the stored status, so this only affects the recorded field.
    """
    cert.status = cert.recheck()
    cert.provenance.setdefault("emitter", "blueberry_circus")
    return cert


def audit_overclaim(cert: Certificate) -> bool:
    """True iff the certificate stores ``PASS`` while the rule re-derives non-``PASS``."""
    return cert.status == PASS and cert.recheck() != PASS


def rel_error_certificate(kind: str, claim: str, value: float, reference: float,
                          tolerance: float, *, method: str = _DEFAULT_METHOD,
                          provenance: Optional[dict] = None,
                          finalize_status: bool = True) -> Certificate:
    """A relative-error claim re-encoded as a canonical ``residual_le_tol`` cert.

    ``residual = |value - reference| / |reference|`` (the relative error) and
    ``tolerance`` is the acceptance threshold, so the canonical rule
    ``residual_le_tol`` reproduces the old ``rel_error_le_tol`` semantics exactly
    while living in the canonical rule set. A non-finite ``value`` or zero
    ``reference`` yields a **finite over-threshold residual sentinel** (so the
    rule re-derives ``FAIL`` *and* the certificate stays serializable -- the
    canonical hash surface rejects inf/NaN); the non-finite value itself is kept
    off the hash surface (recorded as ``None``).
    """
    if reference == 0 or not (math.isfinite(value) and math.isfinite(reference)):
        # No relative error is definable. Record a FINITE residual guaranteed to
        # exceed tolerance (re-derives FAIL) -- never inf, which would make the
        # FAIL cert un-canonicalizable.
        residual = abs(float(tolerance)) * 2.0 + 1.0
    else:
        residual = abs(value - reference) / abs(reference)
    prov = dict(provenance or {})
    prov.setdefault("reference", float(reference) if math.isfinite(reference) else None)
    prov.setdefault("relative_error", True)
    cert = Certificate(
        kind=kind, claim=claim, method=method, rule="residual_le_tol",
        value=value if math.isfinite(value) else None,
        residual=residual,                       # always finite
        tolerance=tolerance, provenance=prov)
    if finalize_status:
        finalize(cert)
    return cert


# --- bundle I/O over canonical certificates ----------------------------------
_BUNDLE_SCHEMA = "blueberry_circus.certificate/2"


def save_bundle(certs, path: str) -> None:
    """Write a JSON bundle of canonical certificates (schema v2)."""
    payload = {"schema": _BUNDLE_SCHEMA,
               "certificates": [c.to_dict() for c in certs]}
    with open(path, "w") as fh:
        # allow_nan=False: refuse to write the non-standard Infinity/NaN tokens
        # (invalid JSON, rejected by Rust/JS/jq) -- fail loudly instead of
        # silently breaking the bundle's cross-language portability.
        json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)


def load_bundle(path: str):
    """Load a bundle written by :func:`save_bundle` into canonical certificates."""
    with open(path) as fh:
        payload = json.load(fh)
    return [Certificate.from_dict(d) for d in payload["certificates"]]


__all__ = [
    "Certificate", "RULES", "PASS", "FAIL", "NULL",
    "finalize", "audit_overclaim", "rel_error_certificate",
    "save_bundle", "load_bundle",
    "canonical_hash", "float_hex_token",
    "to_envelope", "build_chain", "verify_chain", "verify_envelope",
    "recheck_envelope", "GENESIS",
]
