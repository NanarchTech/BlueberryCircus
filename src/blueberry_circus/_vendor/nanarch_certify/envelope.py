"""The cross-language certificate **envelope** (``nanarch-cert-envelope/v1``).

One outer object that both this package's :class:`~nanarch_certify.Certificate`
and the Rust ``certify::Certificate`` serialize into, so a single tamper-evident
hash-chain links certificates emitted by *either* language. This is what gives
the Python certificate the chain-linking it lacks, and gives one re-verifier a
``body_lang``-routed way to recheck both shapes.

Derived from the Nanarch certificate-envelope specification. The buildable subset is specified in ``ENVELOPE.md`` next to this
module; the canonical byte rule (the keystone) is
:mod:`nanarch_certify.canonical`.

Layered guarantee (honest about what crosses a language boundary):

* **chain integrity** -- ``chain_hash = SHA256(canonical({prev_hash, body_hash,
  kind, claim}))``. The material is **strings only**, so the chain verifies
  *byte-identically across Python and Rust* (no float on the hash surface).
  Genesis ``prev_hash`` = 64 hex zeros.
* **body integrity** -- ``body_hash = SHA256(canonical(body))``. Re-derivable by
  any *same-language* reader (Python ``float`` round-trips bit-exactly; the Rust
  body is stored at serde_json's transport fixed point). Cross-language body
  re-hash is **not** asserted -- CPython and Rust can parse the same decimal 1
  ULP apart -- which is exactly why a Rust body is rechecked at the *schema*
  level (recorded defects vs recorded tolerances), never by re-hashing.
* **verdict recheck** -- polyglot, routed by ``body_lang``: Python bodies via the
  existing :data:`~nanarch_certify.certificate.RULES`; Rust bodies via a
  schema-level re-check of the recorded 9-defect contract, **without running
  Rust**.

stdlib-only + this package; no third-party imports.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .canonical import canonical_hash, canonical_str
from .certificate import FAIL, NULL, PASS, Certificate
from .certificate import RULES as _PY_RULES

ENVELOPE_SCHEMA = "nanarch-cert-envelope/v1"
BODY_LANG_PY = "python-cert-ir/v1"
BODY_LANG_RUST = "rust-symplectic-certify/v1"
GENESIS = "0" * 64

# The Rust 9-defect contract field names (seven core + Bloch-Messiah + Siegel)
# and the Heisenberg positivity field. Mirrors `certify::ValidationReport`.
_RUST_DEFECTS_LE_TOL = (
    "symplectic_defect",
    "bog_unitarity_def1",
    "bog_unitarity_def2",
    "sym_bog_reconstruction_defect",
    "cov_consistency_defect",
    "n_consistency_defect",
    "bloch_messiah_defect",
)


def _chain_material(prev_hash: str, body_hash: str, kind: str, claim: str) -> dict:
    """The (string-only) material whose canonical hash is the chain link."""
    return {"prev_hash": prev_hash, "body_hash": body_hash, "kind": kind, "claim": claim}


def chain_hash(prev_hash: str, body_hash: str, kind: str, claim: str) -> str:
    """``SHA256(canonical({prev_hash, body_hash, kind, claim}))`` -- string-only,
    so Python and Rust agree byte-for-byte."""
    return canonical_hash(_chain_material(prev_hash, body_hash, kind, claim))


def to_envelope(cert: Certificate, prev_hash: str = GENESIS) -> dict:
    """Wrap a Python :class:`Certificate` into an envelope linked to ``prev_hash``.

    ``body`` is the verbatim ``cert.to_dict()`` (Python ``float`` round-trips
    bit-exactly, so it is already at its transport fixed point); ``body_hash`` is
    its canonical hash; ``kind`` / ``claim`` pass through; ``flip_distance`` is
    populated by :func:`nanarch_certify.flip.flip_distance` when available.
    """
    body = cert.to_dict()
    body_hash = canonical_hash(body)
    kind = cert.kind
    claim = cert.claim
    env = {
        "schema": ENVELOPE_SCHEMA,
        "kind": kind,
        "claim": claim,
        "method": cert.method,
        "provenance": {
            **dict(cert.provenance),
            "source_lang": "python",
        },
        "body": body,
        "body_lang": BODY_LANG_PY,
        "body_hash": body_hash,
        "prev_hash": prev_hash,
        "chain_hash": chain_hash(prev_hash, body_hash, kind, claim),
    }
    # Best-effort flip-distance enrichment (smallest verdict-flipping perturbation).
    try:
        from .flip import flip_distance as _flip_distance

        fd = _flip_distance(cert)
        if fd is not None:
            env["flip_distance"] = fd
    except Exception:
        # flip is a diagnostic enrichment; never block envelope construction.
        pass
    return env


def build_chain(certs, prev_hash: str = GENESIS) -> list[dict]:
    """Build an envelope chain over a sequence of Python certificates."""
    out = []
    prev = prev_hash
    for c in certs:
        env = to_envelope(c, prev)
        prev = env["chain_hash"]
        out.append(env)
    return out


# --- verification ------------------------------------------------------------
def verify_body_hash(env: dict) -> bool:
    """Re-derive ``body_hash`` from ``body`` (sound same-language; see module doc)."""
    return canonical_hash(env["body"]) == env["body_hash"]


def verify_chain_hash(env: dict) -> bool:
    """Re-derive ``chain_hash`` from the string-only material (always cross-language)."""
    return (
        chain_hash(env["prev_hash"], env["body_hash"], env["kind"], env["claim"])
        == env["chain_hash"]
    )


def verify_envelope(env: dict, *, require_body_hash: bool = True) -> dict:
    """Verify one envelope in isolation.

    Returns a report dict. ``chain_hash`` is always re-derived (string-only, so
    cross-language exact). ``body_hash`` is re-derived too; set
    ``require_body_hash=False`` when the body crossed a *cross-language* boundary
    (CPython vs Rust decimal parse can differ 1 ULP) -- the verdict recheck and
    chain integrity do not depend on body re-hash.
    """
    chain_ok = verify_chain_hash(env)
    body_ok = verify_body_hash(env)
    ok = chain_ok and (body_ok or not require_body_hash)
    return {
        "schema": env.get("schema"),
        "kind": env.get("kind"),
        "body_lang": env.get("body_lang"),
        "chain_hash_ok": chain_ok,
        "body_hash_ok": body_ok,
        "ok": ok,
    }


def verify_chain(envelopes, *, require_body_hash: bool = True) -> dict:
    """Re-verify an envelope chain: link structure + per-envelope hashes.

    ``True`` iff every ``prev_hash`` equals the prior ``chain_hash`` (genesis =
    64 zeros), and every envelope verifies. Returns a structured report.
    """
    rows = []
    expected_prev = GENESIS
    linked = True
    all_ok = True
    for i, env in enumerate(envelopes):
        link_ok = env.get("prev_hash") == expected_prev
        rep = verify_envelope(env, require_body_hash=require_body_hash)
        row = {"index": i, "link_ok": link_ok, **rep}
        rows.append(row)
        if not link_ok:
            linked = False
        if not (link_ok and rep["ok"]):
            all_ok = False
        expected_prev = env.get("chain_hash")
    return {
        "n": len(rows),
        "linked": linked,
        "verified": all_ok,
        "rows": rows,
    }


# --- polyglot verdict recheck (routed by body_lang) --------------------------
def _recheck_python_body(body: dict) -> str:
    """Re-derive a Python certificate's verdict from its recorded numbers + rule.

    Uses the existing :data:`RULES` via a throwaway :class:`Certificate`, so the
    rule semantics are the *same* code the emitter is checked against -- not a
    second copy that could drift. (The independent re-implementation lives in
    ``nanarch-evidence``; this is the certify-side recheck.)
    """
    rule = body.get("rule")
    if rule not in _PY_RULES:
        raise ValueError(f"unknown rule {rule!r}; cannot recheck python body")
    enc = body.get("enclosure")
    cert = Certificate(
        kind=body.get("kind", "?"),
        claim=body.get("claim", ""),
        method=body.get("method", ""),
        rule=rule,
        status=body.get("status", NULL),
        value=body.get("value"),
        enclosure=(float(enc[0]), float(enc[1])) if enc is not None else None,
        residual=body.get("residual"),
        tolerance=body.get("tolerance"),
        provenance=dict(body.get("provenance", {})),
    )
    return cert.recheck()


def _recheck_rust_body(body: dict) -> str:
    """Schema-level recheck of a Rust 9-defect contract body -- WITHOUT running Rust.

    Re-derives ``ok`` from the recorded numbers exactly as ``certify::validate``
    forms it: every named defect ``< tol``, Heisenberg minimum eigenvalue
    ``> -tol``, and the *conditional* Siegel sub-check (gates only when the
    output is pure). Returns ``PASS`` / ``FAIL`` (Rust ``ok`` is two-valued; the
    contract has no NULL), or ``NULL`` only if the record is structurally
    incomplete (an honest "not determinable", never a silent pass).
    """
    report = body.get("report")
    if not isinstance(report, dict):
        return NULL
    # `tol` lives on the Rust ValidationReport (and is mirrored on the
    # Certificate); prefer the report's, fall back to the body's.
    tol = report.get("tol", body.get("tol"))
    if tol is None:
        return NULL
    try:
        for name in _RUST_DEFECTS_LE_TOL:
            if report[name] >= tol:
                return FAIL
        if report["heisenberg_eigval_min"] <= -tol:
            return FAIL
        # Conditional Siegel: gates ok only when the output certifies as pure.
        if report.get("output_is_pure"):
            sa = report.get("siegel_admissibility_defect")
            if sa is None or sa >= tol:
                return FAIL
            # A pure output must yield an admissible Siegel point (Im Omega > 0).
            im_min = report.get("siegel_im_min_eig")
            if im_min is None or im_min <= 0.0:
                return FAIL
    except (KeyError, TypeError):
        return NULL
    return PASS


def recheck_envelope(env: dict) -> str:
    """Polyglot verdict recheck routed by ``body_lang``.

    * ``python-cert-ir/v1`` -> the existing Python recheck rules.
    * ``rust-symplectic-certify/v1`` -> schema-level recheck of the recorded
      9-defect contract (no Rust executed).
    """
    body_lang = env.get("body_lang")
    body = env.get("body", {})
    if body_lang == BODY_LANG_PY:
        return _recheck_python_body(body)
    if body_lang == BODY_LANG_RUST:
        return _recheck_rust_body(body)
    raise ValueError(f"unknown body_lang {body_lang!r}; cannot recheck")


def recorded_verdict(env: dict) -> str | None:
    """The verdict as recorded in the body (``status`` for Python, ``report.ok``
    mapped to PASS/FAIL for Rust) -- what ``recheck_envelope`` is checked against."""
    body = env.get("body", {})
    if env.get("body_lang") == BODY_LANG_PY:
        return body.get("status")
    if env.get("body_lang") == BODY_LANG_RUST:
        report = body.get("report", {})
        ok = report.get("ok")
        if ok is None:
            return None
        return PASS if ok else FAIL
    return None
