"""Unified machine-checkable certificate object for nanarch-certify.

A :class:`Certificate` is the common currency of the Nanarch assurance layer:
a self-describing, JSON-serialisable record asserting that a computed quantity
satisfies a stated correctness claim within an explicit, *re-checkable* bound.

Every certificate kind in :mod:`nanarch_certify.kinds` produces one of these,
and :func:`nanarch_certify.emit.verify_certificate` independently re-derives its
``status`` from the recorded numbers and ``rule`` -- so a reviewer never has to
trust the emitter, only re-run the rule. ``NULL`` is a first-class third state:
"not determinable", distinct from FAIL ("disproven").
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

PASS = "PASS"
FAIL = "FAIL"
NULL = "NULL"  # inconclusive / not determinable (honest third state)
_VALID = (PASS, FAIL, NULL)


# --- recheck rules: pure functions of the recorded numbers -------------------
def _rule_residual_le_tol(c: "Certificate") -> str:
    if c.residual is None or c.tolerance is None:
        return NULL
    return PASS if c.residual <= c.tolerance else FAIL


def _rule_enclosure_pos(c: "Certificate") -> str:
    """PASS iff the enclosure proves strict positivity (lo > 0)."""
    if c.enclosure is None:
        return NULL
    lo, hi = c.enclosure
    if lo > 0.0:
        return PASS
    if hi < 0.0:
        return FAIL
    return NULL  # enclosure straddles 0 -> inconclusive


def _rule_enclosure_contains(c: "Certificate") -> str:
    """PASS iff the reference value lies inside a tight-enough rigorous enclosure."""
    if c.enclosure is None or c.value is None:
        return NULL
    lo, hi = c.enclosure
    if not (lo <= c.value <= hi):
        return FAIL
    if c.tolerance is not None and (hi - lo) > c.tolerance:
        return NULL
    return PASS


def _rule_chern_licensed(c: "Certificate") -> str:
    """PASS iff a verified open spectral gap licenses an integer invariant.

    ``enclosure`` is the spectral-gap bracket ``(gap_lo, gap_hi)`` and ``value``
    is the untrusted (floating-point) Chern number. A proved-open gap
    (``gap_lo > 0``) makes the invariant a well-defined integer; PASS also
    requires the float to round to that integer within ``tolerance``. ``NULL``
    when the gap is not proved open (invariant not licensed).
    """
    if c.enclosure is None or c.value is None:
        return NULL
    gap_lo = c.enclosure[0]
    if gap_lo <= 0.0:
        return NULL
    tol = c.tolerance if c.tolerance is not None else 0.05
    return PASS if abs(c.value - round(c.value)) <= tol else FAIL


def _rule_symplectic_physical(c: "Certificate") -> str:
    """PASS iff the smallest symplectic eigenvalue is provably >= 1/2.

    ``enclosure`` is a sound bracket ``(nu_min_lo, nu_min_hi)`` of the smallest
    symplectic (Williamson) eigenvalue. A covariance is a valid (physical)
    Gaussian state iff every nu_k >= 1/2, equivalently nu_min >= 1/2 (hbar=1).
    A two-sided enclosure that straddles 1/2 (e.g. the pure-state boundary)
    yields ``NULL`` -- honest, not a false PASS.
    """
    if c.enclosure is None:
        return NULL
    lo, hi = c.enclosure
    if lo >= 0.5:
        return PASS
    if hi < 0.5:
        return FAIL
    return NULL


RULES = {
    "residual_le_tol": _rule_residual_le_tol,
    "enclosure_pos": _rule_enclosure_pos,
    "enclosure_contains": _rule_enclosure_contains,
    "chern_licensed": _rule_chern_licensed,
    "symplectic_physical": _rule_symplectic_physical,
}


@dataclass
class Certificate:
    """A machine-checkable correctness certificate.

    Parameters
    ----------
    kind : short identifier of the certificate family (e.g. ``"verified_psd"``).
    claim : human-readable statement of what is asserted.
    method : the numerical method / oracle used.
    rule : key into :data:`RULES`; how ``status`` is (re)derived from the numbers.
    status : PASS / FAIL / NULL (set by the emitter, re-derivable via :meth:`recheck`).
    value : optional reference/point value being certified.
    enclosure : optional rigorous interval ``(lo, hi)`` bounding the true quantity.
    residual : optional observed discrepancy vs an oracle.
    tolerance : optional acceptance threshold.
    provenance : dict; library/version/timestamp + emitter-specific params.
    """

    kind: str
    claim: str
    method: str
    rule: str
    status: str = NULL
    value: float | None = None
    enclosure: tuple[float, float] | None = None
    residual: float | None = None
    tolerance: float | None = None
    provenance: dict = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"unknown rule {self.rule!r}; valid: {sorted(RULES)}")
        if self.status not in _VALID:
            raise ValueError(f"invalid status {self.status!r}; valid: {_VALID}")
        if self.value is not None:
            self.value = float(self.value)
        if self.residual is not None:
            self.residual = float(self.residual)
        if self.tolerance is not None:
            self.tolerance = float(self.tolerance)
        if self.enclosure is not None:
            self.enclosure = (float(self.enclosure[0]), float(self.enclosure[1]))
        self.provenance.setdefault("library", "nanarch-certify")
        self.provenance.setdefault("version", "0.1.0")
        self.provenance.setdefault("emitted_utc", datetime.now(timezone.utc).isoformat())

    def recheck(self) -> str:
        """Recompute status from the recorded numbers + rule, independent of the emitter."""
        return RULES[self.rule](self)

    @property
    def width(self) -> float | None:
        return None if self.enclosure is None else (self.enclosure[1] - self.enclosure[0])

    # --- serialisation -------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        if self.enclosure is not None:
            d["enclosure"] = [self.enclosure[0], self.enclosure[1]]
        return d

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, **kw)

    @classmethod
    def from_dict(cls, d: dict) -> "Certificate":
        d = dict(d)
        enc = d.get("enclosure")
        if enc is not None:
            d["enclosure"] = (float(enc[0]), float(enc[1]))
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "Certificate":
        return cls.from_dict(json.loads(s))
