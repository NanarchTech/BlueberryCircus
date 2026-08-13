"""Canonical byte serialization for the cross-language certificate envelope.

This is the **keystone** of the polyglot ledger: a Python value and the
byte-equal Rust ``serde_json::Value`` must serialize to *exactly the same bytes*
so a SHA-256 over those bytes (the envelope ``body_hash`` / ``chain_hash``)
verifies across both languages. The hash chain is only cross-language if both
sides produce the same bytes for the same logical body.

The rule -- ``nanarch-canonical-json/v1``:

* **UTF-8**, no insignificant whitespace (``,`` and ``:`` separators, no spaces).
* **Object keys sorted** by Unicode code point (Python ``sorted`` on ``str``,
  Rust ``BTreeMap`` / sorted keys -- both order by code point).
* **Strings** JSON-escaped with the minimal escape set (``"`` ``\\`` and the
  C0 controls ``\\b \\t \\n \\f \\r`` plus ``\\uXXXX`` for the rest of U+0000..
  U+001F). Non-ASCII printable characters are emitted **verbatim as UTF-8**
  (not ``\\u`` escaped), so both languages agree without depending on a JSON
  library's escaping policy.
* **Booleans / null** -> ``true`` / ``false`` / ``null``.
* **Integers** -> decimal, no decimal point, no exponent.
* **Floats** -> the exact IEEE-754 hex form emitted **verbatim as a bare token**
  (``float.hex()`` in Python; the bit-identical formatting in Rust). e.g.
  ``1.0`` -> ``0x1.0000000000000p+0``, ``-0.0`` -> ``-0x0.0000000000000p+0``-ish
  (see below), ``1e-30`` -> ``0x1.4484bfeebc2a0p-100``.

**Documented deviation from RFC 8785 (JCS).** JCS specifies the shortest
round-tripping *decimal* (ECMAScript ``Number.prototype.toString``) for the
number production. That decimal algorithm (Ryū / Grisu) is genuinely fragile to
reproduce identically across a Python and a Rust implementation, and a
*one-ULP* disagreement silently breaks the hash chain. We deliberately trade
JCS-compatibility for an **honest, trivially-reproducible** rule: the exact
hex significand. ``float.hex()`` is a total, lossless, bijective function of the
64 bits with a single canonical spelling, so Python and a from-bits Rust port
agree byte-for-byte by construction. The canonical bytes are a *hash input*,
not a storage format -- they are not required to be re-parseable JSON. The
deviation is stated prominently in ``ENVELOPE.md``.

**Float hex spelling (must match Python ``float.hex()`` exactly).** For a finite
``f64`` with sign ``s``, biased exponent field ``e`` (11 bits) and 52-bit
fraction ``f``:

* **zero** (``e == 0 and f == 0``): ``[-]0x0.0p+0`` (mantissa collapses to a
  single ``0``; the unique special case).
* **subnormal** (``e == 0 and f != 0``): ``[-]0x0.{f:013x}p-1022``.
* **normal** (``e != 0``): ``[-]0x1.{f:013x}p{±}{e-1023}``.

The fraction is **always 13 hex digits** for non-zero values (52 bits = 13
nibbles, trailing zeros kept); the exponent sign is always present. Non-finite
floats (``NaN`` / ``±Inf``) are **rejected** -- a certificate hash surface must
never carry a silent sentinel (the no-silent-NULL discipline).

Stdlib-only by charter (``nanarch-certify`` is numpy-only; this module imports
neither numpy nor any third party).
"""
from __future__ import annotations

import math
import struct

CANONICAL_SCHEMA = "nanarch-canonical-json/v1"

# Minimal JSON string escapes (the two structural + the five named C0 controls).
_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def float_hex_token(x: float) -> str:
    """The exact bare hex token for a finite ``f64`` -- identical to ``float.hex()``.

    Implemented from the raw 64 bits (not by calling ``float.hex()``) so the
    spelling contract is explicit and line-by-line mirrorable in Rust.
    """
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(
            f"non-finite float {x!r} cannot appear in a canonical hash surface"
        )
    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    sign = bits >> 63
    exp = (bits >> 52) & 0x7FF
    frac = bits & ((1 << 52) - 1)
    neg = "-" if sign else ""
    if exp == 0 and frac == 0:
        return f"{neg}0x0.0p+0"
    if exp == 0:  # subnormal
        unbiased = -1022
        lead = "0"
    else:  # normal
        unbiased = exp - 1023
        lead = "1"
    esign = "+" if unbiased >= 0 else "-"
    return f"{neg}0x{lead}.{frac:013x}p{esign}{abs(unbiased)}"


def _encode_str(s: str) -> str:
    out = ['"']
    for ch in s:
        esc = _ESCAPES.get(ch)
        if esc is not None:
            out.append(esc)
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)  # printable (ASCII or non-ASCII) emitted verbatim
    out.append('"')
    return "".join(out)


def _encode(value) -> str:
    # bool must be checked before int (bool is an int subclass).
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return _encode_str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return float_hex_token(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_encode(v) for v in value) + "]"
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: kv[0])
        parts = []
        for k, v in items:
            if not isinstance(k, str):
                raise TypeError(f"canonical object keys must be str, got {type(k)}")
            parts.append(_encode_str(k) + ":" + _encode(v))
        return "{" + ",".join(parts) + "}"
    raise TypeError(f"cannot canonicalize value of type {type(value)}")


def canonical_bytes(value) -> bytes:
    """Serialize ``value`` to its canonical UTF-8 bytes (``nanarch-canonical-json/v1``)."""
    return _encode(value).encode("utf-8")


def canonical_str(value) -> str:
    """The canonical serialization as a ``str`` (UTF-8 text)."""
    return _encode(value)


def sha256_hex(data: bytes) -> str:
    """Lowercase hex SHA-256 of raw bytes (stdlib ``hashlib``)."""
    import hashlib

    return hashlib.sha256(data).hexdigest()


def canonical_hash(value) -> str:
    """SHA-256 hex of the canonical bytes of ``value`` -- the hash-surface primitive."""
    return sha256_hex(canonical_bytes(value))
