"""Vendored subset of ``nanarch-certify`` (the Nanarch assurance layer).

This is a **verbatim mirror** of the canonical ``nanarch_certify`` package --
only the three modules BlueberryCircus consumes (``certificate``, ``canonical``,
``envelope``) are vendored, so the repo is self-contained and CI-green with no
sibling checkout on the path. It is NOT a reimplementation: the rule registry here is the
same code, so there is no second-registry drift.

If a canonical ``nanarch_certify`` is installed, it is imported directly and
this vendored copy is unused (the resolver in
``blueberry_circus.certify`` already prefers an installed/sibling import). The
provenance + re-sync policy is recorded in ``PROVENANCE.md``.
"""
from .certificate import Certificate, RULES, PASS, FAIL, NULL
from .canonical import (canonical_hash, canonical_bytes, canonical_str,
                        float_hex_token, sha256_hex)
from .envelope import (to_envelope, build_chain, chain_hash, verify_chain,
                       verify_envelope, recheck_envelope, GENESIS)

__version__ = "0.1.0-vendored"
__all__ = [
    "Certificate", "RULES", "PASS", "FAIL", "NULL",
    "canonical_hash", "canonical_bytes", "canonical_str", "float_hex_token",
    "sha256_hex", "to_envelope", "build_chain", "chain_hash", "verify_chain",
    "verify_envelope", "recheck_envelope", "GENESIS",
]
