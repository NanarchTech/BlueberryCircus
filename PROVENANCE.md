# Provenance

Two buckets: Nanarch-owned code, and cited scientific prior art. No third-party
authors' code is copied into this repository; the SED literature below is cited
as analytic prior art, never vendored.

## 1. Nanarch Technologies, Inc. (owner)

- The BlueberryCircus framework design, API, and all `src/blueberry_circus/`
  implementation and tests. © Joe Pecoraro / Nanarch Technologies, Inc.
  License: **Apache-2.0** (see `LICENSE`). Nanarch trademarks and logos are not
  licensed.
- `src/blueberry_circus/_vendor/nanarch_certify/` is a **near-verbatim mirror** of
  Nanarch's canonical `nanarch_certify` certificate layer (same owner, same
  Apache-2.0 license), vendored inside the package so `import blueberry_circus`
  works from a fresh environment with no sibling checkout. It is not a
  reimplementation. The recheck rule registry is the same code, which avoids
  second-registry drift. The only edit made for publication is a docstring line
  in `envelope.py` that named an internal path; all executable code is
  unchanged. SHA-256 of the files as shipped is recorded in `docs/STATUS.md`.

## 2. Optional external verifier, referenced but not vendored

- `nanarch-verify` (Nanarch's Rust certificate re-checker, Apache-2.0) can
  re-derive every emitted bundle in a different language and numeric stack. It
  is **not** copied into this repository and has no public pinned release yet;
  `tests/test_recheck_parity.py` (marker `verify`) runs only when
  `BLUEBERRY_VERIFY_BIN` points at a built `verify` binary, and no code from it
  ships in the BlueberryCircus wheel.

## 3. Scientific grounding, cited as prior art, not vendored

These results define the acceptance oracles and the honest limitations. They
are external literature and are not claimed to originate here:

- H. E. Puthoff, *Ground state of hydrogen as a zero-point-fluctuation-determined
  state*, Phys. Rev. D **35**, 3266 (1987). Power-balance equilibrium.
- T. H. Boyer, *Random electrodynamics*, Phys. Rev. D **11**, 790 (1975). The
  exact oscillator ground state ⟨x²⟩ = ħ/(2mω₀), the one certified theorem (O2).
- D. C. Cole & Y. Zou, Phys. Lett. A **317**, 14 (2003). Moving-window
  trajectory simulation; short-time radial density.
- T. M. Nieuwenhuizen & M. T. P. Liska, Found. Phys. **45**, 1190 (2015).
  Published full 3-D long-run self-ionization result; not reproduced here at
  its physical coupling or timescale.
- T. M. Nieuwenhuizen, Entropy **18**, 135 (2016), arXiv:1611.10200.
  Near-ionization drift and the independently integrated `L_c` threshold.
- T. M. Nieuwenhuizen, Front. Phys. **8**, 335 (2020).
  Negative assessment of several renormalized-noise schemes.
- G. Moddel & O. Dmitriyeva, arXiv:0910.5893.
  Equilibrium, detailed-balance, and conservation-law assessment.
- B. Setterfield, *ZPE and Atomic Constants' Behavior*. A speculative scaling
  profile represented as a test hypothesis, not an empirical source of truth.

**Do not conflate** R. W. Boyd (nonlinear optics) with T. H. Boyer (the SED
theorist). Different people, different fields.

`NOTICE` lists these under "Scientific grounding (not vendored; cited as prior
art)". No third-party MIT/BSD code is vendored, so no code-attribution NOTICE
entries are required beyond the (same-owner) vendored `nanarch_certify` mirror.
