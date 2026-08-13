# BlueberryCircus verification status

v0.1.0. Verified on a macOS workstation (CPython 3.14.2, numpy 2.3.5,
cargo/rustc 1.91, JAX 0.10.2) on 2026-08-12. Pass counts are
environment-specific; reproduce locally before citing them.

## Suite

| Command | Result | Requires |
|---|---|---|
| `pytest` (default = core) | **89 passed · 3 xfailed (strict) · 0 skipped** | numpy only |
| `pytest -m rust` | 4 passed | `sh scripts/build_rust.sh` |
| `pytest -m jax` | 4 passed | `pip install ".[jax]"` |
| `pytest -m verify` | 3 passed with a verifier binary; skips without | `BLUEBERRY_VERIFY_BIN` |

The core suite runs green in a clean environment with only `src/` on the path,
because `import blueberry_circus` resolves the certificate layer from the
shipped `blueberry_circus/_vendor/`. `pip install .` works too: the wheel
bundles the vendored certify layer, checked by installing into a fresh venv and
importing and running from a neutral directory.

The 3 xfails are strict (`xfail_strict`), and they cover the convergent 3-D
hydrogen radial-density match to the QM 1s state, relativistic corrections, and
stable hydrogen, which by the literature's own verdict never passes. None of
them is silenced work.

## Headline verified numbers

**O1 (Puthoff 1987) measured vs CODATA:** Bohr radius rel-err **1.2e-9**, ground
state **−13.605693 eV** (rel-err 1.0e-8), orbital frequency = Hartree/ħ
(`tests/test_puthoff.py`). These are closed forms in SI evaluated against CODATA,
i.e. the static target the SED power balance lands on. They are not dynamical
measurements, and the dynamical run self-ionizes (O5). The companion
`L/ħ = 1` check is an algebraic identity (`L = m·v(a₀)·a₀` reduces to ħ exactly),
so it locks unit bookkeeping rather than predicting anything.

**O2 (Boyer 1975), the gate:** the SED ground-state integral reproduces
⟨x²⟩ = ħ/(2mω₀) with real SI electron constants to rel-err **4.99e-4**; the
time-domain integrator reproduces the analytic single-mode variance to
**2.2e-3**.

**Cross-language enclosure:** the Rust `cdylib` backend is **bit-identical** to
numpy on the SHO case (max|Δx| = 0.0) and agrees to ~7e-14 on the Kepler orbit;
the JAX backend agrees to ≤1e-6.

## Vacuum-covariance certificate (`symplectic.py`)

The band-limited SED ground-state covariance equals the quantum vacuum ½·I to
residual **3.8e-6**, certified on an equality-to-tolerance rule
(`residual_le_tol`).

What it adds, stated narrowly: numerically it tracks O2, since ⟨v²⟩ = ω₀²⟨x²⟩ is
kinematically forced and ⟨xv⟩ = 0 follows from stationarity, so there is no
*independent* normalization test here. The gain is structural. Pinning the full
covariance rejects off-vacuum states, squeezed or thermal, that a position-only
check accepts.

The single-mode symplectic eigenvalue is the closed form ν = √det σ, so no
eigensolver and no instability at the boundary. Physicality (ν ≥ ½) rides along
as a companion check. It returns `NULL` at the vacuum edge, and it does not
detect ionization: an escaping orbit has huge variance, hence ν ≫ ½. The vacuum
certificate is the departure-from-vacuum detector.

Two more caveats. ⟨x²⟩ is UV-benign but ⟨v²⟩ diverges quadratically in the
cutoff, so the claim lives on a finite band below 1/τ, disclosed on the
certificate. And a diverged or ionized covariance emits a serializable FAIL or
NULL, using a finite sentinel, so inf/NaN never reach the canonical hash.

## Cross-language recheck (`tests/test_recheck_parity.py`)

Emitted bundles can be re-derived by `nanarch-verify`, a separate Rust
implementation of the checking rules. Verified live on the workstation: a
genuine bundle re-derives `PASS` (exit 0); a tampered bundle (`stored=PASS`,
numbers say `FAIL`) is rejected (exit 1). The verifier is an optional external
binary with no public pinned release yet, so the tests sit behind the `verify`
marker and skip without `BLUEBERRY_VERIFY_BIN`. Note the scope: this recheck
re-derives each verdict from the *recorded* residual and rule; the stronger
bit-level audit that recomputes residuals from the covariance is deferred.

## Deferred

1. **Bit-rigorous audit arms.** In the external verifier (recomputing the
   vacuum-covariance residual / symplectic eigenvalue in outward-rounded
   interval arithmetic rather than re-deriving verdicts from recorded numbers).
2. **PyPI publication.** This release is a GitHub source release only.
3. **O3 convergence.** The CPU-day hydrogen radial-density ensembles (strict
   xfail until genuinely reproduced).

## Vendored mirror provenance

`src/blueberry_circus/_vendor/nanarch_certify/` ships in the wheel. The rule
registry and all executable code are a verbatim copy of the canonical
`nanarch_certify`; the only edit for publication is a docstring line in
`envelope.py` that named an internal path. SHA-256 of the files as shipped here:
- `certificate.py` `fe9b065a6a07d7d28cdf8bc4e0bedc9e3009e7d1441b173be1b7fd54f844dcb9`
- `canonical.py`   `37ca1fffb65a9c9aa2288873426da5909503707d5e09fe84567de5c7f54262c2`
- `envelope.py`    `c491ed2e143c5ed8d525f0082f9c65759601afb1628c5aaec60c8c08d4227a49`
