# BlueberryCircus verification status

v0.3.0. Verified on a macOS workstation (CPython 3.14.2; exact dependency and
tool versions are printed by the release gate) on 2026-08-13. Pass counts are
environment-specific; reproduce locally before citing them.

## Suite

| Command | Result | Requires |
|---|---|---|
| `pytest` (default = core) | **134 passed · 2 xfailed (strict) · 0 skipped** | numpy only |
| `pytest -m rust` | 4 passed | `sh scripts/build_rust.sh` |
| `pytest -m jax` | 4 passed | `pip install ".[jax]"` |
| `pytest -m verify` | 3 passed with a verifier binary; skips without | `BLUEBERRY_VERIFY_BIN` |

The core suite runs green in a clean environment with only `src/` on the path,
because `import blueberry_circus` resolves the certificate layer from the
shipped `blueberry_circus/_vendor/`. `pip install .` works too: the wheel
bundles the vendored certify layer, checked by installing into a fresh venv and
importing and running from a neutral directory.

The 2 xfails are strict (`xfail_strict`), and cover the convergent 3-D hydrogen
radial-density match and relativistic corrections. The permanent “stable
hydrogen is unreachable” xfail was removed: a literature conclusion is not a
locally executed acceptance test.

## Headline verified numbers

**O1 (Puthoff 1987 circular approximation):** the implementation evaluates
`P_abs` and `P_rad` separately and gates their relative residual below `1e-12`.
Equality yields `mω₀r₀²=ħ`. Bohr radius and energy are still checked against
CODATA, but neither is described as a nonlinear stability measurement.

**O2 (Boyer 1975), the gate:** the SED ground-state integral reproduces
⟨x²⟩ = ħ/(2mω₀) with real SI electron constants to rel-err **4.99e-4**; the
time-domain integrator reproduces the analytic single-mode variance to
**2.2e-3**.

**O5 (Nieuwenhuizen 2016):** a 96×96 transformed Gauss–Legendre evaluation of
the nested improper integral gives `L_c = 0.5880841551156304`, versus
`16/(5π√3) = 0.5880841551165783` (absolute residual `9.5e-13`). The per-orbit
near-ionization drift is positive below `L_c`, negative above it, and the
critical perihelion is `0.172921... a₀`.

**Setterfield static hypothesis:** at `U=1` and `U=4`, mapped-band trajectories
with identical phases satisfy `x_U(Ut)=x_1(t)`, `Uv_U(Ut)=v_1(t)`, equal
mechanical energy, and equal `L/ħ` below `1e-9`. This certifies static dynamical
inertness, not the empirical scaling proposal.

**Legacy stress fixture:** its dimensionless damping is `13,367.7×` the
physical Bohr value and sustained positive energy begins at `0.06956` orbit.
It is retained only as an accelerated numerical stress test.

## Hypothesis tournament (`tournament.py`)

The baseline report evaluates Nieuwenhuizen's complete finite-energy
point-charge surface over 18 preregistered `(E,L)` cells. It recovers the
near-ionization sign change, agrees with the PR1 asymptote within one percent at
the independently evaluated `E=-0.001` probe, and recovers the exact circular
endpoint. The default stochastic
gate executes 32 stored seeds at 2,048 and 4,096 modes, halving the nominal
timestep and returning `NULL` when normalized drift changes by ten percent or
more.

All four zero-parameter arms recover the identical baseline cell. The driven
Setterfield Hamiltonian records `partial H/partial t` as external work; the
finite shell applies one reciprocal form factor to both powers; the
inverse-square arm integrates the printed source kernel; and the
Rodríguez-inspired multipole surrogate conserves its closed total Hamiltonian.
None can emit `STABLE_GROUND_STATE`.

The inverse-square audit is a new negative reproducibility result. Orders 48
and 64 give `H_max≈7.327` near `mu≈0.590`, hence the defining
`d_c=-H_max²≈-53.69`. The paper's prose `-35.8` squares only its quoted
`H(0)=5.99`, despite stating an all-`mu` criterion. Reports retain both values.

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
3. **O3 convergence.** The CPU-day physical-coupling hydrogen radial-density
   ensembles (strict xfail until genuinely reproduced).

## Vendored mirror provenance

`src/blueberry_circus/_vendor/nanarch_certify/` ships in the wheel. The rule
registry and all executable code are a verbatim copy of the canonical
`nanarch_certify`; the only edit for publication is a docstring line in
`envelope.py` that named an internal path. SHA-256 of the files as shipped here:
- `certificate.py` `fe9b065a6a07d7d28cdf8bc4e0bedc9e3009e7d1441b173be1b7fd54f844dcb9`
- `canonical.py`   `37ca1fffb65a9c9aa2288873426da5909503707d5e09fe84567de5c7f54262c2`
- `envelope.py`    `c491ed2e143c5ed8d525f0082f9c65759601afb1628c5aaec60c8c08d4227a49`
