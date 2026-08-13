# BlueberryCircus reproducibility notes

## Environment

- Python ≥ 3.10 (authored under CPython 3.x).
- Single runtime dependency: `numpy ≥ 1.23`. Test/dev: `pytest ≥ 7`. Optional
  `plot` extra: `matplotlib`.
- No network, no GPU. The integrator (RK4) and all linear algebra are dense and
  deterministic. The only randomness is the ZPF phase/direction draw, which is
  seeded via `numpy.random.default_rng(seed)`; identical seeds give
  byte-identical fields (`test_zpf.py::test_seed_determinism`).

## Commands

```bash
python3 -m pytest                             # core suite (rust/jax/verify are opt-in markers)
python3 -m pytest -m rust                     # after sh scripts/build_rust.sh
python3 -m pytest -m jax                      # after pip install ".[jax]"
python3 examples/demo_sho_ground_state.py     # -> out/sho_ground_state_certificate.json
python3 examples/demo_vacuum_covariance.py    # -> out/vacuum_covariance_certificate.json
python3 examples/demo_hydrogen_coulomb.py     # -> out/hydrogen_certificate.json, out/hydrogen_radial.npz
```

Tests set `pythonpath = ["src"]` via `pyproject.toml`; `conftest.py` and the
demos prepend `src/` to `sys.path` so everything runs from a fresh checkout
without installation.

## Verified result (2026-08-12, macOS workstation: CPython 3.14.2, numpy 2.3.5, rustc 1.91, JAX 0.10.2)

```
pytest              89 passed, 3 xfailed, 0 skipped    (core, default)
pytest -m rust       4 passed                          (Rust cdylib built)
pytest -m jax        4 passed                          (JAX installed)
pytest -m verify     3 skipped                         (no BLUEBERRY_VERIFY_BIN on this run)
```

- **3 xfailed (strict)** = the convergent 3-D SI hydrogen radial-density match to
  the QM 1s state, relativistic corrections, and the (never-passing) stable
  hydrogen ground state. These are declared compute milestones or honest
  negative results, not latent failures. `strict=True` means the suite would
  *fail* if any silently started passing without being promoted.

### Headline verified numbers

| Quantity | Value | Oracle |
|---|---|---|
| SED ground state `∫S_Ex|H_AL|²` vs `ħ/2mω₀` (SI electron) | rel. err **5.0×10⁻⁴** | Boyer 1975 |
| single-mode integrator variance vs `½a²|H|²` | rel. err **2.2×10⁻³** | exact transfer fn |
| LL free-decay energy rate vs `γ=τω₀²` | ratio **1.000** | analytic damping |
| closed SHO energy drift (no RR) | **2.8×10⁻¹⁰** | conservation |
| Kepler energy / angular-momentum drift (no RR) | **2.6×10⁻¹⁴ / 1.3×10⁻¹⁴** (demo dt=0.004; ~1×10⁻¹³ at dt=0.01) | conservation |
| isotropic-ZPF energy density vs band integral | ratio **1.0000** | `∫ρ dω` |
| FDT power balance `⟨P_rad⟩` vs `⟨P_abs⟩` | ≈3% (cited seed); ≤~10% single-realization (gate 20%) | fluctuation–dissipation |

## Caveats (house honesty policy)

- Pass counts are environment-specific; reproduce on the target workstation
  before citing them.
- The many-mode *time-domain* variance has single-realization scatter of order
  10–15% because modes within a resonance half-width equilibrate on times
  `~1/detuning` that exceed short runs; the convergent value is obtained by
  ensemble averaging over seeds and longer integration. The *linear
  ground state itself* is nonetheless certified exactly via the quadrature oracle
  (C2), which needs no time integration.
