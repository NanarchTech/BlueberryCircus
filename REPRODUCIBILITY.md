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
python3 examples/demo_rectification.py         # -> out/rectification_certificate.json
python3 examples/demo_hydrogen_coulomb.py     # -> out/hydrogen_certificate.json, out/hydrogen_radial.npz
```

Tests set `pythonpath = ["src"]` via `pyproject.toml`; `conftest.py` and the
demos prepend `src/` to `sys.path` so everything runs from a fresh checkout
without installation.

## Verified result (2026-08-13, macOS workstation: CPython 3.14.2)

```
pytest             112 passed, 2 xfailed, 0 skipped    (core, default)
pytest -m rust       4 passed                          (Rust cdylib built)
pytest -m jax        4 passed                          (JAX installed)
pytest -m verify     3 skipped                         (no BLUEBERRY_VERIFY_BIN on this run)
```

- **2 xfailed (strict)** = the convergent 3-D SI hydrogen radial-density match
  and relativistic corrections. They are declared compute milestones, not a
  permanent executable assertion of a literature conclusion.

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
| Puthoff circular `P_abs` vs `P_rad` | **<1×10⁻¹²** | analytic balance |
| Nieuwenhuizen improper quadrature vs `16/(5π√3)` | **9.5×10⁻¹³ absolute** | Eq. (2.30) vs Eq. (2.31) |
| Setterfield `U=1` vs `U=4` mapped trajectory | **<1×10⁻⁹** | static conjugacy |

## Caveats (house honesty policy)

- Pass counts are environment-specific; reproduce on the target workstation
  before citing them.
- `Units.scaled()` is an accelerated oscillator/stress normalization. The
  retained Coulomb run is not evidence for physical-timescale self-ionization.
- The many-mode *time-domain* variance has single-realization scatter of order
  10–15% because modes within a resonance half-width equilibrate on times
  `~1/detuning` that exceed short runs; the convergent value is obtained by
  ensemble averaging over seeds and longer integration. The *linear
  ground state itself* is nonetheless certified exactly via the quadrature oracle
  (C2), which needs no time integration.
