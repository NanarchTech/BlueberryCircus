<div align="center">

<img src="docs/assets/blueberry-circus-logo.gif" alt="Blueberry Circus" width="560" />

# 🫐🎪 BlueberryCircus

[![test](https://github.com/NanarchTech/BlueberryCircus/actions/workflows/test.yml/badge.svg)](https://github.com/NanarchTech/BlueberryCircus/actions/workflows/test.yml)
![status](https://img.shields.io/badge/status-simulation--first%20prototype-1f6feb)
![tests](https://img.shields.io/badge/tests-89%20core%20passed%20·%203%20xfail%20·%200%20skip-2ea043)
![python](https://img.shields.io/badge/python-3.10%2B-1f6feb)
![backends](https://img.shields.io/badge/backends-numpy%20·%20rust%20·%20jax-d29922)
![deps](https://img.shields.io/badge/runtime%20deps-numpy%20only-2ea043)
![license](https://img.shields.io/badge/license-Apache--2.0-8b949e)

</div>

---

BlueberryCircus is a Python library for stochastic electrodynamics (SED), the classical theory in which a charged particle obeys ordinary Maxwell electrodynamics while a random background field drives it.

It is designed to test whether classical stochastic electrodynamics can produce atom-like bound behavior when charged-particle dynamics interact with a randomly fluctuating electromagnetic field.

In the analytically tractable harmonic-oscillator case: the implementation reproduces the quantum ground-state variance to approximately 0.05% under its stated assumptions and numerical checks.

However, in the nonlinear Coulomb model of hydrogen: long-time trajectories eventually exhibit electron escape and self-ionization rather than a stable ground state.

## Features

- RK4 integration of the SED equation of motion
- Harmonic and Coulomb binding potentials
- Random-phase plane-wave ZPF backgrounds, 3-D isotropic or 1-D
- Landau–Lifshitz radiation reaction
- numpy, Rust, and JAX backends
- Analytic oracles: Bohr radius, ground-state energy, angular momentum, oscillator variance
- Phase-space covariance with symplectic readout
- Seeded ensemble runs for power balance
- Cole–Zou moving spectral window
- Ionization-time detection
- Re-checkable certificates on every reported number
- SI and scaled unit systems

*Results are accompanied by a re-checkable certificate recording its value, reference rule, tolerance, provenance, and verdict.*

## Install

Not yet on PyPI, so install from a checkout. Distribution name `blueberry-circus`, import name `blueberry_circus`. Python 3.10+.

```bash
# uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"       # extras:  ".[jax]"  ·  ".[all]"

# or plain pip
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

The Rust backend is an optional native build (`sh scripts/build_rust.sh`). The default test run covers the core suite only; the Rust, JAX, and verifier tests are opt-in markers (`pytest -m rust`, `-m jax`, `-m verify`).

## Quick example

```python
import blueberry_circus as bc

# A charged particle on a spring, driven by a random background field.
U = bc.Units.scaled(gamma_over_omega0=0.05, omega0=1.0)
prog = bc.Program(n_particles=1, units=U)
with prog.context as q:
    bc.Harmonic(omega0=1.0)                                       | q[0]
    bc.ZPF(band=(0.3, 3.0), n_modes=400, mode="one_dimensional")  | q[0]
    bc.RadiationReaction("landau_lifshitz")                       | q[0]

result = bc.Engine(dt=0.02, t_max=600).run(prog, x0=[0,0,0], v0=[0,0,0])
print(result.summary())          # trajectory + means/covariance + certificates
```

```python
# Check the engine against the standard ground-state numbers.
from blueberry_circus import oracles as o
print(o.bohr_radius(bc.SI))                  # 5.2917721e-11 m   (rel-err 1.2e-9)
print(o.hydrogen_ground_state_energy(bc.SI) / bc.E_CHARGE)  # -13.605693 (eV)
```

## Results

| Result | Number | vs reference | rule |
|---|---|---|---|
| **Particle on a spring** $\langle x^2\rangle = \hbar/2m\omega_0$ | rel-err **4.99×10⁻⁴** | analytic, real SI electron | `residual_le_tol` |
| **Full vacuum covariance** † | residual **3.8×10⁻⁶** | quantum vacuum $\tfrac12 I$ | `residual_le_tol` |
| **Integrator fidelity** | rel-err **2.2×10⁻³** | analytic transfer function | `residual_le_tol` |
| **Bohr radius** $a_0$ (closed form in SI) | rel-err **1.2×10⁻⁹** | CODATA-2018 | `residual_le_tol` |
| Ground-state energy $E_1$ | **−13.605693 eV** | −13.605693 eV | — |
| Ground-state angular momentum | $L/\hbar = $ **1** (algebraic identity, not a measurement) | Bohr / Puthoff | — |
| **Orbit conservation** (no radiation) | $\Delta E/E \sim 10^{-14}$ | exact | `residual_le_tol` |
| **numpy ↔ Rust agreement** | spring case **bit-identical**, orbit 7×10⁻¹⁴ | — | — |
| **Independent recheck** | Rust re-derives the verdicts | rejects tampered bundles | separate stack |
| **Escape time** $t_{\rm ion}$ | finite (no bound orbit → `NULL`) | N–L 2015 | report / NULL-first |

Only the particle-on-a-spring result is a theorem (Boyer 1975). A certificate says something about *recorded numbers under a recorded rule*. It does not claim physical truth, and it does not claim SED is the correct theory of the atom.

> † The vacuum-covariance certificate pins the whole state rather than one number, so it rejects squeezed and thermal states that a position-only check accepts. It tracks the same physics as the spring result, so it is not an independent measurement. Details and the ultraviolet caveat are in [`docs/theory.md`](docs/theory.md#6-the-vacuum-covariance-certificate).

Bundles can also be re-derived by `nanarch-verify`, a separate Rust implementation of the same checking rules. A passing bundle re-derives as passing; a tampered bundle, where the stored verdict disagrees with the stored numbers, is rejected. The verifier does not ship in this repository and has no public pinned release yet. When a binary is available, point `BLUEBERRY_VERIFY_BIN` at it and run `pytest -m verify`. Without it, every certificate still re-derives its own verdict in-process via `Certificate.recheck()`, and the demos run `audit_overclaim` over the bundle they emit.

## Architecture

```
Operation (| apply)  ─▶  Program.compile()  ─▶  Backend  ─▶  Result
  Harmonic·Coulomb        named, fail-closed     numpy│rust│jax    trajectory
  ZPF·RadiationReaction        passes            (agree to          + means/cov
                                                  tolerance)         + certificates
```

- **`numpy`.** The reference implementation, and the trust root.
- **`rust`.** A dependency-free C-ABI `cdylib` (no PyO3, no crates.io) called through `ctypes`, carrying the integrator's inner loop. Output for the spring case is bit-identical to numpy.
- **`jax`.** A `jit`/`scan` integrator, `vmap`-ready for batched runs on CPU or GPU.
- **Certificates.** Emitted through the canonical `nanarch_certify` envelope, hash-chained and re-checkable across languages. Tamper with a number and the verdict flips.

## Validation ladder (O0 → O5)

| | check | passes if | tier |
|---|---|---|---|
| **O0** | field statistics | discrete → continuum, rel-err < 5% | A |
| **O1** | Puthoff power balance | reproduces $a_0$, −13.6 eV, $L=\hbar$ | A |
| **O2** | **spring variance, the gate** | $\langle x^2\rangle=\hbar/2m\omega_0$ to ~1% | A |
| **O3** | hydrogen radial density | → $4r^2e^{-2r}$ (CPU-day ensembles) | B · `xfail` |
| **O4** | phase-space conjecture | N–L §3 (NULL where the dynamics don't reach) | B |
| **O5** | **escape time, the headline** | reports a finite $t_{\rm ion}$ | report / NULL-first |

O2 is the gate: if the simulated field doesn't give $\hbar/2m\omega_0$ for the spring, every hydrogen number downstream is meaningless. O5 is the headline: the atom falls apart, and the library says so.

## Limitations

- **There is no stable hydrogen ground state here.** At long times the electron escapes (Nieuwenhuizen–Liska 2015). Matching the quantum 1s density is a CPU-day frontier, marked strict-`xfail` rather than faked.
- **The orbit is chaotic.** The code is deterministic and byte-reproducible on a fixed machine, but quantities like `r_max` depend on floating-point summation order and shift across machines. The certified quantities, meaning the conservation laws and the tolerance-gated checks, are stable. Raw chaotic outputs are not, and shouldn't be quoted as if they were.
- Non-relativistic, with dipole and point-charge approximations and a finite, band-limited background field. Each run is faithful only out to bounded times.

## Provenance & citing

Framework and code © Joe Pecoraro / Nanarch Technologies, Inc., Apache-2.0. The physics below is cited prior art, not vendored, and is not claimed to originate here:

- H. E. Puthoff, *Ground state of hydrogen as a zero-point-fluctuation-determined state*, **Phys. Rev. D 35, 3266 (1987)**.
- T. H. Boyer, *Random electrodynamics*, **Phys. Rev. D 11, 790 (1975)**.
- D. C. Cole & Y. Zou, **Phys. Lett. A 317, 14 (2003)**.
- T. M. Nieuwenhuizen & M. T. P. Liska, **Found. Phys. 45, 1190 (2015)**.

The certificate layer is vendored inside the package (`blueberry_circus/_vendor/nanarch_certify`, a near-verbatim mirror of Nanarch's canonical copy), so `import blueberry_circus` works from a fresh checkout. See [`PROVENANCE.md`](PROVENANCE.md) and [`NOTICE`](NOTICE).

## Run it

```bash
pytest                                     # core suite: 89 passed, 3 xfailed, 0 skipped
sh scripts/build_rust.sh && pytest -m rust # optional: Rust backend cross-language tests
pip install ".[jax]" && pytest -m jax      # optional: JAX backend tests
pytest -m verify                           # optional: needs BLUEBERRY_VERIFY_BIN
python examples/demo_sho_ground_state.py   # certified spring ground state
python examples/demo_vacuum_covariance.py  # full vacuum covariance certificate
python examples/demo_hydrogen_coulomb.py   # orbit · radiative collapse · escape
```

<div align="center">

**[docs/STATUS.md](docs/STATUS.md)** · **[docs/theory.md](docs/theory.md)** · **[docs/comparison.md](docs/comparison.md)**

*Nanarch Technologies — Photonic & Quantum Intelligence Systems*

</div>
