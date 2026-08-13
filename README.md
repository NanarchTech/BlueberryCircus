<div align="center">

<img src="docs/assets/blueberry-circus-logo.gif" alt="Blueberry Circus" width="560" />

# 🫐🎪 BlueberryCircus

[![test](https://github.com/NanarchTech/BlueberryCircus/actions/workflows/test.yml/badge.svg)](https://github.com/NanarchTech/BlueberryCircus/actions/workflows/test.yml)
![status](https://img.shields.io/badge/status-simulation--first%20prototype-1f6feb)
![tests](https://img.shields.io/badge/tests-134%20core%20passed%20·%202%20xfail%20·%200%20skip-2ea043)
![python](https://img.shields.io/badge/python-3.10%2B-1f6feb)
![backends](https://img.shields.io/badge/backends-numpy%20·%20rust%20·%20jax-d29922)
![deps](https://img.shields.io/badge/runtime%20deps-numpy%20only-2ea043)
![license](https://img.shields.io/badge/license-Apache--2.0-8b949e)

</div>

---

BlueberryCircus is a Python laboratory for testing stochastic electrodynamics, not a proof that classical vacuum noise reproduces hydrogen. Its linear oscillator remains an exact benchmark, while its Coulomb sector evaluates three narrower claims: Puthoff’s circular-orbit absorption and radiation balance, Nieuwenhuizen’s near-ionization rectification threshold \(L_c=16/(5\pi\sqrt3)\), and Setterfield’s proposed static vacuum co-scaling, which leaves dimensionless hydrogen dynamics unchanged after time rescaling. The v0.1.0 escape run is retained only as an accelerated numerical stress test because its damping ratio is about 13,000 times the physical Bohr-unit value and it unbinds in less than one tenth of an orbit, so it cannot stand as a reproduction of physical hydrogen self-ionization; certificates verify numerical claims under explicit assumptions, not the truth of SED itself.

Version 0.3.0 adds an energy-audited hypothesis tournament. It computes the
complete point-charge `D(E,L)` surface before testing a driven Setterfield map,
a reciprocal finite shell, an inverse-square control, and a conservative
multipole-storage surrogate. Its finite-mode statistics validate the
perturbative response kernel, not a nonlinear long-time trajectory. See the
[tournament specification](docs/tournament.md).

Artifact label: simulation-repo

## Features

- RK4 integration of the SED equation of motion
- Harmonic and Coulomb binding potentials
- Random-phase plane-wave ZPF backgrounds, 3-D isotropic or 1-D
- Landau–Lifshitz radiation reaction
- numpy, Rust, and JAX backends
- Analytic oracles: Bohr radius, ground-state energy, angular momentum, oscillator variance
- Physical Bohr units and an independently integrated near-ionization threshold
- Static Setterfield co-scaling map with a trajectory-conjugacy regression test
- Complete point-charge drift surface with perihelion-adapted quadrature
- Seven-channel energy ledgers and a closed five-state classification vocabulary
- Four preregistered hypothesis arms with exact zero-parameter recovery controls
- Chunkable 32-seed/2,048-mode research command with deterministic JSON/NPZ output
- Phase-space covariance with symplectic readout
- Seeded ensemble runs for power balance
- Cole–Zou moving spectral window
- Generic unbinding-time diagnostics for bounded numerical windows
- Re-checkable certificates on every reported number
- SI and physical Bohr units; accelerated scaled units for oscillator/stress tests

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
| Puthoff circular balance $P_{\rm abs}=P_{\rm rad}$ | residual **<10⁻¹²** | circular harmonic approximation | `residual_le_tol` |
| Ground-state angular momentum | $L/\hbar = $ **1** (algebraic identity, not a measurement) | Bohr / Puthoff | — |
| **Rectification threshold** | $L_c=$ **0.5880841551** | independent improper quadrature vs $16/(5\pi\sqrt3)$ | `residual_le_tol` |
| Critical perihelion | $r_p=$ **0.172921 $a_0$** | $L_c^2/2$ near-ionization asymptote | — |
| Setterfield static co-scaling | trajectory conjugacy **<10⁻⁹** | $x_U(Ut)=x_1(t)$ and mapped velocities | regression |
| Point-charge drift surface | **18 preregistered cells** | full Eq. (2.34), both endpoint limits | regression |
| Inverse-square defining maximum | $H_{\max}=$ **7.327** at $\mu\approx0.590$ | printed kernel gives $d_c\approx-53.69$, not prose −35.8 | quadrature |
| Multipole closed Hamiltonian | relative total-energy error **<10⁻⁶** | no ZPF or damping | conservation |
| **Orbit conservation** (no radiation) | $\Delta E/E \sim 10^{-14}$ | exact | `residual_le_tol` |
| **numpy ↔ Rust agreement** | spring case **bit-identical**, orbit 7×10⁻¹⁴ | — | — |
| **Independent recheck** | Rust re-derives the verdicts | rejects tampered bundles | separate stack |
| Accelerated stress unbinding | **0.0696 orbit** at **13,368×** physical damping | v0.1.0 fixture only | diagnostic, not physical |

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
| **O2** | **spring variance, the gate** | $\langle x^2\rangle=\hbar/2m\omega_0$ to ~1% | A |
| **O3** | hydrogen radial density | → $4r^2e^{-2r}$ (CPU-day ensembles) | B · `xfail` |
| **O4** | phase-space conjecture | N–L §3 (NULL where the dynamics don't reach) | B |
| **O5** | **near-ionization rectification threshold** | quadrature recovers $L_c$ within $10^{-8}$; drift changes sign there | A |

O2 remains the normalization gate: if the simulated field does not give $\hbar/2m\omega_0$ for the spring, every Coulomb result downstream is meaningless. O5 is now the primary reproducible Coulomb result: a conditional near-zero-energy drift asymptote, not a stationary distribution.

## Limitations

- The published long-duration 3-D studies report self-ionization, including later relativistic and renormalized-noise attempts. BlueberryCircus does not claim to reproduce those physical timescales with its accelerated fixture.
- Nieuwenhuizen rectification is a conditional drift in energy space. It does not establish equilibrium vacuum-energy extraction, usable net work, or evasion of detailed balance; broken symmetry alone is insufficient.
- Setterfield co-scaling is represented as a speculative static hypothesis. Its invariants and trajectory conjugacy show that the static profile is dynamically inert after time reparameterization.
- A time-dependent Setterfield profile is an externally driven variable-mass Hamiltonian; any apparent suppression with nonzero parameter work is classified `ACTIVE_CONTROL`.
- The finite-shell and multipole arms are response/surrogate models. They are not validated electron or proton structure models. The stochastic tournament is perturbative and does not replace a physical-timescale nonlinear ensemble.
- Direct quadrature of Nieuwenhuizen's printed inverse-square kernel gives $d_c\approx-53.69$, while the paper's prose quotes −35.8 from its endpoint. Both are retained explicitly; neither establishes a physical inverse-square force.
- Matching the quantum 1s density remains a CPU-day frontier, marked strict-`xfail` rather than faked.
- **The orbit is chaotic.** The code is deterministic and byte-reproducible on a fixed machine, but quantities like `r_max` depend on floating-point summation order and shift across machines. The certified quantities, meaning the conservation laws and the tolerance-gated checks, are stable. Raw chaotic outputs are not, and shouldn't be quoted as if they were.
- Non-relativistic, with dipole and point-charge approximations and a finite, band-limited background field. Each run is faithful only out to bounded times.

## Provenance & citing

Framework and code © Joe Pecoraro / Nanarch Technologies, Inc., Apache-2.0. The physics below is cited prior art, not vendored, and is not claimed to originate here:

- H. E. Puthoff, *Ground state of hydrogen as a zero-point-fluctuation-determined state*, **Phys. Rev. D 35, 3266 (1987)**.
- T. H. Boyer, *Random electrodynamics*, **Phys. Rev. D 11, 790 (1975)**.
- D. C. Cole & Y. Zou, **Phys. Lett. A 317, 14 (2003)**.
- T. M. Nieuwenhuizen & M. T. P. Liska, **Found. Phys. 45, 1190 (2015)**.
- T. M. Nieuwenhuizen, *On the stability of classical orbits of the hydrogen ground state in Stochastic Electrodynamics*, **Entropy 18, 135 (2016)**, [arXiv:1611.10200](https://arxiv.org/abs/1611.10200).
- B. Setterfield, *ZPE and Atomic Constants’ Behavior*, [behaviorzpe3.html](https://www.barrysetterfield.org/behaviorzpe3.html) (speculative scaling proposal, tested here as a hypothesis only).
- T. M. Nieuwenhuizen, *Stochastic Electrodynamics: Renormalized Noise in the Hydrogen Ground-State Problem*, **Front. Phys. 8, 335 (2020)**, [doi:10.3389/fphy.2020.00335](https://doi.org/10.3389/fphy.2020.00335).
- G. Moddel & O. Dmitriyeva, *Extraction of Zero-Point Energy from the Vacuum*, [arXiv:0910.5893](https://arxiv.org/abs/0910.5893) (equilibrium/detailed-balance assessment).
- J. A. E. Rodríguez, extended-charge motivation, [arXiv:1201.6168](https://arxiv.org/abs/1201.6168) (the implemented multipole oscillator is a clearly labeled surrogate).

The certificate layer is vendored inside the package (`blueberry_circus/_vendor/nanarch_certify`, a near-verbatim mirror of Nanarch's canonical copy), so `import blueberry_circus` works from a fresh checkout. See [`PROVENANCE.md`](PROVENANCE.md) and [`NOTICE`](NOTICE).

## Run it

```bash
pytest                                     # core suite: 134 passed, 2 xfailed, 0 skipped
sh scripts/build_rust.sh && pytest -m rust # optional: Rust backend cross-language tests
pip install ".[jax]" && pytest -m jax      # optional: JAX backend tests
pytest -m verify                           # optional: needs BLUEBERRY_VERIFY_BIN
python examples/demo_sho_ground_state.py   # certified spring ground state
python examples/demo_vacuum_covariance.py  # full vacuum covariance certificate
python examples/demo_rectification.py      # certified Puthoff + O5 threshold
python examples/demo_hydrogen_coulomb.py   # accelerated Coulomb stress fixture
python examples/demo_hypothesis_tournament.py # reduced schema/ledger smoke
blueberry-tournament --profile preregistered --arm all # manifest only
```

<div align="center">

**[docs/STATUS.md](docs/STATUS.md)** · **[docs/theory.md](docs/theory.md)** · **[docs/tournament.md](docs/tournament.md)** · **[docs/comparison.md](docs/comparison.md)**

*Nanarch Technologies — Photonic & Quantum Intelligence Systems*

</div>
