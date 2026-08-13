# Physics claim ledger

## Objective and scope

BlueberryCircus is a simulation-repo that retires normalization and accounting
risks in stochastic-electrodynamics calculations. Version 0.2.0 narrowed the
Coulomb claims to three reproducible objects: a circular-orbit power balance, a
near-ionization drift threshold, and a static co-scaling conjugacy. It does not
validate SED as a theory of hydrogen. Version 0.3.0 adds a perturbative,
energy-audited hypothesis tournament without enlarging that physical claim.

The Nanarch technical orientation supplies the systems discipline rather than a
hydrogen equation: page 4, Eq. (1) requires an explicit state space, governing
operator, boundary conditions, constitutive map, measurement layer, and control
policy; page 18, Eqs. (94)--(96) requires proposed models to pass independent
physics verifiers; page 18 §13 separates speculative platforms from measurable
near-term milestones.

## Claim ledger

| Claim | Source and exact scope | Local artifact | Validation command | Status |
|---|---|---|---|---|
| Bohr normalization has `m=hbar=e=a0=omega_B=1`, `c=1/alpha`, and `tau=(2/3)alpha^3=beta^2` | Atomic-unit definitions; dimensional derivation in [`theory.md`](theory.md#53-physical-bohr-normalization-and-the-retained-stress-fixture) | `Units.bohr()`, exported `BOHR` | `pytest tests/test_constants.py` | focused gate passed |
| `P_abs=P_rad` implies `m omega0 r0^2=hbar` for the circular harmonic approximation | [Puthoff, Phys. Rev. D 35, 3266 (1987)](https://doi.org/10.1103/PhysRevD.35.3266); explicitly not nonlinear stability | `oracles.puthoff_power_balance()` | `pytest tests/test_puthoff.py` | focused gate passed |
| `Lc=f(0)=16/(5 pi sqrt(3))` and `Delta<E>=3 pi beta^2(Lc-L)/L^6` near ionization | [Nieuwenhuizen 2016](https://arxiv.org/abs/1611.10200), Eqs. (2.30)--(2.37) | `rectification.py` | `pytest tests/test_rectification.py` | focused gate passed |
| Critical near-parabolic perihelion is `Lc^2/2 = 0.172921... a0` | Nieuwenhuizen 2016, text following Eq. (2.37) | `critical_angular_momentum()` consumer | `pytest tests/test_rectification.py` | focused gate passed |
| Setterfield's static profile preserves `alpha`, `a0`, `E_h`, `beta`, and `tau omega_B` and is a time conjugacy | [Setterfield, §3](https://www.barrysetterfield.org/behaviorzpe3.html); represented as a speculative hypothesis only | `setterfield_rescale()` and mapped ZPF trajectory test | `pytest tests/test_setterfield.py` | focused gate passed |
| The v0.1.0 escape fixture is accelerated by about 13,000 and unbinds in under 0.1 orbit | Direct audit of tagged fixture `e83470b`; no physical-timescale inference | `demo_hydrogen_coulomb.py`, `test_watchdog.py` | `pytest tests/test_watchdog.py` | focused gate passed |
| Full point-charge `D(E,L)` recovers both `f(1)=1/2` and the PR1 near-ionization asymptote | Nieuwenhuizen 2016, Eqs. (2.23)--(2.35); conditional perturbative drift only | `tournament.nieuwenhuizen_gain_function()` and surface report | `pytest tests/test_tournament.py` | focused gate passed |
| All energy reports satisfy the seven-channel ledger identity | Explicit sign convention in [`tournament.md`](tournament.md#energy-identity) | `EnergyLedger` | `pytest tests/test_tournament.py` | deterministic `<1e-6`; stochastic `<1%` |
| Dynamic Setterfield suppression with parameter work is active control | Canonical `H=p²/(2U²)-1/r`; hypothesis, not empirical cosmology | `SetterfieldDrive` | `pytest tests/test_tournament.py` | canonical ledger passed |
| A reciprocal shell applies `sin(kR)/(kR)` identically to absorption and radiation | Spherical-shell Fourier form factor; response hypothesis | `FiniteShellResponse` | `pytest tests/test_tournament.py` | `R=0` exact recovery passed |
| Printed inverse-square kernel has `H_max≈7.327`, `d_c≈-53.69`; paper prose says `-35.8` from `H(0)` | Nieuwenhuizen 2016, Eqs. (3.19)--(3.27); source-level discrepancy, not a physical force claim | `InverseSquareControl.critical_d()` | `pytest tests/test_tournament.py` | orders 48/64 agree near `1e-3` |
| Closed Rodríguez-inspired multipole surrogate conserves particle-plus-mode energy | [Rodríguez 2014](https://arxiv.org/abs/1201.6168) as motivation only; local Hamiltonian is an explicit surrogate | `MultipoleStorage` | `pytest tests/test_tournament.py` | deterministic gate passed |
| 32 stored seeds reproduce 2,048→4,096-mode confidence intervals and convergence decisions | Finite-mode random-phase quadrature of perturbative response, not a nonlinear trajectory | `stochastic_cell()` | `pytest tests/test_tournament.py` | focused gate passed |

## Assumptions, limitations, and rejected claims

- Puthoff's calculation is retained only at circular-orbit harmonic order.
- Nieuwenhuizen's drift is a disorder- and period-averaged, near-zero-energy
  asymptote. It is not a stationary distribution or an equilibrium work cycle.
- Setterfield scaling is not treated as measured cosmology. Static co-scaling
  is dynamically inert; a driven `U(t)` requires an external-work ledger.
- The accelerated fixture is rejected as evidence for physical hydrogen
  self-ionization. Published long-run negative results remain literature
  context rather than locally reproduced timescale claims.
- Broken symmetry alone does not demonstrate net work or evade detailed
  balance. No certificate has that interpretation.
- `CHANNEL_SUPPRESSED` names one preregistered drift channel. It is not a stable
  ground state, and `STABLE_GROUND_STATE` is not an allowed result token.
- The finite-mode layer validates a response-kernel quadrature. It does not
  reproduce the physical timescale of the published nonlinear simulations.
- The calculated inverse-square maximum contradicts the paper's prose
  threshold. Both values are reported; neither is silently privileged as an
  empirical result.

## Executable deliverables and acceptance thresholds

```bash
python -m pytest
python examples/demo_rectification.py
python examples/demo_hydrogen_coulomb.py
python examples/demo_hypothesis_tournament.py
blueberry-tournament --profile preregistered --arm all # manifest, no run
```

The analytic gates are: Bohr identities `<1e-12` relative error, circular power
balance `<1e-12`, improper quadrature `<1e-8`, and static trajectory conjugacy
`<1e-9`. Generated certificate bundles are written under `examples/out/` and
are intentionally excluded from version control because the commands reproduce
them deterministically.

## Nanarch research relevance

The near-term risk retired is methodological: speculative vacuum-field ideas
are converted into dimensionless invariants, conditional drift surfaces, and
auditable energy/accounting tests before compute or hardware expenditure. A
bounded Phase I milestone is a reproducible, independently checkable hypothesis
tournament with null recovery and conservation gates. A Phase II path would add
long-duration compute, calibrated noise spectra, and hardware-facing emulators;
none of those measurements is claimed in this release.
