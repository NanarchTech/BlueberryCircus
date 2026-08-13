# Energy-audited SED hydrogen hypothesis tournament

## Scope

The tournament asks a deliberately narrower question than “is hydrogen stable
in stochastic electrodynamics?” It asks whether one of four explicitly defined
model changes suppresses the positive, near-ionization energy-drift channel of
the point-charge calculation while closing every energy ledger and surviving
resolution and coupling checks.

No result may be called `STABLE_GROUND_STATE`; that token is absent from the
classification vocabulary. The only possible aggregate results are
`CHANNEL_SUPPRESSED`, `ACTIVE_CONTROL`, `NO_EFFECT`, `DESTABILIZED`, and `NULL`.
Channel suppression is not a proof of a stationary distribution. In particular,
nonlinear rectification means conditional drift in energy/angular-momentum
space. It is not equilibrium vacuum-energy extraction. Broken symmetry does not
by itself establish cyclic net work or evade detailed balance.

`CHANNEL_SUPPRESSED` requires an absolutely negative upper confidence bound in
every low-`L` cell. `ACTIVE_CONTROL` takes precedence when such a negative cell
requires recorded external work. `DESTABILIZED` means a positive shift resolved
above the matched point-charge confidence interval. An arm that neither closes
the target channel nor significantly worsens it is `NO_EFFECT`; a missing
cell, open ledger, or failed resolution/coupling gate is `NULL`.

## Point-charge baseline first

Every report begins with the full point-charge surface on the preregistered
Cartesian grid

```text
E = {-0.05, -0.02, -0.01} Hartree
L/hbar = {0.45, 0.55, Lc-0.01, Lc+0.01, 0.65, 0.8}
```

For `k=sqrt(-2E)`, `kappa=kL`, and `epsilon²=1-kappa²`, the implemented rate is
Nieuwenhuizen's Equation (34),

```text
D(E,L) = beta² k⁸ (2+epsilon²) [k f(kappa)-kappa] / (2 kappa⁶).
```

`f(kappa)` is evaluated from the complete nested improper integral in Equation
(29), not interpolated between endpoints. A tangent map concentrates nodes near
the eccentric perihelion. The exact cancellation through cubic order as
`s -> t` is performed as a Taylor-series division before numerical evaluation;
otherwise direct floating-point subtraction fails for the low-`kappa` cells.
The implementation recovers `f(0)=16/(5 pi sqrt(3))`, `f(1)=1/2`, the circular
Puthoff curve, and the PR1 per-revolution asymptote

```text
Delta<E> = 3 pi beta² [Lc-L] / L⁶.
```

## Energy identity

Every cell stores the seven requested channels using one sign convention:

```text
Delta E_mech = W_ZPF - E_rad - Delta E_Schott
               + W_external + W_internal + residual.
```

`E_rad` is a non-negative loss magnitude. `W_external` and `W_internal` are
signed work on the particle. The Schott term is the final-minus-initial boundary
energy and is zero for the closed orbit-averaged calculation. Deterministic
Hamiltonian fixtures must close below `1e-6` Hartree; stochastic response
ledgers must close within one percent. Classification re-derives closure from
the six physical channels and compares it with the stored residual, so a
tampered residual cannot turn an open ledger into a passing one.

## Four preregistered arms

### SetterfieldDrive

The speculative dynamic profile is

```text
U(t) = exp[A sin(Omega t + phi)],    m(t)=U(t)².
```

The static Setterfield map leaves the Coulomb coefficient invariant, so the
canonical driven Hamiltonian is

```text
H(x,p,t) = p²/[2m(t)] - 1/r,
xdot = p/m(t),                 pdot = -x/r³,
partial H/partial t = -mdot p²/(2m²).
```

The last expression is integrated at the same Runge--Kutta stages as the orbit
and entered as external parameter work. A negative drift obtained with nonzero
parameter work is `ACTIVE_CONTROL`, never passive stabilization. The grid is
`A={0.01,0.05,0.1}`, `Omega/omega_B={0.1,1,10}`, and four phases separated by
`pi/2`.

### FiniteShellResponse

The reciprocal spherical-shell amplitude response

```text
F(kR) = sin(kR)/(kR)
```

is applied identically to absorption and radiation; both powers therefore
receive the same `F²` factor at every response frequency. `R=0` is an exact
point-charge recovery control. The hypothesis radii are
`R/a0={alpha², 1e-3, 1e-2, 0.1, 0.3}`. Applying a form factor to only one side
of the ledger is prohibited because that would manufacture nonreciprocal gain.

### InverseSquareControl

This arm uses

```text
V(r) = -1/r - d/(2r²),
d = {alpha², 0, -10, -35.8, -40}.
```

The `d=0` path calls the finite-energy point-charge implementation exactly.
For nonzero `d`, the near-ionization gain and loss are evaluated from
Nieuwenhuizen's published Equations (54)--(64).

There is a material discrepancy in that paper. Its prose obtains `d_c=-35.8`
by using only the quoted endpoint `H(0)=5.99`, while its criterion says
`H(mu)<sqrt(|d|)` for *all* `0<mu<1`. Direct transformed quadrature of its
printed kernel gives, at orders 48 and 64,

```text
H(0)       = 5.98335...
H_max      = 7.3274... at mu approximately 0.590
d_c=-H_max² = -53.69...
```

The report records both the calculated value and the published prose value; it
does not silently select `-35.8`. This is a source-level reproducibility finding,
not a claim that the proposed inverse-square force is physical.

### MultipoleStorage

The Rodríguez-inspired surrogate is explicitly a model-selection probe, not a
validated proton Hamiltonian:

```text
H_Q = P_Q²/(2M_Q) + M_Q Omega_Q² Q²/2 + gQ/r³.
```

`eta` defines the dimensionless coupling `g` in Bohr/Hartree units. A
fourth-order symmetric Forest--Ruth/Yoshida map verifies conservation of the
closed particle-plus-mode Hamiltonian before any stochastic or damping channel
is added. The grid is `Omega_Q/omega_B={0.5,1,2,10}` and
`eta={1e-6,1e-4,1e-2,1e-1}`. Mechanical energy transferred into the mode is
entered as internal-mode exchange, not destroyed.

## Stochastic and convergence protocol

The stochastic layer is a finite-mode Monte Carlo quadrature of the
second-order response kernel, not a nonlinear, long-time hydrogen trajectory.
For every stored seed, the discrete finite-time mean of
`2 cos²(omega t+phi)` is an unbiased unit-mean estimator of the quadratic
random-phase response. A closed-form geometric sum evaluates the time grid, so
halving the timestep changes the estimator without requiring a multi-million
step Python loop. The analytic point-charge calculation fixes the ensemble
mean response; finite modes and finite timesteps estimate its numerical and
phase-sampling uncertainty.

The preregistration uses 32 fixed seeds and 2,048 modes. The same seed produces
nested phases at 4,096 modes while the nominal timestep is halved. Normalized
drift must change by less than ten percent between two successive resolutions;
otherwise the cell is `NULL`. Passive channel suppression additionally requires
convergence after dividing out the consistent accelerated coupling
`s² beta²`, for `s={1,4,8,16}`, and a negative upper 95-percent confidence bound
in every low-`L` cell at every preregistered energy and coupling.

## Commands and output

Inspection is safe by default; a full run requires an explicit flag:

```bash
# Print the job manifest only.
blueberry-tournament --profile preregistered --arm all

# Reproducible reduced execution used as an integration smoke test.
blueberry-tournament --profile smoke --arm all --execute \
  --output tournament-smoke.json --npz tournament-smoke.npz

# One chunk of the full shell sweep (parameter index 3).
blueberry-tournament --profile preregistered --arm finite_shell \
  --parameter-index 3 --execute --output shell-003.json --npz shell-003.npz

# The complete preregistered research sweep; intentionally not part of CI.
blueberry-tournament --profile preregistered --arm all --execute \
  --output tournament-full.json --npz tournament-full.npz
```

JSON contains the complete config, baseline surface, parameters, seed list,
both resolutions, confidence intervals, all ledgers, convergence flags, and
classifications. NPZ is optional and contains only redundant numeric arrays.
Its per-run columns are `E,L,s,mean,CI_low,CI_high,closure_residual`. The JSON is
the authoritative, self-describing artifact.

## References and epistemic status

- T. M. Nieuwenhuizen, *Entropy* **18**, 135 (2016),
  [arXiv:1611.10200](https://arxiv.org/abs/1611.10200).
- J. A. E. Rodríguez, Rodríguez-inspired extended-charge motivation,
  [arXiv:1201.6168](https://arxiv.org/abs/1201.6168). The implemented oscillator
  is BlueberryCircus's surrogate, not that paper's validated proton model.
- B. Setterfield, [ZPE and Atomic Constants' Behavior](https://www.barrysetterfield.org/behaviorzpe3.html).
  This is a speculative hypothesis source.
- T. M. Nieuwenhuizen and M. T. P. Liska,
  [arXiv:1502.06856](https://arxiv.org/abs/1502.06856), and the later
  [renormalized-noise study](https://doi.org/10.3389/fphy.2020.00335), are
  negative long-run evidence that this perturbative tournament does not claim
  to reproduce.
- G. Moddel and O. Dmitriyeva,
  [arXiv:0910.5893](https://arxiv.org/abs/0910.5893), for an equilibrium
  zero-point-energy assessment; see also Nanarch's
  [rectification framing](https://www.pecoraro.dev/?post=rectification).
