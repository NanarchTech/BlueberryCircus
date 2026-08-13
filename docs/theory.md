# Theory: classical stochastic electrodynamics of the ZPF-bound atom

## 1. The classical zero-point field

Stochastic electrodynamics (SED) keeps classical Maxwell electrodynamics but
adds a boundary condition: even at T=0 a real, classical, random radiation
field, the zero-point field (ZPF), fills space. Its spectrum is
fixed (up to the scale `ħ`) by Lorentz invariance: every normal mode carries a
mean energy `½ħω`, giving the spectral energy density

    ρ(ω) = g(ω)·½ħω = ħ ω³ / (2 π² c³),   g(ω) = ω² / (π² c³).

The field is built as a sum of plane waves with random phases, isotropic
directions, and two transverse polarizations per direction, with amplitudes set
so that each mode contributes `½ħω`. The classical amplitudes are normalized to
reproduce the quantum two-point correlation of the electric field.

## 2. Capacitive (dielectric) storage of the electric energy, and the role of ε₀

The fluctuating electric field stores energy capacitively through the vacuum
permittivity,

    u_E = ½ ε₀ ⟨E²⟩,

the electric (capacitive) half of the total ZPF energy density; the magnetic
(inductive) half is `⟨B²⟩/2μ₀`, equal to it, and together they fix the impedance
of free space. There is no separate "capacitor object": ε₀ is the conversion
factor between the fluctuating `E²` and stored energy, and it sets the absolute
scale of the field that supplies the power the orbiting electron absorbs
(Puthoff 1987). BlueberryCircus therefore carries ε₀ explicitly, as a field of
`Units`, never absorbed into an effective coupling, so that
`S_Ex(ω) = ρ(ω)/(3ε₀) = ħω³/(6π²ε₀c³)` is dimensionally exact and the absolute
variance `⟨x²⟩` comes out in metres², not arbitrary units.

## 3. Dynamics and radiation reaction

The point electron obeys `m a = F_pot + q(E + v×B) + F_rad`. The radiative
self-force is the Abraham–Lorentz term, characterized by `τ = q²/6πε₀mc³`. Its
literal third-derivative form `mτẍ̇` has runaway and pre-accelerating solutions
and must not be integrated directly. We use the Landau–Lifshitz reduction of
order, `F_rad = τ dF_ext/dt`, valid when `τω ≪ 1` (always, for bound states).
This is a genuine damping that balances the energy absorbed from the ZPF.

## 4. The exactly-solvable oscillator (Boyer 1975)

For a harmonic binding `U = ½mω₀²x²` the system is linear and solvable. The
stationary variance is

    ⟨x²⟩ = ∫₀^∞ S_Ex(ω) |H_AL(ω)|² dω,
    H_AL(ω) = q / ( m[(ω₀²−ω²) + i τ ω³] ).

The oscillator is a razor-sharp resonant filter (`τω₀ ≪ 1`); the integral is
dominated by `ω ≈ ω₀`, where the Lorentzian integrates to give exactly

    ⟨x²⟩ = ħ / (2 m ω₀),

the quantum-mechanical ground-state value, with the conjugate result
`⟨p²⟩ = ½ħmω₀`. This is the linchpin: classical zero-point radiation, balanced
against radiation reaction, reproduces the quantum oscillator ground state with
**no free parameters**. BlueberryCircus verifies the identity with real electron
SI constants to relative error 5×10⁻⁴, and verifies that the time-domain
integrator reproduces the analytic variance to 2.2×10⁻³, i.e. 0.2% (single mode).

## 5. Coulomb sector: three bounded claims

### 5.1 Puthoff's circular-orbit approximation

Puthoff's 1987 calculation is a circular-orbit, harmonic absorption/radiation
balance at the level of Bohr theory. BlueberryCircus now evaluates both sides,

    P_abs = e² ħ ω₀³ / (6π ε₀ m c³),
    P_rad = e² r₀² ω₀⁴ / (6π ε₀ c³).

Their equality is equivalent to `m ω₀ r₀² = ħ`; at the Coulomb circular orbit
this selects the familiar Bohr targets. It does not establish stability of the
nonlinear stochastic hydrogen problem. That distinction is load-bearing, not
semantic. The exact power residual is gated below `1e-12`.

### 5.2 Nieuwenhuizen's near-ionization rectification

For an almost parabolic Kepler orbit, Nieuwenhuizen derives the disorder- and
period-averaged energy change per revolution

    Δ⟨E⟩ = (3π β²/L⁶) [L_c - L],
    β = √(2/3) Z α^(3/2),
    L_c = f(0) = 16/(5π√3) = 0.5880841551… .

The package independently evaluates the nested improper integral defining
`f(0)` (Eq. 2.30 of [arXiv:1611.10200](https://arxiv.org/abs/1611.10200)) by a
96×96 transformed Gauss–Legendre rule. It recovers the closed form to better
than `1e-8`. The critical near-parabolic perihelion is
`r_p = L_c²/2 = 0.172921… a₀`. Positive drift below `L_c` identifies a
conditional energy-space channel toward ionization; negative drift above it
does not by itself imply a stationary ground state.

This rectification is not equilibrium vacuum-energy extraction. A nonzero
conditional drift, or broken symmetry in one sector, does not establish usable
cycle-averaged work, defeat detailed balance, or close an energy ledger. Those
are separate propositions and require separate accounting; see the equilibrium
assessment in [Moddel and Dmitriyeva, arXiv:0910.5893](https://arxiv.org/abs/0910.5893)
and Nanarch's rectification framing at [pecoraro.dev](https://www.pecoraro.dev/?post=rectification).

### 5.3 Physical Bohr normalization and the retained stress fixture

The physical atomic-unit factory fixes

    m = ħ = e = a₀ = ω_B = 1,   4π ε₀ = 1,   c = 1/α,
    τ ω_B = τ = (2/3)α³ = β².

`Units.scaled()` has a different purpose: it deliberately chooses a moderate
numerical damping and is therefore an accelerated oscillator/stress system.
In the retained v0.1.0 Coulomb fixture, `τ ω_orbit` is `13,367.7` times the
physical Bohr value. The trajectory crosses sustained positive mechanical
energy after `0.06956` orbit. A run that unbinds before one tenth of an orbit at
four orders of magnitude too much damping cannot reproduce physical long-time
hydrogen self-ionization.

The published full-3-D [long simulations](https://arxiv.org/abs/1502.06856)
nevertheless remain important negative evidence: Nieuwenhuizen and Liska's
runs ionized in all attempted modelings, relativistic corrections did not
remove that behavior, and the [2020 renormalized-noise study](https://doi.org/10.3389/fphy.2020.00335)
found no scheme that escaped the prior self-ionization result. These are
literature results, not outputs that BlueberryCircus claims to have reproduced
at their physical timescales.

### 5.4 Setterfield static co-scaling is a conjugacy

Setterfield's speculative profile is represented, without empirical
endorsement, by

    ħ, ε₀ ∝ U,   c ∝ U⁻¹,   e ∝ U^(1/2),   m ∝ U².

Substitution shows that `α`, `a₀`, the Hartree energy `E_h`, `β`, and
`τ ω_B` are invariant, while `ω_B ∝ U⁻¹`. With identical random phases and the
frequency band mapped as `ω → ω/U`, the field amplitudes scale as `U⁻¹/²`
and wavevectors remain fixed. The equations are conjugate under

    x_U(Ut) = x_1(t),   U v_U(Ut) = v_1(t).

The regression test integrates the full Coulomb + ZPF + Landau–Lifshitz system
at `U=1` and `U=4`, including the magnetic term, and requires position,
velocity, mechanical energy, and `L/ħ` agreement below `1e-9`. Thus the static
co-scaling is dynamically inert after time reparameterization. A genuinely
time-dependent `U(t)` is a different, driven problem whose parameter work must
be recorded explicitly.

### 5.5 Full drift surface and hypothesis tournament

Version 0.3.0 evaluates Nieuwenhuizen's complete finite-energy point-charge
drift before changing the model. With `k=sqrt(-2E)`, `kappa=kL`, and
`epsilon²=1-kappa²`, Equation (2.34) is

    D(E,L) = beta² k⁸(2+epsilon²)[k f(kappa)-kappa]/(2 kappa⁶).

The nested definition of `f(kappa)` is numerically nontrivial near a radial
orbit. Equations (2.23)--(2.25) cancel all short-history terms through cubic
order, so evaluating the unfactored numerator loses the signal at the
perihelion. `tournament.py` performs this cancellation coefficient by
coefficient, then uses perihelion-adapted Gauss--Legendre maps. This is why the
same routine can recover both the circular result `f(1)=1/2` and the PR1 radial
limit rather than splicing an empirical interpolation between them.

The tournament ledger is

    Delta E_mech = W_ZPF - E_rad - Delta E_Schott
                   + W_external + W_internal + residual.

This identity blocks three common category errors. Parametric work in a dynamic
Setterfield profile cannot masquerade as passive vacuum stabilization;
energy stored in the multipole surrogate cannot disappear; and a finite shell
cannot attenuate radiation without applying the same reciprocal form factor to
absorption.

The stochastic layer samples the *quadratic perturbative response kernel* with
finite random phases. It does not integrate a nonlinear physical hydrogen atom
for the very long times used in the negative published simulations. Its purpose
is narrower: reproduce confidence intervals from stored seeds, check the
2,048-to-4,096-mode/timestep convergence rule, and refuse a classification when
the response calculation is resolution-dependent. Full equations, parameter
grids, classifications, and commands are in [`tournament.md`](tournament.md).

The inverse-square source has an internal numerical inconsistency worth making
explicit. Direct integration of the kernel printed in Nieuwenhuizen's Eqs.
(3.19)--(3.27) gives `H_max approximately 7.327` near `mu=0.590`, hence the
defining `d_c=-H_max² approximately -53.69`. The prose value `-35.8` follows
from squaring only its quoted endpoint `H(0)=5.99`, even though the stated
criterion is a maximum over the whole repulsive branch. BlueberryCircus records
both values and uses the calculated maximum. This does not validate the added
force; it only makes the paper's mathematical test reproducible.

## 6. The vacuum-covariance certificate

Boyer's oracle (§4) pins one number, `⟨x²⟩`. The vacuum-covariance certificate
restates that result as the *whole* single-mode Gaussian state. In
nondimensional quadratures `q = x√(mω₀/ħ)` and `p = v√(m/ω₀ħ)`, the band-limited
SED ground state reproduces `σ = ½I` to 3.8×10⁻⁶, and the single-mode symplectic
eigenvalue has the closed form `ν = √(det σ)`, so no eigensolver and no
instability at the boundary. That closed form is what makes the claim testable:
an equality-to-tolerance rule can return PASS, whereas the physicality rule
alone can only return NULL at the vacuum edge.

Be precise about what this adds. Numerically it tracks the §4 result: given the
SED spectrum, `⟨v²⟩ = ω₀²⟨x²⟩` is kinematically forced and `⟨xv⟩ = 0` follows
from stationarity, so no *independent* normalization test is added. The gain is
structural. Pinning the full covariance rejects off-vacuum Gaussian states,
squeezed or thermal, that share the vacuum's `⟨x²⟩` and would pass a
position-only check.

Its companion is the physicality check, `ν ≥ ½`. For the vacuum `ν = ½`
*exactly*, the pure-state edge, so a two-sided bracket straddles the boundary
and the rule returns NULL rather than a false PASS. Physicality does **not**
detect self-ionization: an escaping orbit has huge variance, hence `ν ≫ ½`. The
vacuum certificate is a departure-from-vacuum detector, with ionization one
possible cause, and its residual diverges off-vacuum.

**The ultraviolet caveat**, disclosed on the certificate itself: `⟨x²⟩` is
UV-benign (only logarithmically sensitive), but `⟨v²⟩` diverges *quadratically*
in the cutoff, the free-particle jitter that the relativistic cutoff
regularizes, so the vacuum claim is stated on a finite band below `1/τ`.

**Independent recheck.** Every emitted bundle can be re-derived by
`nanarch-verify`, a separate Rust implementation of the checking rules. Note the
scope: the recheck re-derives each verdict from the *recorded* residual and
rule. It does not recompute the residual from the covariance; that bit-level
audit is a stronger, still-deferred layer.
