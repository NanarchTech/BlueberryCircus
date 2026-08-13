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

## 5. Hydrogen: status and honesty boundary

For the Coulomb potential there is no closed form. With radiation reaction and
no ZPF, the orbit radiates and collapses. That is the classical catastrophe,
and BlueberryCircus reproduces it: `r` decreases, energy drops. With the ZPF present,
the absorbed power balances the radiated power and collapse is arrested; Cole &
Zou (2003) reported, for a *planar* orbit, a steady-state radial density
approaching the Schrödinger 1s with no adjustable parameters. Nieuwenhuizen &
Liska (2015) ran the full 3-D problem with relativistic corrections and reached
the opposite conclusion: the atom eventually self-ionizes. They discuss cutoffs
and renormalized noise as candidate repairs; the question is open, not settled.

BlueberryCircus v0.1 makes the hydrogen *engine* runnable and reproduces the
collapse and the ZPF-driven arrest qualitatively (including the known outward
self-ionization wandering at long times). It does **not** claim a converged QM-1s
radial density: that requires frequency-windowed sampling and CPU-day ensembles
and is a compute milestone, marked `xfail(strict=True)` in the test
suite. The package computes what SED predicts; it does not assert SED is the
correct theory of the atom.

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
