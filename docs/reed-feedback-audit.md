# Reed passive-feedback audit

Status: **`NULL` for the full Reed mechanism; `NO_EFFECT` for the bounded
electromagnetic extension.** The experimental implementation was not merged.

A separate experimental branch tested the narrow claim that the subatomic
structure described by Larry Reed can passively reverse BlueberryCircus's
positive low-angular-momentum point-charge drift. This report records the
negative result without treating a qualitative structural picture as a
completed dynamical theory.

## Source constraints

The audit took the electron torus radius as the reduced Compton radius,
`R_e/a0 = alpha`, and the quoted proton constituent separation as
`R_p = 0.33 fm`. Reed's electron discussion says the field is spherically
symmetric and equivalent to a point charge beyond `R_e`. The proton chapter
does not provide a hydrogen Hamiltonian, coupling normalization, mode-energy
span, or a derived phase-locking equation. It also conflicts internally over
whether the proton contains 15 or 18 fields, so the audit evaluated both.

Sources:

- [Quantum Wave Mechanics, fourth-edition sample](https://assets.booklocker.com/pdfs/10176s.pdf)
- [Quantum Wave Mechanics, Chapter 27: Bound Particle States](https://www.researchgate.net/publication/364332579_Quantum_Wave_Mechanics_Ch_27_Bound_Particle_States)

At the near-ionization threshold, `r_p = L_c^2/2`, the orbit remains more than
20 electron-torus radii and more than 20,000 quoted proton-structure scales
away. The audit therefore bounded the leading centered-electron retardation
and zero-dipole proton multipole corrections. It deliberately favored
stabilization:

1. the full `0.33 fm` separation was used as a radius bound;
2. every internal electron-positron charge was allowed to add coherently;
3. absorption was decreased and radiation increased independently by the full
   bound, although a reciprocal passive response cannot generally choose those
   signs independently.

Failure of this envelope excludes the smaller reciprocal electromagnetic
correction under these assumptions. It does not exclude a new interaction
that Reed does not specify.

## Alexander-polynomial role

For a coprime torus knot `T(p,q)`, the audit evaluated

```text
Delta_pq(t) = t^-g (1-t)(1-t^(pq)) / ((1-t^p)(1-t^q)),
g = (p-1)(q-1)/2.
```

It separately recorded the normalized multivariable Alexander polynomial `1`
for the Hopf link `T(2,2)`. Reed's stated two-to-one centerline is `T(2,1)`, an
unknot, not the two-component Hopf link. Both descriptions were retained
rather than conflated. The Alexander polynomial is a discrete isotopy
invariant; it has units of neither energy nor susceptibility. With no
source-derived map from topology to a Hamiltonian coefficient, its only
admissible role was a sector/consistency check. See the torus-link polynomial
treatment by
[Taşköprü and Altıntaş](https://www.combinatorics.org/ojs/index.php/eljc/article/download/v22i4p8/pdf/)
and the Hopf-link normalization discussed in
[L-space surgeries on links](https://londmathsoc.onlinelibrary.wiley.com/doi/full/10.1112/tlm3.12027).

## Preregistered result

The full run used the tournament grid restricted to the three low-`L` cells:

- `E = {-0.05, -0.02, -0.01}`;
- `L = {0.45, 0.55, L_c - 0.01}`;
- coupling scales `s = {1, 4, 8, 16}`;
- 32 fixed seeds;
- 2,048 and 4,096 modes with the timestep halved;
- both 15- and 18-mode source interpretations.

| Reed count | Cells | Bounded-EM class | Largest correction bound | Smallest required/available ratio | Smallest upper 95% drift bound |
|---:|---:|---|---:|---:|---:|
| 15 | 36 | `NO_EFFECT` | `1.1359711184e-7` | `1.6769689565e5` | `5.3346346781e-7` |
| 18 | 36 | `NO_EFFECT` | `1.3631595346e-7` | `1.3995165127e5` | `5.3346284874e-7` |

All upper 95% confidence bounds remained positive. Coupling and resolution
gates passed, external work was exactly zero, and the largest absolute ledger
closure residual was `6.94e-18`. The zero-structure limit recovered the
point-charge channels exactly.

The full claim remains `NULL`, rather than `NO_EFFECT`, because the source does
not define the Hamiltonian and phase-locking dynamics needed for the requested
bounded-storage/horizon-doubling test. Bounded passive storage alone cannot be
a secular sink: if its energy span is `C`, its mean contribution over horizon
`T` is bounded by `C/T`, which tends to zero.

This result concerns conditional drift, not a stationary state or equilibrium
vacuum-energy extraction.
