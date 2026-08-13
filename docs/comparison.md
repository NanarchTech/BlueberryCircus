# Comparison and positioning

## Architectural lineage (quantum-optics stack)

BlueberryCircus borrows the *ergonomics* of the photonic/CV quantum-optics
libraries, not their physics.

| Library | Domain | Model BlueberryCircus borrows |
|---|---|---|
| Strawberry Fields (Xanadu) | quantum (CV photonics) | `Program` + `Engine` + ops piped onto a register |
| Mr Mustard (Xanadu) | quantum (Gaussian/Fock) | state/transformation objects; covariance focus |
| PennyLane | quantum (qubit/CV, autodiff) | device/backend abstraction; future autodiff backend |

BlueberryCircus is the **classical-stochastic** sibling: same declarative
feel, but the "vacuum" is a real classical random field and the dynamics are
Newton + Lorentz + radiation reaction.

## Physics prior art (SED simulators)

| Work | What it did | Form | Gap BlueberryCircus fills |
|---|---|---|---|
| Boyer 1975 | exact oscillator ground state in ZPF | analytic | the certified oracle we anchor to |
| Puthoff 1987 | Bohr-level power-balance equilibrium | analytic | the ε₀-explicit power accounting |
| Cole & Zou 2003 | first hydrogen trajectory sim (planar) | bespoke code | library-grade, tested, documented engine |
| Nieuwenhuizen & Liska 2015 | 3-D + relativistic; self-ionization | bespoke OpenCL/C | reproducible API + assurance certificates |

To our knowledge there is **no pip-installable, documented, tested SED hydrogen
simulator**. EMpy / ElectroMagneticPython (PyPI) is general EM mode-solving, not
SED trajectory dynamics. BlueberryCircus's contribution is the packaging +
dimensional discipline (explicit ε₀) + non-runaway integrator + re-checkable
certificates anchored to the Boyer identity.
