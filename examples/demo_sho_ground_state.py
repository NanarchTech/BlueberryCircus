"""Certified SHO-in-ZPF ground state -- the exactly-solvable anchor.

Emits a certificate bundle to examples/out/sho_ground_state_certificate.json.

Two independent oracles:
  C2  physics      : integral S_Ex |H_AL|^2 dω = hbar/(2 m omega0)   (Boyer 1975)
  C-int integrator : single-mode RK4+LL steady-state variance = (1/2)a^2|H|^2

A finite-time many-mode time-domain run is also shown; it is *intentionally* not
gated, because the modes within a resonance half-width equilibrate on times
~1/detuning that exceed short runs -- the convergent time-domain ground state is
a compute milestone.  Run:  python examples/demo_sho_ground_state.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from blueberry_circus import oracles
from blueberry_circus.constants import Units, SI
from blueberry_circus.dynamics import Particle, integrate
from blueberry_circus.zpf import ZPFBackground
from blueberry_circus.potentials import Harmonic
from blueberry_circus.certify import rel_error_certificate, save_bundle, audit_overclaim

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
certs = []

# ---- C2 : physical ground state (SI electron) -------------------------------
w0 = 2.5e16
val = oracles.sed_ground_state_integral(w0, SI)
ref = oracles.ground_state_variance_target(w0, SI)
c2 = rel_error_certificate(kind="sed_ground_state_variance",
                 claim="stationary <x^2> = hbar/(2 m omega0) for AL oscillator in ZPF",
                 value=val, reference=ref, tolerance=5e-3,
                 method="numerical integral S_Ex |H_AL|^2 dω, Lorentzian substitution (Boyer 1975)",
                 provenance={"omega0": w0, "units": "SI-electron",
                             "reference_paper": "Boyer PRD 11 790 (1975)"})
certs.append(c2)
print(f"[C2] <x^2>_SED={val:.6e}  hbar/2mw0={ref:.6e}  rel.err={abs(val-ref)/ref:.2e}  -> {c2.status}")

# ---- C-int : integrator fidelity (scaled units, single mode) ----------------
U = Units.scaled(gamma_over_omega0=0.05, omega0=1.0)
P = Particle(U.charge, U.mass); pot = Harmonic(1.0, mass=U.mass)
field1 = ZPFBackground.one_dimensional(0.75, 0.85, 1, seed=0, units=U)
t = np.arange(0, 400, 0.04)
tr = integrate(field=field1, potential=pot, particle=P, t_grid=t, x0=[0,0,0],
               v0=[0,0,0], rr="landau_lifshitz", units=U)
meas = tr.x[len(t)//3:, 0].var()
exact = oracles.phase_averaged_variance(field1.amps, field1.omegas,
            lambda w: oracles.transfer_landau_lifshitz(w, 1.0, U))
cint = rel_error_certificate(kind="integrator_fidelity",
                   claim="single-mode RK4+LL steady-state variance = (1/2)a^2|H_LL|^2",
                   value=meas, reference=exact, tolerance=1.5e-2,
                   method="single-mode RK4+LL time-domain run vs analytic transfer function")
certs.append(cint)
print(f"[C-int] measured={meas:.6e}  exact={exact:.6e}  rel.err={abs(meas-exact)/exact:.2e}  -> {cint.status}")

# ---- honest time-domain many-mode trend (NOT gated) -------------------------
print("\nFinite-time many-mode variance vs analytic sum (equilibration-limited):")
ana = None
for seed in (0, 1, 2):
    f = ZPFBackground.one_dimensional(0.3, 3.0, 200, seed=seed, units=U)
    tt = np.arange(0, 350, 0.04)
    trj = integrate(field=f, potential=pot, particle=P, t_grid=tt, x0=[0,0,0],
                    v0=[0,0,0], rr="landau_lifshitz", units=U)
    m = trj.x[len(tt)//3:, 0].var()
    if ana is None:
        ana = oracles.phase_averaged_variance(f.amps, f.omegas,
                  lambda w: oracles.transfer_landau_lifshitz(w, 1.0, U))
    print(f"   seed {seed}: Var(x)={m:.4e}  (analytic sum {ana:.4e}, ratio {m/ana:.3f})")
print("   -> single-realization scatter; the convergent value needs an ensemble.")

save_bundle(certs, os.path.join(OUT, "sho_ground_state_certificate.json"))
assert not any(audit_overclaim(c) for c in certs), "overclaim detected!"
print(f"\nWrote {len(certs)} certificates; overclaim audit clean.")
