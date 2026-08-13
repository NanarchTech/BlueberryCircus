"""3-D Coulomb (hydrogen) SED engine demo.

Shows the three regimes and emits examples/out/hydrogen_certificate.json:
  1. no radiation reaction      -> stable Kepler orbit (E, L conserved)
  2. radiation reaction, no ZPF -> radiative collapse (the classical catastrophe)
  3. accelerated RR + ZPF      -> a numerical stress/unbinding fixture

The third arm uses nonphysical coupling and is not a reproduction of the
long-duration hydrogen simulations in the literature. The convergent
ground-state radial density vs the QM 1s state is a
compute milestone (frequency-windowed sampling + CPU-day ensembles; Cole & Zou
2003, Nieuwenhuizen & Liska 2015) and is *not* claimed here.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from blueberry_circus.constants import BOHR, Units
from blueberry_circus.dynamics import Particle, integrate
from blueberry_circus.potentials import Coulomb
from blueberry_circus.zpf import ZPFBackground
from blueberry_circus.observables import total_energy, angular_momentum, radial_distribution
from blueberry_circus.certify import Certificate, finalize, save_bundle, audit_overclaim
from blueberry_circus.watchdog import ionization_time

OUT = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(OUT, exist_ok=True)
# SCALED units are an accelerated oscillator/stress system. For this Coulomb
# fixture tau*omega_orbit is about 1.34e4 times the physical Bohr-unit value.
# The run is a numerical failure stress, not qualitative physical hydrogen.
U = Units.scaled(gamma_over_omega0=0.02, omega0=1.0)
P = Particle(U.charge, U.mass)
coul = Coulomb(Z=1.0, units=U, charge=U.charge, mass=U.mass, softening=2e-2)
r0 = 1.0
vc = np.sqrt(np.linalg.norm(coul.force([r0, 0, 0])) * r0 / U.mass)
omega_orbit = vc / r0
damping_ratio = U.tau * omega_orbit / BOHR.tau
certs = []

# 1. Kepler (no RR)
t = np.arange(0, 120, 0.004)
k = integrate(field=None, potential=coul, particle=P, t_grid=t, x0=[r0,0,0],
              v0=[0,vc,0], rr="none", units=U, dipole=False)
E = total_energy(k, coul, P); L = angular_momentum(k, P)[:, 2]
dE = (E.max()-E.min())/abs(E.mean()); dL = (L.max()-L.min())/abs(L.mean())
certs.append(finalize(Certificate(kind="kepler_conservation",
    claim="no-RR Coulomb orbit conserves energy and angular momentum",
    method="RK4 Kepler integration; max relative drift of E and L_z",
    rule="residual_le_tol", residual=float(max(dE, dL)), tolerance=1e-6)))
print(f"[Kepler] dE={dE:.2e} dL={dL:.2e}")

# 2. radiative collapse
c = integrate(field=None, potential=coul, particle=P, t_grid=t, x0=[r0,0,0],
              v0=[0,vc,0], rr="landau_lifshitz", units=U, dipole=False)
Ec = total_energy(c, coul, P)
certs.append(finalize(Certificate(kind="radiative_collapse",
    claim="RR without ZPF drives the orbit to lower energy (classical catastrophe)",
    method="RK4+LL Coulomb integration, no ZPF; energy change over the window",
    rule="residual_le_tol", residual=float(max(0.0, Ec[-1]-Ec[0])), tolerance=0.0)))
print(f"[Collapse] E {Ec[0]:.4f} -> {Ec[-1]:.4f}; r {c.r[0]:.3f} -> {c.r[-1]:.3f}")

# 3. RR + ZPF
field = ZPFBackground.isotropic_3d(0.3, 4.0, 150, seed=7, units=U)
t2 = np.arange(0, 80, 0.004)
z = integrate(field=field, potential=coul, particle=P, t_grid=t2, x0=[r0,0,0],
              v0=[0,vc,0], rr="landau_lifshitz", units=U, dipole=False)
finite = bool(np.all(np.isfinite(z.x)))
certs.append(finalize(Certificate(kind="accelerated_coulomb_stress_finite",
    claim="accelerated RR+ZPF Coulomb stress trajectory remains finite over the window",
    method="nonphysical scaled RK4+LL/ZPF stress fixture; not physical hydrogen",
    # finite sentinel (never inf): a FAIL cert must stay serializable on the
    # canonical hash surface. 1.0 > 0.0 -> FAIL.
    rule="residual_le_tol", residual=0.0 if finite else 1.0,
    tolerance=0.0)))
centers, dens = radial_distribution(z, bins=40, burn_in=0.1)
np.savez(os.path.join(OUT, "hydrogen_radial.npz"), r=centers, P=dens)
t_ion, detail = ionization_time(z, coul, P)
orbit_fraction = t_ion * omega_orbit / (2.0 * np.pi) if t_ion is not None else 1.0
certs.append(finalize(Certificate(kind="accelerated_coulomb_stress_unbinding",
    claim="v0.1.0 accelerated fixture unbinds within one tenth of one orbit",
    method="positive mechanical energy sustained for 5% of the stress window",
    rule="residual_le_tol", residual=float(orbit_fraction), tolerance=0.1,
    provenance={"damping_ratio_to_physical_bohr": float(damping_ratio),
                "scope": "numerical stress only"})))
print(f"[Accelerated stress] finite={finite}  r in [{z.r.min():.2f},{z.r.max():.2f}]  "
      f"damping={damping_ratio:.1f}x physical  unbinding={orbit_fraction:.4f} orbit")

save_bundle(certs, os.path.join(OUT, "hydrogen_certificate.json"))
assert not any(audit_overclaim(c) for c in certs)
print(f"Wrote {len(certs)} certificates; overclaim audit clean.")
