"""Certified circular balance and near-ionization rectification diagnostics.

The outputs are analytic/numerical certificates under explicit approximations;
they are not certificates that SED is true or that hydrogen reaches equilibrium.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from blueberry_circus import BOHR, rectification, setterfield_rescale
from blueberry_circus.certify import (Certificate, audit_overclaim, finalize,
                                      rel_error_certificate, save_bundle)
from blueberry_circus.oracles import bohr_radius, puthoff_power_balance


OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

balance = puthoff_power_balance(BOHR)
balance_cert = finalize(Certificate(
    kind="puthoff_circular_power_balance",
    claim="P_abs equals P_rad in Puthoff's circular-orbit harmonic approximation",
    method="closed-form Bohr-unit powers; not nonlinear hydrogen stability",
    rule="residual_le_tol",
    residual=balance["relative_power_residual"],
    tolerance=1e-12,
    provenance={"reference": "Puthoff, Phys. Rev. D 35, 3266 (1987)"},
))

threshold = rectification.threshold_quadrature(order=96)
threshold_exact = rectification.critical_angular_momentum()
threshold_cert = rel_error_certificate(
    "nieuwenhuizen_rectification_threshold",
    "independent improper quadrature recovers Lc = 16/(5 pi sqrt(3))",
    value=threshold,
    reference=threshold_exact,
    tolerance=1e-8,
    method="96x96 transformed Gauss-Legendre evaluation of Eq. (2.30)",
    provenance={"reference": "Nieuwenhuizen, arXiv:1611.10200, Eqs. 2.30-2.37"},
)

scaled = setterfield_rescale(BOHR, 4.0)
a0_scaled = bohr_radius(scaled)
hartree_scaled = scaled.k_e * scaled.charge**2 / a0_scaled
omega_scaled = hartree_scaled / scaled.hbar
invariant_residual = max(
    abs(a0_scaled - 1.0),
    abs(hartree_scaled - 1.0),
    abs(scaled.tau * omega_scaled - BOHR.tau),
)
scaling_cert = finalize(Certificate(
    kind="setterfield_static_coscaling_invariants",
    claim="static U=4 co-scaling preserves a0, Eh, and tau*omegaB",
    method="algebraic invariants; trajectory conjugacy is tested separately",
    rule="residual_le_tol",
    residual=invariant_residual,
    tolerance=1e-12,
    provenance={"hypothesis": "Setterfield scaling profile; not empirically endorsed"},
))

certs = [balance_cert, threshold_cert, scaling_cert]
assert all(c.recheck() == "PASS" for c in certs)
assert not any(audit_overclaim(c) for c in certs)
save_bundle(certs, os.path.join(OUT, "rectification_certificate.json"))

lc = threshold_exact
print(f"[Puthoff circular approximation] residual={balance['relative_power_residual']:.3e}")
print(f"[O5 quadrature] Lc={threshold:.12f}; abs.err={abs(threshold-lc):.3e}")
print(f"[O5 geometry] rp=Lc^2/2={lc**2/2:.12f} a0")
print(f"[Conditional drift] below={rectification.near_ionization_drift(lc-0.01):+.3e}; "
      f"above={rectification.near_ionization_drift(lc+0.01):+.3e}")
print(f"Wrote {len(certs)} certificates; overclaim audit clean.")
