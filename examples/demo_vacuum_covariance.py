"""Certified vacuum-covariance correspondence -- the sharpened linear-sector result.

The Boyer oracle (O2) certifies one number: <x^2> = hbar/(2 m omega0). This demo
re-expresses that result as the WHOLE single-mode Gaussian state in symplectic
(q, p) language: the band-limited SED ground-state covariance equals the quantum
vacuum sigma = (1/2) I, on a PASS-capable rule (the loophole). It tracks O2
numerically (the conjugate <v^2> is kinematically tied to <x^2>, <x v>=0 by
stationarity); its added value is structural -- it rejects off-vacuum states
(squeezed/thermal) a position-only check would accept.

Two honest companions are shown:
  * physicality (nu >= 1/2) is NULL at the vacuum boundary (the pure-state edge),
    and -- corrected from the first synergy sketch -- does NOT detect ionization;
  * the VACUUM cert is the ionization detector: its residual blows up off-vacuum.

If a nanarch-verify binary is available (point BLUEBERRY_VERIFY_BIN at it), the
emitted bundle is re-checked cross-language. Run:
    python examples/demo_vacuum_covariance.py
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from blueberry_circus.constants import SI
from blueberry_circus import symplectic as sp
from blueberry_circus.certify import audit_overclaim, save_bundle

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

w0 = 2.5e16  # an electron-scale binding frequency (SI -- the physical system)

# ---- headline: the band-limited SED ground state IS the quantum vacuum -------
vac = sp.certify_sed_vacuum(w0, w0 / 10, w0 * 10, SI, tolerance=1e-4)
p = vac.provenance
print("VACUUM CORRESPONDENCE  (residual_le_tol on || sigma - (1/2)I ||)")
print(f"  sigma_qq = {p['sigma_qq']:.8f}   (<x^2>, target 0.5)")
print(f"  sigma_pp = {p['sigma_pp']:.8f}   (<v^2> equipartition, target 0.5)")
print(f"  sigma_qp = {p['sigma_qp']:.2e}   (<x v>, target 0)")
print(f"  residual = {vac.residual:.3e}  tol = {vac.tolerance:.0e}  ->  {vac.status}")
print(f"  (tracks O2 numerically; the win is scope -- pins the full sigma, not just sigma_qq)\n")

# ---- physicality: honest NULL at the vacuum boundary ------------------------
Cvac = sp.vacuum_target_covariance_xv(w0, SI)
phys = sp.physicality_certificate(Cvac, mass=SI.mass, omega0=w0, hbar=SI.hbar,
                                  nu_uncertainty=1e-6)
print("PHYSICALITY  (symplectic_physical, nu = sqrt(det sigma))")
print(f"  nu = {phys.value:.10f}  enclosure = ({phys.enclosure[0]:.6f}, "
      f"{phys.enclosure[1]:.6f})  ->  {phys.status}   (boundary nu=1/2 -> honest NULL)")

thermal = sp.physicality_certificate(4.0 * Cvac, mass=SI.mass, omega0=w0,
                                     hbar=SI.hbar, nu_uncertainty=1e-3)
subh = sp.physicality_certificate(0.25 * Cvac, mass=SI.mass, omega0=w0,
                                  hbar=SI.hbar, nu_uncertainty=1e-3)
print(f"  thermal (nu={thermal.value:.2f}) -> {thermal.status} | "
      f"sub-Heisenberg (nu={subh.value:.3f}) -> {subh.status} (no quantum counterpart)\n")

# ---- the ionization detector is the VACUUM cert, not physicality ------------
Cion = 1e6 * Cvac
phys_ion = sp.physicality_certificate(Cion, mass=SI.mass, omega0=w0, hbar=SI.hbar,
                                      nu_uncertainty=1e-3)
vac_ion = sp.vacuum_covariance_certificate(Cion, mass=SI.mass, omega0=w0,
                                           hbar=SI.hbar, tolerance=1e-2)
print("IONIZATION MECHANISM (synthetic huge-variance covariance, not a live trajectory):")
print(f"  physicality nu = {phys_ion.value:.2e} -> {phys_ion.status}  "
      "(valid high-entropy state -- physicality MISSES ionization)")
print(f"  vacuum residual = {vac_ion.residual:.2e} -> {vac_ion.status}  "
      "(departure-from-vacuum detector; ionization is one cause)\n")

# ---- emit bundle + overclaim audit -----------------------------------------
certs = [vac, phys]
path = os.path.join(OUT, "vacuum_covariance_certificate.json")
save_bundle(certs, path)
assert not any(audit_overclaim(c) for c in certs), "overclaim detected!"
print(f"Wrote {len(certs)} certificates to {os.path.relpath(path)}; overclaim audit clean.")

# ---- optional: cross-language recheck ---------------------------------------
vbin = os.environ.get("BLUEBERRY_VERIFY_BIN")
if not (vbin and os.path.isfile(vbin)):
    vbin = None
if vbin:
    r = subprocess.run([vbin, path], capture_output=True, text=True)
    verdict = "PASS" if r.returncode == 0 else f"FAIL (exit {r.returncode})"
    print(f"\ncross-language recheck (nanarch-verify, Rust): RESULT {verdict}")
else:
    print("\ncross-language recheck: nanarch-verify not available "
          "(point BLUEBERRY_VERIFY_BIN at the verify binary to enable).")
