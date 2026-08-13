"""Symplectic / Gaussian-state phase-space certificates.

This module turns BlueberryCircus's phase-space covariance readout into the
language of single-mode Gaussian quantum optics, so the package's central
correspondence claim -- that the linear (Boyer) sector of stochastic
electrodynamics reproduces the *quantum vacuum* -- becomes a re-checkable
certificate rather than prose.

Two distinct, honest certificates are emitted; keeping them distinct is the point:

* **Vacuum correspondence** (rule ``residual_le_tol``). The measured oscillator
  covariance equals the quantum vacuum ``sigma = (1/2) I`` in nondimensional
  ``(q, p)`` coordinates, to a stated tolerance. This is an EQUALITY-to-tolerance
  claim, so it lands on a PASS-capable rule -- the loophole that turns the ``NULL``
  the physicality rule can only return at the vacuum boundary (below) into an
  honest ``PASS``. It re-expresses the certified Boyer result (O2) in the
  symplectic Gaussian-state language. It is NOT numerically stronger
  than O2: it tracks the same ``<x^2>`` relative error, since from the
  band-limited SED spectrum the conjugate ``<v^2> = omega0^2 <x^2>`` is
  kinematically forced and ``<x v> = 0`` is fixed by stationarity (no *independent*
  normalization test is added). Its added value is structural -- pinning the FULL
  covariance, the cert rejects off-vacuum Gaussian states (squeezed / thermal)
  that share the vacuum's ``<x^2>`` and would pass a position-only check.

* **Physicality** (rule ``symplectic_physical``). The smallest symplectic
  (Williamson) eigenvalue ``nu = sqrt(det sigma)`` obeys ``nu >= 1/2``. For the
  vacuum ``nu = 1/2`` EXACTLY -- the pure-state boundary -- so any two-sided
  bracket straddles 1/2 and the rule honestly returns ``NULL``, never a false
  PASS. ``nu < 1/2`` (FAIL) flags a *sub-Heisenberg* classical SED distribution
  with no quantum counterpart.

  IMPORTANT (corrected scope): physicality is **not** an ionization detector. A
  self-ionizing trajectory has *large* variance, hence ``nu >> 1/2`` -- it reads
  as a perfectly valid high-entropy state. The honest ionization detector is the
  vacuum certificate above, whose residual blows up as ``sigma`` leaves ``(1/2)I``.

Nondimensionalization (single mode, ``hbar`` carried explicitly). BlueberryCircus
works in ``(x, v)``; the conjugate momentum is ``p_phys = m v``. With

    q = x sqrt(m omega0 / hbar),     p = v sqrt(m / (omega0 hbar)),

the quantum vacuum ``<x^2> = hbar/(2 m omega0)``, ``<v^2> = hbar omega0/(2 m)``,
``<x v> = 0`` maps to ``sigma = (1/2) I`` and ``[q, p] = i`` (``hbar = 1``). The
2x2 map is ``sigma = D C D`` with ``C`` the measured ``(x, v)`` covariance and
``D = diag(sqrt(m omega0/hbar), sqrt(m/(omega0 hbar)))``; hence
``det sigma = (m/hbar)^2 det C`` and ``nu = (m/hbar) sqrt(det C)`` -- a closed
form, no eigensolver and no eigenvalue-at-a-boundary instability. (This is why the
single-mode certificate sidesteps the "sound Williamson adapter" the multimode
case would need.)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .certify import Certificate, finalize
from .constants import Units, SI

_HALF_I2 = 0.5 * np.eye(2)
_HALF_I2_FRO = 0.5 * np.sqrt(2.0)          # ||(1/2) I_2||_F


def _finite_or_none(x) -> Optional[float]:
    """Keep a value off the canonical hash surface unless it is a finite float
    (the surface rejects inf/NaN -- a diverged covariance must still serialize)."""
    xf = float(x)
    return xf if math.isfinite(xf) else None


# --- phase-space transforms ---------------------------------------------------
def mode_covariance_xv(cov6: np.ndarray, axis: int = 0) -> np.ndarray:
    """Extract the 2x2 ``(x_axis, v_axis)`` sub-covariance from a 6x6 ``(x,v)``
    readout (``Result.covariance()``: indices 0..2 are ``x``, 3..5 are ``v``)."""
    cov6 = np.asarray(cov6, float)
    if cov6.shape != (6, 6):
        raise ValueError(f"expected 6x6 (x,v) covariance, got {cov6.shape}")
    i, j = axis, axis + 3
    return np.array([[cov6[i, i], cov6[i, j]],
                     [cov6[j, i], cov6[j, j]]], float)


def to_quadrature_covariance(cov_xv: np.ndarray, *, mass: float, omega0: float,
                             hbar: float) -> np.ndarray:
    """Map a 2x2 ``(x, v)`` covariance to the nondimensional ``(q, p)`` covariance.

    ``sigma = D C D``, ``D = diag(sqrt(m omega0/hbar), sqrt(m/(omega0 hbar)))``.
    The quantum vacuum is the fixed point ``sigma = (1/2) I``.
    """
    C = np.asarray(cov_xv, float)
    if C.shape != (2, 2):
        raise ValueError(f"expected 2x2 (x,v) covariance, got {C.shape}")
    if not (mass > 0 and omega0 > 0 and hbar > 0):
        raise ValueError("mass, omega0, hbar must be positive")
    d = np.array([np.sqrt(mass * omega0 / hbar), np.sqrt(mass / (omega0 * hbar))])
    return d[:, None] * C * d[None, :]


def vacuum_target_covariance_xv(omega0: float, units: Units = SI) -> np.ndarray:
    """The exact quantum-vacuum ``(x, v)`` covariance
    ``diag(hbar/(2 m omega0), hbar omega0/(2 m))`` (cross term zero)."""
    xx = units.hbar / (2.0 * units.mass * omega0)
    vv = units.hbar * omega0 / (2.0 * units.mass)
    return np.array([[xx, 0.0], [0.0, vv]])


def symplectic_eigenvalue(sigma: np.ndarray) -> float:
    """Smallest symplectic eigenvalue of a covariance matrix.

    Single mode (2x2): the exact closed form ``nu = sqrt(det sigma)`` (no
    eigensolver -- this is what makes the boundary-sensitive physicality
    certificate well-behaved). Multimode (2n x 2n, ``q..q p..p`` ordering): the
    smallest ``|eigenvalue of i Omega sigma|`` via :func:`numpy.linalg.eigvals` --
    a point estimate, provided for completeness and deliberately NOT consumed by
    the physicality certificate (its boundary case would be a float coin-flip).
    """
    sigma = np.asarray(sigma, float)
    n = sigma.shape[0]
    if sigma.shape != (n, n) or n % 2:
        raise ValueError(f"expected square even-dimensional covariance, got {sigma.shape}")
    if n == 2:
        return float(np.sqrt(max(np.linalg.det(sigma), 0.0)))
    half = n // 2
    Omega = np.block([[np.zeros((half, half)), np.eye(half)],
                      [-np.eye(half), np.zeros((half, half))]])
    ev = np.abs(np.linalg.eigvals(1j * Omega @ sigma))
    return float(np.min(ev))


# --- certificates -------------------------------------------------------------
def vacuum_covariance_certificate(cov_xv: np.ndarray, *, mass: float, omega0: float,
                                  hbar: float, tolerance: float,
                                  method: str = "blueberry_circus phase-space readout",
                                  provenance: Optional[dict] = None,
                                  finalize_status: bool = True) -> Certificate:
    """Certify the measured oscillator covariance == quantum vacuum ``(1/2) I``.

    ``residual = ||sigma - (1/2) I||_F / ||(1/2) I||_F`` under rule
    ``residual_le_tol``. PASS means the full 2x2 Gaussian state matches the vacuum
    to ``tolerance`` -- pinning ``<x^2>``, ``<v^2>`` (equipartition) and
    ``<x v> = 0`` -- pinning more of the Gaussian state than the position-only
    Boyer oracle, though numerically tracking the same ``<x^2>`` error.

    A diverged / overflowing covariance (the ionization regime) is handled like
    :func:`~blueberry_circus.certify.rel_error_certificate`: a non-finite residual
    becomes a FINITE over-tolerance sentinel so the cert re-derives ``FAIL`` and
    still canonicalizes (the hash surface rejects inf/NaN); non-finite numbers are
    kept off that surface (recorded as ``None``).
    """
    sigma = to_quadrature_covariance(cov_xv, mass=mass, omega0=omega0, hbar=hbar)
    with np.errstate(over="ignore", invalid="ignore"):
        resid = float(np.linalg.norm(sigma - _HALF_I2) / _HALF_I2_FRO)
        nu = symplectic_eigenvalue(sigma)
    if math.isfinite(resid) and bool(np.all(np.isfinite(sigma))):
        residual = resid
    else:
        # No vacuum residual is definable -- finite sentinel guaranteed > tolerance.
        residual = abs(float(tolerance)) * 2.0 + 1.0
    prov = dict(provenance or {})
    prov.setdefault("sigma_qq", _finite_or_none(sigma[0, 0]))
    prov.setdefault("sigma_pp", _finite_or_none(sigma[1, 1]))
    prov.setdefault("sigma_qp", _finite_or_none(sigma[0, 1]))
    prov.setdefault("symplectic_eigenvalue", _finite_or_none(nu))
    prov.setdefault("mass", float(mass))
    prov.setdefault("omega0", float(omega0))
    prov.setdefault("hbar", float(hbar))
    prov.setdefault("vacuum_target", "0.5 * I_2 in (q,p), hbar=1")
    cert = Certificate(
        kind="vacuum_covariance_correspondence",
        claim=("measured oscillator phase-space covariance equals the quantum "
               "vacuum (1/2) I in (q,p): pins <x^2>, <v^2> equipartition, <x v>=0"),
        method=method, rule="residual_le_tol",
        value=_finite_or_none(nu), residual=residual, tolerance=float(tolerance),
        provenance=prov)
    return finalize(cert) if finalize_status else cert


def physicality_certificate(cov_xv: np.ndarray, *, mass: float, omega0: float,
                            hbar: float, nu_uncertainty: float,
                            method: str = "blueberry_circus phase-space readout",
                            provenance: Optional[dict] = None,
                            finalize_status: bool = True) -> Certificate:
    """Certify the measured covariance is a physical Gaussian state ``nu >= 1/2``.

    ``enclosure = (nu - nu_uncertainty, nu + nu_uncertainty)``,
    ``nu = sqrt(det sigma)`` (rule ``symplectic_physical``). The bracket is a
    *declared* sample/numerical uncertainty, NOT yet interval-arithmetic sound; for
    the vacuum (``nu = 1/2``) it straddles 1/2 and the rule returns ``NULL`` --
    honest, not a false PASS. ``nu + nu_uncertainty < 1/2`` (FAIL) flags a
    sub-Heisenberg classical SED distribution with no quantum counterpart. This is
    NOT an ionization detector (see module docstring).

    A diverged covariance (non-finite ``nu``) yields ``enclosure=None`` -> the rule
    returns ``NULL`` ("not determinable"), which serializes; ``nu`` itself is kept
    off the canonical hash surface.
    """
    sigma = to_quadrature_covariance(cov_xv, mass=mass, omega0=omega0, hbar=hbar)
    with np.errstate(over="ignore", invalid="ignore"):
        nu = symplectic_eigenvalue(sigma)
    d = abs(float(nu_uncertainty))
    prov = dict(provenance or {})
    prov.setdefault("symplectic_eigenvalue", _finite_or_none(nu))
    prov.setdefault("nu_uncertainty", d)
    prov.setdefault("bracket_kind", "declared sample/numerical, not interval-sound")
    prov.setdefault("mass", float(mass))
    prov.setdefault("omega0", float(omega0))
    prov.setdefault("hbar", float(hbar))
    enclosure = (nu - d, nu + d) if math.isfinite(nu) else None
    cert = Certificate(
        kind="gaussian_state_physicality",
        claim="smallest symplectic eigenvalue nu >= 1/2 (valid quantum Gaussian state)",
        method=method, rule="symplectic_physical",
        value=_finite_or_none(nu), enclosure=enclosure, provenance=prov)
    return finalize(cert) if finalize_status else cert


# --- headline convenience: certify the SED vacuum from the spectrum -----------
def certify_sed_vacuum(omega0: float, omega_lo: float, omega_hi: float,
                       units: Units = SI, *, tolerance: float,
                       npts: int = 400001, span: float = 4000.0,
                       provenance: Optional[dict] = None) -> Certificate:
    """The headline vacuum-correspondence certificate, built from the SED spectrum.

    Assembles the full ``(x, v)`` ground-state covariance from
    :func:`~blueberry_circus.oracles.sed_band_covariance_xv` over the finite ZPF
    band ``[omega_lo, omega_hi]`` -- ``<x^2>``, the conjugate ``<v^2>``, and
    ``<x v> = 0`` by stationarity -- then certifies it equals the quantum vacuum
    ``(1/2) I`` in the symplectic Gaussian-state language. This
    re-expresses O2 over the whole 2x2 phase space (so it rejects off-vacuum states
    O2 cannot distinguish); it does NOT add an independent normalization test --
    ``<v^2>`` is kinematically tied to ``<x^2>`` and tracks the same relative error.

    The band is explicit and load-bearing: ``<v^2>`` is UV-divergent and is only
    vacuum-valued within a finite band below ``1/tau`` (see
    :func:`~blueberry_circus.oracles.sed_band_covariance_xv`).
    """
    from . import oracles as o
    C = o.sed_band_covariance_xv(omega0, omega_lo, omega_hi, units, npts=npts, span=span)
    prov = dict(provenance or {})
    prov.setdefault("band", [float(omega_lo), float(omega_hi)])
    prov.setdefault("xx_integral", float(C[0, 0]))
    prov.setdefault("vv_integral", float(C[1, 1]))
    prov.setdefault("xx_target", float(o.ground_state_variance_target(omega0, units)))
    prov.setdefault("vv_target", float(o.ground_state_momentum_target(omega0, units) / units.mass**2))
    prov.setdefault("uv_note", "<v^2> band-limited; UV-sensitive without the NL relativistic cutoff")
    prov.setdefault("reference_paper", "Boyer PRD 11 790 (1975); Puthoff PRD 35 3266 (1987)")
    return vacuum_covariance_certificate(
        C, mass=units.mass, omega0=omega0, hbar=units.hbar, tolerance=tolerance,
        method="band-limited SED transfer-function covariance -> (q,p) vacuum residual",
        provenance=prov)


__all__ = [
    "mode_covariance_xv", "to_quadrature_covariance", "vacuum_target_covariance_xv",
    "symplectic_eigenvalue", "vacuum_covariance_certificate",
    "physicality_certificate", "certify_sed_vacuum",
]
