"""Cole--Zou moving spectral window for the hydrogen problem.

A full isotropic box for hydrogen needs ~1e19 plane waves. The bound electron
only resonantly exchanges power with modes near its *instantaneous orbital
frequency*

    omega(r) = sqrt( k_e q^2 Z / (m r^3) )      (circular-orbit estimate),

so Cole & Zou (2003) retain only modes in a band around omega(r) that slides as r
drifts -- a ~250x cost reduction. The hazard is that a *hard* cutoff makes the
active mode-set change discontinuously and inject spurious energy; Nieuwenhuizen &
Liska (2015) fix this with amplitude/phase continuity corrections (their Eqs.
28-32).

This module uses the equivalent **smooth-taper** window: each mode's amplitude is
multiplied by a raised-cosine weight ``w(omega; r) in [0,1]`` that is 1 inside the
resonant band and ramps to 0 across a taper margin. Because the weight is a
continuous (C1) function of r, no mode ever switches on/off discontinuously,
eliminating the hard-cutoff energy *spike*. (A position-dependent amplitude
``a_m w_m(|x|)`` still makes the field non-Maxwellian, so a small residual
injection from the smooth spatial gradient remains -- it is bounded by the taper
width and is the convergence quantity tracked in the gate, NOT claimed to be
exactly zero.) The hard-cutoff + re-phasing variant and the CPU-day windowed
hydrogen ensemble are the convergence frontier (see README).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .constants import Units, SI
from .zpf import ZPFBackground


def orbital_frequency(r, coef: float, mass: float):
    """Circular-orbit angular frequency ``omega(r) = sqrt(coef / (m r^3))`` where
    ``coef = Z k_e q^2`` (so the Coulomb force magnitude is ``coef / r^2``)."""
    r = np.asarray(r, dtype=float)
    return np.sqrt(coef / (mass * r**3))


def coulomb_coef(coul, units: Units = SI) -> float:
    """``coef = Z k_e q^2`` from a :class:`~blueberry_circus.potentials.Coulomb`."""
    return coul.Z * units.k_e * coul.charge**2


def _band(omega_c: float, f_band: float):
    """Resonant band ``[omega(r(1+f)), omega(r(1-f))]`` since omega ~ r^-3/2."""
    lo = omega_c * (1.0 + f_band) ** (-1.5)
    hi = omega_c * (1.0 - f_band) ** (-1.5)
    return lo, hi


def taper_weights(omegas, omega_c: float, f_band: float, taper: float) -> np.ndarray:
    """Raised-cosine weights in [0,1]: 1 on the resonant band, ramping to 0 over a
    fractional ``taper`` margin on each side. Continuous in ``omega`` (hence in r)."""
    omegas = np.asarray(omegas, dtype=float)
    lo, hi = _band(omega_c, f_band)
    lo_edge, hi_edge = lo * (1.0 - taper), hi * (1.0 + taper)
    w = np.zeros_like(omegas)
    inside = (omegas >= lo) & (omegas <= hi)
    w[inside] = 1.0
    left = (omegas < lo) & (omegas > lo_edge)
    if np.any(left):
        w[left] = 0.5 * (1.0 - np.cos(np.pi * (omegas[left] - lo_edge) / (lo - lo_edge)))
    right = (omegas > hi) & (omegas < hi_edge)
    if np.any(right):
        w[right] = 0.5 * (1.0 + np.cos(np.pi * (omegas[right] - hi) / (hi_edge - hi)))
    return w


@dataclass
class WindowedField:
    """A ZPF background whose modes are smoothly tapered to the resonant band of
    the instantaneous orbital radius. Evaluate at the *actual* particle position
    (``dipole=False``); the radius ``r = |x|`` selects the active band.

    The same frozen realization (omegas, kvecs, evecs, amps, phases) is used at all
    times -- only the per-mode weight slides, so determinism (and the seeded ZPF)
    is preserved.
    """
    base: ZPFBackground
    coef: float
    mass: float
    f_band: float = 0.03
    taper: float = 0.5
    units: Units = None

    def __post_init__(self):
        # The c/eps0 used to read fields out of `base` MUST be the ones `base`
        # was built with; never silently default to SI when they differ.
        if self.units is None:
            self.units = self.base.units

    def _weights(self, r_norm: float) -> np.ndarray:
        if r_norm <= 0.0:
            return np.zeros_like(self.base.omegas)
        omega_c = float(orbital_frequency(r_norm, self.coef, self.mass))
        return taper_weights(self.base.omegas, omega_c, self.f_band, self.taper)

    def _arg(self, r, t):
        kr = self.base.kvecs @ np.asarray(r, dtype=float)
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return kr[:, None] - self.base.omegas[:, None] * t[None, :] + self.base.phases[:, None]

    def E(self, r, t):
        scalar = np.ndim(t) == 0
        w = self._weights(float(np.linalg.norm(r)))
        ph = self._arg(r, t)
        contrib = (self.base.amps * w)[:, None] * np.cos(ph)
        E = self.base.evecs.T @ contrib
        return E[:, 0] if scalar else E

    def dEdt(self, r, t):
        scalar = np.ndim(t) == 0
        w = self._weights(float(np.linalg.norm(r)))
        ph = self._arg(r, t)
        contrib = (self.base.amps * w * self.base.omegas)[:, None] * np.sin(ph)
        dE = self.base.evecs.T @ contrib
        return dE[:, 0] if scalar else dE

    def B(self, r, t):
        scalar = np.ndim(t) == 0
        w = self._weights(float(np.linalg.norm(r)))
        khat = self.base.kvecs / np.maximum(
            np.linalg.norm(self.base.kvecs, axis=1, keepdims=True), 1e-300)
        bvec = np.cross(khat, self.base.evecs) / self.units.c
        ph = self._arg(r, t)
        contrib = (self.base.amps * w)[:, None] * np.cos(ph)
        B = bvec.T @ contrib
        return B[:, 0] if scalar else B

    def active_energy_density(self, r_norm: float) -> float:
        """ZPF electric energy density of the tapered active set at radius ``r``,
        ``eps0 * sum_m (a_m w_m)^2/2 * |e_m|^2`` -- a continuous function of r."""
        w = self._weights(r_norm)
        mse = np.sum(0.5 * (self.base.amps * w) ** 2 * np.sum(self.base.evecs**2, axis=1))
        return float(self.units.eps0 * mse)
