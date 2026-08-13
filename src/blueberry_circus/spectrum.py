"""Zero-point-field spectral densities (the Lorentz-invariant ZPF).

The classical zero-point radiation spectrum is fixed (up to the choice of
``hbar`` as the scale of the fluctuation) by the requirement of Lorentz
invariance: each normal mode carries a mean energy ``(1/2) hbar omega`` and the
spectral energy density is

    rho(omega) = hbar omega^3 / (2 pi^2 c^3)            [J s / m^3]
               = g(omega) * (1/2) hbar omega,

where ``g(omega) = omega^2 / (pi^2 c^3)`` is the two-polarization mode density
per unit volume per unit angular frequency. The electric part stores half of
this energy capacitively, ``u_E = (1/2) eps0 <E^2> = (1/2) integral rho``, hence
the per-Cartesian-component one-sided spectral density of the electric field is

    S_Ex(omega) = hbar omega^3 / (6 pi^2 eps0 c^3).

These two functions are the entire physical input to the ground-state result
(Boyer, Phys. Rev. D 11, 790 (1975); Puthoff 1987). Everything else is dynamics.
"""
from __future__ import annotations

import numpy as np

from .constants import Units, SI


def mode_density(omega, units: Units = SI):
    """Two-polarization mode density ``g(omega) = omega^2 / (pi^2 c^3)``."""
    omega = np.asarray(omega, dtype=float)
    return omega**2 / (np.pi**2 * units.c**3)


def mode_energy(omega, units: Units = SI):
    """Mean energy per mode ``(1/2) hbar omega`` (the zero-point quantum)."""
    omega = np.asarray(omega, dtype=float)
    return 0.5 * units.hbar * omega


def rho(omega, units: Units = SI):
    """ZPF spectral *energy* density ``rho = hbar omega^3 / (2 pi^2 c^3)``."""
    omega = np.asarray(omega, dtype=float)
    return units.hbar * omega**3 / (2.0 * np.pi**2 * units.c**3)


def spectral_density_Ex(omega, units: Units = SI):
    """Per-component one-sided electric-field spectral density.

    Defined so that ``<E_x^2> = integral_0^inf S_Ex(omega) d omega`` and the
    total ``<E^2> = 3 <E_x^2> = (1/eps0) integral rho`` (electric energy =
    half the total ZPF energy density).
    """
    omega = np.asarray(omega, dtype=float)
    return units.hbar * omega**3 / (6.0 * np.pi**2 * units.eps0 * units.c**3)


def energy_density_band(omega_lo: float, omega_hi: float, units: Units = SI,
                        npts: int = 20001) -> float:
    """Total ZPF energy density in a band, ``integral_lo^hi rho d omega``."""
    w = np.linspace(omega_lo, omega_hi, npts)
    integrand = rho(w, units)
    return float(_trapz(integrand, w))


def mode_amplitude(omega, dω, units: Units = SI):
    """1-D random-phase amplitude reproducing ``S_Ex`` over a bin of width ``dω``.

    For a single cosine ``a cos(omega t + phi)`` with uniform phase,
    ``<E_x^2> contribution = (1/2) a^2``. Matching ``S_Ex(omega) dω`` gives
    ``a = sqrt(2 S_Ex(omega) dω)``.
    """
    return np.sqrt(2.0 * spectral_density_Ex(omega, units) * dω)


def _trapz(y, x):
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
