"""Closed-form oracles for the exactly-solvable linear SED problem.

A charged 1-D harmonic oscillator in the ZPF is the *exactly solvable* anchor of
stochastic electrodynamics: its stationary position variance equals the quantum
ground-state value ``<x^2> = hbar / (2 m omega0)`` (Boyer 1975). Because the
system is linear, two independent analytic checks are available and are used as
the verification backbone of the whole package:

1. **Transfer-function oracle** (validates the integrator + sampler): for a sum
   of random-phase modes the phase-averaged variance is ``(1/2) sum_j a_j^2
   |H(omega_j)|^2`` with ``H`` the linear response of the *exact equation being
   integrated*. This is unit-system independent and holds for any damping.

2. **Ground-state normalization oracle** (validates the physics): with the ZPF
   electric spectral density ``S_Ex`` of :mod:`spectrum` and the Abraham--Lorentz
   response, ``integral S_Ex(omega) |H_AL(omega)|^2 d omega = hbar/(2 m omega0)``
   in the radiation-reaction-dominated-resonance limit ``tau omega0 << 1`` (the
   physical regime). Verified numerically to ~5e-4 relative error with SI
   electron constants (the O2 regression lock).
"""
from __future__ import annotations

import math

import numpy as np

from .constants import Units, SI, ALPHA
from .spectrum import spectral_density_Ex, rho


def _trapz(y, x, axis=-1):
    return (np.trapezoid(y, x=x, axis=axis) if hasattr(np, "trapezoid")
            else np.trapz(y, x, axis=axis))


def transfer_abraham_lorentz(omega, omega0: float, units: Units = SI):
    """``H_AL(omega) = q / ( m [ (omega0^2 - omega^2) + i tau omega^3 ] )``.

    Frequency response of ``m xddot = -m omega0^2 x + m tau xdddot + q E``.
    """
    omega = np.asarray(omega, dtype=float)
    m, q, tau = units.mass, units.charge, units.tau
    denom = m * ((omega0**2 - omega**2) + 1j * tau * omega**3)
    return q / denom


def transfer_landau_lifshitz(omega, omega0: float, units: Units = SI):
    """Response of the reduction-of-order (Landau--Lifshitz) oscillator.

    ``m xddot = -m omega0^2 x - m tau omega0^2 xdot + q E + q tau Edot`` gives
    ``H_LL(omega) = (q/m)(1 + i tau omega) / (omega0^2 - omega^2 + i tau omega0^2 omega)``.
    Equal to ``H_AL`` to leading order in ``tau omega`` near resonance; this is
    the response of the *non-runaway* equation actually integrated.
    """
    omega = np.asarray(omega, dtype=float)
    m, q, tau = units.mass, units.charge, units.tau
    num = (q / m) * (1.0 + 1j * tau * omega)
    denom = (omega0**2 - omega**2) + 1j * tau * omega0**2 * omega
    return num / denom


def phase_averaged_variance(amps, omegas, transfer) -> float:
    """``(1/2) sum_j a_j^2 |H(omega_j)|^2`` -- exact phase-averaged ``<x^2>``."""
    amps = np.asarray(amps, dtype=float)
    H = transfer(np.asarray(omegas, dtype=float))
    return float(0.5 * np.sum(amps**2 * np.abs(H)**2))


def ground_state_variance_target(omega0: float, units: Units = SI) -> float:
    """Quantum ground-state oracle ``<x^2> = hbar / (2 m omega0)``."""
    return units.hbar / (2.0 * units.mass * omega0)


def ground_state_momentum_target(omega0: float, units: Units = SI) -> float:
    """``<p^2> = hbar m omega0 / 2`` (the conjugate ground-state oracle)."""
    return units.hbar * units.mass * omega0 / 2.0


def sed_ground_state_integral(omega0: float, units: Units = SI,
                              npts: int = 200001, span: float = 400.0) -> float:
    """Numerically evaluate ``integral_0^inf S_Ex |H_AL|^2 d omega``.

    A Lorentzian (Cauchy) change of variables centred on the resonance,
    ``omega = omega0 + b * span * tan(theta)`` with half-width
    ``b = tau omega0^2 / 2``, resolves the extremely sharp electron resonance so
    that the trapezoid rule converges to the analytic value to ~5e-4.
    """
    tau = units.tau
    b = tau * omega0**2 / 2.0
    th = np.linspace(-np.pi / 2 * (1 - 1e-9), np.pi / 2 * (1 - 1e-9), npts)
    w = omega0 + b * span * np.tan(th)
    keep = w > 0
    w, th = w[keep], th[keep]
    dwdth = b * span / np.cos(th)**2
    S = spectral_density_Ex(w, units)
    H2 = np.abs(transfer_abraham_lorentz(w, omega0, units))**2
    integ = S * H2 * dwdth
    val = np.trapezoid(integ, th) if hasattr(np, "trapezoid") else np.trapz(integ, th)
    return float(val)


def sed_band_covariance_xv(omega0: float, omega_lo: float, omega_hi: float,
                           units: Units = SI, npts: int = 200001,
                           span: float = 400.0, n_log: int = 40001) -> np.ndarray:
    """Phase-averaged ``(x, v)`` ground-state covariance of the AL oscillator over a
    *finite* ZPF band ``[omega_lo, omega_hi]``::

        <x^2> = integral_band         S_Ex |H_AL|^2 d omega   -> hbar/(2 m omega0)
        <v^2> = integral_band omega^2 S_Ex |H_AL|^2 d omega   -> hbar omega0/(2 m)
        <x v> = 0     (stationary random-phase background)

    Returns the 2x2 ``[[<x^2>, 0], [0, <v^2>]]`` -- the full phase-space partner of
    the position-only :func:`sed_ground_state_integral`.

    The grid is a UNION of (1) a tangent grid concentrated on ``omega0`` to resolve
    the razor-sharp electron resonance and (2) a dense log grid over the whole band
    so the high-``omega`` tail is genuinely sampled -- the tangent grid alone
    undersamples the UV and would silently report the resonance-core value for wide
    bands. Integration is a non-uniform trapezoid over the merged frequencies.

    UV NOTE (load-bearing honesty). ``<x^2>`` is UV-benign: its integrand ``~1/omega``
    in the UV is only *logarithmically* sensitive (tail ~1e-6 out to ``1/tau``), so
    it is effectively the Boyer value for any sane band. ``<v^2>`` -- the kinetic /
    momentum quadrature -- is *quadratically* UV-DIVERGENT: a bound charge inherits
    the free-particle ZPF jitter at high ``omega`` (the divergence the
    Nieuwenhuizen--Liska relativistic cutoff ``~ m c^2/hbar`` regularizes). On a
    band bracketing the resonance and ``omega_hi`` well below ``1/tau`` both
    quadratures sit at the quantum vacuum (e.g. ``[omega0/10, 10 omega0]`` reproduces
    ``<v^2>`` to ~1e-5; ``1/tau ~ 6e6 omega0`` for the electron); push ``omega_hi``
    toward ``1/tau`` and ``<v^2>`` grows ``~ omega_hi^2`` as the real physics
    demands. The vacuum correspondence is therefore a *finite-band* statement.
    """
    if not (0.0 < omega_lo < omega0 < omega_hi):
        raise ValueError("require 0 < omega_lo < omega0 < omega_hi")
    tau = units.tau
    b = tau * omega0**2 / 2.0
    # (1) resonance core: tangent grid concentrated at omega0, clamped to the band.
    th = np.linspace(-np.pi / 2 * (1 - 1e-12), np.pi / 2 * (1 - 1e-12), npts)
    w_core = omega0 + b * span * np.tan(th)
    w_core = w_core[(w_core >= omega_lo) & (w_core <= omega_hi)]
    # (2) broadband / UV coverage over the WHOLE band (so <v^2>'s tail is sampled).
    w_log = np.geomspace(omega_lo, omega_hi, n_log)
    w = np.unique(np.concatenate([w_core, w_log, [omega_lo, omega_hi]]))
    base = spectral_density_Ex(w, units) * np.abs(transfer_abraham_lorentz(w, omega0, units))**2
    xx = float(_trapz(base, w))
    vv = float(_trapz(w**2 * base, w))
    return np.array([[xx, 0.0], [0.0, vv]])


def larmor_power(accel, units: Units = SI):
    """Nonrelativistic Larmor radiated power ``P = m tau a^2 = q^2 a^2/(6 pi eps0 c^3)``."""
    accel = np.asarray(accel, dtype=float)
    return units.mass * units.tau * accel**2


# --- O0: the field two-point function ----------------------------------------
def field_autocorrelation(field, t, component: int = 0) -> np.ndarray:
    """Phase-averaged electric-field autocorrelation ``<E_i(0,t) E_i(0,0)>`` of a
    random-phase :class:`~blueberry_circus.zpf.ZPFBackground`, Cartesian comp ``i``.

    With ``E_i(0,t) = sum_m a_m e_{m,i} cos(omega_m t - phi_m)`` and independent
    uniform phases, ``<E_i(t)E_i(0)> = sum_m (a_m^2/2) e_{m,i}^2 cos(omega_m t)``
    -- the exact two-point function of the discrete realization (no sampling).
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    w = 0.5 * field.amps**2 * field.evecs[:, component]**2            # (M,)
    return (w[None, :] * np.cos(field.omegas[None, :] * t[:, None])).sum(axis=1)


def field_two_point_continuum(t, omega_lo: float, omega_hi: float,
                              units: Units = SI, npts: int = 40001) -> np.ndarray:
    """Band-limited continuum two-point function, per Cartesian component:
    ``C(t) = integral_{omega_lo}^{omega_hi} S_Ex(omega) cos(omega t) d omega``
    (Wiener--Khinchin for the one-sided electric PSD ``S_Ex``).

    The discrete :func:`field_autocorrelation` converges to this as the mode count
    grows; the full-spectrum UV-regularized Nieuwenhuizen--Liska form is the
    ``omega_hi -> m c^2 / hbar`` limit. ``C(0) = <E_i^2>``.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    w = np.linspace(omega_lo, omega_hi, npts)
    S = spectral_density_Ex(w, units)
    integ = S[None, :] * np.cos(w[None, :] * t[:, None])
    return _trapz(integ, w, axis=1)


# --- ground-state polarizability (analytic-oracle standalone oracle) ------------------
def polarizability(omega, omega0: float, units: Units = SI):
    """Complex dynamic polarizability of the charged AL/LL oscillator,
    ``alpha(omega) = q^2 / ( m [ (omega0^2 - omega^2) + i tau omega^3 ] ) = q H_AL``.

    The static limit ``alpha(0) = q^2 / (m omega0^2)`` is exactly known
    (:func:`static_polarizability_target`) and is the analytic-oracle acceptance oracle.
    """
    omega = np.asarray(omega, dtype=float)
    m, q, tau = units.mass, units.charge, units.tau
    return q**2 / (m * ((omega0**2 - omega**2) + 1j * tau * omega**3))


def static_polarizability_target(omega0: float, units: Units = SI) -> float:
    """Exact static polarizability of the charged oscillator ``q^2/(m omega0^2)``."""
    return units.charge**2 / (units.mass * omega0**2)


# --- Coulomb Landau-Lifshitz damping coefficient (analytic oracle) -------------
def beta_coefficient(Z: float = 1.0) -> float:
    """Nieuwenhuizen--Liska Bohr-unit Coulomb-LL damping coefficient (Eq. 9),
    ``beta = sqrt(2/3) Z alpha^{3/2} = Z / 1964.71``. Depends only on the
    fine-structure constant (unit-system independent)."""
    return math.sqrt(2.0 / 3.0) * Z * ALPHA**1.5


def coulomb_ll_damping_accel(x, v, coul, units: Units = SI) -> np.ndarray:
    """LL radiation-reaction acceleration for a Coulomb binding via the GENERIC
    force-Jacobian path the integrator actually uses: ``a = (tau/m) J(x) v``."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    return (units.tau / coul.mass) * (coul.force_jacobian(x) @ v)


def coulomb_ll_damping_closed_form(x, v, coul, units: Units = SI) -> np.ndarray:
    """The same LL damping acceleration written out for the Coulomb Jacobian,
    ``a = -(tau coef / m) ( v/rs^3 - 3 x (x.v)/rs^5 )``, ``coef = Z k_e q^2``.

    Equality with :func:`coulomb_ll_damping_accel` is the analytic oracle that the
    *hydrogen* damping term is correct -- the one place the Coulomb physics can be
    silently wrong while the harmonic oracle (O2) stays green.
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    rs = math.sqrt(float(x @ x) + coul.softening**2)
    # Use the SAME coefficient the integrator's Jacobian uses (frozen on the
    # potential at construction), not one recomputed from the passed units --
    # otherwise this oracle silently diverges from the path it is meant to check.
    coef = coul._coef
    return -(units.tau * coef / coul.mass) * (v / rs**3 - 3.0 * x * float(x @ v) / rs**5)


# --- O3: hydrogen radial density vs the QM 1s state --------------------------
def hydrogen_1s_radial(r):
    """QM ground-state radial probability density ``P_r(r) = 4 r^2 e^{-2r}``
    (Bohr units, a0 = 1); ``integral_0^inf P_r dr = 1``."""
    r = np.asarray(r, dtype=float)
    return 4.0 * r**2 * np.exp(-2.0 * r)


def radial_l1_distance(centers, density) -> float:
    """L1 distance ``integral |P_SED - P_1s| dr`` between a measured radial density
    (on bin ``centers``, Bohr units) and the QM 1s density. Both are renormalized
    to unit integral on the given grid, so this is a shape distance in [0, 2]."""
    centers = np.asarray(centers, dtype=float)
    density = np.asarray(density, dtype=float)
    dr = np.gradient(centers)
    p = density / max(float(np.sum(density * dr)), 1e-300)
    q = hydrogen_1s_radial(centers)
    q = q / max(float(np.sum(q * dr)), 1e-300)
    return float(np.sum(np.abs(p - q) * dr))


# --- O1: Puthoff power-balance ground state (Bohr-level analytic targets) -----
# Puthoff (PRD 35, 3266, 1987) evaluated a circular-orbit harmonic power
# balance at the Bohr level. Its analytic targets are the Bohr radius, the
# -13.6 eV binding energy, and angular momentum hbar. O1 is not a claim that a
# stable nonlinear stochastic orbit has been reproduced dynamically.
def bohr_radius(units: Units = SI, Z: float = 1.0) -> float:
    """ZPF-determined ground-state radius ``a0 = 4 pi eps0 hbar^2 / (m e^2) / Z``."""
    a0 = 4.0 * math.pi * units.eps0 * units.hbar**2 / (units.mass * units.charge**2)
    return a0 / Z


def hydrogen_ground_state_energy(units: Units = SI, Z: float = 1.0) -> float:
    """``E_1 = -Z^2 k_e e^2 / (2 a0) = -1/2 Z^2 alpha^2 m c^2`` (-13.6 eV for Z=1)."""
    a0 = bohr_radius(units, 1.0)
    return -(Z**2) * units.k_e * units.charge**2 / (2.0 * a0)


def bohr_angular_momentum(units: Units = SI, Z: float = 1.0) -> float:
    """Angular momentum of the circular ground-state orbit; SED/Bohr -> ``hbar``."""
    a0 = bohr_radius(units, Z)
    coef = Z * units.k_e * units.charge**2
    v = math.sqrt(coef / (units.mass * a0))
    return units.mass * v * a0


def puthoff_power_balance(units: Units = SI, Z: float = 1.0) -> dict:
    """Evaluate Puthoff's circular-orbit harmonic power-balance approximation.

    This is the Bohr-level calculation in Puthoff (1987), not a nonlinear
    hydrogen-stability result. At radius ``r0`` and circular frequency ``omega0``
    it compares

    ``P_abs = e^2 hbar omega0^3 / (6 pi eps0 m c^3)`` and
    ``P_rad = e^2 r0^2 omega0^4 / (6 pi eps0 c^3)``.

    Equality is equivalent to ``m omega0 r0^2 = hbar``. The legacy
    ``larmor_power`` key is retained as an alias for ``radiated_power``.
    """
    a0 = bohr_radius(units, Z)
    coef = Z * units.k_e * units.charge**2
    omega0 = math.sqrt(coef / (units.mass * a0**3))          # orbital angular freq
    p_abs = (units.charge**2 * units.hbar * omega0**3 /
             (6.0 * math.pi * units.eps0 * units.mass * units.c**3))
    p_rad = (units.charge**2 * a0**2 * omega0**4 /
             (6.0 * math.pi * units.eps0 * units.c**3))
    action = units.mass * omega0 * a0**2
    rel_residual = abs(p_abs - p_rad) / max(abs(p_abs), abs(p_rad), 1e-300)
    return {
        "a0": a0,
        "omega0": omega0,
        "absorbed_power": p_abs,
        "radiated_power": p_rad,
        "larmor_power": p_rad,
        "relative_power_residual": rel_residual,
        "action_m_omega_r2": action,
        "action_over_hbar": action / units.hbar,
        "rho_zpf_at_omega0": float(rho(omega0, units)),
        "angular_momentum_over_hbar": bohr_angular_momentum(units, Z) / units.hbar,
        "energy_joule": hydrogen_ground_state_energy(units, Z),
    }
