"""Energy-audited hypothesis tournament for perturbative SED hydrogen.

This module deliberately separates three levels of claim:

* :func:`point_charge_drift` is Nieuwenhuizen's orbit-averaged, second-order
  point-charge calculation, evaluated over the complete elliptic domain.
* The four arm classes are preregistered response models.  Setterfield and the
  Rodriguez-inspired multipole arm are hypotheses, not established physics.
* :func:`stochastic_cell` is a finite-mode Monte Carlo evaluation of the
  *perturbative response kernel*.  It is not a long-time nonlinear trajectory
  and cannot establish a stationary hydrogen ground state.

The ledger identity is

``dE_mech = W_ZPF - E_rad - dE_Schott + W_ext + W_internal + residual``.

Radiative loss is stored as a non-negative number.  All other channels are
signed contributions to mechanical energy.  This convention makes hidden
external work, especially in a time-dependent Setterfield profile, explicit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import itertools
import json
import math
from typing import Iterable, Sequence

import numpy as np

from .constants import ALPHA
from .oracles import beta_coefficient
from .rectification import critical_angular_momentum


CHANNEL_SUPPRESSED = "CHANNEL_SUPPRESSED"
ACTIVE_CONTROL = "ACTIVE_CONTROL"
NO_EFFECT = "NO_EFFECT"
DESTABILIZED = "DESTABILIZED"
NULL = "NULL"
CLASSIFICATIONS = frozenset({
    CHANNEL_SUPPRESSED, ACTIVE_CONTROL, NO_EFFECT, DESTABILIZED, NULL,
})
TOURNAMENT_SCHEMA_VERSION = "0.3.0"


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class OrbitState:
    """A bound Kepler orbit in physical Bohr geometry.

    ``energy`` and ``angular_momentum`` are in Hartree and ``hbar`` units.
    For ``V=-Z/r``, ``k=sqrt(-2E)`` and ``kappa=kL/Z``.
    """

    energy: float
    angular_momentum: float
    Z: float = 1.0

    def __post_init__(self) -> None:
        energy = _finite(self.energy, "energy")
        angular_momentum = _finite(self.angular_momentum, "angular_momentum")
        Z = _finite(self.Z, "Z")
        if energy >= 0.0:
            raise ValueError("energy must be negative for a bound orbit")
        if angular_momentum <= 0.0:
            raise ValueError("angular_momentum must be positive")
        if Z <= 0.0:
            raise ValueError("Z must be positive")
        if self.kappa > 1.0 + 2e-14:
            raise ValueError("angular momentum exceeds the circular-orbit limit")

    @property
    def k(self) -> float:
        return math.sqrt(-2.0 * self.energy)

    @property
    def kappa(self) -> float:
        return self.k * self.angular_momentum / self.Z

    @property
    def eccentricity(self) -> float:
        return math.sqrt(max(0.0, 1.0 - self.kappa**2))

    @property
    def semimajor_axis(self) -> float:
        return -self.Z / (2.0 * self.energy)

    @property
    def perihelion(self) -> float:
        return self.angular_momentum**2 / (
            self.Z * (1.0 + self.eccentricity)
        )

    @property
    def period(self) -> float:
        return 2.0 * math.pi * self.Z / self.k**3

    def perihelion_phase_point(self) -> tuple[np.ndarray, np.ndarray]:
        """Return canonical ``(x,p)`` at perihelion for unit electron mass."""
        radius = self.perihelion
        x = np.array([radius, 0.0, 0.0])
        p = np.array([0.0, self.angular_momentum / radius, 0.0])
        return x, p


@dataclass(frozen=True)
class EnergyLedger:
    """Signed energy channels and their explicitly computed closure residual."""

    mechanical_energy_change: float
    zpf_work: float
    radiative_loss: float
    schott_boundary_energy: float = 0.0
    external_parameter_work: float = 0.0
    internal_mode_exchange: float = 0.0
    numerical_closure_residual: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(value)) for value in asdict(self).values()):
            raise ValueError("all ledger fields must be finite")
        if self.radiative_loss < 0.0:
            raise ValueError("radiative_loss is a non-negative magnitude")

    @classmethod
    def from_channels(
        cls,
        mechanical_energy_change: float,
        zpf_work: float,
        radiative_loss: float,
        schott_boundary_energy: float = 0.0,
        external_parameter_work: float = 0.0,
        internal_mode_exchange: float = 0.0,
    ) -> "EnergyLedger":
        values = [
            mechanical_energy_change, zpf_work, radiative_loss,
            schott_boundary_energy, external_parameter_work,
            internal_mode_exchange,
        ]
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("all ledger channels must be finite")
        if radiative_loss < 0.0:
            raise ValueError("radiative_loss is a non-negative magnitude")
        accounted = (
            zpf_work - radiative_loss - schott_boundary_energy
            + external_parameter_work + internal_mode_exchange
        )
        residual = mechanical_energy_change - accounted
        return cls(*(float(value) for value in values), float(residual))

    @property
    def accounted_change(self) -> float:
        return (
            self.zpf_work - self.radiative_loss
            - self.schott_boundary_energy + self.external_parameter_work
            + self.internal_mode_exchange
        )

    @property
    def rederived_closure_residual(self) -> float:
        """Recompute closure without trusting the stored residual field."""
        return self.mechanical_energy_change - self.accounted_change

    @property
    def relative_closure(self) -> float:
        scale = max(
            abs(self.mechanical_energy_change), abs(self.zpf_work),
            abs(self.radiative_loss), abs(self.schott_boundary_energy),
            abs(self.external_parameter_work), abs(self.internal_mode_exchange),
            # Tournament energies are normalized to one Hartree.  Retaining a
            # unit floor prevents an arbitrarily tiny, physically irrelevant
            # channel exchange from inflating an otherwise closed ledger.
            1.0,
        )
        audit_residual = max(
            abs(self.rederived_closure_residual),
            abs(self.numerical_closure_residual
                - self.rederived_closure_residual),
        )
        return audit_residual / scale


def _default_momenta() -> tuple[float, ...]:
    lc = critical_angular_momentum()
    return (0.45, 0.55, lc - 0.01, lc + 0.01, 0.65, 0.8)


@dataclass(frozen=True)
class TournamentConfig:
    """Immutable preregistration and numerical convergence policy."""

    energies: tuple[float, ...] = (-0.05, -0.02, -0.01)
    angular_momenta: tuple[float, ...] = field(default_factory=_default_momenta)
    coupling_scales: tuple[float, ...] = (1.0, 4.0, 8.0, 16.0)
    seeds: tuple[int, ...] = tuple(range(101, 133))
    n_modes: int = 2048
    timestep: float = 2e-3
    max_resolution_levels: int = 2
    convergence_rtol: float = 0.10
    setterfield_amplitudes: tuple[float, ...] = (0.01, 0.05, 0.1)
    setterfield_omega_ratios: tuple[float, ...] = (0.1, 1.0, 10.0)
    setterfield_phases: tuple[float, ...] = (
        0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi,
    )
    shell_radii: tuple[float, ...] = (ALPHA**2, 1e-3, 1e-2, 0.1, 0.3)
    inverse_square_d: tuple[float, ...] = (ALPHA**2, 0.0, -10.0, -35.8, -40.0)
    multipole_omega_ratios: tuple[float, ...] = (0.5, 1.0, 2.0, 10.0)
    multipole_eta: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1e-1)

    def __post_init__(self) -> None:
        tuple_fields = (
            "energies", "angular_momenta", "coupling_scales", "seeds",
            "setterfield_amplitudes", "setterfield_omega_ratios",
            "setterfield_phases", "shell_radii", "inverse_square_d",
            "multipole_omega_ratios", "multipole_eta",
        )
        for name in tuple_fields:
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.n_modes < 2:
            raise ValueError("n_modes must be at least 2")
        if self.timestep <= 0.0 or not math.isfinite(self.timestep):
            raise ValueError("timestep must be finite and positive")
        if self.max_resolution_levels < 2:
            raise ValueError("at least two resolution levels are required")
        if not 0.0 < self.convergence_rtol < 1.0:
            raise ValueError("convergence_rtol must lie in (0, 1)")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        for E in self.energies:
            for L in self.angular_momenta:
                OrbitState(E, L)
        if any(scale <= 0.0 for scale in self.coupling_scales):
            raise ValueError("coupling scales must be positive")

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=indent,
                          allow_nan=False)

    @classmethod
    def from_json(cls, payload: str) -> "TournamentConfig":
        data = json.loads(payload)
        return cls(**data)


@dataclass(frozen=True)
class HypothesisResult:
    """One auditable cell result, including stored seeds and resolutions."""

    arm: str
    energy: float
    angular_momentum: float
    mean_drift: float
    confidence_low: float
    confidence_high: float
    ledger: EnergyLedger
    converged: bool
    coupling_scale: float = 1.0
    classification: str = NO_EFFECT
    seeds: tuple[int, ...] = ()
    resolutions: tuple[tuple[int, float], ...] = ()
    resolution_drifts: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.classification not in CLASSIFICATIONS:
            raise ValueError(f"unknown classification {self.classification!r}")
        numeric = (
            self.energy, self.angular_momentum, self.mean_drift,
            self.confidence_low, self.confidence_high,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("result coordinates and interval must be finite")
        if not self.confidence_low <= self.mean_drift <= self.confidence_high:
            raise ValueError("mean drift must lie inside its confidence interval")
        if self.coupling_scale <= 0.0 or not math.isfinite(self.coupling_scale):
            raise ValueError("coupling_scale must be finite and positive")
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "resolutions", tuple(
            (int(modes), float(dt)) for modes, dt in self.resolutions
        ))
        object.__setattr__(self, "resolution_drifts",
                           tuple(float(x) for x in self.resolution_drifts))

    @classmethod
    def cell(
        cls, arm: str, energy: float, angular_momentum: float,
        confidence_low: float, confidence_high: float, ledger: EnergyLedger,
        *, converged: bool, classification: str = NO_EFFECT,
        coupling_scale: float = 1.0,
        seeds: Sequence[int] = (),
        resolutions: Sequence[tuple[int, float]] = (),
        resolution_drifts: Sequence[float] = (),
        mean_drift: float | None = None,
    ) -> "HypothesisResult":
        if mean_drift is None:
            mean_drift = 0.5 * (confidence_low + confidence_high)
        return cls(
            arm, float(energy), float(angular_momentum), float(mean_drift),
            float(confidence_low), float(confidence_high), ledger,
            bool(converged), float(coupling_scale), classification,
            tuple(int(s) for s in seeds),
            tuple(resolutions), tuple(resolution_drifts),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=indent,
                          allow_nan=False)

    @classmethod
    def from_json(cls, payload: str) -> "HypothesisResult":
        data = json.loads(payload)
        data["ledger"] = EnergyLedger(**data["ledger"])
        return cls(**data)


# -- Stable evaluation of Nieuwenhuizen's complete point-charge surface -------

_SERIES_ORDER = 18
_LD = np.longdouble


def _poly_constant(value, shape: tuple[int, ...]) -> np.ndarray:
    result = np.zeros((_SERIES_ORDER + 1,) + shape, dtype=_LD)
    result[0] = value
    return result


def _poly_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.zeros_like(left)
    for degree in range(_SERIES_ORDER + 1):
        for index in range(degree + 1):
            result[degree] += left[index] * right[degree - index]
    return result


def _poly_power(value: np.ndarray, exponent: int) -> np.ndarray:
    result = _poly_constant(1.0, value.shape[1:])
    for _ in range(exponent):
        result = _poly_multiply(result, value)
    return result


def _linear_trig(constant: np.ndarray, slope: float, sine: bool) -> np.ndarray:
    constant = np.asarray(constant, dtype=_LD)
    result = np.empty((_SERIES_ORDER + 1,) + constant.shape, dtype=_LD)
    for degree in range(_SERIES_ORDER + 1):
        phase = constant + _LD(degree) * np.pi / 2
        trig = np.sin(phase) if sine else np.cos(phase)
        result[degree] = (
            trig * _LD(slope) ** degree / _LD(math.factorial(degree))
        )
    return result


def _evaluate_series(coefficients: np.ndarray, argument: np.ndarray) -> np.ndarray:
    result = np.zeros((coefficients.shape[1], len(argument)), dtype=_LD)
    for coefficient in coefficients[::-1]:
        result = result * argument[None, :] + coefficient[:, None]
    return result


def _small_separation_drift_ratio(a: np.ndarray, epsilon: _LD) -> np.ndarray:
    """Series for ``rho_a rho_b (Gdot-3) / tau^4`` around ``b=a``.

    Eq. (2.23)--(2.25) cancels terms through cubic order.  Evaluating the
    unfactored expression loses many digits at eccentric perihelia, so the
    cancellation is performed coefficient-by-coefficient before division.
    """
    a = np.asarray(a, dtype=_LD)
    zeros = np.zeros_like(a)
    shape = a.shape
    sd = _linear_trig(zeros, 1.0, True)
    s2d = _linear_trig(zeros, 2.0, True)
    cd = _linear_trig(zeros, 1.0, False)
    c2d = _linear_trig(zeros, 2.0, False)
    sa = _linear_trig(a, 0.0, True)
    ca = _linear_trig(a, 0.0, False)
    s2a = _linear_trig(2.0 * a, 0.0, True)
    c2a = _linear_trig(2.0 * a, 0.0, False)
    sb = _linear_trig(a, -1.0, True)
    cb = _linear_trig(a, -1.0, False)
    s2b = _linear_trig(2.0 * a, -2.0, True)
    sin_a_minus_2b = _linear_trig(-a, 2.0, True)
    cos_a_minus_2b = _linear_trig(-a, 2.0, False)
    sin_2a_minus_b = _linear_trig(a, 1.0, True)
    cos_2a_minus_b = _linear_trig(a, 1.0, False)

    rho_a = _poly_constant(1.0 - epsilon * np.cos(a), shape)
    rho_b = _poly_constant(1.0, shape) - epsilon * cb
    tau = _poly_constant(0.0, shape)
    tau[1] = 1.0
    tau -= epsilon * (sa - sb)
    A = (
        5.0 * sd + 0.5 * s2d
        + 1.5 * epsilon**2 * (s2a - s2b + 2.0 * sd)
        - 2.0 * epsilon * (
            3.0 * (sa - sb) + sin_a_minus_2b + sin_2a_minus_b
        )
    )
    B = -3.0 * cd + 3.0 * epsilon**2 * _poly_multiply(ca, cb)
    dA_da = (
        5.0 * cd + c2d
        + 1.5 * epsilon**2 * (2.0 * c2a + 2.0 * cd)
        - 2.0 * epsilon * (
            3.0 * ca + cos_a_minus_2b + 2.0 * cos_2a_minus_b
        )
    )
    dB_da = 3.0 * sd - 3.0 * epsilon**2 * _poly_multiply(sa, cb)
    numerator = A + _poly_multiply(B, tau)
    derivative = (
        dA_da + _poly_multiply(dB_da, tau)
        + _poly_multiply(B, rho_a)
    )
    cancellation = (
        _poly_multiply(derivative, rho_a)
        - numerator * (epsilon * np.sin(a))
    ) / rho_a[0] ** 2 - 3.0 * _poly_multiply(rho_a, rho_b)
    denominator = _poly_power(tau, 4)

    # Both series begin at d^4.  Divide only after factoring it exactly.
    numerator4 = cancellation[4:]
    denominator4 = denominator[4:]
    quotient = np.zeros_like(numerator4)
    for degree in range(len(quotient)):
        quotient[degree] = numerator4[degree]
        for index in range(degree):
            quotient[degree] -= (
                quotient[index] * denominator4[degree - index]
            )
        quotient[degree] /= denominator4[0]
    return quotient


def nieuwenhuizen_gain_function(kappa: float, order: int = 96) -> float:
    """Evaluate Nieuwenhuizen's smooth ``f(kappa)`` from Eq. (2.29).

    A tangent map concentrates the outer integral at eccentric perihelia and a
    scaled separation map resolves the causal history.  The exactly cancelled
    short-time series above is used where direct subtraction is ill-conditioned.
    The analytic endpoint values are ``f(0)=Lc`` and ``f(1)=1/2``.
    """
    kappa = _finite(kappa, "kappa")
    if not 0.0 <= kappa <= 1.0:
        raise ValueError("kappa must lie in [0, 1]")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 16:
        raise ValueError("order must be an integer >= 16")
    if kappa == 0.0:
        return critical_angular_momentum()
    # The transformed finite-kappa formula becomes an ill-conditioned
    # subtraction before its O(kappa^2) correction is numerically resolvable.
    # In that controlled boundary layer use the independently integrated
    # endpoint.  The tournament's smallest preregistered kappa is > 0.06.
    if kappa < 0.02:
        return critical_angular_momentum()
    if kappa == 1.0:
        return 0.5

    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    nodes = nodes.astype(_LD)
    weights = weights.astype(_LD)
    kap = _LD(kappa)
    epsilon = np.sqrt(1.0 - kap**2)

    u = np.tan(np.pi * nodes / 2.0)
    a = 2.0 * np.arctan(kap * u)
    da = np.pi * kap * (1.0 + u**2) / (1.0 + (kap * u) ** 2)
    history = (1.0 + nodes) / (1.0 - nodes)
    separation = kap * history
    dd = separation[None, :]
    db = kap * 2.0 / (1.0 - nodes) ** 2
    aa = a[:, None]
    bb = aa - dd
    rho_a = 1.0 - epsilon * np.cos(aa)
    rho_b = 1.0 - epsilon * np.cos(bb)
    tau = dd - epsilon * (np.sin(aa) - np.sin(bb))

    A = (
        5.0 * np.sin(dd) + 0.5 * np.sin(2.0 * dd)
        + 1.5 * epsilon**2 * (
            np.sin(2.0 * aa) - np.sin(2.0 * bb) + 2.0 * np.sin(dd)
        )
        - 2.0 * epsilon * (
            3.0 * (np.sin(aa) - np.sin(bb))
            + np.sin(aa - 2.0 * bb) + np.sin(2.0 * aa - bb)
        )
    )
    B = -3.0 * np.cos(dd) + 3.0 * epsilon**2 * np.cos(aa) * np.cos(bb)
    dA_da = (
        5.0 * np.cos(dd) + np.cos(2.0 * dd)
        + 1.5 * epsilon**2 * (
            2.0 * np.cos(2.0 * aa) + 2.0 * np.cos(dd)
        )
        - 2.0 * epsilon * (
            3.0 * np.cos(aa) + np.cos(aa - 2.0 * bb)
            + 2.0 * np.cos(2.0 * aa - bb)
        )
    )
    dB_da = 3.0 * np.sin(dd) - 3.0 * epsilon**2 * np.sin(aa) * np.cos(bb)
    numerator = A + B * tau
    derivative = dA_da + dB_da * tau + B * rho_a
    ratio = (
        (derivative * rho_a - numerator * epsilon * np.sin(aa))
        / rho_a**2 - 3.0 * rho_a * rho_b
    ) / tau**4

    series = _small_separation_drift_ratio(a, epsilon)
    stable_ratio = _evaluate_series(series, separation)
    cutoff = min(0.03, 0.2 * kappa)
    ratio[:, separation < cutoff] = stable_ratio[:, separation < cutoff]

    integral = np.sum(
        weights[:, None] * weights[None, :]
        * da[:, None] * db[None, :] * ratio
    )
    prefactor = 6.0 * kap**6 / (np.pi**2 * (3.0 - kap**2))
    return float(prefactor * integral)


def _drift_channels(
    state: OrbitState, *, order: int = 96, per_orbit: bool = False,
) -> tuple[float, float]:
    """Return positive ZPF gain and positive radiative-loss magnitude."""
    k = state.k
    kap = state.kappa
    epsilon2 = 1.0 - kap**2
    beta2 = beta_coefficient(state.Z) ** 2
    gain = (
        beta2 * k**9 / (2.0 * kap**6)
        * (2.0 + epsilon2) * nieuwenhuizen_gain_function(kap, order)
    )
    loss = beta2 * k**8 * (3.0 - kap**2) / (2.0 * kap**5)
    if per_orbit:
        gain *= state.period
        loss *= state.period
    return gain, loss


def point_charge_drift(
    energy: float, angular_momentum: float, *, Z: float = 1.0,
    order: int = 96, per_orbit: bool = False,
) -> float:
    """Compute the full point-charge drift ``D(E,L)`` from Eq. (2.34)."""
    state = OrbitState(energy, angular_momentum, Z)
    gain, loss = _drift_channels(state, order=order, per_orbit=per_orbit)
    return gain - loss


def point_charge_channels(
    state: OrbitState, *, order: int = 96, per_orbit: bool = False,
) -> tuple[float, float]:
    """Public positive ``(ZPF gain, radiative loss)`` channel decomposition."""
    return _drift_channels(state, order=order, per_orbit=per_orbit)


def point_charge_drift_surface(
    energies: Iterable[float], angular_momenta: Iterable[float], *,
    Z: float = 1.0, order: int = 96, per_orbit: bool = False,
) -> np.ndarray:
    """Return the complete Cartesian ``D(E,L)`` surface for valid cells."""
    energies = tuple(float(value) for value in energies)
    momenta = tuple(float(value) for value in angular_momenta)
    return np.array([
        [point_charge_drift(E, L, Z=Z, order=order, per_orbit=per_orbit)
         for L in momenta]
        for E in energies
    ])


# -- Preregistered arms -------------------------------------------------------

@dataclass(frozen=True)
class DrivenOrbitResult:
    ledger: EnergyLedger
    initial_hamiltonian: float
    final_hamiltonian: float


@dataclass(frozen=True)
class SetterfieldDrive:
    """Speculative time-dependent Setterfield co-scaling drive.

    With ``m(t)=U(t)^2`` and invariant Coulomb coefficient, the canonical
    Hamiltonian is ``p^2/(2m(t))-1/r``.  Its explicit time derivative is the
    parameter-work channel; omitting it would falsely label active control as
    passive stabilization.
    """

    amplitude: float
    omega_ratio: float
    phase: float

    def scale(self, t: float) -> float:
        return math.exp(self.amplitude * math.sin(self.omega_ratio * t + self.phase))

    def scale_rate(self, t: float) -> float:
        angle = self.omega_ratio * t + self.phase
        return self.scale(t) * self.amplitude * self.omega_ratio * math.cos(angle)

    def mass(self, t: float) -> float:
        return self.scale(t) ** 2

    def mass_rate(self, t: float) -> float:
        return 2.0 * self.scale(t) * self.scale_rate(t)

    def canonical_rhs(
        self, t: float, x: np.ndarray, p: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        p = np.asarray(p, dtype=float)
        radius = np.linalg.norm(x)
        return p / self.mass(t), -x / radius**3

    def external_power(self, t: float, p: np.ndarray) -> float:
        p = np.asarray(p, dtype=float)
        mass = self.mass(t)
        return -self.mass_rate(t) * float(np.dot(p, p)) / (2.0 * mass**2)

    def hamiltonian(self, t: float, x: np.ndarray, p: np.ndarray) -> float:
        radius = float(np.linalg.norm(x))
        return float(np.dot(p, p)) / (2.0 * self.mass(t)) - 1.0 / radius

    def integrate_canonical(
        self, state: OrbitState, *, periods: float = 1.0, steps: int = 16_384,
    ) -> DrivenOrbitResult:
        """Integrate the explicitly time-dependent canonical Hamiltonian.

        Parameter work is integrated as a seventh state variable with the same
        fourth-order Runge--Kutta stages, so the energy ledger tests the derived
        ``partial H/partial t`` rather than a posteriori finite differences.
        """
        if periods <= 0.0 or steps < 2:
            raise ValueError("periods must be positive and steps >= 2")
        x0, p0 = state.perihelion_phase_point()
        y = np.concatenate((x0, p0, np.array([0.0])))
        duration = periods * state.period
        dt = duration / int(steps)

        def rhs(t: float, value: np.ndarray) -> np.ndarray:
            xdot, pdot = self.canonical_rhs(t, value[:3], value[3:6])
            power = self.external_power(t, value[3:6])
            return np.concatenate((xdot, pdot, np.array([power])))

        initial_hamiltonian = self.hamiltonian(0.0, x0, p0)
        t = 0.0
        for _ in range(int(steps)):
            k1 = rhs(t, y)
            k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = rhs(t + dt, y + dt * k3)
            y = y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            t += dt
        final_hamiltonian = self.hamiltonian(t, y[:3], y[3:6])
        delta = final_hamiltonian - initial_hamiltonian
        ledger = EnergyLedger.from_channels(
            mechanical_energy_change=delta,
            zpf_work=0.0,
            radiative_loss=0.0,
            external_parameter_work=float(y[6]),
        )
        return DrivenOrbitResult(ledger, initial_hamiltonian, final_hamiltonian)


@dataclass(frozen=True)
class FiniteShellResponse:
    """Reciprocal spherical-shell response ``sin(kR)/(kR)``."""

    radius: float
    c: float = 1.0 / ALPHA

    def __post_init__(self) -> None:
        if self.radius < 0.0 or not math.isfinite(self.radius):
            raise ValueError("radius must be finite and non-negative")
        if self.c <= 0.0 or not math.isfinite(self.c):
            raise ValueError("c must be finite and positive")

    def form_factor(self, omega):
        omega = np.asarray(omega, dtype=float)
        argument = omega * self.radius / self.c
        result = np.sinc(argument / math.pi)
        return float(result) if result.ndim == 0 else result

    def filter_channel_amplitudes(
        self, omega, absorption_amplitude, radiation_amplitude,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the *same amplitude response* to reciprocal channels."""
        response = self.form_factor(omega)
        return (np.asarray(absorption_amplitude) * response,
                np.asarray(radiation_amplitude) * response)


# The inverse-square H(mu) quadrature is kept private; public callers receive
# only the defining maximum and its convergence-controlled critical d.
def _inverse_square_G(
    mu: float, order: int = 64, *, phase_branch: str = "difference",
    algebra_branch: str = "difference",
) -> float:
    """Numerically evaluate Nieuwenhuizen's gain kernel ``G(mu)`` (Eq. 56)."""
    mu = _finite(mu, "mu")
    if mu < 0.0:
        raise ValueError("mu must be non-negative")
    if order < 16:
        raise ValueError("order must be >= 16")

    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    nodes = nodes.astype(_LD)
    weights = weights.astype(_LD)
    x = np.tan(np.pi * nodes / 2.0)
    dx = np.pi * (1.0 + x**2) / 2.0
    separation = (1.0 + nodes) / (1.0 - nodes)
    dseparation = 2.0 / (1.0 - nodes) ** 2
    X = x[:, None]
    Y = X - separation[None, :]
    m = _LD(mu)
    m2 = m * m
    m4 = m2 * m2
    # The difference form fixes the branch continuously along the causal
    # history.  Using arctan((x-y)/(1+xy)) naively loses pi at a branch cut.
    angle_difference = np.arctan(X) - np.arctan(Y)
    angle_ratio = np.arctan((X - Y) / (1.0 + X * Y))
    branches = {"difference": angle_difference, "principal": angle_ratio}
    if phase_branch not in branches or algebra_branch not in branches:
        raise ValueError("angle branch must be 'difference' or 'principal'")
    phase_angle = branches[phase_branch]
    algebra_angle = branches[algebra_branch]
    phase = 2.0 * m * phase_angle
    sine_over_mu = (
        2.0 * phase_angle if mu == 0.0 else np.sin(phase) / m
    )

    h_c = (
        15.0 + Y**6 * (2.0 * m2 + X**2 - 1.0)
        + 5.0 * Y**4 * (
            X**2 * (-2.0 * m2 * (X**2 + 2.0) + 2.0 * X**2 + 5.0)
            + 1.0
        )
        + 10.0 * m2 * X * (X**2 + 1.0) ** 2 * Y**3
        + 5.0 * Y**2 * (
            X**2 * (-6.0 * m2 * (X**2 + 2.0) + 4.0 * X**2 + 11.0)
            + 1.0
        )
        + 2.0 * X * Y * (
            m2 * (19.0 * X**4 + 30.0 * X**2 - 5.0)
            + 2.0 * (X**6 + 4.0 * X**4 + 5.0 * X**2 + 10.0)
        )
        + 5.0 * X**2 * (
            -2.0 * m2 * (X**4 + X**2 - 1.0) + 2.0 * X**2 + 3.0
        )
        - 10.0 * (m2 - 1.0) * X * (X**2 + 1.0) ** 2
        * (Y**2 + 1.0) ** 2 * algebra_angle
    )
    h_s = (
        4.0 * m2 * X**7 + 10.0 * m2 * X**6 * Y
        + X**5 * (
            8.0 * m4 + m2 * (26.0 - 10.0 * Y**2)
            + 5.0 * (1.0 + Y**2) ** 2
        )
        + 10.0 * X**3 * (
            -2.0 * m2 * (-2.0 + Y**2) + (1.0 + Y**2) ** 2
            + m4 * (1.0 + Y**2) ** 2
        )
        - 10.0 * m2 * X**4 * Y * (-1.0 + m2 * (3.0 + Y**2))
        - 2.0 * m2 * X**2 * Y * (
            -5.0 + 2.0 * Y**4 + 10.0 * m2 * (3.0 + Y**2)
        )
        + 5.0 * X * (
            (1.0 + Y**2) ** 2
            - 2.0 * m2 * (-3.0 + 5.0 * Y**2 + 2.0 * Y**4)
            + 2.0 * m4 * (-1.0 + 6.0 * Y**2 + 3.0 * Y**4)
        )
        - 2.0 * m2 * Y * (
            15.0 - 2.0 * Y**4
            + m2 * (-5.0 + 5.0 * Y**2 + 4.0 * Y**4)
        )
        + 10.0 * m2 * (m2 - 1.0) * (1.0 + X**2) ** 2
        * (1.0 + Y**2) ** 2 * algebra_angle
    )
    h_0 = 5.0 * (Y**2 + 1.0) / (9.0 * (X**2 + 1.0)) * (
        -2.0 * m2 * X**6 - 12.0 * m2 * X**4
        + 4.0 * (m2 - 1.0) * X**3 * Y * (Y**2 + 3.0)
        - 18.0 * m2 * X**2
        + 12.0 * (m2 - 1.0) * X * Y * (Y**2 + 3.0)
        - 2.0 * (m2 - 1.0) * Y**2 * (Y**2 + 3.0) ** 2
        + 2.0 * X**6 + 12.0 * X**4 + 18.0 * X**2
        - 27.0 * (X**2 + 1.0) ** 4
    )
    denominator = (
        5.0 / (2.0**2 * 3.0**4) * (1.0 + X**2) ** 2
        * (3.0 * X + X**3 - (3.0 * Y + Y**3)) ** 4
    )
    regulator = (
        64.0 * (1.0 - m2) * X / (3.0 * (1.0 + X**2) ** 5)
        / ((X - Y) * (X - Y + 1.0))
    )
    integrand = (
        (h_c * np.cos(phase) + h_s * sine_over_mu + h_0) / denominator
        + regulator
    )
    G = 3.0 / np.pi**2 * np.sum(
        weights[:, None] * weights[None, :]
        * dx[:, None] * dseparation[None, :] * integrand
    )
    return float(G)


def _inverse_square_H(
    mu: float, order: int = 64, *, phase_branch: str = "difference",
    algebra_branch: str = "difference",
) -> float:
    """Evaluate the repulsive-branch ``H(mu)`` in published Eq. (64)."""
    mu = _finite(mu, "mu")
    if not 0.0 <= mu <= 1.0:
        raise ValueError("mu must lie in [0, 1]")
    if mu == 1.0:
        return 0.0
    G = _inverse_square_G(
        mu, order, phase_branch=phase_branch, algebra_branch=algebra_branch,
    )
    polynomial = 7.0 - 30.0 * mu**2 + 35.0 * mu**4
    return 8.0 * math.sqrt(1.0 - mu**2) * G / polynomial


def inverse_square_gain_function(mu: float, order: int = 64) -> float:
    """Public evaluator for the printed inverse-square gain kernel ``G(mu)``."""
    return _inverse_square_G(mu, order)


@dataclass(frozen=True)
class InverseSquareControl:
    """Control potential ``V=-1/r-d/(2r^2)`` from Nieuwenhuizen Sec. 3."""

    d: float

    def potential(self, x) -> float:
        radius = float(np.linalg.norm(x))
        return -1.0 / radius - self.d / (2.0 * radius**2)

    def force(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        radius = np.linalg.norm(x)
        return -x / radius**3 - self.d * x / radius**4

    def drift_channels(
        self, state: OrbitState, *, order: int = 64,
    ) -> tuple[float, float]:
        """Near-ionization per-orbit channels from published Eqs. (54)--(56).

        The exact ``d=0`` branch recovers the complete finite-energy baseline,
        making the null limit stronger than an asymptotic comparison.
        """
        if self.d == 0.0:
            return point_charge_channels(state, order=order, per_orbit=True)
        effective_squared = state.angular_momentum**2 - self.d
        if effective_squared <= 0.0:
            raise ValueError("L^2-d must be positive for the resolved orbit family")
        effective = math.sqrt(effective_squared)
        kbar = 1.0 / effective
        mu = state.angular_momentum / effective
        G = _inverse_square_G(mu, order)
        beta2 = beta_coefficient(state.Z) ** 2
        gain = 2.0 * math.pi * beta2 * kbar**6 * G
        polynomial = 7.0 - 30.0 * mu**2 + 35.0 * mu**4
        loss = 2.0 * math.pi * beta2 * kbar**5 * polynomial / 8.0
        return gain, loss

    @classmethod
    def critical_d(cls, order: int = 64) -> tuple[float, float, float]:
        """Calculate ``d_c=-[max_{0<=mu<=1} H(mu)]^2``.

        No value from the paper's contradictory prose is inserted.  A nested
        deterministic grid evaluates the defining maximum, including both
        endpoints.
        """
        grid = np.linspace(0.0, 1.0, 65)
        values = np.array([_inverse_square_H(float(mu), order) for mu in grid])
        index = int(np.argmax(values))
        left = float(grid[max(0, index - 1)])
        right = float(grid[min(len(grid) - 1, index + 1)])
        # Golden-section maximization refines the grid result without importing
        # an optimizer or inserting the paper's rounded endpoint value.
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        x1 = right - ratio * (right - left)
        x2 = left + ratio * (right - left)
        f1 = _inverse_square_H(x1, order)
        f2 = _inverse_square_H(x2, order)
        for _ in range(24):
            if f1 < f2:
                left, x1, f1 = x1, x2, f2
                x2 = left + ratio * (right - left)
                f2 = _inverse_square_H(x2, order)
            else:
                right, x2, f2 = x2, x1, f1
                x1 = right - ratio * (right - left)
                f1 = _inverse_square_H(x1, order)
        mu_max = 0.5 * (left + right)
        hmax = _inverse_square_H(mu_max, order)
        return -hmax**2, mu_max, hmax


@dataclass(frozen=True)
class ConservativeMultipoleResult:
    ledger: EnergyLedger
    relative_total_energy_error: float
    initial_total_energy: float
    final_total_energy: float


@dataclass(frozen=True)
class MultipoleStorage:
    """Rodriguez-inspired Hamiltonian surrogate, not a validated proton model."""

    omega_ratio: float
    eta: float
    mode_mass: float = 1.0

    def __post_init__(self) -> None:
        if self.omega_ratio <= 0.0 or self.mode_mass <= 0.0:
            raise ValueError("mode frequency and mass must be positive")
        if self.eta < 0.0:
            raise ValueError("eta must be non-negative")

    @property
    def coupling(self) -> float:
        return self.eta

    def _energies(
        self, x: np.ndarray, p: np.ndarray, Q: float, P_Q: float,
    ) -> tuple[float, float, float, float]:
        radius = np.linalg.norm(x)
        mechanical = 0.5 * float(np.dot(p, p)) - 1.0 / radius
        internal = (
            P_Q**2 / (2.0 * self.mode_mass)
            + 0.5 * self.mode_mass * self.omega_ratio**2 * Q**2
        )
        coupling = self.coupling * Q / radius**3
        return mechanical, internal, coupling, mechanical + internal + coupling

    def integrate_conservative(
        self, state: OrbitState, *, periods: float = 1.0, steps: int = 20_000,
        Q0: float = 0.0, P_Q0: float = 0.0,
    ) -> ConservativeMultipoleResult:
        """Velocity-Verlet integration of the closed surrogate Hamiltonian."""
        if periods <= 0.0 or steps < 2:
            raise ValueError("periods must be positive and steps >= 2")
        x, p = state.perihelion_phase_point()
        Q = float(Q0)
        P_Q = float(P_Q0)
        dt = periods * state.period / int(steps)
        initial = self._energies(x, p, Q, P_Q)

        def forces(position: np.ndarray, mode: float):
            radius = np.linalg.norm(position)
            particle_force = (
                -position / radius**3
                + 3.0 * self.coupling * mode * position / radius**5
            )
            mode_force = (
                -self.mode_mass * self.omega_ratio**2 * mode
                - self.coupling / radius**3
            )
            return particle_force, mode_force

        def verlet(substep: float, position, momentum, mode, mode_momentum):
            force, force_Q = forces(position, mode)
            momentum_half = momentum + 0.5 * substep * force
            mode_momentum_half = mode_momentum + 0.5 * substep * force_Q
            position = position + substep * momentum_half
            mode = mode + substep * mode_momentum_half / self.mode_mass
            force, force_Q = forces(position, mode)
            momentum = momentum_half + 0.5 * substep * force
            mode_momentum = mode_momentum_half + 0.5 * substep * force_Q
            return position, momentum, mode, mode_momentum

        # Fourth-order Forest--Ruth/Yoshida composition of the reversible
        # velocity-Verlet map.  The closed Hamiltonian test is stringent near
        # eccentric perihelia, where a single second-order step is inadequate.
        cube_root_two = 2.0 ** (1.0 / 3.0)
        w1 = 1.0 / (2.0 - cube_root_two)
        w0 = -cube_root_two / (2.0 - cube_root_two)
        for _ in range(int(steps)):
            for coefficient in (w1, w0, w1):
                x, p, Q, P_Q = verlet(
                    coefficient * dt, x, p, Q, P_Q,
                )

        final = self._energies(x, p, Q, P_Q)
        delta_mechanical = final[0] - initial[0]
        internal_exchange = -(
            (final[1] + final[2]) - (initial[1] + initial[2])
        )
        ledger = EnergyLedger.from_channels(
            delta_mechanical, 0.0, 0.0,
            internal_mode_exchange=internal_exchange,
        )
        total_error = final[3] - initial[3]
        relative_error = abs(total_error) / max(abs(initial[3]), 1e-300)
        return ConservativeMultipoleResult(
            ledger, relative_error, initial[3], final[3],
        )


# -- Finite-mode stochastic response validation ------------------------------

def _confidence_interval(samples: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(samples))
    if len(samples) < 2:
        return mean, mean, mean
    stderr = float(np.std(samples, ddof=1) / math.sqrt(len(samples)))
    return mean, mean - 1.96 * stderr, mean + 1.96 * stderr


def _discrete_phase_response(
    omega: np.ndarray, phases: np.ndarray, timestep: float, duration: float,
) -> np.ndarray:
    """Finite-time discrete mean of ``2 cos²(omega*t+phase)``.

    The geometric sum is evaluated in closed form, so millions of physical
    perihelion-resolution timesteps do not become a Python loop.  Unlike a
    single phase draw, this estimator genuinely changes when ``dt`` is halved.
    """
    n_steps = max(2, int(math.ceil(duration / timestep)))
    dt = duration / n_steps
    theta = 2.0 * np.asarray(omega, dtype=float) * dt
    amplitude = np.sinc(n_steps * theta / (2.0 * math.pi)) / np.sinc(
        theta / (2.0 * math.pi)
    )
    midpoint_phase = theta * (n_steps - 1) / 2.0
    return 1.0 + amplitude * np.cos(2.0 * phases + midpoint_phase)


def stochastic_cell(
    state: OrbitState, *, coupling_scale: float, config: TournamentConfig,
    arm: str = "PointCharge", response: FiniteShellResponse | None = None,
    channels: tuple[float, float] | None = None,
) -> HypothesisResult:
    """Validate one perturbative cell with nested random-phase mode sets.

    Each finite-mode realization evaluates the discrete finite-time mean of the
    quadratic response-kernel factor ``2 cos(omega*t+phi)^2``. It has unit phase
    expectation and converges to the analytic ZPF gain. The same seed produces
    nested phases as the mode count doubles.
    This is a stochastic quadrature of the PR1/PR2 response calculation, not a
    replacement for nonlinear long-time SED dynamics.
    """
    coupling_scale = _finite(coupling_scale, "coupling_scale")
    if coupling_scale <= 0.0:
        raise ValueError("coupling_scale must be positive")
    gain, loss = (
        point_charge_channels(state, order=64, per_orbit=True)
        if channels is None else (float(channels[0]), float(channels[1]))
    )
    if gain < 0.0 or loss < 0.0:
        raise ValueError("gain and loss channels must be non-negative")
    scale2 = coupling_scale**2
    resolution_means: list[float] = []
    final_samples = None
    final_gain_samples = None
    resolutions: list[tuple[int, float]] = []
    converged = False
    final_loss = loss * scale2

    for level in range(config.max_resolution_levels):
        modes = config.n_modes * 2**level
        timestep = config.timestep / 2**level
        resolutions.append((modes, timestep))
        gain_samples = []
        # Deterministic midpoint bins in the causal response band.  Both the
        # field quadrature and optional shell response use this same grid.
        omega = np.linspace(0.0, 16.0 * state.k**3, modes,
                            endpoint=False) + 8.0 * state.k**3 / modes
        response_power = np.ones(modes)
        if response is not None:
            # Equal-weight frequency bins span the orbit's perturbative
            # response band.  The identical F^2 multiplier below is the
            # reciprocity gate for absorbed and emitted power.
            response_power = response.form_factor(omega) ** 2
        level_loss = loss * scale2 * float(np.mean(response_power))
        for seed in config.seeds:
            phases = np.random.default_rng(seed).uniform(0.0, 2.0 * math.pi,
                                                          modes)
            factors = _discrete_phase_response(
                omega, phases, timestep, state.period,
            )
            factors *= response_power
            gain_samples.append(gain * scale2 * float(np.mean(factors)))
        gain_samples = np.asarray(gain_samples)
        samples = gain_samples - level_loss
        normalized_mean = float(np.mean(samples) / scale2)
        resolution_means.append(normalized_mean)
        final_samples = samples
        final_gain_samples = gain_samples
        final_loss = level_loss
        if level:
            reference = max(
                abs(resolution_means[-1]), abs(resolution_means[-2]),
                abs(gain), abs(loss), 1e-300,
            )
            converged = (
                abs(resolution_means[-1] - resolution_means[-2]) / reference
                < config.convergence_rtol
            )
            if converged:
                break

    assert final_samples is not None and final_gain_samples is not None
    mean, low, high = _confidence_interval(final_samples)
    ledger = EnergyLedger.from_channels(
        mechanical_energy_change=mean,
        zpf_work=float(np.mean(final_gain_samples)),
        radiative_loss=final_loss,
    )
    return HypothesisResult.cell(
        arm, state.energy, state.angular_momentum, low, high, ledger,
        mean_drift=mean, converged=converged,
        classification=NO_EFFECT if converged else NULL,
        coupling_scale=coupling_scale,
        seeds=config.seeds, resolutions=resolutions,
        resolution_drifts=resolution_means,
    )


def classify_hypothesis(
    results: Sequence[HypothesisResult], *, low_l_cells: Sequence[float],
    coupling_converged: bool, closure_limit: float = 0.01,
    baseline_results: Sequence[HypothesisResult] | None = None,
) -> str:
    """Apply the preregistered closed-vocabulary classification gate."""
    if not results:
        return NULL
    target_L = tuple(float(value) for value in low_l_cells)
    selected = [
        result for result in results
        if any(math.isclose(result.angular_momentum, value,
                            rel_tol=0.0, abs_tol=1e-12) for value in target_L)
    ]
    covered = {
        value for value in target_L
        if any(math.isclose(result.angular_momentum, value,
                            rel_tol=0.0, abs_tol=1e-12) for result in selected)
    }
    expected_coordinates = {
        (result.energy, result.coupling_scale) for result in results
    }
    complete = all(
        expected_coordinates <= {
            (result.energy, result.coupling_scale) for result in selected
            if math.isclose(result.angular_momentum, value,
                            rel_tol=0.0, abs_tol=1e-12)
        }
        for value in target_L
    )
    if (
        len(covered) != len(target_L) or not complete
        or any(not result.converged for result in selected)
        or any(result.ledger.relative_closure >= closure_limit for result in selected)
    ):
        return NULL
    all_negative = all(result.confidence_high < 0.0 for result in selected)
    if any(
        result.confidence_high < 0.0
        and abs(result.ledger.external_parameter_work) > 1e-12
        for result in selected
    ):
        return ACTIVE_CONTROL
    if not coupling_converged:
        return NULL
    if all_negative:
        return CHANNEL_SUPPRESSED
    if baseline_results is not None:
        baseline = {
            (result.energy, result.angular_momentum, result.coupling_scale): result
            for result in baseline_results
        }
        comparisons = []
        for result in selected:
            key = (result.energy, result.angular_momentum,
                   result.coupling_scale)
            if key not in baseline:
                return NULL
            comparisons.append((result, baseline[key]))
        if any(
            not reference.converged
            or reference.ledger.relative_closure >= closure_limit
            for _, reference in comparisons
        ):
            return NULL
        if any(
            result.confidence_low > reference.confidence_high
            for result, reference in comparisons
        ):
            return DESTABILIZED
        return NO_EFFECT
    if all(result.confidence_low > 0.0 for result in selected):
        return DESTABILIZED
    return NO_EFFECT


def _combine_ledgers(
    stochastic: HypothesisResult, deterministic: EnergyLedger,
) -> HypothesisResult:
    """Add a deterministic arm exchange to a stochastic baseline cell."""
    shift = deterministic.mechanical_energy_change
    ledger = EnergyLedger.from_channels(
        mechanical_energy_change=(
            stochastic.ledger.mechanical_energy_change + shift
        ),
        zpf_work=stochastic.ledger.zpf_work + deterministic.zpf_work,
        radiative_loss=(
            stochastic.ledger.radiative_loss + deterministic.radiative_loss
        ),
        schott_boundary_energy=(
            stochastic.ledger.schott_boundary_energy
            + deterministic.schott_boundary_energy
        ),
        external_parameter_work=(
            stochastic.ledger.external_parameter_work
            + deterministic.external_parameter_work
        ),
        internal_mode_exchange=(
            stochastic.ledger.internal_mode_exchange
            + deterministic.internal_mode_exchange
        ),
    )
    converged = (
        stochastic.converged and deterministic.relative_closure < 1e-6
    )
    return HypothesisResult.cell(
        stochastic.arm, stochastic.energy, stochastic.angular_momentum,
        stochastic.confidence_low + shift,
        stochastic.confidence_high + shift,
        ledger,
        mean_drift=stochastic.mean_drift + shift,
        converged=converged,
        classification=NO_EFFECT if converged else NULL,
        coupling_scale=stochastic.coupling_scale,
        seeds=stochastic.seeds,
        resolutions=stochastic.resolutions,
        resolution_drifts=stochastic.resolution_drifts,
    )


def _coupling_convergence(
    results: Sequence[HypothesisResult], scales: Sequence[float],
    tolerance: float,
) -> bool:
    if len(scales) < 2:
        return True
    grouped: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for result in results:
        key = (result.energy, result.angular_momentum)
        grouped.setdefault(key, []).append((
            result.coupling_scale,
            result.mean_drift / result.coupling_scale**2,
        ))
    for values in grouped.values():
        if {value[0] for value in values} != {float(scale) for scale in scales}:
            return False
        normalized = np.array([value for _, value in values])
        scale = max(float(np.max(np.abs(normalized))), 1e-300)
        if float(np.ptp(normalized)) / scale >= tolerance:
            return False
    return True


def run_hypothesis_arm(
    arm: str, parameters: dict, *, config: TournamentConfig | None = None,
    quadrature_order: int = 64,
) -> dict:
    """Run one parameterization of a preregistered arm.

    Full preregistered sweeps call this function once per parameter tuple.  The
    return value is composed only of JSON-native values and contains every seed,
    resolution, confidence interval, and energy ledger required to reproduce a
    classification.
    """
    config = TournamentConfig() if config is None else config
    aliases = {
        "setterfielddrive": "setterfield",
        "finiteshellresponse": "finite_shell",
        "inversesquarecontrol": "inverse_square",
        "multipolestorage": "multipole",
    }
    normalized_arm = aliases.get(arm.lower().replace("_", ""), arm.lower())
    valid_arms = {"setterfield", "finite_shell", "inverse_square", "multipole"}
    if normalized_arm not in valid_arms:
        raise ValueError(f"arm must be one of {sorted(valid_arms)}")
    parameters = dict(parameters)

    setterfield = None
    shell = None
    inverse = None
    multipole = None
    integration_steps = int(parameters.pop("integration_steps", 16_384))
    integration_periods = float(parameters.pop("integration_periods", 1.0))
    if normalized_arm == "setterfield":
        setterfield = SetterfieldDrive(**parameters)
    elif normalized_arm == "finite_shell":
        shell = FiniteShellResponse(**parameters)
    elif normalized_arm == "inverse_square":
        inverse = InverseSquareControl(**parameters)
    else:
        multipole = MultipoleStorage(**parameters)

    cell_results: list[HypothesisResult] = []
    baseline_results: list[HypothesisResult] = []
    for energy in config.energies:
        for angular_momentum in config.angular_momenta:
            state = OrbitState(energy, angular_momentum)
            point_channels = point_charge_channels(
                state, order=quadrature_order, per_orbit=True,
            )
            baseline_channels = point_channels
            deterministic = None
            if setterfield is not None:
                deterministic = (
                    EnergyLedger.from_channels(0.0, 0.0, 0.0)
                    if setterfield.amplitude == 0.0 else
                    setterfield.integrate_canonical(
                        state, periods=integration_periods,
                        steps=integration_steps,
                    ).ledger
                )
            elif inverse is not None:
                baseline_channels = inverse.drift_channels(
                    state, order=quadrature_order,
                )
            elif multipole is not None:
                deterministic = (
                    EnergyLedger.from_channels(0.0, 0.0, 0.0)
                    if multipole.eta == 0.0 else
                    multipole.integrate_conservative(
                        state, periods=integration_periods,
                        steps=integration_steps,
                    ).ledger
                )

            for coupling_scale in config.coupling_scales:
                baseline_results.append(stochastic_cell(
                    state, coupling_scale=coupling_scale, config=config,
                    arm="point_charge", channels=point_channels,
                ))
                result = stochastic_cell(
                    state, coupling_scale=coupling_scale, config=config,
                    arm=normalized_arm, response=shell,
                    channels=baseline_channels,
                )
                if deterministic is not None:
                    result = _combine_ledgers(result, deterministic)
                cell_results.append(result)

    coupling_converged = _coupling_convergence(
        cell_results, config.coupling_scales, config.convergence_rtol,
    )
    low_l = tuple(
        value for value in config.angular_momenta
        if value < critical_angular_momentum()
    )
    classification = classify_hypothesis(
        cell_results, low_l_cells=low_l,
        coupling_converged=coupling_converged,
        baseline_results=baseline_results,
    )
    report = {
        "schema": "blueberry-circus/hypothesis-arm/v1",
        "implementation_version": TOURNAMENT_SCHEMA_VERSION,
        "arm": normalized_arm,
        "parameters": json.loads(json.dumps(parameters, allow_nan=False)),
        "quadrature_order": quadrature_order,
        "response_grid": {
            "omega_over_orbital_frequency": [0.0, 16.0],
            "rule": "equal-width midpoint bins",
            "time_rule": "exact geometric sum on the recorded timestep",
        },
        "integration": {
            "periods": integration_periods,
            "steps": integration_steps,
        } if normalized_arm in {"setterfield", "multipole"} else None,
        "classification": classification,
        "coupling_converged": coupling_converged,
        "config": json.loads(config.to_json()),
        "baseline_cells": [
            json.loads(result.to_json()) for result in baseline_results
        ],
        "cells": [json.loads(result.to_json()) for result in cell_results],
    }
    if inverse is not None:
        dc, mu_max, hmax = inverse.critical_d(order=quadrature_order)
        report["critical_d"] = {
            "calculated": dc,
            "mu_at_max": mu_max,
            "H_max": hmax,
            "published_prose": -35.8,
            "published_endpoint_H0": _inverse_square_H(0.0, quadrature_order),
        }
    return report


def preregistered_parameters(
    arm: str, config: TournamentConfig | None = None,
) -> tuple[dict, ...]:
    """Return the immutable parameter grid for one named hypothesis arm."""
    config = TournamentConfig() if config is None else config
    arm = arm.lower()
    if arm == "setterfield":
        return tuple({
            "amplitude": amplitude,
            "omega_ratio": omega,
            "phase": phase,
        } for amplitude, omega, phase in itertools.product(
            config.setterfield_amplitudes,
            config.setterfield_omega_ratios,
            config.setterfield_phases,
        ))
    if arm == "finite_shell":
        # R=0 is a recovery control, not one of the five hypothesis radii.
        return tuple({"radius": radius} for radius in (0.0,) + config.shell_radii)
    if arm == "inverse_square":
        return tuple({"d": value} for value in config.inverse_square_d)
    if arm == "multipole":
        return tuple({
            "omega_ratio": omega,
            "eta": eta,
        } for omega, eta in itertools.product(
            config.multipole_omega_ratios, config.multipole_eta,
        ))
    raise ValueError("unknown hypothesis arm")


def point_charge_surface_report(
    config: TournamentConfig, *, quadrature_order: int = 64,
) -> dict:
    """Machine-readable baseline that is always computed before any arm."""
    surface = point_charge_drift_surface(
        config.energies, config.angular_momenta, order=quadrature_order,
        per_orbit=True,
    )
    lc = critical_angular_momentum()
    probe_energy = -1e-3
    probe_order = max(128, quadrature_order)
    probes = []
    for angular_momentum in (lc - 0.01, lc + 0.01):
        finite = point_charge_drift(
            # The small-kappa subtraction is more demanding than the
            # preregistered surface, so certify this asymptotic probe at a
            # separately recorded higher order.
            probe_energy, angular_momentum, order=probe_order,
            per_orbit=True,
        )
        asymptotic = (
            3.0 * math.pi * beta_coefficient() ** 2
            * (lc - angular_momentum) / angular_momentum**6
        )
        probes.append({
            "energy": probe_energy,
            "quadrature_order": probe_order,
            "angular_momentum": angular_momentum,
            "finite_drift": finite,
            "asymptotic_drift": asymptotic,
            "relative_residual": abs(finite - asymptotic)
            / max(abs(asymptotic), 1e-300),
        })
    return {
        "equation": "Nieuwenhuizen 2016 Eq. (34), per revolution",
        "quadrature_order": quadrature_order,
        "energies": list(config.energies),
        "angular_momenta": list(config.angular_momenta),
        "drift": surface.tolist(),
        "near_ionization_probes": probes,
    }


def run_tournament(
    *, config: TournamentConfig | None = None,
    arms: Sequence[str] = (
        "setterfield", "finite_shell", "inverse_square", "multipole",
    ),
    parameter_index: int | None = None,
    quadrature_order: int = 64,
    integration_steps: int | None = None,
    integration_periods: float | None = None,
) -> dict:
    """Run selected parameter chunks after computing the baseline surface."""
    config = TournamentConfig() if config is None else config
    report = {
        "schema": "blueberry-circus/hypothesis-tournament/v1",
        "implementation_version": TOURNAMENT_SCHEMA_VERSION,
        "claim_scope": (
            "conditional perturbative energy drift; not a stationary ground "
            "state and not equilibrium vacuum-energy extraction"
        ),
        "config": json.loads(config.to_json()),
        "point_charge_baseline": point_charge_surface_report(
            config, quadrature_order=quadrature_order,
        ),
        "runs": [],
    }
    for arm in arms:
        grid = preregistered_parameters(arm, config)
        if parameter_index is not None:
            if not 0 <= parameter_index < len(grid):
                raise IndexError(
                    f"parameter index {parameter_index} outside {arm} grid "
                    f"of length {len(grid)}"
                )
            grid = (grid[parameter_index],)
        for parameters in grid:
            parameters = dict(parameters)
            if arm in {"setterfield", "multipole"}:
                if integration_steps is not None:
                    parameters["integration_steps"] = integration_steps
                if integration_periods is not None:
                    parameters["integration_periods"] = integration_periods
            report["runs"].append(run_hypothesis_arm(
                arm, parameters, config=config,
                quadrature_order=quadrature_order,
            ))
    return report


__all__ = [
    "ACTIVE_CONTROL", "CHANNEL_SUPPRESSED", "CLASSIFICATIONS",
    "DESTABILIZED", "NO_EFFECT", "NULL", "OrbitState", "EnergyLedger",
    "TournamentConfig", "HypothesisResult", "SetterfieldDrive",
    "FiniteShellResponse", "InverseSquareControl", "MultipoleStorage",
    "ConservativeMultipoleResult", "DrivenOrbitResult",
    "nieuwenhuizen_gain_function", "inverse_square_gain_function",
    "point_charge_channels",
    "point_charge_drift", "point_charge_drift_surface", "stochastic_cell",
    "TOURNAMENT_SCHEMA_VERSION", "classify_hypothesis",
    "preregistered_parameters",
    "point_charge_surface_report", "run_hypothesis_arm", "run_tournament",
]
