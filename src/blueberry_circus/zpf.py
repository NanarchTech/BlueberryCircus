"""The classical zero-point electromagnetic background.

A :class:`ZPFBackground` is a finite random-phase plane-wave realization of the
classical zero-point radiation field. Each plane-wave component has a frequency
``omega``, a propagation direction ``khat`` (so ``k = (omega/c) khat``), a
transverse polarization unit vector ``e`` (``e . khat = 0``), an amplitude fixed
by the ZPF spectral density, and an independent uniform random phase:

    E(r,t) = sum_m  a_m e_m cos(k_m . r - omega_m t + phi_m)
    B(r,t) = sum_m  (a_m / c) (khat_m x e_m) cos(k_m . r - omega_m t + phi_m)

Two constructors are provided:

* :meth:`isotropic_3d` -- physical 3-D isotropic background: random directions on
  the sphere, two transverse polarizations per direction. Per-component amplitude
  ``a = sqrt(rho(omega)/eps0 * dω)`` so that, after isotropic averaging,
  ``<E_x^2> = integral S_Ex dω`` and each mode carries ``(1/2) hbar omega``.
* :meth:`one_dimensional` -- a band of modes all polarized along one axis,
  amplitude ``a = sqrt(2 S_Ex dω)``; reproduces ``S_Ex`` along that axis with no
  direction-averaging variance. Used for the exactly-solvable oscillator oracle.

The time derivative ``dEdt`` is analytic (``d/dt cos = omega sin``), which keeps
the Runge--Kutta radiation-reaction term exact rather than finite-differenced.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .constants import Units, SI
from .spectrum import spectral_density_Ex, rho


def _transverse_pair(khat: np.ndarray):
    """Return two orthonormal vectors spanning the plane transverse to khat."""
    a = np.array([1.0, 0.0, 0.0]) if abs(khat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = a - np.dot(a, khat) * khat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(khat, e1)
    return e1, e2


@dataclass
class ZPFBackground:
    omegas: np.ndarray      # (M,) angular frequency of each plane-wave component
    kvecs: np.ndarray       # (M,3) wavevector  k = (omega/c) khat
    evecs: np.ndarray       # (M,3) polarization unit vector (transverse)
    amps: np.ndarray        # (M,) amplitude
    phases: np.ndarray      # (M,) phase
    units: Units = SI

    # ----- constructors -------------------------------------------------------
    @classmethod
    def one_dimensional(cls, omega_lo: float, omega_hi: float, n_modes: int,
                        seed: int = 0, units: Units = SI, axis: int = 0,
                        log_spaced: bool = False) -> "ZPFBackground":
        rng = np.random.default_rng(seed)
        if log_spaced:
            edges = np.logspace(np.log10(omega_lo), np.log10(omega_hi), n_modes + 1)
        else:
            edges = np.linspace(omega_lo, omega_hi, n_modes + 1)
        omegas = 0.5 * (edges[:-1] + edges[1:])
        dω = np.diff(edges)
        amps = np.sqrt(2.0 * spectral_density_Ex(omegas, units) * dω)
        evec = np.zeros(3); evec[axis] = 1.0
        evecs = np.tile(evec, (n_modes, 1))
        khat = np.zeros(3); khat[(axis + 1) % 3] = 1.0           # k perp to e
        kvecs = (omegas[:, None] / units.c) * khat[None, :]
        phases = rng.uniform(0, 2 * np.pi, n_modes)
        return cls(omegas, kvecs, evecs, amps, phases, units)

    @classmethod
    def isotropic_3d(cls, omega_lo: float, omega_hi: float, n_modes: int,
                     seed: int = 0, units: Units = SI,
                     log_spaced: bool = False) -> "ZPFBackground":
        rng = np.random.default_rng(seed)
        if log_spaced:
            edges = np.logspace(np.log10(omega_lo), np.log10(omega_hi), n_modes + 1)
        else:
            edges = np.linspace(omega_lo, omega_hi, n_modes + 1)
        omegas = 0.5 * (edges[:-1] + edges[1:])
        dω = np.diff(edges)
        # isotropic directions
        u = rng.uniform(-1, 1, n_modes)
        az = rng.uniform(0, 2 * np.pi, n_modes)
        st = np.sqrt(1 - u**2)
        khat = np.stack([st * np.cos(az), st * np.sin(az), u], axis=1)
        # per-pol amplitude a: (1/2)a^2 over 2 pols = a^2 = rho/eps0 * dω (= <E^2> per mode)
        a = np.sqrt(rho(omegas, units) / units.eps0 * dω)     # per polarization
        O, K, Ev, A, Ph = [], [], [], [], []
        for j in range(n_modes):
            e1, e2 = _transverse_pair(khat[j])
            for e in (e1, e2):
                O.append(omegas[j]); K.append((omegas[j] / units.c) * khat[j])
                Ev.append(e); A.append(a[j])
                Ph.append(rng.uniform(0, 2 * np.pi))
        return cls(np.array(O), np.array(K), np.array(Ev), np.array(A),
                   np.array(Ph), units)

    # ----- field evaluation ---------------------------------------------------
    def _arg(self, r, t):
        # r: (3,), t: scalar or (T,)
        kr = self.kvecs @ np.asarray(r, dtype=float)           # (M,)
        t = np.atleast_1d(np.asarray(t, dtype=float))          # (T,)
        return kr[:, None] - self.omegas[:, None] * t[None, :] + self.phases[:, None]

    def E(self, r, t):
        """Electric field. Returns (3,) for scalar t, else (3,T)."""
        scalar = np.ndim(t) == 0
        ph = self._arg(r, t)                                   # (M,T)
        contrib = (self.amps[:, None] * np.cos(ph))            # (M,T)
        E = self.evecs.T @ contrib                             # (3,T)
        return E[:, 0] if scalar else E

    def dEdt(self, r, t):
        """Analytic time derivative of E."""
        scalar = np.ndim(t) == 0
        ph = self._arg(r, t)
        contrib = (self.amps[:, None] * self.omegas[:, None] * np.sin(ph))
        dE = self.evecs.T @ contrib
        return dE[:, 0] if scalar else dE

    def B(self, r, t):
        """Magnetic field, B = (khat x e) a cos(...) / c."""
        scalar = np.ndim(t) == 0
        khat = self.kvecs / np.maximum(np.linalg.norm(self.kvecs, axis=1, keepdims=True), 1e-300)
        bvec = np.cross(khat, self.evecs) / self.units.c       # (M,3)
        ph = self._arg(r, t)
        contrib = (self.amps[:, None] * np.cos(ph))
        B = bvec.T @ contrib
        return B[:, 0] if scalar else B

    # ----- diagnostics --------------------------------------------------------
    def mean_square_field_components(self):
        """Phase-averaged (<E_x^2>,<E_y^2>,<E_z^2>) over the realization."""
        w = 0.5 * self.amps**2                                  # (M,)
        comp = (self.evecs**2) * w[:, None]                     # (M,3)
        return comp.sum(axis=0)

    def mean_energy_density(self) -> float:
        """Total ZPF energy density eps0 <E^2> (electric + magnetic equal)."""
        mse = self.mean_square_field_components().sum()         # <E^2>
        return float(self.units.eps0 * mse)

    @property
    def n_components(self) -> int:
        return len(self.omegas)
