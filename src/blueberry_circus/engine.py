"""Execution engine: compile a Program, integrate it on a bound Backend.

The engine binds a :class:`~blueberry_circus.backends.Backend` at *construction*
(the PennyLane device model), then ``run`` lowers a :class:`Program` through its
named :meth:`~blueberry_circus.program.Program.compile` passes and hands the
pure-data result to the backend. The op-collection that used to live as an inline
``isinstance`` interpreter is now the fail-closed ``compile()`` pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np

from .dynamics import Particle, Trajectory
from . import observables as obs
from .backends import get_backend
from .certify import Certificate, PASS, finalize


@dataclass
class Result:
    trajectory: Trajectory
    certificates: List[Certificate] = field(default_factory=list)
    observables: dict = field(default_factory=dict)
    compiled_passes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"steps={len(self.trajectory.t)}  "
                 f"t_max={self.trajectory.t[-1]:.4g}"]
        if self.compiled_passes:
            lines.append(f"  passes: {' -> '.join(self.compiled_passes)}")
        for k, v in self.observables.items():
            lines.append(f"  {k}: {v}")
        for c in self.certificates:
            lines.append(f"  [{c.recheck()}] {c.kind}: {c.claim}")
        return "\n".join(lines)

    # --- Gaussian-style phase-space readout (means / covariance) -------------
    def _tail(self, burn_in: float):
        i0 = int(burn_in * len(self.trajectory.t))
        return self.trajectory.x[i0:], self.trajectory.v[i0:]

    def means(self, burn_in: float = 0.3) -> np.ndarray:
        """First moments of the (x, v) phase-space state after burn-in (length 6)."""
        x, v = self._tail(burn_in)
        return np.concatenate([x.mean(axis=0), v.mean(axis=0)])

    def covariance(self, burn_in: float = 0.3) -> np.ndarray:
        """6x6 phase-space covariance of (x, v) after burn-in (the Gaussian readout
        used by the optional Bogoliubov/FDT covariance cross-check)."""
        x, v = self._tail(burn_in)
        z = np.concatenate([x, v], axis=1)             # (T, 6)
        return np.cov(z, rowvar=False)


class Engine:
    def __init__(self, backend="numpy", dt: float = None,
                 t_max: float = None, n_steps: int = None, burn_in: float = 0.3):
        # Bind the backend at construction (not at run); unknown/unbuilt backends
        # raise here (NotImplementedError for planned 'rust'/'jax'/'cuda').
        self.backend = get_backend(backend)
        self.dt = dt
        self.t_max = t_max
        self.n_steps = n_steps
        self.burn_in = burn_in

    def _time_grid(self):
        if self.n_steps is not None and self.t_max is not None:
            return np.linspace(0.0, self.t_max, self.n_steps)
        if self.dt is not None and self.t_max is not None:
            return np.arange(0.0, self.t_max + 0.5 * self.dt, self.dt)
        raise ValueError("specify (dt,t_max) or (n_steps,t_max)")

    def run(self, program: "object", x0, v0, index: int = 0,
            dipole: bool = True, certify: bool = True) -> Result:
        compiled = program.compile(index)             # named, fail-closed passes
        units = compiled.units
        particle = Particle(units.charge, units.mass)
        t_grid = self._time_grid()
        traj = self.backend.integrate(
            field=compiled.field, potential=compiled.potential, particle=particle,
            t_grid=t_grid, x0=x0, v0=v0, rr=compiled.rr, units=units, dipole=dipole)
        observables = {}
        certs: List[Certificate] = []
        finite = bool(np.all(np.isfinite(traj.x)) and np.all(np.isfinite(traj.v)))
        observables["trajectory_finite"] = finite
        if certify:
            c = Certificate(
                kind="trajectory_finite",
                claim="integration produced no NaN/Inf over the window",
                method=f"{self.backend.name} RK4 + Landau-Lifshitz fixed-step integration",
                rule="residual_le_tol",
                # finite (not inf): a diverged-trajectory FAIL cert must stay
                # serializable on the canonical hash surface. 1.0 > 0.0 -> FAIL.
                residual=0.0 if finite else 1.0,
                tolerance=0.0,
                provenance={"index": index, "rr": compiled.rr,
                            "steps": len(t_grid), "backend": self.backend.name})
            certs.append(finalize(c))
        if compiled.potential is not None:
            observables["position_variance"] = obs.position_variance(traj, self.burn_in)
        return Result(trajectory=traj, certificates=certs, observables=observables,
                      compiled_passes=compiled.passes)
