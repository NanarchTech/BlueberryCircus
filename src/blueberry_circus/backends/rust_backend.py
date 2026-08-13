"""The Rust LL-RK4 integrator backend, called over a C ABI via ctypes.

The native library (``rust/blueberry_rs``) is a dependency-free ``cdylib`` that
ports ``dynamics.integrate`` line-for-line. The boundary is pure data: a frozen
ZPF realization + SI scalars in, an ``(n,3)`` trajectory out. NumPy is the
reference oracle; this backend is diffed against it under enclosure tolerance
(1-ULP CPython/Rust divergence is expected — never bit-equality).

Import-or-raise: constructing a :class:`RustBackend` when the library is not built
raises ``NotImplementedError`` with the build command (never a silent fallback to
NumPy).
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

from .base import Backend
from ..dynamics import Trajectory
from ..potentials import Harmonic as _Harmonic, Coulomb as _Coulomb

_C = ctypes.c_double
_PTR = ctypes.POINTER(ctypes.c_double)


def _find_library() -> "str | None":
    here = os.path.dirname(os.path.abspath(__file__))           # .../src/blueberry_circus/backends
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    base = os.path.join(repo_root, "rust", "blueberry_rs", "target", "release")
    for name in ("libblueberry_rs.dylib", "libblueberry_rs.so", "blueberry_rs.dll"):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            return p
    env = os.environ.get("BLUEBERRY_RS_LIB")
    return env if env and os.path.isfile(env) else None


def is_available() -> bool:
    return _find_library() is not None


class RustBackend(Backend):
    name = "rust"

    def __init__(self):
        lib_path = _find_library()
        if lib_path is None:
            raise NotImplementedError(
                "rust backend not built. Build it with:\n"
                "  (cd rust/blueberry_rs && cargo build --release)\n"
                "or run scripts/build_rust.sh, then retry.")
        self._lib = ctypes.CDLL(lib_path)
        self._fn = self._lib.bc_integrate
        self._fn.restype = None
        self._fn.argtypes = [
            ctypes.c_int, _C, _C, _C, _C,          # pot_kind, p0, p1, charge, mass
            ctypes.c_longlong, _PTR,               # n_modes, field
            ctypes.c_longlong, _PTR,               # n, t_grid
            _PTR, _PTR,                            # x0, v0
            ctypes.c_int, ctypes.c_int,            # rr, dipole
            _C, _C,                                # tau, c_light
            _PTR, _PTR,                            # out_x, out_v
        ]

    def integrate(self, *, field, potential, particle, t_grid, x0, v0,
                  rr: str, units, dipole: bool):
        # --- potential params (mirror potentials.{Harmonic,Coulomb}) ---------
        if isinstance(potential, _Harmonic):
            # p1 carries the potential's OWN mass (force = -m_pot omega0^2 x),
            # which is independent of particle mass; the kernel divides by the
            # particle mass separately. Matches NumPy's potentials.Harmonic.force.
            pot_kind, p0, p1 = 0, float(potential.omega0), float(potential.mass)
        elif isinstance(potential, _Coulomb):
            # Use the coefficient frozen on the potential at construction (what
            # NumPy's force/jacobian use), NOT one recomputed from integrate-time
            # units -- otherwise the two backends diverge when the units differ.
            pot_kind, p0, p1 = 1, float(potential._coef), float(potential.softening)
        elif potential is None:
            raise ValueError("rust backend requires a potential")
        else:
            raise TypeError(f"rust backend cannot integrate potential {potential!r}")

        # --- frozen ZPF realization as (M,9) [omega,kx,ky,kz,ex,ey,ez,amp,phase]
        if field is None:
            n_modes = 0
            field_arr = np.zeros(0, dtype=np.float64)
        else:
            M = len(field.omegas)
            packed = np.empty((M, 9), dtype=np.float64)
            packed[:, 0] = field.omegas
            packed[:, 1:4] = field.kvecs
            packed[:, 4:7] = field.evecs
            packed[:, 7] = field.amps
            packed[:, 8] = field.phases
            n_modes = M
            field_arr = np.ascontiguousarray(packed.ravel(), dtype=np.float64)

        tg = np.ascontiguousarray(np.asarray(t_grid, dtype=np.float64))
        n = len(tg)
        x0a = np.ascontiguousarray(np.asarray(x0, dtype=np.float64))
        v0a = np.ascontiguousarray(np.asarray(v0, dtype=np.float64))
        if x0a.shape != (3,) or v0a.shape != (3,):
            raise ValueError("x0 and v0 must be length-3 vectors")  # fail loudly, not Rust UB
        out_x = np.empty(n * 3, dtype=np.float64)
        out_v = np.empty(n * 3, dtype=np.float64)

        # The B field is read out of the frozen realization using the c the field
        # was BUILT with (matches NumPy ZPFBackground.B); fall back to units.c
        # when there is no field.
        c_light = float(field.units.c) if field is not None else float(units.c)
        self._fn(
            pot_kind, p0, p1, float(particle.charge), float(particle.mass),
            ctypes.c_longlong(n_modes),
            field_arr.ctypes.data_as(_PTR) if n_modes else None,
            ctypes.c_longlong(n), tg.ctypes.data_as(_PTR),
            x0a.ctypes.data_as(_PTR), v0a.ctypes.data_as(_PTR),
            1 if rr == "landau_lifshitz" else 0, 1 if dipole else 0,
            float(units.tau), c_light,
            out_x.ctypes.data_as(_PTR), out_v.ctypes.data_as(_PTR),
        )
        return Trajectory(tg, out_x.reshape(n, 3), out_v.reshape(n, 3), units)
