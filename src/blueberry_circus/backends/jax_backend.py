"""JAX backend: a jit/scan RK4+LL integrator, vmap-ready for batched ensembles.

Ports the reference integrator to JAX (float64) with ``lax.scan`` over the time
grid, so a single trajectory JITs and an ensemble ``vmap``s over seeds on CPU or
GPU. NumPy remains the reference oracle; this backend is diffed against it under
enclosure tolerance (XLA transcendental functions differ from libm at the ULP
level -- never bit-equality).

Import-or-raise: constructing a :class:`JaxBackend` without JAX installed raises
``NotImplementedError`` (never a silent fallback).
"""
from __future__ import annotations

import numpy as np

from .base import Backend
from ..dynamics import Trajectory
from ..potentials import Harmonic as _Harmonic, Coulomb as _Coulomb

_JAX = None


def _import_jax():
    global _JAX
    if _JAX is None:
        try:
            import jax
            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
            from jax import lax
            _JAX = (jax, jnp, lax)
        except Exception:  # pragma: no cover
            _JAX = False
    return _JAX


def is_available() -> bool:
    return _import_jax() is not False


class JaxBackend(Backend):
    name = "jax"

    def __init__(self):
        if not is_available():
            raise NotImplementedError(
                "jax backend requires JAX (pip install jax). Not importable.")
        self._jax, self._jnp, self._lax = _JAX

    def integrate(self, *, field, potential, particle, t_grid, x0, v0,
                  rr: str, units, dipole: bool):
        jax, jnp, lax = self._jax, self._jnp, self._lax

        pm = 1.0  # potential's own mass (harmonic force uses it, not particle mass)
        if isinstance(potential, _Harmonic):
            pot_kind, p0, p1, pm = 0, float(potential.omega0), 0.0, float(potential.mass)
        elif isinstance(potential, _Coulomb):
            # frozen coefficient (matches NumPy), not recomputed from integrate units
            pot_kind, p0, p1 = 1, float(potential._coef), float(potential.softening)
        elif potential is None:
            raise ValueError("jax backend requires a potential")
        else:
            raise TypeError(f"jax backend cannot integrate potential {potential!r}")

        q = float(particle.charge)
        m = float(particle.mass)
        tau = float(units.tau)
        # B is read out with the c the field was BUILT with (matches ZPFBackground.B)
        c = float(field.units.c) if field is not None else float(units.c)
        is_ll = rr == "landau_lifshitz"
        has_field = field is not None

        if has_field:
            omegas = jnp.asarray(field.omegas)
            K = jnp.asarray(field.kvecs)
            Ev = jnp.asarray(field.evecs)
            amps = jnp.asarray(field.amps)
            phases = jnp.asarray(field.phases)
            khat = K / jnp.maximum(jnp.linalg.norm(K, axis=1, keepdims=True), 1e-300)
            bvec = jnp.cross(khat, Ev) / c

        tg = jnp.asarray(np.asarray(t_grid, dtype=float))
        x0j = jnp.asarray(np.asarray(x0, dtype=float))
        v0j = jnp.asarray(np.asarray(v0, dtype=float))

        def pot_force(x):
            if pot_kind == 0:
                return -pm * p0 * p0 * x          # pm = potential mass
            rs = jnp.sqrt(jnp.dot(x, x) + p1 * p1)
            return -p0 * x / rs**3

        def pot_jac_v(x, v):
            if pot_kind == 0:
                return -pm * p0 * p0 * v          # pm = potential mass
            rs = jnp.sqrt(jnp.dot(x, x) + p1 * p1)
            return -p0 * (v / rs**3 - 3.0 * x * jnp.dot(x, v) / rs**5)

        def accel(x, v, t):
            r_eval = jnp.zeros(3) if dipole else x
            F = pot_force(x)
            dedt = jnp.zeros(3)
            if has_field:
                arg = K @ r_eval - omegas * t + phases
                ca, sa = jnp.cos(arg), jnp.sin(arg)
                E = (Ev * (amps * ca)[:, None]).sum(axis=0)
                F = F + q * E
                if not dipole:
                    B = (bvec * (amps * ca)[:, None]).sum(axis=0)
                    F = F + q * jnp.cross(v, B)
                dedt = (Ev * (amps * omegas * sa)[:, None]).sum(axis=0)
            if is_ll:
                dfdt = pot_jac_v(x, v)
                if has_field:
                    dfdt = dfdt + q * dedt
                F = F + tau * dfdt
            return F / m

        def step(carry, i):
            x, v = carry
            t = tg[i]
            h = tg[i + 1] - t
            k1x, k1v = v, accel(x, v, t)
            k2x = v + 0.5 * h * k1v
            k2v = accel(x + 0.5 * h * k1x, v + 0.5 * h * k1v, t + 0.5 * h)
            k3x = v + 0.5 * h * k2v
            k3v = accel(x + 0.5 * h * k2x, v + 0.5 * h * k2v, t + 0.5 * h)
            k4x = v + h * k3v
            k4v = accel(x + h * k3x, v + h * k3v, t + h)
            xn = x + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            vn = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
            return (xn, vn), (xn, vn)

        n = len(tg)

        @jax.jit
        def run(x0j, v0j):
            (_, _), (xs, vs) = lax.scan(step, (x0j, v0j), jnp.arange(n - 1))
            X = jnp.concatenate([x0j[None, :], xs], axis=0)
            V = jnp.concatenate([v0j[None, :], vs], axis=0)
            return X, V

        X, V = run(x0j, v0j)
        return Trajectory(np.asarray(tg), np.asarray(X), np.asarray(V), units)
