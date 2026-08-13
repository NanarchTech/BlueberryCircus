"""Numeric backends + a small registry bound at Engine construction.

Three backends ship: :class:`NumpyBackend` (the deterministic trust root and
reference implementation), ``rust`` (a dependency-free C-ABI ``cdylib`` reached
through ``ctypes``), and ``jax`` (a ``jit``/``scan`` integrator). The latter two
are optional native/optional-dependency builds: asking for one that is not built
raises an honest ``NotImplementedError`` rather than silently falling back to
numpy. Names that are planned but unimplemented raise the same way.
"""
from __future__ import annotations

from .base import Backend
from .numpy_backend import NumpyBackend

AVAILABLE = ("numpy", "rust", "jax")

# name -> Backend subclass. 'rust' and 'jax' are imported lazily in get_backend
# (their modules raise if the toolchain is absent); the planned-but-unimplemented
# names raise NotImplementedError so a typo never silently degrades.
_REGISTRY = {"numpy": NumpyBackend}
_PLANNED = {
    "cuda": "native CUDA batched-ensemble backend (use 'jax' on GPU instead)",
}


def get_backend(name) -> Backend:
    """Resolve a backend name (or pass through a Backend instance).

    ``"rust"`` -> native LL-RK4 cdylib (when built); ``"jax"`` -> JAX scan
    integrator (when JAX is installed). Each raises ``NotImplementedError`` with
    the remediation when its dependency is absent (import-or-raise, never a silent
    fallback to numpy).
    """
    if isinstance(name, Backend):
        return name
    if name in _REGISTRY:
        return _REGISTRY[name]()
    if name == "rust":
        from .rust_backend import RustBackend     # raises NotImplementedError if unbuilt
        return RustBackend()
    if name == "jax":
        from .jax_backend import JaxBackend       # raises NotImplementedError if no JAX
        return JaxBackend()
    if name in _PLANNED:
        raise NotImplementedError(f"backend {name!r} not built yet ({_PLANNED[name]})")
    raise ValueError(f"unknown backend {name!r}; available: {AVAILABLE}")


def register_backend(name: str, cls) -> None:
    """Register a backend subclass under ``name``."""
    global AVAILABLE
    _REGISTRY[name] = cls
    if name not in AVAILABLE:
        AVAILABLE = AVAILABLE + (name,)


__all__ = ["Backend", "NumpyBackend", "AVAILABLE", "get_backend", "register_backend"]
