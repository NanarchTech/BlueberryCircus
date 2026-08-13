"""Program/Engine/Backend/compile() architecture + the Boyer regression lock."""
import numpy as np
import pytest

import blueberry_circus as bc
from blueberry_circus.constants import Units, SI
from blueberry_circus import oracles
from blueberry_circus.backends import get_backend, NumpyBackend, Backend


def _sho_program():
    U = Units.scaled(0.05, 1.0)
    prog = bc.Program(n_particles=1, units=U)
    with prog.context as q:
        bc.Harmonic(1.0) | q[0]
        bc.ZPF(band=(0.5, 2.0), n_modes=40, seed=1, mode="one_dimensional") | q[0]
        bc.RadiationReaction("landau_lifshitz") | q[0]
    return prog


def test_compile_pass_list_is_inspectable():
    compiled = _sho_program().compile()
    assert compiled.passes == ["validate_single_particle", "collect_potential",
                               "collect_field", "resolve_reaction",
                               "validate_ops_exhaustive"]
    assert compiled.rr == "landau_lifshitz"
    assert compiled.field is not None and compiled.potential is not None


def test_compile_allows_coulomb_without_field():
    U = Units.scaled(0.02, 1.0)
    prog = bc.Program(n_particles=1, units=U)
    with prog.context as q:
        bc.Coulomb(Z=1.0) | q[0]
    compiled = prog.compile()
    assert compiled.field is None and compiled.rr == "none"


def test_compile_rejects_duplicate_potential():
    U = Units.scaled(0.05, 1.0)
    prog = bc.Program(n_particles=1, units=U)
    with prog.context as q:
        bc.Harmonic(1.0) | q[0]
        bc.Harmonic(2.0) | q[0]
    with pytest.raises(ValueError):
        prog.compile()


def test_radiation_reaction_build_rejects_bare_al():
    assert bc.RadiationReaction("landau_lifshitz").build() == "landau_lifshitz"
    assert bc.RadiationReaction("none").build() == "none"
    with pytest.raises(ValueError):
        bc.RadiationReaction("abraham_lorentz").build()      # runaway trap


def test_backend_registry():
    assert isinstance(get_backend("numpy"), NumpyBackend)
    assert isinstance(get_backend("numpy"), Backend)
    # rust / jax resolve to a Backend when their dependency is present, else
    # NotImplementedError (import-or-raise).
    for opt in ("rust", "jax"):
        try:
            assert isinstance(get_backend(opt), Backend)
        except NotImplementedError:
            pass
    with pytest.raises(NotImplementedError):
        get_backend("cuda")                 # planned, unbuilt; use 'jax' on GPU
    with pytest.raises(ValueError):
        get_backend("nonsense")


def test_engine_result_carries_passes_and_covariance():
    res = bc.Engine(dt=0.05, t_max=60).run(_sho_program(), x0=[0, 0, 0], v0=[0, 0, 0])
    assert res.compiled_passes[0] == "validate_single_particle"
    cov = res.covariance()
    assert cov.shape == (6, 6) and np.all(np.isfinite(cov))
    assert res.means().shape == (6,)


def test_boyer_regression_lock():
    # The certified Boyer oscillator number must not drift across the refactor.
    # Pre-refactor: rel-err = 4.99e-4 at w0=2.5e16 (SI electron). Lock it.
    w0 = 2.5e16
    val = oracles.sed_ground_state_integral(w0, SI)
    ref = oracles.ground_state_variance_target(w0, SI)
    rel = abs(val - ref) / ref
    assert rel < 5.5e-4, f"Boyer rel-err {rel:.3e} regressed past the 4.99e-4 lock"
    assert rel < 5e-3                        # the certified tolerance still holds
