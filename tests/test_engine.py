import numpy as np
import blueberry_circus as bc
from blueberry_circus.constants import Units


def test_program_context_records_ops():
    prog = bc.Program(n_particles=1, units=Units.scaled(0.05, 1.0))
    with prog.context as q:
        bc.Harmonic(1.0) | q[0]
        bc.ZPF(band=(0.5, 2.0), n_modes=50, seed=1, mode="one_dimensional") | q[0]
        bc.RadiationReaction("landau_lifshitz") | q[0]
    kinds = [type(op).__name__ for op in prog.ops_for(0)]
    assert kinds == ["Harmonic", "ZPF", "RadiationReaction"]


def test_engine_runs_harmonic_in_zpf():
    U = Units.scaled(0.05, 1.0)
    prog = bc.Program(n_particles=1, units=U)
    with prog.context as q:
        bc.Harmonic(1.0) | q[0]
        bc.ZPF(band=(0.5, 2.0), n_modes=40, seed=1, mode="one_dimensional") | q[0]
        bc.RadiationReaction("landau_lifshitz") | q[0]
    res = bc.Engine(dt=0.05, t_max=120).run(prog, x0=[0, 0, 0], v0=[0, 0, 0])
    assert res.trajectory.x.shape[1] == 3
    assert res.observables["trajectory_finite"]
    assert all(c.recheck() == bc.PASS for c in res.certificates)


def test_engine_rejects_unknown_backend():
    import pytest
    with pytest.raises(NotImplementedError):
        bc.Engine(backend="cuda", dt=0.1, t_max=1.0)


def test_engine_fails_closed_on_multiple_potentials():
    # The v0.1 interpreter must NOT silently keep only the last of several
    # potentials (a real measurement-integrity defect).
    import pytest
    U = Units.scaled(0.05, 1.0)
    prog = bc.Program(n_particles=1, units=U)
    with prog.context as q:
        bc.Harmonic(1.0) | q[0]
        bc.Harmonic(2.0) | q[0]
    with pytest.raises(ValueError):
        bc.Engine(dt=0.1, t_max=1.0).run(prog, x0=[0, 0, 0], v0=[0, 0, 0])


def test_diverged_trajectory_certificate_is_serializable():
    # A blown-up integration must yield a FAIL trajectory_finite cert that still
    # canonicalizes (finite residual, never inf on the hash surface).
    from blueberry_circus.certify import build_chain, verify_chain, FAIL
    U = Units.scaled(0.05, 1.0)
    prog = bc.Program(n_particles=1, units=U)
    with prog.context as q:
        bc.Harmonic(1.0e3) | q[0]          # omega0*dt huge -> RK4 unstable -> inf
    with np.errstate(all="ignore"):        # divergence to inf is the point here
        res = bc.Engine(dt=0.5, t_max=50).run(prog, x0=[1, 0, 0], v0=[0, 0, 0])
    assert res.observables["trajectory_finite"] is False
    (cert,) = res.certificates
    assert cert.recheck() == FAIL
    envs = build_chain(res.certificates)   # must not raise on a diverged-run cert
    assert verify_chain(envs)["verified"] is True
