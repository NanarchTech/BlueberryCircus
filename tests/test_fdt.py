"""Fluctuation--dissipation: at stationarity, ZPF power in = Larmor power out."""
import numpy as np
from blueberry_circus.constants import Units
from blueberry_circus.dynamics import Particle, integrate
from blueberry_circus.zpf import ZPFBackground
from blueberry_circus.potentials import Harmonic
from blueberry_circus import observables as O
from blueberry_circus.ensemble import fdt_balance_ensemble
from blueberry_circus.backends import NumpyBackend

try:
    from blueberry_circus.backends.rust_backend import RustBackend, is_available
except Exception:  # pragma: no cover
    def is_available():
        return False


def test_power_balance_at_stationarity():
    U = Units.scaled(0.05, 1.0); w0 = 1.0
    P = Particle(U.charge, U.mass); pot = Harmonic(w0, mass=U.mass)
    field = ZPFBackground.one_dimensional(0.3, 3.0, 120, seed=11, units=U)
    t = np.arange(0, 500, 0.03)
    tr = integrate(field=field, potential=pot, particle=P, t_grid=t,
                   x0=[0, 0, 0], v0=[0, 0, 0], rr="landau_lifshitz", units=U)
    P_rad = O.mean_radiated_power(tr, U, burn_in=0.4)
    P_abs = O.mean_absorbed_power(tr, field, P, burn_in=0.4)
    assert P_rad > 0 and P_abs > 0
    assert abs(P_rad - P_abs) / P_abs < 0.20      # single-realization tolerance


def test_ensemble_fdt_balance_tightens_below_5pct():
    # The FDT is an ensemble statement; averaging over seeds tightens the balance
    # from the single-realization ~20% scatter to <5%. Use the fast Rust backend
    # with more seeds when available, else a smaller NumPy ensemble.
    U = Units.scaled(0.05, 1.0)
    if is_available():
        res = fdt_balance_ensemble(U, n_seeds=32, n_modes=120, t_max=500, dt=0.03,
                                   backend=RustBackend())
    else:
        res = fdt_balance_ensemble(U, n_seeds=12, n_modes=80, t_max=400, dt=0.03,
                                   backend=NumpyBackend())
    assert res["P_rad"] > 0 and res["P_abs"] > 0
    assert res["rel_imbalance"] < 0.05
