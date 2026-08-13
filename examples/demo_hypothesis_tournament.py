"""Reduced, deterministic integration smoke test for all four tournament arms.

The preregistered research sweep is intentionally a separate explicit command;
see ``docs/tournament.md``.  This demo verifies schema, ledgers, and null limits
without presenting its tiny sample as a scientific tournament result.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from blueberry_circus.rectification import critical_angular_momentum
from blueberry_circus.tournament import TournamentConfig, run_tournament


OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

lc = critical_angular_momentum()
config = TournamentConfig(
    energies=(-0.02,),
    angular_momenta=(0.55, lc + 0.01),
    coupling_scales=(1.0, 4.0),
    seeds=(101, 107, 109, 113),
    n_modes=64,
    timestep=0.01,
    max_resolution_levels=2,
    convergence_rtol=0.8,
)
report = run_tournament(
    config=config,
    parameter_index=0,
    quadrature_order=32,
    integration_steps=512,
    integration_periods=0.002,
)
path = os.path.join(OUT, "hypothesis_tournament_smoke.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
    handle.write("\n")

assert report["schema"] == "blueberry-circus/hypothesis-tournament/v1"
assert len(report["runs"]) == 4
assert all(run["classification"] in {
    "CHANNEL_SUPPRESSED", "ACTIVE_CONTROL", "NO_EFFECT", "DESTABILIZED", "NULL",
} for run in report["runs"])
assert not any(run["classification"] == "STABLE_GROUND_STATE"
               for run in report["runs"])

print("Reduced schema/ledger smoke only; not a scientific sweep:")
for run in report["runs"]:
    print(f"  {run['arm']}: {run['classification']}")
print(f"Wrote {path}")
