"""Command-line entry point for explicit, chunkable tournament research runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .rectification import critical_angular_momentum
from .tournament import (
    TournamentConfig,
    preregistered_parameters,
    run_tournament,
)


ARMS = ("setterfield", "finite_shell", "inverse_square", "multipole")


def _smoke_config() -> TournamentConfig:
    lc = critical_angular_momentum()
    return TournamentConfig(
        energies=(-0.02,),
        angular_momenta=(0.55, lc + 0.01),
        coupling_scales=(1.0, 4.0),
        seeds=(101, 107, 109, 113),
        n_modes=64,
        timestep=0.01,
        max_resolution_levels=2,
        convergence_rtol=0.8,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blueberry-tournament",
        description=(
            "Energy-audited perturbative SED hypothesis tournament. Full "
            "preregistered sweeps are never started without --execute."
        ),
    )
    parser.add_argument("--arm", choices=("all",) + ARMS, default="all")
    parser.add_argument(
        "--profile", choices=("smoke", "preregistered"), default="smoke",
    )
    parser.add_argument("--parameter-index", type=int)
    parser.add_argument("--quadrature-order", type=int, default=64)
    parser.add_argument("--integration-steps", type=int)
    parser.add_argument("--integration-periods", type=float)
    parser.add_argument("--output", type=Path,
                        default=Path("tournament-summary.json"))
    parser.add_argument("--npz", type=Path,
                        help="optional raw numeric arrays")
    parser.add_argument(
        "--execute", action="store_true",
        help="perform the run; without this flag only the manifest is printed",
    )
    return parser


def _manifest(config: TournamentConfig, arms: tuple[str, ...], profile: str) -> dict:
    return {
        "profile": profile,
        "arms": {
            arm: len(preregistered_parameters(arm, config)) for arm in arms
        },
        "orbit_cells": len(config.energies) * len(config.angular_momenta),
        "coupling_scales": len(config.coupling_scales),
        "seeds": len(config.seeds),
        "base_modes": config.n_modes,
        "resolution_levels": config.max_resolution_levels,
        "execute_required": True,
    }


def _write_npz(path: Path, report: dict) -> None:
    arrays = {
        "point_charge_drift": np.asarray(
            report["point_charge_baseline"]["drift"], dtype=float,
        ),
    }
    for index, run in enumerate(report["runs"]):
        rows = []
        for cell in run["cells"]:
            rows.append([
                cell["energy"], cell["angular_momentum"],
                cell["coupling_scale"], cell["mean_drift"],
                cell["confidence_low"], cell["confidence_high"],
                cell["ledger"]["numerical_closure_residual"],
            ])
        arrays[f"run_{index:03d}_{run['arm']}"] = np.asarray(rows, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _smoke_config() if args.profile == "smoke" else TournamentConfig()
    arms = ARMS if args.arm == "all" else (args.arm,)
    manifest = _manifest(config, arms, args.profile)
    if not args.execute:
        print(json.dumps(manifest, sort_keys=True, indent=2))
        return 0

    parameter_index = args.parameter_index
    integration_steps = args.integration_steps
    integration_periods = args.integration_periods
    if args.profile == "smoke":
        parameter_index = 0 if parameter_index is None else parameter_index
        integration_steps = 512 if integration_steps is None else integration_steps
        integration_periods = (
            0.002 if integration_periods is None else integration_periods
        )
    report = run_tournament(
        config=config,
        arms=arms,
        parameter_index=parameter_index,
        quadrature_order=args.quadrature_order,
        integration_steps=integration_steps,
        integration_periods=integration_periods,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.npz is not None:
        _write_npz(args.npz, report)
    classifications = [run["classification"] for run in report["runs"]]
    print(json.dumps({
        "output": str(args.output),
        "npz": None if args.npz is None else str(args.npz),
        "runs": len(report["runs"]),
        "classifications": classifications,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
