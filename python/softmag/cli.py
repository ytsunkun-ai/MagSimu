"""Command-line interface for the soft magnetic powder-core predictor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .configs import load_project
from .pipeline import run_all
from .schemas import ConfigError, load_config


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="softmag",
        description="Multi-scale soft magnetic powder-core predictor (DEM + micromagnetics + calibration)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-all", help="run the configured end-to-end pipeline")
    run.add_argument("--project", default="configs/project.yaml")
    run.add_argument("--run-id", default=None)
    run.add_argument(
        "--steps",
        default=None,
        help="comma-separated subset of steps (default: project file)",
    )
    run.add_argument("--quiet", action="store_true")

    val = sub.add_parser("validate", help="validate the input contract without running models")
    val.add_argument("--config", default="configs/material.yaml")
    val.add_argument("--project", default=None)

    demo = sub.add_parser("demo", help="run the demo project (same as run-all)")
    demo.add_argument("--project", default="configs/project.yaml")
    demo.add_argument("--run-id", default="demo")

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            if args.project:
                project, path = load_project(args.project)
                prediction = load_config(project.prediction_config)
                print(f"project: {path}")
            else:
                prediction = load_config(args.config)
            missing = prediction.missing_parameters()
            print(f"material: {prediction.material.material_id}")
            quality = prediction.material.parameter_quality
            print(f"quality: {getattr(quality, 'value', quality)}")
            print("missing:")
            for layer, items in missing.items():
                print(f"  {layer}: {items if items else 'none'}")
            for w in prediction.quality_warnings():
                print(f"WARNING: {w}")
            return 0
        project = args.project
        if args.command == "demo" and args.run_id == "demo":
            run_id = "demo"
        else:
            run_id = getattr(args, "run_id", None)
        steps = None
        if getattr(args, "steps", None):
            steps = [s.strip() for s in args.steps.split(",") if s.strip()]
        log = (lambda *a, **k: None) if getattr(args, "quiet", False) else print
        results = run_all(project, steps=steps, run_id=run_id, log=log)
        print(results.get("run_dir"))
        if results.get("report"):
            print(results["report"].get("markdown"))
        return 0
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
