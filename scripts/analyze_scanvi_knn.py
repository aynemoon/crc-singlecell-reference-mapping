"""Reproduce the PR 1 analysis from saved artifacts, without importing training code."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import matplotlib
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")

from scmap.scanvi_knn_analysis import run_analysis, sha256  # noqa: E402
from scmap.scanvi_knn_figures import plot_review  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "predictions",
        "embeddings",
        "reference-annotation",
        "query-annotation",
        "historical-dir",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--native-classes",
        type=Path,
        help="JSON column list or static reference-checkpoint metadata report",
    )
    parser.add_argument(
        "--provenance", type=Path, help="Original training-run provenance, hashed only"
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="Also write bootstrap draws, derived group deltas and PNG previews",
    )
    args = parser.parse_args()
    # An explicit fresh directory prevents accidental replacement of archived or prior outputs.
    if args.output.exists():
        parser.error("--output must be a new directory; existing results will not be overwritten")
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve() in (
        (root / "results/tables").resolve(),
        (root / "results/figures").resolve(),
    ):
        parser.error("archived result directories cannot be used as --output")
    if args.threads < 1:
        parser.error("--threads must be positive")
    with threadpool_limits(limits=args.threads):
        tables, checks = run_analysis(
            args.predictions,
            args.embeddings,
            args.reference_annotation,
            args.query_annotation,
            args.historical_dir,
            args.native_classes,
            args.n_boot,
            args.seed,
        )
    inputs = {
        name: getattr(args, name)
        for name in (
            "predictions",
            "embeddings",
            "reference_annotation",
            "query_annotation",
            "native_classes",
            "provenance",
        )
        if getattr(args, name) is not None
    }
    inputs.update(
        {
            f"historical_{name}": args.historical_dir / name
            for name in ("metrics_fine.csv", "per_class_f1_fine.csv")
        }
    )
    inputs["historical_snapshot_manifest"] = args.historical_dir / "historical_snapshot.json"
    manifest = dict(
        analysis_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        analysis_worktree=subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True
        ).splitlines(),
        input_sha256={name: sha256(path) for name, path in inputs.items()},
        python=platform.python_version(),
        packages={
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "anndata", "matplotlib")
        },
        analysis_source_sha256={
            path: sha256(root / path)
            for path in (
                "scripts/analyze_scanvi_knn.py",
                "src/scmap/scanvi_knn_analysis.py",
                "src/scmap/scanvi_knn_figures.py",
                "src/scmap/baselines.py",
                "src/scmap/evaluate.py",
                "src/scmap/plotting.py",
                "src/scmap/historical_snapshot.py",
                "src/scmap/checkpoint_metadata.py",
            )
        },
        threads=args.threads,
        repository_sha256={
            path: sha256(root / path) for path in ("uv.lock", "configs/default.yaml")
        },
        checks=checks,
        full_output=args.full_output,
    )
    args.output.mkdir(parents=True)
    for name, frame in tables.items():
        if not args.full_output and name in {"bootstrap_draws", "group_differences"}:
            continue
        frame.to_csv(args.output / f"{name}.csv", index=False, float_format="%.17g")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_review(tables, args.output / "figures", preview_png=args.full_output)
    print(tables["bootstrap"].to_string(index=False))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
