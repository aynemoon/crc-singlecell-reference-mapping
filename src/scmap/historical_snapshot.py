"""Pinned, content-verified historical results, independent of regenerated working tables."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

BASELINE_COMMIT = "63bcf65028fb2597a23742c1f52a21d963697681"
EXPECTED_SHA256 = {
    "metrics_fine.csv": "bfe11414a6a9b82b218aa30e227490c422a6d87ef1e8fd3993fe61d975e154d4",
    "per_class_f1_fine.csv": "bab92a1240b2ff2095f04c533ef2ea5c8e81016233198935b181c8d26ef9be00",
}


def verify_snapshot(directory: Path) -> dict:
    try:
        manifest = json.loads((directory / "historical_snapshot.json").read_text())
        if manifest["commit"] != BASELINE_COMMIT or manifest["sha256"] != EXPECTED_SHA256:
            raise ValueError("historical snapshot identity mismatch")
        for filename, expected in EXPECTED_SHA256.items():
            if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f"historical snapshot content mismatch: {filename}")
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError("historical-dir must be a verified original Git snapshot") from error
    return manifest


def create_snapshot(repository: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("snapshot output already exists")
    contents = {
        name: subprocess.check_output(
            ["git", "show", f"{BASELINE_COMMIT}:results/tables/{name}"], cwd=repository
        )
        for name in EXPECTED_SHA256
    }
    if any(
        hashlib.sha256(data).hexdigest() != EXPECTED_SHA256[name] for name, data in contents.items()
    ):
        raise ValueError("Git baseline blobs do not match the approved historical hashes")
    output.mkdir(parents=True)
    for name, data in contents.items():
        (output / name).write_bytes(data)
    (output / "historical_snapshot.json").write_text(
        json.dumps(dict(commit=BASELINE_COMMIT, sha256=EXPECTED_SHA256), indent=2) + "\n"
    )
