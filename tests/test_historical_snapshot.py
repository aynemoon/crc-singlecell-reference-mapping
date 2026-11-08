import json
import shutil
from pathlib import Path

import pytest

from scmap.historical_snapshot import (
    BASELINE_COMMIT,
    EXPECTED_SHA256,
    create_snapshot,
    verify_snapshot,
)
from scmap.scanvi_knn_analysis import run_analysis

ARCHIVE = Path(__file__).resolve().parents[1] / "results/historical/63bcf65028fb"


def test_snapshot_uses_git_blobs_even_when_working_tables_are_regenerated(tmp_path, monkeypatch):
    def git_show(command, cwd):
        assert command[:2] == ["git", "show"]
        assert command[2].startswith(f"{BASELINE_COMMIT}:results/tables/")
        return (ARCHIVE / command[2].split("/")[-1]).read_bytes()

    monkeypatch.setattr("scmap.historical_snapshot.subprocess.check_output", git_show)
    working = tmp_path / "results/tables"
    working.mkdir(parents=True)
    (working / "metrics_fine.csv").write_text("regenerated local results")
    snapshot = tmp_path / "snapshot"
    create_snapshot(tmp_path, snapshot)
    assert verify_snapshot(snapshot)["sha256"] == EXPECTED_SHA256
    assert (working / "metrics_fine.csv").read_text() == "regenerated local results"


def test_analysis_rejects_unverified_working_tables_before_loading_data(tmp_path):
    (tmp_path / "metrics_fine.csv").write_text("regenerated")
    with pytest.raises(ValueError, match="verified original Git snapshot"):
        run_analysis(
            tmp_path / "missing",
            tmp_path / "missing",
            tmp_path / "missing",
            tmp_path / "missing",
            tmp_path,
        )


@pytest.mark.parametrize("damage", ["content", "manifest", "forged_hash"])
def test_snapshot_rejects_modified_files_and_forged_identity(tmp_path, damage):
    snapshot = tmp_path / "snapshot"
    shutil.copytree(ARCHIVE, snapshot)
    path = snapshot / "historical_snapshot.json"
    manifest = json.loads(path.read_text())
    if damage == "manifest":
        manifest["commit"] = "regenerated"
    else:
        (snapshot / "metrics_fine.csv").write_text("local regenerated data")
        if damage == "forged_hash":
            import hashlib

            manifest["sha256"]["metrics_fine.csv"] = hashlib.sha256(
                b"local regenerated data"
            ).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="snapshot .* mismatch"):
        verify_snapshot(snapshot)
