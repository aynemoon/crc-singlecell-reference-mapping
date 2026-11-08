import importlib.util
import json
import shutil
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

matplotlib.use("Agg")

from scmap.baselines import scanvi_knn  # noqa: E402
from scmap.scanvi_knn_analysis import (  # noqa: E402
    analyse,
    paired_bootstrap,
    reconstruct,
    run_analysis,
    validate_predictions,
)
from scmap.scanvi_knn_figures import plot_review  # noqa: E402


@pytest.fixture
def experiment():
    def annotations(ids, labels, patients):
        return pd.DataFrame(
            dict(
                patient=patients,
                sample=patients,
                tissue="Tumor",
                cell_type="Epithelial cells",
                cell_subtype=labels,
            ),
            index=ids,
        )

    ref = annotations(
        ["r1", "r2", "r3", "r4"], ["CMS1", "CMS2", "CMS4", "Unknown"], ["R1", "R2", "R2", "R1"]
    )
    qry = annotations(
        ["q1", "q2", "q3", "q4", "q5", "q6"],
        ["CMS1", "CMS2", "CMS1", "CMS2", "Tuft cells", "Unknown"],
        ["P1", "P1", "P2", "P2", "P1", "P2"],
    )
    zr = np.array([[0], [2], [100]], dtype=np.float32)
    zq = np.array([[0.1], [1.9], [0.8], [2.6], [5.1], [5.5]], dtype=np.float32)
    classes = np.array(["CMS4", "CMS1", "CMS2"])
    vote = scanvi_knn(zr, ref.cell_subtype.iloc[:3].to_numpy(), zq, 2, classes)
    rows = []
    for method in (
        "logreg",
        "knn_pca",
        "scvi_knn",
        "scanvi",
        "scanvi_knn",
        "scanvi_latent_distance",
    ):
        predicted = vote.predicted.copy()
        if method != "scanvi_knn":
            predicted[0] = "CMS2"
        rows.append(
            pd.DataFrame(
                dict(
                    cell=qry.index,
                    method=method,
                    pred_fine=predicted,
                    pred_coarse="Epithelial cells",
                    confidence=vote.proba.max(1),
                    novelty=vote.novelty,
                )
            )
        )
    return ref, qry, pd.concat(rows, ignore_index=True), zr, zq, classes


def test_paired_bootstrap_matches_direct_cell_resampling():
    truth = np.array(["a", "a", "b", "b", "b"])
    native = np.array(["a", "b", "b", "a", "c"])
    knn = np.array(["a", "a", "a", "b", "b"])
    patients = np.array(["P1", "P1", "P2", "P3", "P3"])
    summary, samples = paired_bootstrap(truth, native, knn, patients, 40, 0)
    donors = np.unique(patients)
    draws = np.random.default_rng(0).choice(donors, (40, 3))
    expected = []
    for draw in draws:
        idx = np.concatenate([np.flatnonzero(patients == p) for p in draw])
        f1 = [
            f1_score(
                truth[idx], p[idx], labels=np.unique(truth[idx]), average="macro", zero_division=0
            )
            for p in (native, knn)
        ]
        acc = [accuracy_score(truth[idx], p[idx]) for p in (native, knn)]
        expected.append([f1[1] - f1[0], acc[1] - acc[0]])
    np.testing.assert_allclose(
        samples[["knn_minus_native_macro_f1", "knn_minus_native_accuracy"]], expected
    )
    assert samples.filter(like="draw_").sum(axis=1).eq(3).all()
    assert samples.filter(like="draw_").max().max() > 1
    row = summary.query("method == 'knn_minus_native' and metric == 'macro_f1'").iloc[0]
    np.testing.assert_allclose(
        [row.ci_low, row.ci_high], np.percentile(np.array(expected)[:, 0], [2.5, 97.5])
    )


@pytest.mark.parametrize(
    "damage,match",
    [
        ("duplicate", "duplicate"),
        ("missing", "missing or extra"),
        ("method", "method names"),
        ("label", "label mismatch"),
        ("coarse", "coarse label"),
        ("confidence", "invalid confidence"),
    ],
)
def test_prediction_input_checks(experiment, damage, match):
    ref, qry, predictions, *_ = experiment
    reference = ref[ref.cell_subtype != "Unknown"]
    if damage == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[:1]])
    elif damage == "missing":
        predictions = predictions.iloc[1:]
    elif damage == "method":
        predictions.loc[0, "method"] = "typo"
    elif damage == "label":
        predictions.loc[0, "pred_fine"] = "made up"
    elif damage == "coarse":
        predictions.loc[0, "pred_coarse"] = "wrong parent"
    else:
        predictions.loc[0, "confidence"] = 1.1
    with pytest.raises(ValueError, match=match):
        validate_predictions(predictions, qry, reference)


def test_analysis_keeps_reference_only_class_and_excluded_cells_separate(experiment):
    ref, qry, predictions, *_ = experiment
    tables, checks = analyse(predictions.sample(frac=1, random_state=3), ref, qry, 20)
    assert checks["shared_query_classes"] == 2
    assert checks["shared_query_cells"] == 4
    assert set(tables["support"].subtype) == {"CMS1", "CMS2", "CMS4"}
    assert set(tables["subtypes"].subtype) == {"CMS1", "CMS2"}
    cms1 = tables["subtypes"].query("method == 'scanvi_knn' and subtype == 'CMS1'").iloc[0]
    assert (cms1.tp, cms1.fp, cms1.fn, cms1.query_support) == (2, 0, 0, 2)
    assert tables["disagreements"].corrected.sum() == 1
    assert tables["disagreements"].introduced.sum() == 0
    with pytest.raises(ValueError, match="duplicate"):
        analyse(predictions, pd.concat([ref, ref.iloc[:1]]), qry)
    qry.loc["q1", "cell_subtype"] = "unrecognised truth"
    with pytest.raises(ValueError, match="has no role"):
        analyse(predictions, ref, qry)


def test_reconstruction_checks_class_list_and_saved_values(experiment):
    ref, qry, predictions, zr, zq, classes = experiment
    saved = predictions.query("method == 'scanvi_knn'").set_index("cell")
    args = (
        zr,
        ref.cell_subtype.iloc[:3].to_numpy(),
        zq,
        saved,
        ref.patient.iloc[:3].to_numpy(),
        qry.cell_subtype.to_numpy(),
    )
    checks, _ = reconstruct(*args, classes, 2)
    assert checks["maximum_confidence_error"] == 0
    assert checks["native_order_metadata_supplied"]
    for bad in (["CMS1", "CMS2"], ["CMS1", "CMS1", "CMS4"]):
        with pytest.raises(ValueError, match="class-order"):
            reconstruct(*args, np.array(bad), 2)
    saved.loc["q1", "novelty"] += 1
    with pytest.raises(ValueError, match="novelty distances"):
        reconstruct(*args, classes, 2)


@pytest.mark.parametrize("preview_png", [False, True])
def test_end_to_end_saved_files_figures_and_annotation_mismatch(experiment, tmp_path, preview_png):
    ref, qry, predictions, zr, zq, classes = experiment
    for name, frame in (("ref", ref), ("qry", qry)):
        frame.rename(
            columns={
                "patient": "Patient",
                "sample": "Sample",
                "tissue": "Class",
                "cell_type": "Cell_type",
                "cell_subtype": "Cell_subtype",
            }
        ).rename_axis("Index").to_csv(tmp_path / f"{name}.tsv", sep="\t")
    obs = pd.concat([ref.iloc[:3].assign(cohort="SMC"), qry.assign(cohort="KUL3")])
    ad.AnnData(obs=obs, obsm={"scANVI": np.vstack([zr, zq])}).write_h5ad(tmp_path / "emb.h5ad")
    predictions.to_parquet(tmp_path / "pred.parquet")
    (tmp_path / "classes.json").write_text(json.dumps(classes.tolist()))
    historic = tmp_path / "historical"
    original = Path(__file__).resolve().parents[1] / "results/historical/63bcf65028fb"
    shutil.copytree(original, historic)
    args = (
        tmp_path / "pred.parquet",
        tmp_path / "emb.h5ad",
        tmp_path / "ref.tsv",
        tmp_path / "qry.tsv",
        historic,
    )
    tables, checks = run_analysis(*args, tmp_path / "classes.json", n_boot=20, k=2)
    assert checks["reference_classes"] == 3 and checks["shared_query_classes"] == 2
    assert checks["prediction_mismatches"] == 0
    (tmp_path / "classes.json").write_text(
        json.dumps(
            dict(
                model_name="SCANVI",
                scvi_version="1.5.1",
                native_classes=classes.tolist(),
                order_basis="reference checkpoint registry",
            )
        )
    )
    _, recovered = run_analysis(*args, tmp_path / "classes.json", n_boot=20, k=2)
    assert recovered["native_order_status"] == "reference checkpoint registry"
    plot_review(tables, tmp_path / "figures", preview_png=preview_png)
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 6
    assert len(list((tmp_path / "figures").glob("*.png"))) == (6 if preview_png else 0)
    obs.loc["q1", "cell_subtype"] = "CMS2"
    ad.AnnData(obs=obs, obsm={"scANVI": np.vstack([zr, zq])}).write_h5ad(tmp_path / "emb.h5ad")
    with pytest.raises(ValueError, match="label or donor mismatch"):
        run_analysis(*args, n_boot=20, k=2)


def test_cli_refuses_to_overwrite_existing_results(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_scanvi_knn.py"
    spec = importlib.util.spec_from_file_location("review_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    marker = tmp_path / "keep.txt"
    marker.write_text("original")
    argv = [str(script)]
    for name in (
        "predictions",
        "embeddings",
        "reference-annotation",
        "query-annotation",
        "historical-dir",
    ):
        argv.extend([f"--{name}", str(tmp_path / "unused")])
    argv.extend(["--output", str(tmp_path)])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert marker.read_text() == "original"
