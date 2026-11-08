"""Deterministic analysis of saved transfers. No model training or query adaptation."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors

from scmap.baselines import scanvi_knn
from scmap.data import read_annotation
from scmap.evaluate import classification_scores, novelty_scores
from scmap.historical_snapshot import verify_snapshot
from scmap.labels import UNKNOWN, harmonise_query, harmonise_reference, query_role
from scmap.plotting import CLASSIFIERS

PAIR = ("scanvi", "scanvi_knn")
FOCUS = ("CMS1", "CMS2", "CMS3", "CMS4", "cDC", "T follicular helper cells", "T helper 17 cells")
KNOWN = (*CLASSIFIERS, "scanvi_latent_distance")


def score_group(truth: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    # This warning is expected for patient/tissue groups missing a predicted class.
    # Keep the original present-truth-class scoring definition unchanged.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return classification_scores(truth, predictions)


def validate_annotations(frame: pd.DataFrame, name: str) -> None:
    columns = ["patient", "sample", "tissue", "cell_type", "cell_subtype"]
    if not frame.index.is_unique or frame.index.hasnans:
        raise ValueError(f"{name}: duplicate or missing cell IDs")
    if not set(columns) <= set(frame.columns) or frame[columns].isna().any().any():
        raise ValueError(f"{name}: missing annotation fields")
    if frame.empty:
        raise ValueError(f"{name}: empty annotations")


def validate_predictions(
    predictions: pd.DataFrame, query: pd.DataFrame, reference: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    required = {"cell", "method", "pred_fine", "pred_coarse", "confidence", "novelty"}
    if not required <= set(predictions) or predictions[list(required)].isna().any().any():
        raise ValueError("predictions: missing fields or values")
    if predictions.duplicated(["cell", "method"]).any():
        raise ValueError("predictions: duplicate cell IDs within a method")
    methods = set(predictions.method)
    if methods - set(KNOWN) or not set(CLASSIFIERS) <= methods:
        raise ValueError("predictions: invalid or missing method names")
    values = predictions[["confidence", "novelty"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not predictions.confidence.between(0, 1).all():
        raise ValueError("predictions: invalid confidence or novelty")
    if (predictions.novelty < 0).any():
        raise ValueError("predictions: negative novelty")
    pairs = reference[["cell_subtype", "cell_type"]].drop_duplicates()
    if pairs.cell_subtype.duplicated().any():
        raise ValueError("reference: fine labels have inconsistent coarse parents")
    parent = pairs.set_index("cell_subtype").cell_type
    if not predictions.pred_fine.isin(parent.index).all():
        raise ValueError("predictions: label mismatch with reference classes")
    if not np.array_equal(predictions.pred_fine.map(parent), predictions.pred_coarse):
        raise ValueError("predictions: coarse label mismatch")
    result = {}
    for method in KNOWN:
        if method not in methods:
            continue
        frame = predictions[predictions.method == method].set_index("cell")
        if set(frame.index) != set(query.index):
            raise ValueError(f"{method}: missing or extra query cells")
        result[method] = frame.loc[query.index]
    return result


def _scores_from_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same present-truth-class Macro F1 as classification_scores, including false positives."""
    tp = np.diagonal(counts, axis1=-2, axis2=-1)
    support = counts.sum(axis=-1)
    denominator = support + counts.sum(axis=-2)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
    present = support > 0
    return (f1 * present).sum(-1) / present.sum(-1), tp.sum(-1) / counts.sum(axis=(-2, -1))


def paired_bootstrap(
    truth: np.ndarray,
    native: np.ndarray,
    knn: np.ndarray,
    patients: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired cluster draws with multiplicity; recompute pooled metrics on each draw."""
    if (
        n_boot < 1
        or len(truth) == 0
        or not (len(truth) == len(native) == len(knn) == len(patients))
    ):
        raise ValueError("bootstrap requires nonempty aligned arrays and positive n_boot")
    donors = np.unique(patients)
    labels = np.unique(np.concatenate([truth, native, knn]))
    cubes = np.array(
        [
            [
                confusion_matrix(truth[patients == p], pred[patients == p], labels=labels)
                for p in donors
            ]
            for pred in (native, knn)
        ]
    )
    draws = np.random.default_rng(seed).choice(len(donors), (n_boot, len(donors)))
    weights = np.eye(len(donors), dtype=int)[draws].sum(axis=1)
    counts = np.einsum("bd,mdij->mbij", weights, cubes)
    boot = dict(zip(("macro_f1", "accuracy"), _scores_from_counts(counts), strict=True))
    point = dict(zip(("macro_f1", "accuracy"), _scores_from_counts(cubes.sum(1)), strict=True))
    rows, samples = [], pd.DataFrame(weights, columns=[f"draw_{p}" for p in donors])
    for metric, values in boot.items():
        for method, value, distribution in (
            (PAIR[0], point[metric][0], values[0]),
            (PAIR[1], point[metric][1], values[1]),
            ("knn_minus_native", point[metric][1] - point[metric][0], values[1] - values[0]),
        ):
            low, high = np.percentile(distribution, [2.5, 97.5])
            rows.append(dict(method=method, metric=metric, value=value, ci_low=low, ci_high=high))
            samples[f"{method}_{metric}"] = distribution
    samples.index.name = "resample"
    return pd.DataFrame(rows), samples.reset_index()


def support_table(reference: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in reference.groupby("cell_subtype", sort=True):
        counts = group.patient.value_counts()
        shares = counts / counts.sum()
        query_counts = query.loc[query.cell_subtype == label, "patient"].value_counts()
        rows.append(
            dict(
                subtype=label,
                reference_cells=len(group),
                reference_donors=len(counts),
                reference_largest_donor=counts.index[0],
                reference_largest_donor_share=shares.iloc[0],
                reference_effective_donors=1 / (shares**2).sum(),
                query_cells=int(query_counts.sum()),
                query_donors=len(query_counts),
                query_largest_donor=query_counts.index[0] if len(query_counts) else "",
                query_largest_donor_share=query_counts.max() / query_counts.sum()
                if len(query_counts)
                else np.nan,
            )
        )
    return pd.DataFrame(rows)


def reconstruct(
    z_ref: np.ndarray,
    labels: np.ndarray,
    z_query: np.ndarray,
    saved: pd.DataFrame,
    reference_patients: np.ndarray,
    query_truth: np.ndarray,
    classes: np.ndarray | None = None,
    k: int = 15,
) -> tuple[dict, pd.DataFrame]:
    """Check exported kNN results; a missing native order is reported, never inferred."""
    native_order_supplied = classes is not None
    if classes is None:
        classes = np.unique(labels)
    classes = np.asarray(classes)
    if classes.ndim != 1 or len(set(classes)) != len(classes) or set(classes) != set(labels):
        raise ValueError("class-order inconsistency with labelled reference")
    vote = scanvi_knn(z_ref, labels, z_query, k, classes)
    confidence_error = np.max(np.abs(vote.proba.max(1) - saved.confidence.to_numpy()))
    distance_error = np.max(np.abs(vote.novelty - saved.novelty.to_numpy()))
    mismatches = int(np.sum(vote.predicted != saved.pred_fine.to_numpy()))
    if mismatches or not np.allclose(vote.proba.max(1), saved.confidence, rtol=1e-6, atol=1e-8):
        raise ValueError("reconstructed predictions or confidence disagree with saved output")
    if not np.allclose(vote.novelty, saved.novelty, rtol=1e-6, atol=1e-8):
        raise ValueError("reconstructed novelty distances disagree with saved output")
    distances, neighbours = NearestNeighbors(n_neighbors=k).fit(z_ref).kneighbors(z_query)
    weights = np.divide(1, distances, out=np.zeros_like(distances), where=distances != 0)
    zeros = distances == 0
    weights[zeros.any(1)] = zeros[zeros.any(1)]
    weights /= weights.sum(1, keepdims=True)
    rows = []
    for truth in FOCUS:
        mask = query_truth == truth
        if not mask.any():
            continue
        neighbour_labels = labels[neighbours[mask]]
        neighbour_donors = reference_patients[neighbours[mask]]
        for label in np.unique(neighbour_labels):
            is_label = neighbour_labels == label
            rows.append(
                dict(
                    true_subtype=truth,
                    neighbour_subtype=label,
                    query_cells=int(mask.sum()),
                    neighbour_fraction=is_label.mean(),
                    mean_vote_weight=(weights[mask] * is_label).sum(1).mean(),
                    cells_with_neighbour=int(is_label.any(1).sum()),
                    mean_unique_neighbour_donors=np.mean(
                        [len(np.unique(row)) for row in neighbour_donors]
                    ),
                )
            )
        # Include an explicit zero-support row, particularly for CMS4.
        if truth not in neighbour_labels:
            rows.append(
                dict(
                    true_subtype=truth,
                    neighbour_subtype=truth,
                    query_cells=int(mask.sum()),
                    neighbour_fraction=0.0,
                    mean_vote_weight=0.0,
                    cells_with_neighbour=0,
                    mean_unique_neighbour_donors=np.mean(
                        [len(np.unique(row)) for row in neighbour_donors]
                    ),
                )
            )
    checks = dict(
        reference_cells=len(z_ref),
        query_cells=len(z_query),
        reference_classes=len(classes),
        classes=classes.tolist(),
        native_order_metadata_supplied=native_order_supplied,
        native_order_status="supplied class list validated against reference"
        if native_order_supplied
        else "unverified; sorted reference fallback used",
        k=k,
        prediction_mismatches=mismatches,
        maximum_confidence_error=float(confidence_error),
        maximum_novelty_error=float(distance_error),
        maximum_probability_sum_error=float(np.abs(vote.proba.sum(1) - 1).max()),
        probability_ties=int(((vote.proba == vote.proba.max(1, keepdims=True)).sum(1) > 1).sum()),
        zero_distance_neighbours=int(zeros.sum()),
        tolerance=dict(rtol=1e-6, atol=1e-8),
    )
    return checks, pd.DataFrame(rows)


def analyse(
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    query: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Summarise an aligned experiment using the unchanged benchmark label rules."""
    validate_annotations(reference, "reference")
    validate_annotations(query, "query")
    reference, query = reference.copy(), query.copy()
    reference_total_cells = len(reference)
    reference.cell_subtype = harmonise_reference(reference.cell_subtype)
    query.cell_subtype = harmonise_query(query.cell_subtype)
    reference = reference[reference.cell_subtype != UNKNOWN]
    role = query_role(query.cell_subtype, set(reference.cell_subtype))
    frames = validate_predictions(predictions, query, reference)
    shared = query[role == "shared"]
    if shared.empty:
        raise ValueError("no shared query cells")
    if shared.patient.nunique() < 2:
        raise ValueError("patient sensitivity analysis requires at least two query patients")
    support = support_table(reference, shared)
    tables, metrics, per_class, confusion, grouped = {}, [], [], [], []
    for method, frame in frames.items():
        pred = frame.loc[shared.index, "pred_fine"].to_numpy()
        truth = shared.cell_subtype.to_numpy()
        metrics.append(dict(method=method, **score_group(truth, pred)))
        labels = np.unique(truth)
        precision, recall, f1, n = precision_recall_fscore_support(
            truth, pred, labels=labels, zero_division=0
        )
        for label, p, r, f, count in zip(labels, precision, recall, f1, n, strict=True):
            tp = int(np.sum((truth == label) & (pred == label)))
            per_class.append(
                dict(
                    method=method,
                    subtype=label,
                    precision=p,
                    recall=r,
                    f1=f,
                    tp=tp,
                    fp=int(np.sum(pred == label)) - tp,
                    fn=int(count) - tp,
                    query_support=int(count),
                )
            )
        for factor in ("patient", "tissue"):
            for level, group in shared.groupby(factor, sort=True):
                scores = score_group(
                    group.cell_subtype.to_numpy(), frame.loc[group.index, "pred_fine"].to_numpy()
                )
                grouped.append(
                    dict(method=method, factor=factor, level=level, n_cells=len(group), **scores)
                )
        if method in PAIR:
            counts = pd.crosstab(pd.Series(truth, name="truth"), pd.Series(pred, name="prediction"))
            for source in FOCUS:
                if source not in counts.index:
                    continue
                for target, count in counts.loc[source].items():
                    if count:
                        confusion.append(
                            dict(
                                method=method,
                                truth=source,
                                prediction=target,
                                count=int(count),
                                fraction=count / np.sum(truth == source),
                            )
                        )
    native, knn = [frames[m].loc[shared.index, "pred_fine"].to_numpy() for m in PAIR]
    truth = shared.cell_subtype.to_numpy()
    tables["bootstrap"], tables["bootstrap_draws"] = paired_bootstrap(
        truth, native, knn, shared.patient.to_numpy(), n_boot, seed
    )
    tables["metrics"] = pd.DataFrame(metrics)
    tables["subtypes"] = pd.DataFrame(per_class).merge(
        support, on="subtype", validate="many_to_one"
    )
    tables["support"] = support
    tables["groups"] = pd.DataFrame(grouped)
    group_rows = []
    for factor in ("patient", "tissue"):
        group = tables["groups"].query("factor == @factor")
        for metric in ("macro_f1", "accuracy"):
            wide = group.pivot(index="level", columns="method", values=metric)
            for level, row in wide.iterrows():
                group_rows.append(
                    dict(
                        factor=factor, level=level, metric=metric, delta=row[PAIR[1]] - row[PAIR[0]]
                    )
                )
    tables["group_differences"] = pd.DataFrame(group_rows)
    sensitivities = []
    for metric in ("macro_f1", "accuracy"):
        delta = tables["group_differences"].query("factor == 'patient' and metric == @metric")
        delta = delta.set_index("level").delta
        weights = tables["bootstrap_draws"][[f"draw_{p}" for p in delta.index]].to_numpy()
        low, high = np.percentile(weights @ delta.to_numpy() / len(delta), [2.5, 97.5])
        sensitivities.append(
            dict(
                analysis="equal_patient_mean",
                omitted_patient="",
                metric=metric,
                value=delta.mean(),
                ci_low=low,
                ci_high=high,
            )
        )
    for patient in sorted(shared.patient.unique()):
        mask = shared.patient.to_numpy() != patient
        scores = [score_group(truth[mask], p[mask]) for p in (native, knn)]
        for metric in ("macro_f1", "accuracy"):
            sensitivities.append(
                dict(
                    analysis="leave_one_patient_out",
                    omitted_patient=patient,
                    metric=metric,
                    value=scores[1][metric] - scores[0][metric],
                )
            )
    tables["patient_sensitivity"] = pd.DataFrame(sensitivities)
    class_f1 = tables["subtypes"].pivot(index="subtype", columns="method", values="f1")
    effect = class_f1[PAIR[1]] - class_f1[PAIR[0]]
    abundance = support.set_index("subtype").loc[effect.index, "reference_cells"]
    if len(effect) >= 3 and abundance.nunique() > 1 and effect.nunique() > 1:
        rho, pvalue = spearmanr(abundance, effect)
    else:
        rho, pvalue = np.nan, np.nan
    tables["support_association"] = pd.DataFrame(
        [dict(n_subtypes=len(effect), spearman_rho=rho, nominal_pvalue=pvalue)]
    )
    tables["confusions"] = pd.DataFrame(confusion)
    tables["donors"] = pd.concat(
        [
            frame.groupby(["cell_subtype", "patient"])
            .size()
            .rename("cells")
            .reset_index()
            .assign(cohort=name)
            for name, frame in (("reference", reference), ("query_shared", shared))
        ],
        ignore_index=True,
    )
    disagreement = pd.DataFrame(
        dict(
            cell=shared.index,
            patient=shared.patient.to_numpy(),
            truth=truth,
            native=native,
            knn=knn,
        )
    )
    disagreement["disagree"] = native != knn
    disagreement["corrected"] = (native != truth) & (knn == truth)
    disagreement["introduced"] = (native == truth) & (knn != truth)
    # Aggregate counts retain the scientific result without duplicating per-cell predictions.
    tables["disagreements"] = (
        disagreement.groupby("patient", sort=True)[["disagree", "corrected", "introduced"]]
        .sum()
        .reset_index()
    )
    scored = role.isin(["shared", "novel"])
    tables["novelty"] = pd.DataFrame(
        [
            dict(
                method=m,
                **novelty_scores(
                    role[scored].to_numpy() == "novel", f.loc[scored, "novelty"].to_numpy()
                ),
            )
            for m, f in frames.items()
        ]
    )
    if "scanvi_latent_distance" in frames:
        if not np.array_equal(
            frames["scanvi_knn"].novelty, frames["scanvi_latent_distance"].novelty
        ):
            raise ValueError("reused latent-distance novelty arrays differ")
    checks = dict(
        reference_total_cells=reference_total_cells,
        query_patients=query.patient.nunique(),
        shared_query_cells=len(shared),
        shared_query_classes=shared.cell_subtype.nunique(),
        query_role_counts=role.value_counts().to_dict(),
        seed=seed,
        n_boot=n_boot,
        reused_novelty_identical=True if "scanvi_latent_distance" in frames else None,
    )
    return tables, checks


def run_analysis(
    predictions_path: Path,
    embeddings_path: Path,
    reference_path: Path,
    query_path: Path,
    historical_dir: Path,
    native_classes_path: Path | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    k: int = 15,
) -> tuple[dict[str, pd.DataFrame], dict]:
    historical_evidence = verify_snapshot(historical_dir)
    ref, qry = read_annotation(reference_path), read_annotation(query_path)
    validate_annotations(ref, "reference")
    validate_annotations(qry, "query")
    ref.cell_subtype, qry.cell_subtype = (
        harmonise_reference(ref.cell_subtype),
        harmonise_query(qry.cell_subtype),
    )
    labelled = ref[ref.cell_subtype != UNKNOWN]
    embeddings = ad.read_h5ad(embeddings_path)
    obs = embeddings.obs.astype(str)
    if not obs.index.is_unique or obs.index.hasnans:
        raise ValueError("embeddings: duplicate or missing cell IDs")
    required_obs = {"cohort", "patient", "sample", "cell_type", "cell_subtype"}
    if (
        not required_obs <= set(obs.columns)
        or embeddings.obs[list(required_obs)].isna().any().any()
    ):
        raise ValueError("embeddings: missing observation metadata")
    if "scANVI" not in embeddings.obsm or set(obs.cohort) != {"SMC", "KUL3"}:
        raise ValueError("embeddings: missing scANVI space or invalid cohorts")
    masks = [obs.cohort == cohort for cohort in ("SMC", "KUL3")]
    for mask, ann in zip(masks, (labelled, qry), strict=True):
        cells = obs.index[mask]
        if set(cells) != set(ann.index):
            raise ValueError("embeddings: missing or extra annotation cells")
        columns = ["patient", "sample", "cell_type", "cell_subtype"]
        if not np.array_equal(
            obs.loc[cells, columns].to_numpy(), ann.loc[cells, columns].astype(str).to_numpy()
        ):
            raise ValueError("embeddings: label or donor mismatch with annotations")
    latent = np.asarray(embeddings.obsm["scANVI"])
    if (
        latent.ndim != 2
        or latent.shape[0] != len(obs)
        or latent.shape[1] == 0
        or not np.isfinite(latent).all()
    ):
        raise ValueError("embeddings: invalid latent array")
    predictions = pd.read_parquet(predictions_path)
    tables, checks = analyse(predictions, ref, qry, n_boot, seed)
    ids = obs.index[masks[1]]
    saved = predictions[predictions.method == "scanvi_knn"].set_index("cell").loc[ids]
    evidence = None if native_classes_path is None else json.loads(native_classes_path.read_text())
    if isinstance(evidence, dict):
        if evidence.get("model_name") != "SCANVI" or evidence.get("scvi_version") != "1.5.1":
            raise ValueError("unsupported checkpoint class-order evidence")
        classes = np.asarray(evidence["native_classes"])
    else:
        classes = None if evidence is None else np.asarray(evidence)
    reconstruction, tables["neighbours"] = reconstruct(
        latent[masks[0]],
        labelled.loc[obs.index[masks[0]], "cell_subtype"].to_numpy(),
        latent[masks[1]],
        saved,
        obs.loc[masks[0], "patient"].to_numpy(),
        qry.loc[ids, "cell_subtype"].to_numpy(),
        classes,
        k,
    )
    checks.update(reconstruction)
    checks["historical_snapshot"] = historical_evidence
    if isinstance(evidence, dict):
        checks["native_order_status"] = evidence["order_basis"]
        checks["native_order_evidence"] = evidence
    historic = pd.read_csv(historical_dir / "metrics_fine.csv")
    historic = historic.pivot(index="method", columns="metric", values="value").reset_index()
    tables["historical_metrics"] = tables["metrics"].merge(
        historic, on="method", how="left", suffixes=("_local", "_archived"), validate="one_to_one"
    )
    historic_class = (
        pd.read_csv(historical_dir / "per_class_f1_fine.csv", index_col=0)
        .rename_axis("subtype")
        .reset_index()
        .melt(id_vars="subtype", var_name="method", value_name="f1_archived")
    )
    tables["historical_subtypes"] = (
        tables["subtypes"][["method", "subtype", "f1"]]
        .merge(historic_class, on=["method", "subtype"], how="left", validate="one_to_one")
        .rename(columns={"f1": "f1_local"})
    )
    tables["historical_subtypes"]["delta_local_minus_archived"] = (
        tables["historical_subtypes"].f1_local - tables["historical_subtypes"].f1_archived
    )
    return tables, checks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
