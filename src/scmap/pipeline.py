"""One reference -> query label-transfer run for every method.

`run_transfer` is used both for the main experiment and, with a cell type removed from the
reference, for the open-set experiment.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scmap.baselines import Transfer, knn_on_embedding, knn_pca, logreg, scanvi_knn
from scmap.models import map_query, scanvi_soft, train_reference
from scmap.preprocess import lognorm, select_hvgs

METHODS = ("logreg", "knn_pca", "scvi_knn", "scanvi", "scanvi_knn")


def run_transfer(
    ref: ad.AnnData,
    qry: ad.AnnData,
    cfg: dict,
    seed: int,
    model_dir: Path | None = None,
    max_epochs_scvi: int | None = None,
) -> tuple[dict[str, Transfer], dict[str, np.ndarray]]:
    """Return per-method transfers (on the query) and embeddings (reference+query)."""
    lab = cfg["labels"]
    fine, unlabeled = lab["fine"], lab["unlabeled"]
    genes = select_hvgs(ref, cfg["data"]["n_hvg"])
    ref_h, qry_h = ref[:, genes].copy(), qry[:, genes].copy()
    # The mapping models only need the query batch. Truth and label roles stay with the scorer.
    qry_h.obs = qry_h.obs[["sample"]].copy()

    labelled = (ref.obs[fine] != unlabeled).to_numpy()
    y_ref = ref.obs[fine].astype(str).to_numpy()
    X_ref, X_qry = lognorm(ref, genes), lognorm(qry, genes)
    b = cfg["baselines"]

    out: dict[str, Transfer] = {}
    emb: dict[str, np.ndarray] = {}
    out["logreg"] = logreg(X_ref[labelled], y_ref[labelled], X_qry, b["logreg_C"], seed)
    t, z_ref, z_qry = knn_pca(X_ref[labelled], y_ref[labelled], X_qry, b["n_pcs"], b["knn_k"], seed)
    out["knn_pca"] = t
    emb["PCA (uncorrected)"] = (z_ref, z_qry, labelled)

    tmp = tempfile.TemporaryDirectory() if model_dir is None else None
    mdir = Path(tmp.name) if tmp else model_dir
    mdir.mkdir(parents=True, exist_ok=True)
    vae, lvae = train_reference(ref_h, fine, unlabeled, cfg, seed, mdir, max_epochs_scvi)

    q_vae = map_query(qry_h, mdir / "scvi", "scvi", unlabeled, cfg, seed)
    zr, zq = vae.get_latent_representation(), q_vae.get_latent_representation()
    out["scvi_knn"] = knn_on_embedding(zr[labelled], y_ref[labelled], zq, b["knn_k"])
    emb["scVI"] = (zr[labelled], zq, labelled)

    q_lvae = map_query(qry_h, mdir / "scanvi", "scanvi", unlabeled, cfg, seed)
    proba, classes = scanvi_soft(q_lvae)
    keep = classes != unlabeled
    proba, classes = proba[:, keep], classes[keep]
    proba = proba / proba.sum(1, keepdims=True)
    out["scanvi"] = Transfer(proba, classes, 1.0 - proba.max(1))
    zr, zq = lvae.get_latent_representation(), q_lvae.get_latent_representation()
    emb["scANVI"] = (zr[labelled], zq, labelled)
    vote = scanvi_knn(zr[labelled], y_ref[labelled], zq, b["knn_k"], classes)
    # Keep the native probabilities for the published distance-only variant.
    out["scanvi_latent_distance"] = Transfer(proba, classes, vote.novelty)
    out["scanvi_knn"] = vote
    if tmp:
        tmp.cleanup()
    return out, emb


def coarse_of(ref: ad.AnnData, fine_col: str, coarse_col: str) -> dict[str, str]:
    """Map every fine reference label to its coarse parent."""
    pairs = ref.obs[[fine_col, coarse_col]].astype(str).drop_duplicates()
    return dict(zip(pairs[fine_col], pairs[coarse_col], strict=True))


def predictions_frame(
    qry: ad.AnnData, transfers: dict[str, Transfer], parent: dict[str, str]
) -> pd.DataFrame:
    rows = []
    for name, t in transfers.items():
        pred = t.predicted
        rows.append(
            pd.DataFrame(
                {
                    "cell": qry.obs_names,
                    "method": name,
                    "pred_fine": pred,
                    "pred_coarse": [parent.get(p, p) for p in pred],
                    "confidence": t.proba.max(1),
                    "novelty": t.novelty,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
