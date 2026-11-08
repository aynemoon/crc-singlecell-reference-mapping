"""Main experiment: annotate the KUL3 query with an SMC reference, for every method.

Writes
  results/tables/predictions.parquet            per-cell predictions, confidence, novelty
  results/tables/metrics_{fine,coarse}.csv      patient-bootstrap 95% CIs
  results/tables/per_class_f1_fine.csv
  results/tables/novelty_natural.csv            AUROC for cell types the reference never had
  data/processed/embeddings.h5ad                reference+query embeddings for scib-metrics
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd

from scmap.config import load_config
from scmap.evaluate import novelty_scores, patient_bootstrap, per_class_f1
from scmap.pipeline import coarse_of, predictions_frame, run_transfer


def main() -> None:
    cfg = load_config()
    proc, tables = cfg.path("processed"), cfg.path("results") / "tables"
    ref = ad.read_h5ad(proc / "reference.h5ad")
    qry = ad.read_h5ad(proc / "query.h5ad")
    lab = cfg["labels"]

    transfers, emb = run_transfer(ref, qry, cfg.raw, cfg.seed, model_dir=cfg.path("models"))
    parent = coarse_of(ref, lab["fine"], lab["coarse"])
    preds = predictions_frame(qry, transfers, parent)
    preds.to_parquet(tables / "predictions.parquet")
    # Preserve the actual native probability order for later frozen-output verification.
    (tables / "native_class_order.json").write_text(
        json.dumps(transfers["scanvi"].classes.tolist(), indent=2) + "\n"
    )

    obs = qry.obs
    role = obs["label_role"].to_numpy()
    fine_true = obs[lab["fine"]].astype(str).to_numpy()
    coarse_true = obs[lab["coarse"]].astype(str).to_numpy()
    patients = obs["patient"].astype(str).to_numpy()
    n_boot, seed = cfg["evaluation"]["n_boot"], cfg.seed

    fine_rows, coarse_rows, pcf, nov = [], [], {}, []
    for method, p in preds.groupby("method", sort=False):
        p = p.set_index("cell").loc[obs.index]
        shared = role == "shared"
        fine_rows.append(
            patient_bootstrap(
                fine_true[shared], p["pred_fine"].to_numpy()[shared], patients[shared], n_boot, seed
            ).assign(method=method)
        )
        coarse_rows.append(
            patient_bootstrap(
                coarse_true, p["pred_coarse"].to_numpy(), patients, n_boot, seed
            ).assign(method=method)
        )
        pcf[method] = per_class_f1(fine_true[shared], p["pred_fine"].to_numpy()[shared])
        scored = np.isin(role, ["shared", "novel"])
        nov.append(
            {
                "method": method,
                **novelty_scores(role[scored] == "novel", p["novelty"].to_numpy()[scored]),
            }
        )

    pd.concat(fine_rows).to_csv(tables / "metrics_fine.csv", index=False)
    pd.concat(coarse_rows).to_csv(tables / "metrics_coarse.csv", index=False)
    pd.DataFrame(pcf).to_csv(tables / "per_class_f1_fine.csv")
    pd.DataFrame(nov).to_csv(tables / "novelty_natural.csv", index=False)

    # embeddings of labelled reference cells + all query cells, for the integration benchmark
    cols = ["cohort", "patient", "sample", lab["coarse"], lab["fine"]]
    ref_obs = ref.obs.loc[ref.obs[lab["fine"]] != lab["unlabeled"], cols]
    joint_obs = pd.concat([ref_obs, qry.obs[cols]]).astype(str)
    obsm = {name: np.vstack([zr, zq]).astype(np.float32) for name, (zr, zq, _) in emb.items()}
    ad.AnnData(obs=joint_obs, obsm=obsm).write_h5ad(proc / "embeddings.h5ad")

    summary = pd.concat(fine_rows).query("metric == 'macro_f1'")[
        ["method", "value", "ci_low", "ci_high"]
    ]
    print("fine-level macro-F1 (shared subtypes):\n", summary.round(3).to_string(index=False))
    print(
        "novelty AUROC (natural unseen types):\n", pd.DataFrame(nov).round(3).to_string(index=False)
    )


if __name__ == "__main__":
    main()
