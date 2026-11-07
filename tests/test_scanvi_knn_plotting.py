import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from scmap.plotting import CLASSIFIERS  # noqa: E402


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("include_knn", [False, True])
def test_figures_accept_archived_and_expanded_results(tmp_path, monkeypatch, include_knn):
    figures = load_script("05_figures.py")
    patients = load_script("08_per_patient.py")
    methods = CLASSIFIERS if include_knn else CLASSIFIERS[:-1]
    subtypes = ["CMS2", "CMS4", "T helper 17 cells"]
    for level in ("fine", "coarse"):
        pd.DataFrame(
            [dict(method=m, metric="macro_f1", value=0.7, ci_low=0.6, ci_high=0.8) for m in methods]
        ).to_csv(tmp_path / f"metrics_{level}.csv", index=False)
    pd.DataFrame({m: [0.5, 0.0, 0.8] for m in methods}, index=subtypes).to_csv(
        tmp_path / "per_class_f1_fine.csv"
    )
    pd.DataFrame({"cell_subtype": subtypes, "cell_type": ["Epithelial", "Epithelial", "T"]}).to_csv(
        tmp_path / "label_roles.csv", index=False
    )
    pd.DataFrame(
        [dict(method=m, held_out="CMS4", auroc=0.6) for m in figures.NOVELTY_METHODS]
    ).to_csv(tmp_path / "open_set_auroc.csv", index=False)
    obs = pd.DataFrame(
        {
            "cell_subtype": [*subtypes, "Anti-inflammatory", "BEST4+ Enterocytes", "Tuft cells"],
            "label_role": ["shared"] * 3 + ["novel"] * 3,
        },
        index=[f"q{i}" for i in range(6)],
    )
    predictions = pd.DataFrame(
        [
            dict(cell=cell, method=method, pred_fine="CMS2", confidence=0.7, novelty=i + 1.0)
            for method in (*methods, "scanvi_latent_distance")
            for i, cell in enumerate(obs.index)
        ]
    )
    pd.DataFrame(
        dict(
            umap_1=np.arange(6),
            umap_2=np.arange(6),
            cohort=["SMC", "KUL3"] * 3,
            cell_type=["Epithelial", "T"] * 3,
        )
    ).to_csv(tmp_path / "umap_scanvi.csv", index=False)
    monkeypatch.setattr(figures, "TAB", tmp_path)
    monkeypatch.setattr(figures, "FIG", tmp_path)
    monkeypatch.setattr(figures, "_query_predictions", lambda: (predictions, obs))
    figures.main()

    patient_frame = pd.DataFrame(
        [
            dict(method=m, patient=p, n_cells=3, fine_accuracy=0.7)
            for m in methods
            for p in ("P1", "P2")
        ]
    )
    patients.plot(patient_frame, patient_frame.rename(columns={"patient": "tissue"}), tmp_path)
    assert len(list(tmp_path.glob("*.pdf"))) == 7
    assert len(list(tmp_path.glob("*.png"))) == 7
