import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from scmap import pipeline
from scmap.config import load_config
from scmap.plotting import CLASSIFIERS


def test_transfer_uses_reference_labels_and_keeps_query_order(monkeypatch, tmp_path):
    def cells(ids, values, labels, sample):
        counts = csr_matrix(np.column_stack([values, 12 - np.array(values)]), dtype=np.float32)
        obs = pd.DataFrame({"sample": sample, "cell_subtype": labels}, index=ids)
        return ad.AnnData(X=counts, obs=obs, layers={"counts": counts.copy()})

    ref = cells(
        ["r1", "r2", "r3", "r4", "r5"], [1, 10, 9, 2, 8], ["z", "Unknown", "a", "z", "a"], "SMC"
    )
    query = cells(["q9", "q2", "q7"], [8, 1, 9], ["z", "a", "z"], "KUL")
    query.obs["label_role"] = "shared"
    query.obs["cell_type"] = "hidden lineage"
    cfg = load_config().raw
    cfg["baselines"].update(knn_k=2, n_pcs=1)

    class FrozenModel:
        def __init__(self, data):
            self.data = data

        def get_latent_representation(self):
            return self.data.X[:, :1].toarray()

    def train(data, *args):
        return FrozenModel(data), FrozenModel(data)

    def map_unlabelled(data, *args):
        assert list(data.obs.columns) == ["sample"]
        return FrozenModel(data)

    def native(model):
        # Native predictions deliberately differ from the kNN predictions.
        p = np.tile([0.8, 0.1, 0.1], (model.data.n_obs, 1))
        return p, np.array(["z", "Unknown", "a"])

    monkeypatch.setattr(pipeline, "select_hvgs", lambda data, _: list(data.var_names))
    monkeypatch.setattr(pipeline, "train_reference", train)
    monkeypatch.setattr(pipeline, "map_query", map_unlabelled)
    monkeypatch.setattr(pipeline, "scanvi_soft", native)
    transfers, embeddings = pipeline.run_transfer(ref, query, cfg, 0, tmp_path)
    vote = transfers["scanvi_knn"]
    assert CLASSIFIERS == pipeline.METHODS
    np.testing.assert_array_equal(vote.classes, transfers["scanvi"].classes)
    np.testing.assert_array_equal(vote.predicted, ["a", "z", "a"])
    np.testing.assert_array_equal(transfers["scanvi"].predicted, ["z", "z", "z"])
    np.testing.assert_array_equal(
        transfers["scanvi_latent_distance"].proba, transfers["scanvi"].proba
    )
    assert transfers["scanvi_latent_distance"].novelty is vote.novelty
    np.testing.assert_array_equal(embeddings["scANVI"][0].ravel(), [1, 9, 2, 8])
    frame = pipeline.predictions_frame(query, {"scanvi_knn": vote}, {"z": "Z", "a": "A"})
    assert frame["cell"].tolist() == ["q9", "q2", "q7"]

    changed = query.copy()
    changed.obs["cell_subtype"] = ["a", "z", "a"]
    changed.obs["cell_type"] = "adversarial lineage"
    changed.obs["label_role"] = "novel"
    altered, _ = pipeline.run_transfer(ref, changed, cfg, 0, tmp_path)
    for method, original in transfers.items():
        np.testing.assert_array_equal(altered[method].classes, original.classes)
        np.testing.assert_array_equal(altered[method].proba, original.proba)
        np.testing.assert_array_equal(altered[method].novelty, original.novelty)
    assert query.obs["label_role"].eq("shared").all()
    assert query.obs["cell_subtype"].tolist() == ["z", "a", "z"]

    masked = query[::-1].copy()
    masked.obs = masked.obs[["sample"]].copy()
    repeated, _ = pipeline.run_transfer(ref, masked, cfg, 0, tmp_path)
    np.testing.assert_array_equal(repeated["scanvi_knn"].proba, vote.proba[::-1])
