"""The kNN head and the existing native classifier share only the mapped geometry."""

import numpy as np
import pytest

from scmap.baselines import scanvi_knn


def test_scanvi_knn_probabilities_and_order():
    ref = np.array([[0.0], [1.0], [3.0], [4.0]])
    labels = np.array(["z", "z", "a", "a"])
    query = np.array([[4.0], [2.25], [0.0]])
    classes = np.array(["z", "a"])  # native order differs from sklearn's sorted classes
    vote = scanvi_knn(ref, labels, query, k=2, classes=classes)

    assert vote.proba.shape == (3, 2)
    np.testing.assert_array_equal(vote.classes, classes)
    np.testing.assert_allclose(vote.proba, [[0, 1], [0.375, 0.625], [1, 0]])
    np.testing.assert_array_equal(vote.predicted, ["a", "a", "z"])
    np.testing.assert_allclose(vote.proba.sum(axis=1), 1)
    assert np.all((vote.proba >= 0) & (vote.proba <= 1))

    np.testing.assert_allclose(vote.novelty, [0.5, 1.0, 0.5])
    shuffled = scanvi_knn(ref, labels, query[::-1], k=2, classes=classes)
    np.testing.assert_array_equal(shuffled.proba, vote.proba[::-1])


@pytest.mark.parametrize("classes", [["z"], ["z", "q"], ["z", "z", "a"]])
def test_scanvi_knn_rejects_incompatible_class_order(classes):
    with pytest.raises(ValueError, match="labelled reference classes"):
        scanvi_knn(
            np.array([[0.0], [1.0]]), np.array(["z", "a"]), np.array([[0.0]]), 1, np.array(classes)
        )
