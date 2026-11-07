"""Non-deep-learning label-transfer baselines.

* `logreg`: L2 multinomial logistic regression on scaled log-normalised HVGs, the approach
  behind CellTypist-style classifiers.
* `knn_pca`: k-nearest-neighbour vote in a PCA space fitted on the reference, the classic
  projection-based transfer.
Both return class probabilities and a novelty score (higher = more likely unseen).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class Transfer:
    proba: np.ndarray  # (n_query, n_classes), columns follow `classes`
    classes: np.ndarray
    novelty: np.ndarray  # higher = less like anything in the reference

    @property
    def predicted(self) -> np.ndarray:
        return self.classes[self.proba.argmax(1)]


def _scale(X_ref: np.ndarray, X_qry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sc = StandardScaler().fit(X_ref)
    return np.clip(sc.transform(X_ref), -10, 10), np.clip(sc.transform(X_qry), -10, 10)


def logreg(X_ref, y_ref, X_qry, C: float, seed: int) -> Transfer:
    A, B = _scale(X_ref, X_qry)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(A, y_ref)
    proba = clf.predict_proba(B)
    return Transfer(proba, clf.classes_, 1.0 - proba.max(1))


def knn_on_embedding(Z_ref, y_ref, Z_qry, k: int) -> Transfer:
    """kNN label vote in a shared embedding; novelty = mean distance to the k references."""
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(Z_ref, y_ref)
    proba = clf.predict_proba(Z_qry)
    dist, _ = NearestNeighbors(n_neighbors=k).fit(Z_ref).kneighbors(Z_qry)
    return Transfer(proba, clf.classes_, dist.mean(1))


def scanvi_knn(
    z_ref: np.ndarray, y_ref: np.ndarray, z_qry: np.ndarray, k: int, classes: np.ndarray
) -> Transfer:
    """Vote on frozen scANVI latents, with probability columns in native scANVI order."""
    vote = knn_on_embedding(z_ref, y_ref, z_qry, k)
    column = {label: i for i, label in enumerate(vote.classes)}
    if set(classes) != column.keys() or len(classes) != len(column):
        raise ValueError("scANVI classes must match the labelled reference classes")
    return Transfer(vote.proba[:, [column[label] for label in classes]], classes, vote.novelty)


def knn_pca(
    X_ref, y_ref, X_qry, n_pcs: int, k: int, seed: int
) -> tuple[Transfer, np.ndarray, np.ndarray]:
    A, B = _scale(X_ref, X_qry)
    pca = PCA(n_components=n_pcs, random_state=seed).fit(A)
    Z_ref, Z_qry = pca.transform(A), pca.transform(B)
    return knn_on_embedding(Z_ref, y_ref, Z_qry, k), Z_ref, Z_qry
