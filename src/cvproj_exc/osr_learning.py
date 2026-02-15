from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression

from cvproj_exc.config import Config

UNKNOWN_LABEL: Final[int] = -1


@dataclass
class _OpenSetModel:
    mean: np.ndarray
    projection: np.ndarray
    transform_scale: np.ndarray
    known_labels: np.ndarray
    known_centers: np.ndarray
    unknown_centers: np.ndarray | None
    gate: LogisticRegression | None
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    threshold: float


def _pairwise_l2(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-values))


def _fit_feature_transform(x: np.ndarray, target_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a compact linear transform:
    1) center
    2) project to top principal components
    3) per-dimension scaling
    """
    mean = np.mean(x, axis=0)
    xc = x - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    reduced_dim = max(1, min(target_dim, vt.shape[0], x.shape[1]))
    projection = vt[:reduced_dim].T.astype(np.float64)
    projected = xc @ projection
    transform_scale = np.std(projected, axis=0) + 1e-6
    return mean.astype(np.float64), projection, transform_scale.astype(np.float64)


def _transform_features(
    x: np.ndarray, mean: np.ndarray, projection: np.ndarray, transform_scale: np.ndarray
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x_t = ((x - mean) @ projection) / transform_scale
    return _normalize_rows(x_t)


def _compute_centroids(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.unique(y)
    centers = np.vstack([x[y == label].mean(axis=0) for label in labels]).astype(np.float64)
    centers = _normalize_rows(centers)
    return labels.astype(int), centers


def _compute_distance_features(
    x: np.ndarray, known_centers: np.ndarray, unknown_centers: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    known_dists = _pairwise_l2(x, known_centers)
    nearest_known_idx = np.argmin(known_dists, axis=1)
    d1 = known_dists[np.arange(x.shape[0]), nearest_known_idx]

    if known_centers.shape[0] > 1:
        d2 = np.partition(known_dists, kth=1, axis=1)[:, 1]
    else:
        d2 = d1 + 1.0

    if unknown_centers is None or unknown_centers.size == 0:
        du = d1 + np.median(d2 - d1) + 1e-3
    else:
        unknown_dists = _pairwise_l2(x, unknown_centers)
        du = np.min(unknown_dists, axis=1)

    # Features for known-vs-unknown gating.
    # - d1: nearest known prototype distance (smaller is more likely known)
    # - d2-d1: margin inside known classes (larger is more likely known)
    # - du-d1: relative distance to unknown prototypes (larger is more likely known)
    features = np.column_stack([d1, d2 - d1, du - d1]).astype(np.float64)
    return nearest_known_idx, d1, features


def _select_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    known_mask = labels == 1
    unknown_mask = labels == 0
    if not np.any(known_mask) or not np.any(unknown_mask):
        return 0.5

    candidate_thresholds = np.quantile(scores, np.linspace(0.01, 0.99, 99))
    best_threshold = 0.5
    best_objective = -np.inf
    best_far = np.inf

    for threshold in candidate_thresholds:
        tpr = float(np.mean(scores[known_mask] >= threshold))
        far = float(np.mean(scores[unknown_mask] >= threshold))
        tnr = 1.0 - far
        balanced_acc = 0.5 * (tpr + tnr)

        # Slight FAR penalty biases to safer unknown rejection.
        objective = balanced_acc - 0.05 * far
        if objective > best_objective or (
            np.isclose(objective, best_objective) and far < best_far
        ):
            best_threshold = float(threshold)
            best_objective = objective
            best_far = far

    return best_threshold


def _fit_gate(
    known_features: np.ndarray, unknown_features: np.ndarray
) -> tuple[LogisticRegression | None, np.ndarray, np.ndarray, float]:
    if unknown_features.size == 0:
        feature_mean = np.mean(known_features, axis=0)
        feature_scale = np.std(known_features, axis=0) + 1e-6
        return None, feature_mean, feature_scale, 0.5

    x_gate = np.vstack([known_features, unknown_features]).astype(np.float64)
    y_gate = np.concatenate(
        [
            np.ones(known_features.shape[0], dtype=int),
            np.zeros(unknown_features.shape[0], dtype=int),
        ]
    )

    feature_mean = np.mean(x_gate, axis=0)
    feature_scale = np.std(x_gate, axis=0) + 1e-6
    x_gate_norm = (x_gate - feature_mean) / feature_scale

    gate = LogisticRegression(
        random_state=42,
        max_iter=500,
        solver="lbfgs",
        class_weight="balanced",
        C=3.0,
    )
    gate.fit(x_gate_norm, y_gate)
    known_scores = gate.predict_proba(x_gate_norm)[:, 1]
    unknown_scores = known_scores[y_gate == 0]
    threshold_balanced = _select_threshold(known_scores, y_gate)
    threshold_far10 = float(np.quantile(unknown_scores, 0.90))
    threshold = 0.6 * threshold_far10 + 0.4 * threshold_balanced
    threshold = float(np.clip(0.6 * threshold, 0.03, 0.4))
    return gate, feature_mean, feature_scale, threshold


def _kmeans_centroids(
    x: np.ndarray, n_clusters: int, max_iter: int = 100, seed: int = 42
) -> np.ndarray:
    n_samples = x.shape[0]
    n_clusters = max(1, min(n_clusters, n_samples))
    rng = np.random.default_rng(seed)

    center_idx = rng.choice(n_samples, size=n_clusters, replace=False)
    centers = x[center_idx].astype(np.float64)

    for _ in range(max_iter):
        dists = _pairwise_l2(x, centers)
        assignment = np.argmin(dists, axis=1)

        new_centers = centers.copy()
        for cluster_idx in range(n_clusters):
            members = x[assignment == cluster_idx]
            if members.size == 0:
                new_centers[cluster_idx] = x[rng.integers(0, n_samples)]
            else:
                new_centers[cluster_idx] = members.mean(axis=0)

        if np.allclose(new_centers, centers, rtol=1e-4, atol=1e-6):
            break
        centers = new_centers

    return centers


def _build_unknown_centers(
    x_unknown: np.ndarray, n_unknown_clusters: int
) -> np.ndarray | None:
    if x_unknown.shape[0] == 0:
        return None
    if n_unknown_clusters <= 1 or x_unknown.shape[0] == 1:
        return _normalize_rows(np.mean(x_unknown, axis=0, keepdims=True))

    n_clusters = min(n_unknown_clusters, x_unknown.shape[0])
    try:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=5,
            batch_size=min(1024, x_unknown.shape[0]),
            max_iter=200,
        )
        centers = kmeans.fit(x_unknown).cluster_centers_
    except Exception:
        centers = _kmeans_centroids(x_unknown, n_clusters=n_clusters, max_iter=100, seed=42)
    return _normalize_rows(centers.astype(np.float64))


def _choose_projection_dim(n_features: int) -> int:
    # For 128-d embeddings this keeps a compact, high-signal subspace.
    if n_features >= 96:
        return 48
    if n_features >= 64:
        return 40
    if n_features >= 32:
        return 24
    return n_features


def _train_open_set_model(
    x_train: np.ndarray, y_train: np.ndarray, n_unknown_clusters: int, target_dim: int
) -> _OpenSetModel | None:
    known_mask = y_train != UNKNOWN_LABEL
    if not np.any(known_mask):
        return None

    x_known = x_train[known_mask]
    y_known = y_train[known_mask]

    mean, projection, transform_scale = _fit_feature_transform(x_known, target_dim=target_dim)
    x_known_t = _transform_features(x_known, mean, projection, transform_scale)
    x_unknown_t = _transform_features(x_train[~known_mask], mean, projection, transform_scale)

    known_labels, known_centers = _compute_centroids(x_known_t, y_known)
    unknown_centers = _build_unknown_centers(x_unknown_t, n_unknown_clusters=n_unknown_clusters)

    _, _, known_features = _compute_distance_features(x_known_t, known_centers, unknown_centers)
    if x_unknown_t.shape[0] > 0:
        _, _, unknown_features = _compute_distance_features(x_unknown_t, known_centers, unknown_centers)
    else:
        unknown_features = np.empty((0, 3), dtype=np.float64)

    gate, feature_mean, feature_scale, threshold = _fit_gate(known_features, unknown_features)
    return _OpenSetModel(
        mean=mean,
        projection=projection,
        transform_scale=transform_scale,
        known_labels=known_labels,
        known_centers=known_centers,
        unknown_centers=unknown_centers,
        gate=gate,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        threshold=threshold,
    )


def _predict_with_model(model: _OpenSetModel | None, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_test = np.asarray(x_test, dtype=np.float64)
    if x_test.ndim != 2:
        raise ValueError("x_test must be a 2D array of shape (n_samples, n_features).")

    if model is None:
        n_test = x_test.shape[0]
        return (
            np.full(n_test, UNKNOWN_LABEL, dtype=int),
            np.zeros(n_test, dtype=np.float64),
        )

    x_test_t = _transform_features(x_test, model.mean, model.projection, model.transform_scale)
    nearest_known_idx, _, features = _compute_distance_features(
        x_test_t, model.known_centers, model.unknown_centers
    )

    features_norm = (features - model.feature_mean) / model.feature_scale
    if model.gate is None:
        known_prob = _sigmoid(-3.0 * features_norm[:, 0] + 2.0 * features_norm[:, 1])
    else:
        known_prob = model.gate.predict_proba(features_norm)[:, 1]

    y_pred = model.known_labels[nearest_known_idx].astype(int, copy=True)
    reject_by_threshold = known_prob < model.threshold
    reject_by_unknown_prototype = features[:, 2] < 0.0
    y_pred[reject_by_threshold | reject_by_unknown_prototype] = UNKNOWN_LABEL
    y_score = known_prob.astype(np.float64)
    return y_pred, y_score


def spl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the single pseudo label (SPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    spl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """

    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=int)
    target_dim = _choose_projection_dim(x_train.shape[1])
    model = _train_open_set_model(
        x_train, y_train, n_unknown_clusters=1, target_dim=target_dim
    )

    def spl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _predict_with_model(model, x_test)

    return spl_predict_fn


def mpl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the multi pseudo label (MPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    mpl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """

    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=int)

    n_unknown = int(np.sum(y_train == UNKNOWN_LABEL))
    n_unknown_clusters = 1 if n_unknown <= 1 else min(128, max(8, int(np.sqrt(n_unknown) * 2)))
    target_dim = _choose_projection_dim(x_train.shape[1])
    model = _train_open_set_model(
        x_train,
        y_train,
        n_unknown_clusters=n_unknown_clusters,
        target_dim=target_dim,
    )

    def mpl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _predict_with_model(model, x_test)

    return mpl_predict_fn


def load_challenge_train_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the challenge training data.

    Returns
    -------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.
    """
    import pandas as pd

    df = pd.read_csv(Config.CHAL_TRAIN_DATA, header=None).values
    x = df[:, :-1]
    y = df[:, -1].astype(int)
    return x, y


def main():
    x_train, y_train = load_challenge_train_data()

    spl_predict_fn = spl_training(x_train, y_train)

    mpl_predict_fn = mpl_training(x_train, y_train)

    # This is roughly how the trained predictors are evaluated (with real data).
    x_test = np.random.rand(50, x_train.shape[1])
    y_test = np.random.randint(-1, 5, 50)
    for predict_fn in (spl_predict_fn, mpl_predict_fn):
        y_pred, y_score = predict_fn(x_test)
        print("Acc: {}".format(np.equal(y_test, y_pred).sum() / len(x_test)))


if __name__ == "__main__":
    main()
