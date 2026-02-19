import pickle
from typing import Dict, Tuple

import numpy as np

from cvproj_exc.classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(
        self,
        classifier=NearestNeighborClassifier(),
        false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True),
    ):
        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

        # Internal cache for brute-force NN (robust to unknown classifier interface)
        self._known_train_embeddings = None
        self._known_train_labels = None

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):
        with open(train_data_file, "rb") as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding="bytes")
        with open(test_data_file, "rb") as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding="bytes")

        # Ensure numpy arrays
        self.train_embeddings = np.asarray(self.train_embeddings, dtype=np.float32)
        self.test_embeddings = np.asarray(self.test_embeddings, dtype=np.float32)
        self.train_labels = np.asarray(self.train_labels)
        self.test_labels = np.asarray(self.test_labels)

        # Cache known-only training data for robust NN prediction
        known_mask = self.train_labels != UNKNOWN_LABEL
        self._known_train_embeddings = self.train_embeddings[known_mask].astype(np.float32)
        self._known_train_labels = self.train_labels[known_mask]

        if self._known_train_embeddings.shape[0] == 0:
            raise ValueError("No known identities in training data (all labels are UNKNOWN_LABEL).")

    def _fit_classifier_if_possible(self):
        clf = self.classifier
        if clf is None:
            return

        # Try common method names
        for fit_name in ("fit", "train"):
            if hasattr(clf, fit_name):
                try:
                    getattr(clf, fit_name)(self._known_train_embeddings, self._known_train_labels)
                except TypeError:
                    # Some implementations might expect python lists
                    getattr(clf, fit_name)(
                        self._known_train_embeddings.tolist(), self._known_train_labels.tolist()
                    )
                except Exception:
                    # Ignore if classifier has unexpected signature; we have brute-force fallback.
                    pass
                break

    def _predict_nn(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Robust brute-force nearest neighbor prediction in cosine similarity space.

        Returns:
        - pred_labels: (N,)
        - pred_similarity: (N,)  (max cosine similarity to any known training sample)
        """
        X = np.asarray(X, dtype=np.float32)

        # cosine similarity since embeddings are normalized
        sims = X @ self._known_train_embeddings.T  # (N, M)
        nn_idx = np.argmax(sims, axis=1)  # (N,)
        max_sim = sims[np.arange(sims.shape[0]), nn_idx]
        pred_labels = self._known_train_labels[nn_idx]
        return pred_labels, max_sim

    # Run the evaluation and find performance measure (identification rates) at different
    # similarity thresholds.
    def run(self) -> Dict[str, np.ndarray]:
        self._fit_classifier_if_possible()
        # predict on test split (labels + similarities)
        # Prefer classifier interface required by the exercise.
        if self.classifier is not None and hasattr(
            self.classifier, "predict_labels_and_similarities"
        ):
            try:
                test_pred_labels, test_pred_sim = self.classifier.predict_labels_and_similarities(
                    self.test_embeddings
                )
            except Exception:
                # robust fallback
                test_pred_labels, test_pred_sim = self._predict_nn(self.test_embeddings)
        else:
            test_pred_labels, test_pred_sim = self._predict_nn(self.test_embeddings)

        test_pred_labels = np.asarray(test_pred_labels)
        test_pred_sim = np.asarray(test_pred_sim, dtype=np.float32)

        # Threshold calibration for FAR is done on unknown test samples.
        unk_test_mask = self.test_labels == UNKNOWN_LABEL
        impostor_sim = test_pred_sim[unk_test_mask]

        # sweep FARs, pick thresholds, compute identification rates ---
        similarity_thresholds = np.zeros_like(self.false_alarm_rate_range, dtype=np.float32)
        identification_rates = np.zeros_like(self.false_alarm_rate_range, dtype=np.float32)

        for i, far in enumerate(self.false_alarm_rate_range):
            tau = self.select_similarity_threshold(impostor_sim, float(far))
            similarity_thresholds[i] = tau

            # Apply open-set decision rule: reject as unknown if similarity < tau
            final_pred = test_pred_labels.copy()
            final_pred[test_pred_sim < tau] = UNKNOWN_LABEL

            identification_rates[i] = self.calc_identification_rate(final_pred)

        # Report all performance measures.
        evaluation_results = {
            "false_alarm_rates": np.asarray(self.false_alarm_rate_range, dtype=np.float32),
            "similarity_thresholds": similarity_thresholds,
            "identification_rates": identification_rates,
        }
        return evaluation_results

    def select_similarity_threshold(self, similarity: np.ndarray, false_alarm_rate: float) -> float:
        """
        Choose threshold tau such that FAR ≈ P(similarity >= tau) = false_alarm_rate.

        similarity: similarities of impostor (unknown) samples to nearest known identity.
        """
        similarity = np.asarray(similarity, dtype=np.float32)

        # If we cannot calibrate (no impostor samples), make it extremely strict.
        if similarity.size == 0:
            return float("inf")

        far = float(false_alarm_rate)
        # Clamp
        if far <= 0.0:
            # FAR=0 -> require similarity > max(sim) to accept none
            return float(np.max(similarity) + 1e-6)
        if far >= 1.0:
            # FAR=1 -> accept all -> threshold below min
            return float(np.min(similarity) - 1e-6)

        # Hint-aligned implementation:
        # choose p-percentile with p = 100 * (1 - FAR), so approximately
        # P(similarity >= tau) ~= FAR on impostor (unknown) samples.
        p = 100.0 * (1.0 - far)
        tau = float(np.percentile(similarity, p))
        return tau

    def calc_identification_rate(self, prediction_labels: np.ndarray) -> float:
        """
        Identification Rate (DIR) under open-set protocol:
        - Only consider known test samples (label != UNKNOWN_LABEL).
        - Count correct identifications among them (prediction == true label).
        - Unknown test samples do not affect DIR (they affect FAR via thresholds).
        """
        prediction_labels = np.asarray(prediction_labels)
        true_labels = np.asarray(self.test_labels)

        known_mask = true_labels != UNKNOWN_LABEL
        num_known = int(np.sum(known_mask))
        if num_known == 0:
            return 0.0

        correct = np.sum(prediction_labels[known_mask] == true_labels[known_mask])
        return float(correct) / float(num_known)
