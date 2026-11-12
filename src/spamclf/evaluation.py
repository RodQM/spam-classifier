"""Held-out evaluation metrics and model introspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Metrics:
    """Test-set metrics with spam as the positive class.

    Accuracy is reported but is a weak signal here: predicting "ham" for every
    message already scores ~87%. Precision (how many flagged messages really are
    spam) and recall (how much spam is caught) are the numbers that matter.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    average_precision: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def confusion(self) -> npt.NDArray[np.int_]:
        """Confusion matrix as ``[[TN, FP], [FN, TP]]`` (rows: truth, columns: prediction)."""
        return np.array(
            [
                [self.true_negatives, self.false_positives],
                [self.false_negatives, self.true_positives],
            ]
        )


def decision_scores(pipeline: Pipeline, X: npt.ArrayLike) -> FloatArray:
    """Continuous "spamminess" score per message, higher meaning more spam-like.

    Uses the spam probability when the model provides one and the signed distance
    to the decision boundary otherwise (e.g. linear SVMs).
    """
    if hasattr(pipeline, "predict_proba"):
        scores: FloatArray = pipeline.predict_proba(X)[:, 1]
    else:
        scores = pipeline.decision_function(X)
    return scores


def evaluate(pipeline: Pipeline, X: npt.ArrayLike, y: npt.ArrayLike) -> Metrics:
    """Compute test-set metrics for a fitted pipeline."""
    predictions = pipeline.predict(X)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return Metrics(
        accuracy=float(accuracy_score(y, predictions)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        average_precision=float(average_precision_score(y, decision_scores(pipeline, X))),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )


def top_spam_tokens(pipeline: Pipeline, k: int = 15) -> list[tuple[str, float]]:
    """Return the ``k`` tokens that push a message hardest towards "spam".

    For linear models the weight is the coefficient; for Naive Bayes it is the
    log-likelihood ratio ``log P(token | spam) - log P(token | ham)``.
    """
    vectorizer, classifier = pipeline[0], pipeline[-1]
    tokens = vectorizer.get_feature_names_out()

    if hasattr(classifier, "coef_"):
        weights = np.ravel(classifier.coef_)
    elif hasattr(classifier, "feature_log_prob_"):
        weights = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    else:
        raise TypeError(f"{type(classifier).__name__} exposes no per-token weights.")

    top = np.argsort(weights)[::-1][:k]
    return [(str(tokens[i]), float(weights[i])) for i in top]
