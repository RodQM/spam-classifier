"""Multinomial Naive Bayes implemented from scratch with NumPy.

The estimator follows the scikit-learn API (``fit`` / ``predict`` /
``predict_proba``, learned attributes ending in ``_``), so it can be dropped into
a :class:`~sklearn.pipeline.Pipeline`, cross-validated and grid-searched exactly
like a built-in model.

The model
---------
For a document with token counts ``x`` and a class ``c``, Bayes' rule plus the
"naive" assumption that tokens are independent given the class yields::

    log P(c | x)  ∝  log P(c) + Σ_j  x_j · log P(token_j | c)

Both terms are estimated by counting. ``P(c)`` is the fraction of training
documents in class ``c``; ``P(token_j | c)`` is the fraction of all tokens seen in
class ``c`` that are token ``j``, with additive (Laplace) smoothing ``alpha`` so a
token never seen in a class does not zero out the whole product.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_non_negative, validate_data

FloatArray = npt.NDArray[np.float64]
# Dense arrays or SciPy sparse matrices; SciPy ships no type information.
MatrixLike = Any


class ScratchMultinomialNB(ClassifierMixin, BaseEstimator):
    """Multinomial Naive Bayes classifier for count features.

    Parameters
    ----------
    alpha:
        Additive smoothing strength. ``1.0`` is Laplace smoothing; smaller values
        trust the observed counts more.

    Attributes
    ----------
    classes_:
        Class labels, sorted.
    class_count_:
        Number of training samples per class.
    feature_count_:
        Total count of each feature per class, shape ``(n_classes, n_features)``.
    class_log_prior_:
        ``log P(c)`` for each class.
    feature_log_prob_:
        ``log P(feature | c)``, shape ``(n_classes, n_features)``.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.input_tags.positive_only = True
        # Same declaration as scikit-learn's MultinomialNB: the conformance suite
        # scores classifiers on continuous Gaussian blobs, which a count model is
        # not expected to separate well.
        tags.classifier_tags.poor_score = True
        return tags

    def fit(self, X: MatrixLike, y: npt.ArrayLike) -> Self:
        """Estimate class priors and per-class token distributions by counting."""
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha!r}.")

        X, y = validate_data(self, X, y, accept_sparse="csr")
        check_classification_targets(y)
        check_non_negative(X, "ScratchMultinomialNB (input X)")

        self.classes_, y_index = np.unique(y, return_inverse=True)
        n_samples = X.shape[0]

        # One-hot class membership, shape (n_samples, n_classes). Multiplying it
        # with X sums the rows of X per class in one (sparse-friendly) product.
        membership = np.zeros((n_samples, len(self.classes_)))
        membership[np.arange(n_samples), y_index] = 1.0

        self.class_count_ = membership.sum(axis=0)
        self.feature_count_ = np.asarray(membership.T @ X)

        smoothed = self.feature_count_ + self.alpha
        self.feature_log_prob_ = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
        self.class_log_prior_ = np.log(self.class_count_) - np.log(n_samples)
        return self

    def _joint_log_likelihood(self, X: MatrixLike) -> FloatArray:
        """Return ``log P(c) + log P(x | c)`` with shape ``(n_samples, n_classes)``."""
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse="csr", reset=False)
        scores: FloatArray = np.asarray(X @ self.feature_log_prob_.T) + self.class_log_prior_
        return scores

    def predict_log_proba(self, X: MatrixLike) -> FloatArray:
        """Return ``log P(c | x)`` for every class."""
        joint = self._joint_log_likelihood(X)
        # Normalise in log space: P(c | x) = P(c, x) / Σ_c' P(c', x).
        normalised: FloatArray = joint - logsumexp(joint, axis=1, keepdims=True)
        return normalised

    def predict_proba(self, X: MatrixLike) -> FloatArray:
        """Return ``P(c | x)`` for every class; each row sums to 1."""
        proba: FloatArray = np.exp(self.predict_log_proba(X))
        return proba

    def predict(self, X: MatrixLike) -> npt.NDArray[Any]:
        """Return the most probable class for each sample."""
        joint = self._joint_log_likelihood(X)
        predictions: npt.NDArray[Any] = self.classes_[np.argmax(joint, axis=1)]
        return predictions
