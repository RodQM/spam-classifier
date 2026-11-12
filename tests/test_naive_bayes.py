from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.estimator_checks import parametrize_with_checks

from spamclf.naive_bayes import ScratchMultinomialNB


@pytest.fixture
def counts() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Random count matrix with three classes of different sizes."""
    rng = np.random.default_rng(0)
    X = rng.poisson(lam=1.5, size=(120, 25))
    y = rng.choice(["ham", "promo", "spam"], size=120, p=[0.6, 0.1, 0.3])
    return X, y


@parametrize_with_checks([ScratchMultinomialNB()])
def test_follows_scikit_learn_estimator_contract(estimator: Any, check: Any) -> None:
    """Run scikit-learn's own API conformance suite against the estimator."""
    check(estimator)


@pytest.mark.parametrize("alpha", [0.01, 0.5, 1.0, 10.0])
def test_matches_scikit_learn_reference(
    counts: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]], alpha: float
) -> None:
    X, y = counts
    ours = ScratchMultinomialNB(alpha=alpha).fit(X, y)
    reference = MultinomialNB(alpha=alpha).fit(X, y)

    np.testing.assert_allclose(ours.class_log_prior_, reference.class_log_prior_)
    np.testing.assert_allclose(ours.feature_log_prob_, reference.feature_log_prob_)
    np.testing.assert_allclose(ours.predict_proba(X), reference.predict_proba(X))
    np.testing.assert_array_equal(ours.predict(X), reference.predict(X))


def test_sparse_and_dense_inputs_agree(
    counts: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
) -> None:
    X, y = counts
    dense = ScratchMultinomialNB().fit(X, y)
    from_sparse = ScratchMultinomialNB().fit(sparse.csr_matrix(X), y)

    np.testing.assert_allclose(dense.feature_log_prob_, from_sparse.feature_log_prob_)
    np.testing.assert_allclose(
        dense.predict_proba(X), from_sparse.predict_proba(sparse.csr_matrix(X))
    )


def test_probabilities_are_normalised(
    counts: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
) -> None:
    X, y = counts
    proba = ScratchMultinomialNB().fit(X, y).predict_proba(X)

    assert proba.shape == (120, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


def test_smoothing_handles_tokens_unseen_in_a_class() -> None:
    # Token 1 never appears in class "a": without smoothing its likelihood is
    # zero and log(0) would poison every prediction containing it.
    X = np.array([[3, 0], [2, 0], [0, 4], [1, 3]])
    y = np.array(["a", "a", "b", "b"])
    model = ScratchMultinomialNB(alpha=1.0).fit(X, y)

    assert np.isfinite(model.feature_log_prob_).all()
    assert model.predict([[0, 5]])[0] == "b"
    assert model.predict([[5, 0]])[0] == "a"


@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_rejects_non_positive_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be positive"):
        ScratchMultinomialNB(alpha=alpha).fit([[1, 2], [3, 4]], [0, 1])


def test_rejects_negative_features() -> None:
    with pytest.raises(ValueError, match="Negative values"):
        ScratchMultinomialNB().fit([[1, -2], [3, 4]], [0, 1])


def test_predict_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        ScratchMultinomialNB().predict([[1, 2]])
