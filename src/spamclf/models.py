"""Model registry: every candidate is a text → features → classifier pipeline."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from spamclf.naive_bayes import ScratchMultinomialNB


class ModelName(StrEnum):
    """Identifiers of the available models (also the CLI choices)."""

    NB_SCRATCH = "nb-scratch"
    NB = "nb"
    LOGREG = "logreg"
    SVM = "svm"


# Unigrams + bigrams capture phrases such as "free entry" or "call now".
# min_df=2 drops tokens seen in a single message, which cannot generalise.
_VECTORIZER_OPTIONS = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),
    "min_df": 2,
}


def _counts() -> CountVectorizer:
    """Raw token counts: the natural input for multinomial Naive Bayes."""
    return CountVectorizer(**_VECTORIZER_OPTIONS)


def _tfidf() -> TfidfVectorizer:
    """TF-IDF weights: down-weights ubiquitous tokens for the linear models."""
    return TfidfVectorizer(sublinear_tf=True, **_VECTORIZER_OPTIONS)


def _nb_scratch() -> Pipeline:
    return Pipeline([("features", _counts()), ("clf", ScratchMultinomialNB(alpha=1.0))])


def _nb() -> Pipeline:
    return Pipeline([("features", _counts()), ("clf", MultinomialNB(alpha=1.0))])


def _logreg() -> Pipeline:
    # class_weight="balanced" compensates for spam being only ~13% of messages.
    clf = LogisticRegression(C=10.0, class_weight="balanced", max_iter=1000)
    return Pipeline([("features", _tfidf()), ("clf", clf)])


def _svm() -> Pipeline:
    return Pipeline([("features", _tfidf()), ("clf", LinearSVC(C=1.0, class_weight="balanced"))])


_BUILDERS: dict[ModelName, Callable[[], Pipeline]] = {
    ModelName.NB_SCRATCH: _nb_scratch,
    ModelName.NB: _nb,
    ModelName.LOGREG: _logreg,
    ModelName.SVM: _svm,
}

MODEL_DESCRIPTIONS: dict[ModelName, str] = {
    ModelName.NB_SCRATCH: "Naive Bayes (from scratch)",
    ModelName.NB: "Naive Bayes (scikit-learn)",
    ModelName.LOGREG: "Logistic regression",
    ModelName.SVM: "Linear SVM",
}


def build_pipeline(name: ModelName | str) -> Pipeline:
    """Return a fresh, unfitted pipeline for the given model name."""
    try:
        return _BUILDERS[ModelName(name)]()
    except ValueError:
        choices = ", ".join(model.value for model in ModelName)
        raise ValueError(f"Unknown model {name!r}. Choose one of: {choices}.") from None
