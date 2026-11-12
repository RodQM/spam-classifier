"""Models, training, evaluation and plots, exercised end to end on the tiny corpus."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spamclf.data import Split
from spamclf.evaluation import Metrics, decision_scores, evaluate, top_spam_tokens
from spamclf.models import MODEL_DESCRIPTIONS, ModelName, build_pipeline
from spamclf.plots import plot_confusion_matrix, plot_model_comparison, plot_precision_recall
from spamclf.training import (
    CV_METRICS,
    compare_models,
    cross_validate_model,
    load_model,
    save_model,
    train_model,
)

ALL_MODELS = list(ModelName)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_every_model_has_a_description() -> None:
    assert set(MODEL_DESCRIPTIONS) == set(ModelName)


def test_build_pipeline_accepts_plain_strings() -> None:
    assert build_pipeline("logreg").steps[-1][0] == "clf"


def test_build_pipeline_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Choose one of: nb-scratch, nb, logreg, svm"):
        build_pipeline("transformer")


@pytest.mark.parametrize("name", ALL_MODELS)
def test_model_separates_the_toy_corpus(name: ModelName, split: Split) -> None:
    pipeline = train_model(name, split.X_train, split.y_train)
    metrics = evaluate(pipeline, split.X_test, split.y_test)

    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.confusion.sum() == len(split.y_test)
    assert metrics.false_positives == metrics.false_negatives == 0


@pytest.mark.parametrize("name", ALL_MODELS)
def test_decision_scores_rank_spam_above_ham(name: ModelName, split: Split) -> None:
    pipeline = train_model(name, split.X_train, split.y_train)
    scores = decision_scores(pipeline, split.X_test)

    assert scores.shape == (len(split.y_test),)
    assert scores[split.y_test == 1].min() > scores[split.y_test == 0].max()


@pytest.mark.parametrize("name", ALL_MODELS)
def test_top_spam_tokens_come_from_spam_messages(name: ModelName, split: Split) -> None:
    pipeline = train_model(name, split.X_train, split.y_train)
    tokens = top_spam_tokens(pipeline, k=5)
    spam_text = " ".join(split.X_train[split.y_train == 1]).lower()

    assert len(tokens) == 5
    assert all(token in spam_text for token, _ in tokens)
    weights = [weight for _, weight in tokens]
    assert weights == sorted(weights, reverse=True)


def test_scratch_and_reference_naive_bayes_agree_on_text(split: Split) -> None:
    ours = train_model(ModelName.NB_SCRATCH, split.X_train, split.y_train)
    reference = train_model(ModelName.NB, split.X_train, split.y_train)

    np.testing.assert_allclose(
        ours.predict_proba(split.X_test), reference.predict_proba(split.X_test)
    )


def test_evaluate_counts_errors_correctly(split: Split) -> None:
    pipeline = train_model(ModelName.LOGREG, split.X_train, split.y_train)
    flipped = 1 - split.y_test  # every prediction is now "wrong"
    metrics = evaluate(pipeline, split.X_test, flipped)

    assert metrics.accuracy == 0.0
    assert metrics.precision == metrics.recall == metrics.f1 == 0.0
    assert metrics.true_positives == metrics.true_negatives == 0
    assert metrics.false_positives + metrics.false_negatives == len(split.y_test)


def test_cross_validation_reports_every_metric(split: Split) -> None:
    result = cross_validate_model(ModelName.NB_SCRATCH, split.X_train, split.y_train, folds=3)

    assert set(result.mean) == set(result.std) == set(CV_METRICS)
    assert all(0.0 <= value <= 1.0 for value in result.mean.values())
    assert result.fit_seconds > 0


def test_compare_models_sorts_by_f1(split: Split) -> None:
    results = compare_models(split.X_train, split.y_train, folds=3)

    assert {result.model for result in results} == set(ModelName)
    f1_scores = [result.mean["f1"] for result in results]
    assert f1_scores == sorted(f1_scores, reverse=True)


def test_saved_model_round_trips(split: Split, tmp_path: Path) -> None:
    pipeline = train_model(ModelName.SVM, split.X_train, split.y_train)
    path = save_model(pipeline, tmp_path / "models", ModelName.SVM, {"seed": 0})
    restored = load_model(tmp_path / "models", ModelName.SVM)

    assert path.name == "svm.joblib"
    assert path.with_suffix(".json").read_text(encoding="utf-8") == '{\n  "seed": 0\n}'
    np.testing.assert_array_equal(restored.predict(split.X_test), pipeline.predict(split.X_test))


def test_load_model_missing_points_to_train_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="spamclf train --model nb"):
        load_model(tmp_path, ModelName.NB)


def test_plots_are_written_as_png(split: Split, tmp_path: Path) -> None:
    pipeline = train_model(ModelName.LOGREG, split.X_train, split.y_train)
    metrics: Metrics = evaluate(pipeline, split.X_test, split.y_test)
    scores = decision_scores(pipeline, split.X_test)
    results = compare_models(split.X_train, split.y_train, folds=3)

    figures = [
        plot_confusion_matrix(metrics, tmp_path / "cm.png", model_label="Test"),
        plot_precision_recall(split.y_test, scores, metrics, tmp_path / "pr.png", model_label="T"),
        plot_model_comparison(results, tmp_path / "nested" / "cmp.png", folds=3),
    ]

    for figure in figures:
        assert figure.read_bytes().startswith(PNG_SIGNATURE)
