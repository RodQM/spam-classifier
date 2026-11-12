"""Cross-validation, training and model persistence."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from spamclf.models import ModelName, build_pipeline

CV_METRICS = ("precision", "recall", "f1", "average_precision")
"""Scorers computed during cross-validation; all treat spam (1) as the positive class."""

Texts = npt.NDArray[np.object_]
Targets = npt.NDArray[np.int_]


@dataclass(frozen=True)
class CVResult:
    """Cross-validated scores of one model: mean and standard deviation per metric."""

    model: ModelName
    mean: dict[str, float]
    std: dict[str, float]
    fit_seconds: float


def cross_validate_model(
    name: ModelName, X: Texts, y: Targets, *, folds: int = 5, seed: int = 42
) -> CVResult:
    """Score a model with stratified k-fold cross-validation.

    The whole pipeline (vectoriser included) is re-fitted inside every fold, so
    the vocabulary never sees the held-out messages.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_validate(build_pipeline(name), X, y, cv=cv, scoring=list(CV_METRICS))
    return CVResult(
        model=name,
        mean={metric: float(np.mean(scores[f"test_{metric}"])) for metric in CV_METRICS},
        std={metric: float(np.std(scores[f"test_{metric}"])) for metric in CV_METRICS},
        fit_seconds=float(np.mean(scores["fit_time"])),
    )


def compare_models(
    X: Texts,
    y: Targets,
    *,
    names: Iterable[ModelName] = tuple(ModelName),
    folds: int = 5,
    seed: int = 42,
) -> list[CVResult]:
    """Cross-validate several models on the same folds, best F1 first."""
    results = [cross_validate_model(name, X, y, folds=folds, seed=seed) for name in names]
    return sorted(results, key=lambda result: result.mean["f1"], reverse=True)


def train_model(name: ModelName, X: Texts, y: Targets) -> Pipeline:
    """Fit a fresh pipeline on the given data."""
    return build_pipeline(name).fit(X, y)


def model_path(model_dir: Path, name: ModelName) -> Path:
    """Location of the serialised pipeline for a model."""
    return model_dir / f"{name.value}.joblib"


def save_model(
    pipeline: Pipeline, model_dir: Path, name: ModelName, metadata: dict[str, Any]
) -> Path:
    """Serialise the pipeline and write its metadata next to it as JSON."""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_path(model_dir, name)
    joblib.dump(pipeline, path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def load_model(model_dir: Path, name: ModelName) -> Pipeline:
    """Load a pipeline saved by :func:`save_model`.

    joblib files are pickles: only load models you trained yourself.
    """
    path = model_path(model_dir, name)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `spamclf train --model {name.value}` first."
        )
    pipeline: Pipeline = joblib.load(path)
    return pipeline
