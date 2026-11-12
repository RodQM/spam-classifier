from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from spamclf import __version__
from spamclf.cli import app

runner = CliRunner()


@pytest.fixture
def workspace(dataset_path: Path, tmp_path: Path) -> dict[str, str]:
    """Common CLI options pointing every artefact at the temporary directory."""
    return {
        "data": str(dataset_path),
        "models": str(tmp_path / "models"),
        "reports": str(tmp_path / "reports"),
    }


def test_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_train_then_predict(workspace: dict[str, str], tmp_path: Path) -> None:
    trained = runner.invoke(
        app,
        ["train", "--model", "nb-scratch", "--data-path", workspace["data"],
         "--model-dir", workspace["models"], "--reports-dir", workspace["reports"]],
    )  # fmt: skip
    assert trained.exit_code == 0, trained.output
    assert "Strongest spam signals" in trained.output

    metadata = json.loads((tmp_path / "models" / "nb-scratch.json").read_text(encoding="utf-8"))
    assert metadata["model"] == "nb-scratch"
    assert metadata["n_train"] + metadata["n_test"] == 40
    assert metadata["test_metrics"]["f1"] == pytest.approx(1.0)
    assert (tmp_path / "reports" / "confusion_matrix_nb-scratch.png").exists()
    assert (tmp_path / "reports" / "precision_recall_nb-scratch.png").exists()

    predicted = runner.invoke(
        app,
        ["predict", "claim your free prize now", "see you at lunch today",
         "--model", "nb-scratch", "--model-dir", workspace["models"]],
    )  # fmt: skip
    assert predicted.exit_code == 0, predicted.output
    assert "spam" in predicted.output
    assert "ham" in predicted.output


def test_compare_writes_results(workspace: dict[str, str], tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["compare", "--folds", "3", "--data-path", workspace["data"],
         "--reports-dir", workspace["reports"]],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / "reports" / "cv_results.json").read_text(encoding="utf-8"))
    assert {entry["model"] for entry in saved} == {"nb-scratch", "nb", "logreg", "svm"}
    assert (tmp_path / "reports" / "model_comparison.png").exists()


def test_train_without_dataset_fails_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["train", "--data-path", str(tmp_path / "missing.tsv")])

    assert result.exit_code == 1
    assert "spamclf download" in result.output


def test_predict_without_model_fails_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["predict", "hello", "--model-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "spamclf train" in result.output


def test_unknown_model_is_rejected(workspace: dict[str, str]) -> None:
    result = runner.invoke(app, ["train", "--model", "gpt", "--data-path", workspace["data"]])

    assert result.exit_code == 2
