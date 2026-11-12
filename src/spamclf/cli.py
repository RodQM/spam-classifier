"""Command-line interface: ``spamclf download | compare | train | predict``."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import sklearn
import typer
from rich.console import Console
from rich.table import Table

from spamclf import __version__
from spamclf.data import (
    DEFAULT_DATA_PATH,
    LABELS,
    ChecksumError,
    Split,
    download_dataset,
    load_dataset,
    split_dataset,
)
from spamclf.evaluation import decision_scores, evaluate, top_spam_tokens
from spamclf.models import MODEL_DESCRIPTIONS, ModelName
from spamclf.plots import plot_confusion_matrix, plot_model_comparison, plot_precision_recall
from spamclf.training import compare_models, load_model, save_model, train_model

app = typer.Typer(
    help="Train, compare and use SMS spam classifiers.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

DataPath = Annotated[Path, typer.Option(help="Location of the dataset TSV file.")]
ModelDir = Annotated[Path, typer.Option(help="Directory for trained models.")]
ReportsDir = Annotated[Path, typer.Option(help="Directory for figures and metric files.")]
Seed = Annotated[int, typer.Option(help="Random seed for the train/test split and CV folds.")]
Model = Annotated[ModelName, typer.Option("--model", "-m", help="Which model to use.")]


def _fail(message: str) -> NoReturn:
    console.print(f"[bold red]Error:[/] {message}")
    raise typer.Exit(code=1)


def _load_split(data_path: Path, seed: int) -> Split:
    try:
        return split_dataset(load_dataset(data_path), seed=seed)
    except FileNotFoundError as error:
        _fail(str(error))


def _version(value: bool) -> None:
    if value:
        console.print(f"spamclf {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version.")
    ] = False,
) -> None:
    """Train, compare and use SMS spam classifiers."""


@app.command()
def download(
    data_path: DataPath = DEFAULT_DATA_PATH,
    force: Annotated[bool, typer.Option(help="Download again even if the file exists.")] = False,
) -> None:
    """Download the SMS Spam Collection from the UCI repository."""
    try:
        path = download_dataset(data_path, force=force)
    except (ChecksumError, OSError) as error:
        _fail(str(error))
    df = load_dataset(path, deduplicate=False)
    spam_share = (df["label"] == "spam").mean()
    console.print(f"Dataset ready at [bold]{path}[/]: {len(df):,} messages, {spam_share:.1%} spam.")


@app.command()
def compare(
    data_path: DataPath = DEFAULT_DATA_PATH,
    reports_dir: ReportsDir = Path("reports"),
    folds: Annotated[int, typer.Option(min=2, help="Number of cross-validation folds.")] = 5,
    seed: Seed = 42,
) -> None:
    """Cross-validate every model on the training split and rank them by F1."""
    split = _load_split(data_path, seed)
    with console.status("Cross-validating models..."):
        results = compare_models(split.X_train, split.y_train, folds=folds, seed=seed)

    table = Table(title=f"{folds}-fold cross-validation ({len(split.y_train):,} messages)")
    table.add_column("Model")
    for column in ("Precision", "Recall", "F1", "Avg. precision", "Fit time"):
        table.add_column(column, justify="right")
    for result in results:
        table.add_row(
            MODEL_DESCRIPTIONS[result.model],
            *(
                f"{result.mean[metric]:.3f} ± {result.std[metric]:.3f}"
                for metric in ("precision", "recall", "f1", "average_precision")
            ),
            f"{result.fit_seconds * 1000:.0f} ms",
        )
    console.print(table)

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "cv_results.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8"
    )
    figure = plot_model_comparison(results, reports_dir / "model_comparison.png", folds=folds)
    console.print(f"Saved [bold]{figure}[/] and cv_results.json")


@app.command()
def train(
    model: Model = ModelName.LOGREG,
    data_path: DataPath = DEFAULT_DATA_PATH,
    model_dir: ModelDir = Path("models"),
    reports_dir: ReportsDir = Path("reports"),
    seed: Seed = 42,
) -> None:
    """Train a model, evaluate it on the held-out test set and save it."""
    split = _load_split(data_path, seed)
    label = MODEL_DESCRIPTIONS[model]

    with console.status(f"Training {label}..."):
        pipeline = train_model(model, split.X_train, split.y_train)
    metrics = evaluate(pipeline, split.X_test, split.y_test)

    table = Table(title=f"{label}: test set ({len(split.y_test):,} messages)")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for name in ("accuracy", "precision", "recall", "f1", "average_precision"):
        table.add_row(name.replace("_", " ").capitalize(), f"{getattr(metrics, name):.3f}")
    table.add_row("False alarms (ham flagged)", str(metrics.false_positives))
    table.add_row("Missed spam", str(metrics.false_negatives))
    console.print(table)

    tokens = ", ".join(token for token, _ in top_spam_tokens(pipeline, k=12))
    console.print(f"[bold]Strongest spam signals:[/] {tokens}")

    metadata = {
        "model": model.value,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "n_train": len(split.y_train),
        "n_test": len(split.y_test),
        "scikit_learn": sklearn.__version__,
        "test_metrics": metrics.to_dict(),
    }
    saved = save_model(pipeline, model_dir, model, metadata)

    plot_confusion_matrix(
        metrics, reports_dir / f"confusion_matrix_{model.value}.png", model_label=label
    )
    plot_precision_recall(
        split.y_test,
        decision_scores(pipeline, split.X_test),
        metrics,
        reports_dir / f"precision_recall_{model.value}.png",
        model_label=label,
    )
    console.print(f"Model saved to [bold]{saved}[/]; figures in [bold]{reports_dir}[/]")


@app.command()
def predict(
    texts: Annotated[list[str], typer.Argument(help="One or more messages to classify.")],
    model: Model = ModelName.LOGREG,
    model_dir: ModelDir = Path("models"),
) -> None:
    """Classify messages with a previously trained model."""
    try:
        pipeline = load_model(model_dir, model)
    except FileNotFoundError as error:
        _fail(str(error))

    predictions = pipeline.predict(texts)
    has_probability = hasattr(pipeline, "predict_proba")
    scores = decision_scores(pipeline, texts)

    table = Table(title=MODEL_DESCRIPTIONS[model])
    table.add_column("Message", overflow="fold")
    table.add_column("Verdict")
    table.add_column("P(spam)" if has_probability else "Score", justify="right")
    for text, prediction, score in zip(texts, predictions, scores, strict=True):
        verdict = LABELS[int(prediction)]
        style = "bold red" if verdict == "spam" else "green"
        formatted = f"{score:.1%}" if has_probability else f"{score:+.2f}"
        table.add_row(text, f"[{style}]{verdict}[/]", formatted)
    console.print(table)
