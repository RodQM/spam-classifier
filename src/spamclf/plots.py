"""Report figures, rendered with matplotlib's object-oriented API (no pyplot state)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve

from spamclf.evaluation import Metrics
from spamclf.models import MODEL_DESCRIPTIONS
from spamclf.training import CVResult

if TYPE_CHECKING:
    from matplotlib.typing import RcKeyType

# Palette: neutral chrome, one blue for single-series and sequential encodings,
# and three categorical hues that stay distinguishable under colour blindness.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "spamclf_blues", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
)

STYLE: dict[RcKeyType, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "legend.frameon": False,
}


@contextmanager
def _figure(path: Path, figsize: tuple[float, float]) -> Iterator[tuple[Figure, Axes]]:
    """Yield a styled figure/axes pair and save it to ``path`` on exit."""
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=figsize, dpi=160, layout="constrained")
        ax = fig.add_subplot()
        yield fig, ax
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)


def _titles(ax: Axes, title: str, subtitle: str) -> None:
    """Left-aligned title with a quieter subtitle underneath."""
    ax.set_title(f"{title}\n", loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=10, color=INK_SECONDARY)


def plot_confusion_matrix(metrics: Metrics, path: Path, *, model_label: str) -> Path:
    """Draw the 2x2 confusion matrix, naming what each cell means for a user."""
    matrix = metrics.confusion
    meanings = [["correctly delivered", "false alarm"], ["missed spam", "spam caught"]]

    with _figure(path, (5.2, 4.6)) as (_, ax):
        ax.imshow(matrix, cmap=_SEQUENTIAL, vmin=0)
        threshold = float(np.max(matrix)) / 2
        for (row, col), count in np.ndenumerate(matrix):
            color = "#ffffff" if count > threshold else INK
            ax.text(col, row - 0.08, f"{count:,}", ha="center", va="center", color=color,
                    fontsize=20, fontweight="bold")  # fmt: skip
            ax.text(col, row + 0.2, meanings[row][col], ha="center", va="center", color=color,
                    fontsize=9)  # fmt: skip

        ax.set_xticks([0, 1], ["ham", "spam"])
        ax.set_yticks([0, 1], ["ham", "spam"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        # Thin surface-coloured gaps between the cells instead of drawn borders.
        ax.set_xticks([0.5], minor=True)
        ax.set_yticks([0.5], minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=3)
        ax.tick_params(which="both", length=0)
        ax.spines[:].set_visible(False)
        total = int(matrix.sum())
        _titles(ax, f"Confusion matrix: {model_label}", f"{total:,} held-out test messages")
    return path


def plot_precision_recall(
    y_true: npt.ArrayLike,
    scores: npt.ArrayLike,
    metrics: Metrics,
    path: Path,
    *,
    model_label: str,
) -> Path:
    """Draw the precision-recall curve and mark the model's operating point."""
    precision, recall, _ = precision_recall_curve(y_true, scores)

    with _figure(path, (6.2, 4.4)) as (_, ax):
        ax.plot(recall, precision, color=BLUE, linewidth=2, drawstyle="steps-post")
        ax.scatter([metrics.recall], [metrics.precision], s=70, color=BLUE, edgecolor=SURFACE,
                   linewidth=2, zorder=3)  # fmt: skip
        ax.annotate(
            f"default threshold\nprecision {metrics.precision:.3f} · recall {metrics.recall:.3f}",
            xy=(metrics.recall, metrics.precision),
            xytext=(-14, -38),
            textcoords="offset points",
            ha="right",
            color=INK_SECONDARY,
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": AXIS, "linewidth": 1},
        )
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.04)
        ax.set_xlabel("Recall (share of spam caught)")
        ax.set_ylabel("Precision (share of flags that are spam)")
        ax.grid(True)
        _titles(
            ax,
            f"Precision-recall: {model_label}",
            f"Average precision {metrics.average_precision:.3f} on the held-out test set",
        )
    return path


def plot_model_comparison(results: Sequence[CVResult], path: Path, *, folds: int) -> Path:
    """Dot plot of cross-validated precision, recall and F1 for every model.

    Dots rather than bars: all scores sit between ~0.85 and 1.0, and a bar chart
    would either hide the differences (zero baseline) or exaggerate them
    (truncated baseline). Position on a zoomed axis has no such problem.
    """
    series = [("precision", BLUE, "o"), ("recall", ORANGE, "s"), ("f1", AQUA, "D")]
    labels = {"precision": "Precision", "recall": "Recall", "f1": "F1"}
    ordered = list(reversed(results))  # best model on the top row
    rows = np.arange(len(ordered))

    with _figure(path, (7.0, 1.4 + 0.62 * len(ordered))) as (fig, ax):
        lowest = 1.0
        for row, result in zip(rows, ordered, strict=True):
            values = [result.mean[metric] for metric, _, _ in series]
            lowest = min(lowest, *values)
            ax.plot([min(values), max(values)], [row, row], color=AXIS, linewidth=1.5, zorder=1)
        for metric, color, marker in series:
            ax.scatter([result.mean[metric] for result in ordered], rows, s=80, color=color,
                       marker=marker, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=labels[metric])  # fmt: skip

        ax.set_yticks(rows, [MODEL_DESCRIPTIONS[result.model] for result in ordered])
        ax.set_ylim(-0.6, len(ordered) - 0.4)
        ax.set_xlim(np.floor(lowest * 20) / 20 - 0.01, 1.005)
        ax.set_xlabel(f"Score (mean of {folds}-fold cross-validation, spam = positive class)")
        ax.grid(True, axis="x")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title("Model comparison", loc="left", fontsize=13, fontweight="bold", color=INK)
        fig.legend(loc="outside upper right", ncols=3, handletextpad=0.2, columnspacing=1.2)
    return path
