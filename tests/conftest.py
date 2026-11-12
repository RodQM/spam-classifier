"""Shared fixtures: a tiny synthetic corpus so the suite never touches the network."""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
import pytest

from spamclf.data import Split, load_dataset, split_dataset

_SPAM_OPENERS = ("WINNER!", "URGENT!", "Congratulations,", "FREE entry:", "Final notice,")
_SPAM_BODIES = (
    "claim your free prize now, call 09061701461",
    "you have won a cash award, text WIN to 80086",
    "free ringtone waiting, reply STOP to cancel",
    "txt CLAIM to 87121 to receive your free prize",
)
_HAM_OPENERS = ("Hey,", "Ok,", "Sorry,", "Mum,", "Hi love,")
_HAM_BODIES = (
    "are we still meeting for lunch today?",
    "i will be home late tonight, see you later",
    "can you pick up some milk on the way home?",
    "thanks for dinner, see you at the weekend",
)


def _corpus() -> list[tuple[str, str]]:
    spam = [("spam", f"{a} {b}") for a, b in itertools.product(_SPAM_OPENERS, _SPAM_BODIES)]
    ham = [("ham", f"{a} {b}") for a, b in itertools.product(_HAM_OPENERS, _HAM_BODIES)]
    return spam + ham


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    """A 40-message dataset (20 spam, 20 ham) in the same TSV format as the real one."""
    path = tmp_path / "data" / "sms.tsv"
    path.parent.mkdir()
    path.write_text("".join(f"{label}\t{text}\n" for label, text in _corpus()), encoding="utf-8")
    return path


@pytest.fixture
def dataset(dataset_path: Path) -> pd.DataFrame:
    return load_dataset(dataset_path)


@pytest.fixture
def split(dataset: pd.DataFrame) -> Split:
    return split_dataset(dataset, test_size=0.25, seed=0)
