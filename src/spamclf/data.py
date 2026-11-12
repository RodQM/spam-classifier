"""Dataset download, loading and train/test splitting.

The project uses the *SMS Spam Collection* (Almeida & Hidalgo, 2011) hosted by the
UCI Machine Learning Repository: 5,574 English SMS messages labelled ham or spam.
"""

from __future__ import annotations

import csv
import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATASET_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
ARCHIVE_MEMBER = "SMSSpamCollection"
DEFAULT_DATA_PATH = Path("data") / "SMSSpamCollection.tsv"

LABELS = ("ham", "spam")
"""Class names, indexed by the integer target (0 = ham, 1 = spam)."""


class ChecksumError(RuntimeError):
    """The downloaded archive does not match the pinned SHA-256 digest."""


@dataclass(frozen=True)
class Split:
    """A stratified train/test partition of the dataset."""

    X_train: npt.NDArray[np.object_]
    X_test: npt.NDArray[np.object_]
    y_train: npt.NDArray[np.int_]
    y_test: npt.NDArray[np.int_]


def download_dataset(
    dest: Path = DEFAULT_DATA_PATH,
    *,
    url: str = DATASET_URL,
    expected_sha256: str | None = DATASET_SHA256,
    force: bool = False,
) -> Path:
    """Download the dataset archive, verify it and extract the TSV file to ``dest``.

    The archive is verified against ``expected_sha256`` *before* anything is
    extracted, and only the single expected member is read from it. Pass
    ``expected_sha256=None`` to skip verification (e.g. for a mirror).
    """
    if dest.exists() and not force:
        return dest

    with urllib.request.urlopen(url, timeout=30) as response:
        payload: bytes = response.read()

    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ChecksumError(
            f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {digest}. "
            "The file may have changed upstream or been corrupted in transit."
        )

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        raw = archive.read(ARCHIVE_MEMBER)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return dest


def load_dataset(path: Path = DEFAULT_DATA_PATH, *, deduplicate: bool = True) -> pd.DataFrame:
    """Load the tab-separated dataset into a DataFrame.

    Returns the columns ``label`` ("ham"/"spam"), ``text`` and ``is_spam`` (0/1).

    About 7% of the messages are exact duplicates. They are dropped by default:
    otherwise the same message can land in both the training and the test set,
    which leaks information and inflates every metric.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run `spamclf download` first.")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        dtype=str,
        quoting=csv.QUOTE_NONE,  # messages contain unbalanced quotes
        encoding="utf-8",
        keep_default_na=False,
    )

    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unexpected labels in {path}: {unknown}. Expected only {LABELS}.")

    if deduplicate:
        df = df.drop_duplicates(subset="text").reset_index(drop=True)

    df["is_spam"] = (df["label"] == "spam").astype(int)
    return df


def split_dataset(df: pd.DataFrame, *, test_size: float = 0.2, seed: int = 42) -> Split:
    """Split into train and test sets, preserving the spam/ham ratio in both."""
    X = df["text"].to_numpy(dtype=object)
    y = df["is_spam"].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return Split(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
