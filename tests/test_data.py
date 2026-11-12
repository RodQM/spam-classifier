from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from spamclf.data import (
    ARCHIVE_MEMBER,
    ChecksumError,
    download_dataset,
    load_dataset,
    split_dataset,
)


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """A local zip that mimics the UCI archive, served through a file:// URL."""
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(ARCHIVE_MEMBER, "ham\thello there\nspam\tfree prize\n")
        bundle.writestr("readme", "not the data")
    return path


def test_load_returns_expected_columns(dataset: pd.DataFrame) -> None:
    assert list(dataset.columns) == ["label", "text", "is_spam"]
    assert len(dataset) == 40
    assert dataset["is_spam"].sum() == 20
    assert set(dataset.loc[dataset["is_spam"] == 1, "label"]) == {"spam"}


def test_load_drops_duplicate_messages(tmp_path: Path) -> None:
    path = tmp_path / "dupes.tsv"
    path.write_text("ham\tsee you soon\nham\tsee you soon\nspam\twin cash\n", encoding="utf-8")

    assert len(load_dataset(path)) == 2
    assert len(load_dataset(path, deduplicate=False)) == 3


def test_load_keeps_quotes_and_literal_na(tmp_path: Path) -> None:
    path = tmp_path / "tricky.tsv"
    path.write_text('ham\t"unbalanced quote\nham\tNA\n', encoding="utf-8")

    assert list(load_dataset(path)["text"]) == ['"unbalanced quote', "NA"]


def test_load_rejects_unknown_labels(tmp_path: Path) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text("ham\thello\nphishing\tclick here\n", encoding="utf-8")

    with pytest.raises(ValueError, match="phishing"):
        load_dataset(path)


def test_load_missing_file_points_to_download_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="spamclf download"):
        load_dataset(tmp_path / "missing.tsv")


def test_split_is_stratified_and_reproducible(dataset: pd.DataFrame) -> None:
    first = split_dataset(dataset, test_size=0.25, seed=7)
    second = split_dataset(dataset, test_size=0.25, seed=7)

    assert len(first.X_train) == 30
    assert len(first.X_test) == 10
    assert first.y_test.mean() == pytest.approx(0.5)
    assert list(first.X_test) == list(second.X_test)
    assert set(first.X_train).isdisjoint(first.X_test)


def test_download_verifies_and_extracts(archive: Path, tmp_path: Path) -> None:
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    dest = tmp_path / "out" / "sms.tsv"

    result = download_dataset(dest, url=archive.as_uri(), expected_sha256=digest)

    assert result == dest
    assert dest.read_text(encoding="utf-8").startswith("ham\thello there")


def test_download_rejects_checksum_mismatch(archive: Path, tmp_path: Path) -> None:
    dest = tmp_path / "sms.tsv"

    with pytest.raises(ChecksumError, match="SHA-256 mismatch"):
        download_dataset(dest, url=archive.as_uri(), expected_sha256="0" * 64)
    assert not dest.exists()


def test_download_skips_existing_file_unless_forced(archive: Path, tmp_path: Path) -> None:
    dest = tmp_path / "sms.tsv"
    dest.write_text("cached", encoding="utf-8")

    download_dataset(dest, url=archive.as_uri(), expected_sha256=None)
    assert dest.read_text(encoding="utf-8") == "cached"

    download_dataset(dest, url=archive.as_uri(), expected_sha256=None, force=True)
    assert dest.read_text(encoding="utf-8").startswith("ham")
