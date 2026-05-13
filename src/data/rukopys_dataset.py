from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


DATASET_ID = "UkrainianCatholicUniversity/rukopys"


def load_rukopys(
    config: str = "gt_only",
    split: str | None = None,
    revision: str | None = None,
    streaming: bool = False,
) -> Dataset | DatasetDict:
    """Load RUKOPYS through HuggingFace datasets.

    The stable first-baseline path is `config="gt_only"` for ground-truth train
    data and `config="test"` for test images. The full config currently needs
    separate verification in this environment because train/silver metadata
    schemas differ.
    """
    kwargs: dict[str, Any] = {}
    if split is not None:
        kwargs["split"] = split
    if revision is not None:
        kwargs["revision"] = revision
    if streaming:
        kwargs["streaming"] = True
    return load_dataset(DATASET_ID, config, **kwargs)


def iter_records(dataset: Dataset | Iterable[dict[str, Any]], limit: int | None = None) -> Iterable[dict[str, Any]]:
    for index, record in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        record = dict(record)
        record.setdefault("_row_index", index)
        yield record
