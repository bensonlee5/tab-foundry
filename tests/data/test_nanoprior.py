from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from tab_foundry.data.factory import build_task_loader
from tab_foundry.data.nanoprior import NanoPriorTaskDataset, inspect_nano_prior_dump


def _write_prior_dump(path: Path, *, max_num_classes: int = 2) -> None:
    x = np.arange(8 * 6 * 4, dtype=np.float32).reshape(8, 6, 4)
    y = np.tile(np.asarray([[0, 1, 0, 1, 0, 1]], dtype=np.float32), (8, 1))
    num_features = np.asarray([1, 2, 3, 4, 2, 4, 3, 1], dtype=np.int32)
    num_datapoints = np.asarray([6, 5, 6, 4, 6, 6, 5, 6], dtype=np.int32)
    single_eval_pos = np.asarray([2, 3, 4, 2, 3, 4, 2, 3], dtype=np.int32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=x)
        handle.create_dataset("y", data=y)
        handle.create_dataset("num_features", data=num_features)
        handle.create_dataset("num_datapoints", data=num_datapoints)
        handle.create_dataset("single_eval_pos", data=single_eval_pos)
        handle.create_dataset("max_num_classes", data=np.asarray([max_num_classes], dtype=np.int64))


def test_nanoprior_dataset_slices_rows_features_and_split(tmp_path: Path) -> None:
    prior_dump = tmp_path / "tiny.h5"
    _write_prior_dump(prior_dump)

    summary = inspect_nano_prior_dump(prior_dump)
    dataset = NanoPriorTaskDataset(prior_dump, offset=1, size=2)
    batch = dataset[0]

    assert summary.num_tasks == 8
    assert batch.metadata["task_index"] == 1
    assert batch.x_train.shape == (3, 2)
    assert batch.x_test.shape == (2, 2)
    assert batch.y_train.shape == (3,)
    assert batch.y_test.shape == (2,)


def test_nanoprior_dataset_rejects_nonbinary_dump(tmp_path: Path) -> None:
    prior_dump = tmp_path / "bad.h5"
    _write_prior_dump(prior_dump, max_num_classes=3)

    try:
        _ = NanoPriorTaskDataset(prior_dump, offset=0, size=1)
    except RuntimeError as exc:
        assert "binary classification" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError for non-binary nano prior dump")


def test_build_task_loader_shuffle_is_seeded(tmp_path: Path) -> None:
    prior_dump = tmp_path / "tiny.h5"
    _write_prior_dump(prior_dump)
    dataset = NanoPriorTaskDataset(prior_dump, offset=0, size=6)

    loader = build_task_loader(dataset, num_workers=0, shuffle=True, seed=7)
    first_epoch = [int(batch.metadata["task_index"]) for batch in loader]
    second_epoch = [int(batch.metadata["task_index"]) for batch in loader]
    repeat_loader = build_task_loader(dataset, num_workers=0, shuffle=True, seed=7)
    repeat_epoch = [int(batch.metadata["task_index"]) for batch in repeat_loader]

    assert first_epoch == repeat_epoch
    assert first_epoch != second_epoch
