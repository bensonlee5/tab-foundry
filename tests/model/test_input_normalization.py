from __future__ import annotations

import numpy as np

from tab_foundry.input_normalization import normalize_train_test_arrays


def test_train_zscore_clip_normalizes_from_train_only_and_clips() -> None:
    x_train = np.asarray([[0.0, 1.0], [2.0, 1.0]], dtype=np.float32)
    x_test = np.asarray([[1000.0, 1.0]], dtype=np.float32)

    train_norm, test_norm = normalize_train_test_arrays(
        x_train,
        x_test,
        mode="train_zscore_clip",
    )

    assert np.allclose(train_norm[:, 0], np.asarray([-1.0, 1.0], dtype=np.float32))
    assert np.allclose(train_norm[:, 1], np.asarray([0.0, 0.0], dtype=np.float32))
    assert test_norm[0, 0] == np.float32(100.0)
    assert test_norm[0, 1] == np.float32(0.0)
