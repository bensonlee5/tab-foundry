"""H5-based stability acceptance tests for the stability ladder.

These tests load real training data from the figshare H5 file and run short
training loops to verify numerical stability. Marked as slow.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

H5_PATH = Path(__file__).resolve().parent.parent.parent.parent / "nanoTabPFN" / "figshare_58932628.h5"


@pytest.fixture
def h5_file() -> Path:
    if not H5_PATH.exists():
        pytest.skip("H5 file not found at ../nanoTabPFN/figshare_58932628.h5")
    return H5_PATH


def _make_batch_from_h5(h5_path: Path, idx: int = 0):
    """Load one task from the H5 file and return a TaskBatch."""
    import h5py
    from tab_foundry.types import TaskBatch

    with h5py.File(str(h5_path), "r") as f:
        keys = list(f.keys())
        if idx >= len(keys):
            idx = 0
        group = f[keys[idx]]
        x_train = torch.tensor(group["x_train"][:], dtype=torch.float32)
        y_train = torch.tensor(group["y_train"][:].flatten(), dtype=torch.int64)
        x_test = torch.tensor(group["x_test"][:], dtype=torch.float32)
        y_test = torch.tensor(group["y_test"][:].flatten(), dtype=torch.int64)

    num_classes = max(2, int(max(y_train.max(), y_test.max()).item()) + 1)
    return TaskBatch(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metadata={},
        num_classes=num_classes,
    )


def _build_ladder_model(rung: str):
    from tab_foundry.model.architectures.tabfoundry_staged.model import TabFoundryStagedClassifier

    base_kwargs: dict[str, object] = {
        "stage": "prenorm_block",
        "d_icl": 96,
        "input_normalization": "train_zscore_clip",
        "many_class_base": 2,
        "tficl_n_heads": 4,
        "tficl_n_layers": 3,
        "head_hidden_dim": 192,
        "module_overrides": {"table_block_style": "prenorm"},
    }
    if rung in ("C", "D"):
        base_kwargs["staged_dropout"] = 0.1
    if rung == "D":
        base_kwargs["pre_encoder_clip"] = 10.0
    return TabFoundryStagedClassifier(**base_kwargs)


@pytest.mark.slow
class TestStabilityLadderH5:
    @pytest.mark.parametrize("rung", ["A", "B", "C", "D"])
    def test_short_training_loop_no_nan(self, h5_file: Path, rung: str) -> None:
        """Run 50 training steps and verify no NaN/Inf in loss history."""
        from tab_foundry.training.losses import classification_loss

        model = _build_ladder_model(rung)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        grad_clip = 1.0 if rung in ("B", "C", "D") else 0.0

        loss_history: list[float] = []
        grad_norm_history: list[float] = []

        batch = _make_batch_from_h5(h5_file)
        # Limit classes to model capacity
        if batch.num_classes is not None and batch.num_classes > 2:
            # Filter to binary
            mask_train = batch.y_train < 2
            mask_test = batch.y_test < 2
            batch = type(batch)(
                x_train=batch.x_train[mask_train],
                y_train=batch.y_train[mask_train],
                x_test=batch.x_test[mask_test],
                y_test=batch.y_test[mask_test],
                metadata={},
                num_classes=2,
            )

        if batch.x_train.shape[0] < 2 or batch.x_test.shape[0] < 1:
            pytest.skip("Not enough samples for binary training after filtering")

        for step in range(50):
            optimizer.zero_grad()
            output = model(batch)
            if output.logits is not None:
                logits = output.logits[:, :2]
                target = batch.y_test.to(torch.int64)
                loss = classification_loss(logits, target)
            else:
                pytest.skip("Model did not produce logits")

            loss_val = float(loss.item())
            loss_history.append(loss_val)

            loss.backward()

            # Compute grad norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += float(p.grad.data.norm(2).item()) ** 2
            total_norm = total_norm**0.5
            grad_norm_history.append(total_norm)

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        # Assertions
        for i, loss_val in enumerate(loss_history):
            assert not (loss_val != loss_val), f"NaN loss at step {i} for rung {rung}"
            assert loss_val != float("inf"), f"Inf loss at step {i} for rung {rung}"

        # Grad norms should stay reasonable
        max_grad = max(grad_norm_history)
        assert max_grad < 1e6, f"Grad norm spike to {max_grad} for rung {rung}"
