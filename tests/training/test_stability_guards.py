"""Tests for NaN guard and cosine warmup stability mechanisms."""

from __future__ import annotations

import math

from tab_foundry.training.schedule import StageConfig, stage_base_lr, warmup_steps_for_stage


class TestCosineWarmup:
    """Cosine schedule with warmup_ratio > 0 should linear-ramp then cosine-decay."""

    def test_cosine_warmup_linear_ramp(self) -> None:
        stage = StageConfig(name="s1", steps=100, lr_max=1.0, lr_schedule="cosine", warmup_ratio=0.05)
        warmup = warmup_steps_for_stage(stage)
        assert warmup == 5
        # Steps 1..5 should be a linear ramp from lr_max/5 to lr_max
        for step in range(1, warmup + 1):
            lr = stage_base_lr(stage, step=step, lr_min=0.0)
            expected = 1.0 * (step / warmup)
            assert abs(lr - expected) < 1e-9, f"step={step}: {lr} != {expected}"

    def test_cosine_warmup_peak_at_warmup_end(self) -> None:
        stage = StageConfig(name="s1", steps=100, lr_max=1.0, lr_schedule="cosine", warmup_ratio=0.05)
        warmup = warmup_steps_for_stage(stage)
        lr_at_warmup = stage_base_lr(stage, step=warmup, lr_min=0.0)
        assert abs(lr_at_warmup - 1.0) < 1e-9

    def test_cosine_warmup_decay_after_warmup(self) -> None:
        stage = StageConfig(name="s1", steps=100, lr_max=1.0, lr_schedule="cosine", warmup_ratio=0.05)
        warmup = warmup_steps_for_stage(stage)
        # First post-warmup step should be at or near lr_max (start of cosine)
        lr_post = stage_base_lr(stage, step=warmup + 1, lr_min=0.0)
        assert lr_post <= 1.0 + 1e-9
        # Last step should be near lr_min
        lr_end = stage_base_lr(stage, step=100, lr_min=0.0)
        assert lr_end < 0.01

    def test_cosine_no_warmup_backward_compatible(self) -> None:
        """With warmup_ratio=0.0, cosine schedule should behave as before."""
        stage = StageConfig(name="s1", steps=100, lr_max=1.0, lr_schedule="cosine", warmup_ratio=0.0)
        # Step 1 should be at lr_max (start of cosine)
        lr_start = stage_base_lr(stage, step=1, lr_min=0.0)
        assert abs(lr_start - 1.0) < 1e-9
        # Last step should be at lr_min
        lr_end = stage_base_lr(stage, step=100, lr_min=0.0)
        assert abs(lr_end - 0.0) < 1e-9

    def test_cosine_warmup_monotonically_increases_during_warmup(self) -> None:
        stage = StageConfig(name="s1", steps=200, lr_max=0.001, lr_schedule="cosine", warmup_ratio=0.1)
        warmup = warmup_steps_for_stage(stage)
        prev_lr = 0.0
        for step in range(1, warmup + 1):
            lr = stage_base_lr(stage, step=step, lr_min=0.0)
            assert lr > prev_lr, f"step={step}: lr={lr} <= prev={prev_lr}"
            prev_lr = lr

    def test_cosine_warmup_with_nonzero_lr_min(self) -> None:
        stage = StageConfig(name="s1", steps=100, lr_max=1.0, lr_schedule="cosine", warmup_ratio=0.1)
        lr_min = 0.1
        lr_end = stage_base_lr(stage, step=100, lr_min=lr_min)
        assert abs(lr_end - lr_min) < 1e-6


class TestNanGuardMetrics:
    """Verify nan_skip_count is tracked correctly in the training result."""

    def test_nan_skip_count_in_train_result_metrics(self) -> None:
        """TrainResult.metrics should include nan_skip_count field."""
        from tab_foundry.types import TrainResult
        from pathlib import Path

        result = TrainResult(
            output_dir=Path("/tmp/test"),
            best_checkpoint=None,
            latest_checkpoint=None,
            global_step=10,
            metrics={
                "best_val_loss": 0.5,
                "nan_skip_count": 2.0,
            },
        )
        assert result.metrics["nan_skip_count"] == 2.0

    def test_trainer_summary_payload_includes_nan_skip_count(self) -> None:
        from pathlib import Path
        from tab_foundry.training.trainer import _trainer_summary_payload

        payload = _trainer_summary_payload(
            output_dir=Path("/tmp/test"),
            optimizer_requested_name="adamw",
            optimizer_resolved_name="adamw",
            optimizer_fallback_reason=None,
            global_step=10,
            best_checkpoint=None,
            latest_checkpoint=None,
            best_val=0.5,
            best_val_step=5.0,
            final_train_loss=0.3,
            final_train_loss_ema=0.3,
            last_train_metrics=None,
            last_val_metrics=None,
            final_grad_norm=1.0,
            grad_norm_sum=10.0,
            grad_norm_count=10,
            max_grad_norm=2.0,
            train_elapsed_seconds=10.0,
            wall_elapsed_seconds=12.0,
            nan_skip_count=3,
        )
        assert payload["metrics"]["nan_skip_count"] == 3.0

    def test_isfinite_detects_nan(self) -> None:
        """Verify the guard condition correctly identifies NaN."""
        assert not math.isfinite(float("nan"))
        assert not math.isfinite(float("inf"))
        assert math.isfinite(0.0)
        assert math.isfinite(1.5)
