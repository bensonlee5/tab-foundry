from __future__ import annotations

from tab_foundry.training.artifacts import gradient_history_record


def test_gradient_history_record_includes_grad_norm_alias() -> None:
    record = gradient_history_record(
        global_step=3,
        stage_name="prior_dump",
        train_loss=1.25,
        train_acc=0.5,
        lr=1.0e-3,
        global_grad_norm=0.875,
        global_grad_norm_kind="finite",
        module_grad_norms={"feature_encoder": 0.25},
        activation_norms=None,
        elapsed_seconds=12.0,
        train_elapsed_seconds=10.0,
        grad_clip_threshold=0.0,
        grad_clip_triggered=False,
        task_batch_size_requested=16,
        task_batch_size_actual=64,
        task_batch_batched_count=4,
        task_batch_singleton_fallback_count=0,
        task_batch_singleton_fallback_fraction=0.0,
        task_batch_signature_counts={"192x64x20x2": 1},
    )

    assert record["grad_norm"] == 0.875
    assert record["global_grad_norm"] == 0.875
    assert record["global_grad_norm_kind"] == "finite"


def test_gradient_history_record_keeps_grad_norm_alias_null_for_non_finite_values() -> None:
    record = gradient_history_record(
        global_step=4,
        stage_name="prior_dump",
        train_loss=1.5,
        train_acc=None,
        lr=1.0e-3,
        global_grad_norm=None,
        global_grad_norm_kind="nan",
        module_grad_norms={},
        activation_norms=None,
        elapsed_seconds=15.0,
        train_elapsed_seconds=12.0,
        grad_clip_threshold=0.0,
        grad_clip_triggered=False,
        task_batch_size_requested=16,
        task_batch_size_actual=0,
        task_batch_batched_count=0,
        task_batch_singleton_fallback_count=0,
        task_batch_singleton_fallback_fraction=0.0,
        task_batch_signature_counts={},
    )

    assert record["grad_norm"] is None
    assert record["global_grad_norm"] is None
    assert record["global_grad_norm_kind"] == "nan"
