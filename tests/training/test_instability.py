from __future__ import annotations

from pathlib import Path

from tab_foundry.training.instability import build_training_telemetry


def test_build_training_telemetry_adds_windowed_diagnostics(tmp_path: Path) -> None:
    history_records = [
        {
            "step": step,
            "train_loss": 1.0 + (0.01 * step),
            "train_loss_delta": None if step == 1 else 0.01,
        }
        for step in range(1, 101)
    ]
    gradient_records = [
        {
            "step": step,
            "global_grad_norm": 0.1 * step,
            "grad_clip_triggered": step % 10 == 0,
            "module_grad_norms": {
                "feature_encoder": 1.0,
                "direct_head": 5.0,
                "column_encoder": 0.5 + (0.01 * step),
                "row_pool": 0.75 + (0.01 * step),
                "context_encoder": 1.25 + (0.01 * step),
            },
            "activation_norms": {
                "post_feature_encoder": 1.0 + (0.1 * step),
                "pre_transformer": 2.0 + (0.2 * step),
                "post_column_encoder": 3.0 + (0.15 * step),
                "post_row_pool": 4.0 + (0.15 * step),
                "post_context_encoder": 5.0 + (0.15 * step),
                "post_transformer_block_8": 10.0 + (0.3 * step),
                "post_transformer_block_9": 11.0 + (0.3 * step),
                "post_transformer_block_10": 12.0 + (0.3 * step),
                "post_transformer_block_11": 13.0 + (0.3 * step),
            },
        }
        for step in range(1, 101)
    ]
    training_surface_record = {
        "training": {
            "schedule_stages": [
                {
                    "name": "prior_dump",
                    "steps": 2500,
                    "lr_max": 0.004,
                    "lr_schedule": "linear",
                    "warmup_ratio": 0.05,
                }
            ]
        }
    }

    telemetry = build_training_telemetry(
        run_dir=tmp_path,
        success=True,
        artifacts={},
        checkpoint_snapshots=[],
        history_records=history_records,
        gradient_records=gradient_records,
        training_surface_record=training_surface_record,
    )

    diagnostics = telemetry["diagnostics"]
    assert diagnostics["windowing"]["warmup_end_step"] == 125
    assert diagnostics["windowing"]["window_record_counts"] == {
        "early_1_25": 25,
        "post_warmup_100": 0,
        "final_10pct": 10,
    }
    assert diagnostics["grad_clip"] == {
        "record_count": 100,
        "clipped_step_count": 10,
        "clipped_step_fraction": 0.1,
    }
    module_balance = diagnostics["module_balance"]["feature_encoder_vs_direct_head"]
    assert module_balance["windows"]["early_1_25"]["feature_encoder_to_direct_head_mean_ratio"] == 0.2
    assert module_balance["windows"]["early_1_25"]["direct_head_to_feature_encoder_mean_ratio"] == 5.0
    stage_local_gradients = diagnostics["stage_local_gradients"]["modules"]
    assert stage_local_gradients["column_encoder"]["windows"]["early_1_25"]["record_count"] == 25
    assert stage_local_gradients["row_pool"]["windows"]["final_10pct"]["mean_grad_norm"] > 0.0
    assert stage_local_gradients["context_encoder"]["windows"]["final_10pct"]["final_grad_norm"] > 0.0
    activations = diagnostics["activation_windows"]["tracked_activations"]
    assert activations["post_feature_encoder"]["windows"]["early_1_25"]["record_count"] == 25
    assert activations["pre_transformer"]["windows"]["final_10pct"]["record_count"] == 10
    assert activations["post_column_encoder"]["windows"]["final_10pct"]["mean"] > 0.0
    assert activations["post_row_pool"]["early_to_final_mean_delta"] > 0.0
    assert activations["post_context_encoder"]["early_to_final_mean_delta"] > 0.0
    assert activations["post_feature_encoder"]["early_to_final_mean_delta"] > 0.0
    upper_blocks = diagnostics["activation_windows"]["upper_transformer_blocks"]
    assert upper_blocks["block_names"] == [
        "post_transformer_block_8",
        "post_transformer_block_9",
        "post_transformer_block_10",
        "post_transformer_block_11",
    ]
    assert upper_blocks["aggregate"]["final_window_mean"] > 0.0
    assert upper_blocks["aggregate"]["post_warmup_mean_slope"] is None


def test_build_training_telemetry_handles_missing_context_stage_metrics(tmp_path: Path) -> None:
    telemetry = build_training_telemetry(
        run_dir=tmp_path,
        success=True,
        artifacts={},
        checkpoint_snapshots=[],
        history_records=[{"step": 1, "train_loss": 1.0, "train_loss_delta": None}],
        gradient_records=[
            {
                "step": 1,
                "global_grad_norm": 0.5,
                "grad_clip_triggered": False,
                "module_grad_norms": {
                    "column_encoder": 0.25,
                    "row_pool": 0.4,
                },
                "activation_norms": {
                    "post_column_encoder": 1.5,
                    "post_row_pool": 2.0,
                },
            }
        ],
    )

    diagnostics = telemetry["diagnostics"]
    assert (
        diagnostics["stage_local_gradients"]["modules"]["context_encoder"]["windows"]["early_1_25"]["record_count"]
        == 0
    )
    assert (
        diagnostics["activation_windows"]["tracked_activations"]["post_context_encoder"]["windows"]["early_1_25"][
            "record_count"
        ]
        == 0
    )
