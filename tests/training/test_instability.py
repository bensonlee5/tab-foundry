from __future__ import annotations

from pathlib import Path

import pytest
from torch import nn

from tab_foundry.training.instability import (
    build_regime_budget_summary,
    build_runtime_summary,
    build_training_telemetry,
    build_utilization_summary,
    gradient_module_map,
    history_loss_summary,
)


def test_build_training_telemetry_adds_windowed_diagnostics(tmp_path: Path) -> None:
    history_records = [
        {
            "step": step,
            "train_loss": 1.0 + (0.01 * step),
            "train_loss_delta": None if step == 1 else 0.01,
            "task_batch_size_requested": 8,
            "task_batch_size_actual": 8 if step % 5 else 1,
            "task_batch_batched_count": 1 if step % 5 else 0,
            "task_batch_singleton_fallback_count": 0 if step % 5 else 1,
            "task_batch_singleton_fallback_fraction": 0.0 if step % 5 else 1.0,
            "task_batch_signature_counts": {
                "24x8x6x2": 1 if step % 3 else 0,
                "18x6x6x2": 0 if step % 3 else 1,
            },
        }
        for step in range(1, 101)
    ]
    gradient_records = [
        {
            "step": step,
            "global_grad_norm": 0.1 * step,
            "grad_clip_triggered": step % 10 == 0,
            "task_batch_size_requested": 8,
            "task_batch_size_actual": 8 if step % 5 else 1,
            "task_batch_batched_count": 1 if step % 5 else 0,
            "task_batch_singleton_fallback_count": 0 if step % 5 else 1,
            "task_batch_singleton_fallback_fraction": 0.0 if step % 5 else 1.0,
            "task_batch_signature_counts": {
                "24x8x6x2": 1 if step % 3 else 0,
                "18x6x6x2": 0 if step % 3 else 1,
            },
            "timing_seconds": {
                "data_wait": 0.1,
                "batch_diagnostics": 0.05,
                "h2d_transfer": 0.02,
                "forward_backward": 0.5,
                "activation_trace": 0.0,
                "grad_diagnostics": 0.03,
                "optimizer": 0.08,
                "checkpoint": 0.04 if step % 10 == 0 else 0.0,
            },
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
    assert diagnostics["task_batching"] == {
        "record_count": 100,
        "requested_task_batch_sizes": [8],
        "actual_task_batch_size_counts": {"1": 20, "8": 80},
        "batched_step_count": 80,
        "singleton_fallback_count": 20,
        "singleton_fallback_fraction": 0.2,
        "signature_counts": {"18x6x6x2": 33, "24x8x6x2": 67},
        "signature_family_steps": {
            "one_family_step_count": 100,
            "mixed_family_step_count": 0,
            "consecutive_repeated_family_step_count": 33,
            "consecutive_switched_family_step_count": 66,
            "family_block_count": 67,
            "estimated_family_switch_count": 66,
        },
    }
    step_timing_summary = diagnostics["step_timing_summary"]
    assert step_timing_summary["profiled_step_count"] == 100
    assert step_timing_summary["mean_profiled_step_seconds"] == pytest.approx(0.784)
    assert step_timing_summary["buckets"]["data_wait"] == {
        "mean_seconds": pytest.approx(0.1),
        "fraction_of_profiled_step_time": pytest.approx(0.1275510204),
    }
    assert step_timing_summary["buckets"]["forward_backward"] == {
        "mean_seconds": pytest.approx(0.5),
        "fraction_of_profiled_step_time": pytest.approx(0.6377551020),
    }
    assert step_timing_summary["buckets"]["checkpoint"] == {
        "mean_seconds": pytest.approx(0.004),
        "fraction_of_profiled_step_time": pytest.approx(0.0051020408),
    }


def test_build_training_telemetry_uses_sandwich_stage_activations_for_upper_blocks(
    tmp_path: Path,
) -> None:
    history_records = [
        {
            "step": step,
            "train_loss": 1.0 - (0.01 * step),
            "train_loss_delta": None if step == 1 else -0.01,
        }
        for step in range(1, 41)
    ]
    gradient_records = [
        {
            "step": step,
            "global_grad_norm": 0.2 * step,
            "grad_clip_triggered": False,
            "activation_norms": {
                "post_stage_0_self": 6.0 + (0.02 * step),
                "post_stage_1_self": 6.5 + (0.02 * step),
            },
        }
        for step in range(1, 41)
    ]
    training_surface_record = {
        "model": {
            "arch": "tabfoundry_sandwich",
        },
        "training": {
            "loss_surface": "classification",
            "schedule_stages": [
                {
                    "name": "stage1",
                    "steps": 40,
                    "lr_max": 8.0e-4,
                    "warmup_ratio": 0.1,
                }
            ],
        },
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

    upper_blocks = telemetry["diagnostics"]["activation_windows"]["upper_transformer_blocks"]
    assert upper_blocks["block_names"] == [
        "post_stage_0_self",
        "post_stage_1_self",
    ]
    assert upper_blocks["aggregate"]["final_window_mean"] is not None
    assert upper_blocks["aggregate"]["post_warmup_mean_slope"] is not None


def test_history_loss_summary_weights_losses_by_actual_task_count() -> None:
    summary = history_loss_summary(
        [
            {
                "step": 1,
                "train_loss": 1.0,
                "train_loss_ema": 1.0,
                "train_loss_delta": None,
                "task_batch_size_actual": 2,
            },
            {
                "step": 2,
                "train_loss": 3.0,
                "train_loss_ema": 2.5,
                "train_loss_delta": 2.0,
                "task_batch_size_actual": 1,
            },
        ]
    )

    assert summary["record_count"] == 2
    assert summary["initial_train_loss"] == 1.0
    assert summary["final_train_loss"] == 3.0
    assert summary["final_train_loss_ema"] == 2.5
    assert summary["mean_train_loss"] == pytest.approx(5.0 / 3.0)
    assert summary["train_loss_variance"] == pytest.approx(8.0 / 9.0)
    assert summary["max_abs_train_loss_delta"] == 2.0
    assert summary["final_tail_record_count"] == 1
    assert summary["final_tail_mean_train_loss"] == 3.0
    assert summary["final_tail_mean_train_loss_ema"] == 2.5


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


def test_build_training_telemetry_omits_direct_head_balance_when_head_is_inactive(
    tmp_path: Path,
) -> None:
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
                    "feature_encoder": 0.25,
                    "gaussian_head": 0.4,
                    "cell_decoder_blocks.0": 0.6,
                },
            }
        ],
    )

    assert telemetry["diagnostics"]["module_balance"] == {}
    assert telemetry["gradient_summary"]["modules"]["gaussian_head"]["final_grad_norm"] == 0.4


class _TinySandwichTelemetryModel(nn.Module):
    def __init__(self, *, loss_surface: str) -> None:
        super().__init__()
        self.arch = "tabfoundry_sandwich"
        self.loss_surface = loss_surface
        self.tokenizer = nn.Linear(1, 1)
        self.feature_encoder = nn.Linear(1, 1)
        self.feature_type_film = nn.Linear(1, 1)
        self.row_summary_builder = nn.Linear(1, 1)
        self.column_summary_builder = nn.Linear(1, 1)
        self.y_conditioner = nn.Linear(1, 1)
        self.y_role_embedding = nn.Embedding(2, 1)
        self.token_type_embedding = nn.Embedding(2, 1)
        self.pre_row_attention_blocks = nn.ModuleList([nn.Linear(1, 1)])
        self.pre_column_attention_blocks = nn.ModuleList([nn.Linear(1, 1)])
        self.perceiver_stages = nn.ModuleList([nn.Linear(1, 1)])
        self.latent_readout = nn.Linear(1, 1)
        self.cell_readout = nn.Linear(1, 1)
        self.test_row_pool = nn.Linear(1, 1)
        self.direct_head = nn.Linear(1, 1)
        self.cell_decoder_blocks = nn.ModuleList([nn.Linear(1, 1)])
        self.gaussian_head = nn.Linear(1, 1)
        self.discrete_query = nn.Linear(1, 1)
        self.discrete_oov = nn.Linear(1, 1)
        self.integer_gate = nn.Linear(1, 1)


def test_gradient_module_map_tracks_only_active_cell_bpc_sandwich_modules() -> None:
    modules = gradient_module_map(_TinySandwichTelemetryModel(loss_surface="cell_bpc"))

    assert set(modules) == {
        "tokenizer",
        "feature_encoder",
        "feature_type_film",
        "y_conditioner",
        "y_role_embedding",
        "token_type_embedding",
        "pre_row_attention_blocks.0",
        "pre_column_attention_blocks.0",
        "cell_decoder_blocks.0",
        "gaussian_head",
        "discrete_query",
        "discrete_oov",
        "integer_gate",
    }


def test_build_training_telemetry_tracks_non_finite_global_grad_norm_kinds(tmp_path: Path) -> None:
    telemetry = build_training_telemetry(
        run_dir=tmp_path,
        success=True,
        artifacts={},
        checkpoint_snapshots=[],
        history_records=[{"step": 1, "train_loss": 1.0, "train_loss_delta": None}],
        gradient_records=[
            {"step": 1, "global_grad_norm": None, "global_grad_norm_kind": "pos_inf"},
            {"step": 2, "global_grad_norm": 0.5, "global_grad_norm_kind": "finite"},
            {"step": 3, "global_grad_norm": None, "global_grad_norm_kind": "nan"},
            {"step": 4, "global_grad_norm": None, "global_grad_norm_kind": "neg_inf"},
        ],
    )

    gradient_summary = telemetry["gradient_summary"]
    assert gradient_summary["global"]["mean_grad_norm"] == 0.5
    assert gradient_summary["global"]["max_grad_norm"] == 0.5
    assert gradient_summary["global"]["final_grad_norm"] == 0.5
    assert gradient_summary["non_finite_global_grad_norm_counts"] == {
        "nan": 1,
        "pos_inf": 1,
        "neg_inf": 1,
    }
    assert gradient_summary["final_global_grad_norm_kind"] == "neg_inf"


def test_build_training_telemetry_persists_wandb_identity_when_available(tmp_path: Path) -> None:
    telemetry = build_training_telemetry(
        run_dir=tmp_path,
        success=True,
        artifacts={},
        checkpoint_snapshots=[],
        history_records=[{"step": 1, "train_loss": 1.0, "train_loss_delta": None}],
        gradient_records=[],
        wandb={
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-123",
            "run_name": "demo-run",
            "mode": "online",
        },
    )

    assert telemetry["wandb"] == {
        "entity": "test-entity",
        "project": "test-project",
        "run_id": "run-123",
        "run_name": "demo-run",
        "mode": "online",
    }


def test_build_training_telemetry_leaves_wandb_empty_when_unavailable(tmp_path: Path) -> None:
    telemetry = build_training_telemetry(
        run_dir=tmp_path,
        success=True,
        artifacts={},
        checkpoint_snapshots=[],
        history_records=[{"step": 1, "train_loss": 1.0, "train_loss_delta": None}],
        gradient_records=[],
    )

    assert telemetry["wandb"] is None


def test_build_training_telemetry_persists_runtime_and_regime_budget_metadata(
    tmp_path: Path,
) -> None:
    training_surface_record = {
        "data": {
            "surface_label": "dagzoo_shape_aware_multi_invocation",
            "dagzoo_provenance": {
                "corpus_variant": "dagzoo_shape_aware_multi_invocation",
                "config_refs": ["configs/dagzoo/binary.yaml"],
                "invocations": [
                    {
                        "invocation_id": "shape_aware",
                        "requested_config_ref": "configs/dagzoo/binary.yaml",
                        "num_datasets": 48,
                        "rows": 256,
                        "handoff": {
                            "source_family": "dagzoo.fixed_layout_scm",
                            "generate_run_id": "1" * 32,
                            "generated_corpus_id": "2" * 32,
                        },
                    }
                ],
            },
            "manifest": {
                "characteristics": {
                    "split_counts": {"train": 96, "val": 12},
                    "row_count_distribution": {"min": 24, "max": 40},
                    "feature_count_distribution": {"min": 6, "max": 8},
                    "class_count_distribution": {"min": 2, "max": 2},
                }
            },
        }
    }
    telemetry = build_training_telemetry(
        run_dir=tmp_path,
        task="classification",
        global_step=75,
        success=True,
        artifacts={},
        checkpoint_snapshots=[],
        history_records=[{"step": 75, "train_loss": 0.4, "train_loss_delta": -0.01}],
        gradient_records=[],
        runtime_summary=build_runtime_summary(
            train_elapsed_seconds=3.0,
            wall_elapsed_seconds=3.8,
            examples_seen=96,
            tokens_seen=38400,
            peak_memory_summary={"peak_vram_allocated": 1024, "peak_vram_reserved": 2048},
            total_device_vram_bytes=80 * 1024**3,
        ),
        hardware_summary={
            "device_type": "cuda",
            "raw_device_name": "NVIDIA A100-SXM4-80GB",
            "gpu_class": "a100",
            "total_device_vram_bytes": 80 * 1024**3,
            "vram_class_gb": 80,
            "hardware_profile_id": "a100_80gb",
        },
        regime_budget=build_regime_budget_summary(
            task="classification",
            loss_surface="classification",
            training_surface_record=training_surface_record,
            global_step=75,
            tokens_seen=38400,
        ),
        training_surface_record=training_surface_record,
    )

    assert telemetry["runtime_summary"] == {
        "peak_vram_allocated": 1024,
        "peak_vram_reserved": 2048,
        "peak_vram_allocated_fraction": pytest.approx(1024 / float(80 * 1024**3)),
        "peak_vram_reserved_fraction": pytest.approx(2048 / float(80 * 1024**3)),
        "throughput_examples_per_second": 32.0,
        "throughput_tokens_per_second": 12800.0,
        "non_train_overhead_seconds": pytest.approx(0.8),
        "non_train_overhead_fraction": pytest.approx(0.8 / 3.8),
    }
    assert telemetry["utilization_summary"] == {
        "peak_vram_allocated_fraction": pytest.approx(1024 / float(80 * 1024**3)),
        "peak_vram_reserved_fraction": pytest.approx(2048 / float(80 * 1024**3)),
        "non_train_overhead_fraction": pytest.approx(0.8 / 3.8),
        "achieved_train_tflops_per_second": None,
        "theoretical_peak_tflops_per_second": None,
        "compute_utilization_fraction": None,
        "theoretical_hbm_bandwidth_gbps": None,
        "roofline_knee_flops_per_byte": None,
        "peak_compute_basis": None,
    }
    assert telemetry["hardware_summary"]["gpu_class"] == "a100"
    assert telemetry["hardware_summary"]["vram_class_gb"] == 80
    assert telemetry["regime_budget"]["tokens_per_step"] == pytest.approx(512.0)
    assert telemetry["regime_budget"]["unique_task_budget"] == 96
    assert telemetry["regime_budget"]["objective_metric"] == "final_log_loss_at_matched_regime_budget"
    assert telemetry["regime_budget"]["curriculum_id"] == "1" * 32


def test_build_runtime_summary_records_loader_wall_metadata() -> None:
    summary = build_runtime_summary(
        train_elapsed_seconds=3.0,
        wall_elapsed_seconds=3.8,
        end_to_end_wall_seconds=4.2,
        loader_setup_seconds=0.4,
        examples_seen=96,
        tokens_seen=38400,
        peak_memory_summary={"peak_vram_allocated": 1024, "peak_vram_reserved": 2048},
        total_device_vram_bytes=80 * 1024**3,
        loader_effective_num_workers=8,
        loader_effective_prefetch_factor=4,
        loader_task_batch_cache_mode="bounded_streaming",
        compile_shape_dispatch_mode="signature_family",
        compile_shape_dispatch_max_families=16,
        compile_shape_dispatch_summary={"compiled_family_count": 3, "family_switch_count": 7},
    )

    assert summary == {
        "peak_vram_allocated": 1024,
        "peak_vram_reserved": 2048,
        "peak_vram_allocated_fraction": pytest.approx(1024 / float(80 * 1024**3)),
        "peak_vram_reserved_fraction": pytest.approx(2048 / float(80 * 1024**3)),
        "throughput_examples_per_second": 32.0,
        "throughput_tokens_per_second": 12800.0,
        "non_train_overhead_seconds": pytest.approx(0.8),
        "non_train_overhead_fraction": pytest.approx(0.8 / 3.8),
        "end_to_end_wall_seconds": 4.2,
        "loader_setup_seconds": 0.4,
        "loader_effective_num_workers": 8,
        "loader_effective_prefetch_factor": 4,
        "loader_task_batch_cache_mode": "bounded_streaming",
        "compile_shape_dispatch_mode": "signature_family",
        "compile_shape_dispatch_max_families": 16,
        "compile_shape_dispatch": {
            "compiled_family_count": 3,
            "family_switch_count": 7,
        },
    }


def test_build_utilization_summary_enriches_compute_utilization_from_compute_accounting() -> None:
    utilization = build_utilization_summary(
        runtime_summary={
            "peak_vram_allocated": 40 * 1024**3,
            "peak_vram_reserved": 50 * 1024**3,
            "peak_vram_allocated_fraction": 0.5,
            "peak_vram_reserved_fraction": 0.625,
            "throughput_tokens_per_second": 160000.0,
            "non_train_overhead_fraction": 0.2,
        },
        hardware_summary={
            "device_type": "cuda",
            "raw_device_name": "NVIDIA A100-SXM4-80GB",
            "gpu_class": "a100",
            "total_device_vram_bytes": 80 * 1024**3,
            "hardware_profile_id": "a100_80gb",
        },
        training_surface_record={
            "runtime": {
                "mixed_precision": "bf16",
            }
        },
        compute_accounting={
            "train_flops_per_token": 2.5e7,
        },
    )

    assert utilization == {
        "peak_vram_allocated_fraction": 0.5,
        "peak_vram_reserved_fraction": 0.625,
        "non_train_overhead_fraction": 0.2,
        "achieved_train_tflops_per_second": pytest.approx(4.0),
        "theoretical_peak_tflops_per_second": 312.0,
        "compute_utilization_fraction": pytest.approx(4.0 / 312.0),
        "theoretical_hbm_bandwidth_gbps": 2039.0,
        "roofline_knee_flops_per_byte": pytest.approx(153.0161844031388),
        "peak_compute_basis": "tensorcore_bf16_dense",
    }


def test_build_utilization_summary_keeps_ambiguous_gpu_peaks_null() -> None:
    utilization = build_utilization_summary(
        runtime_summary={
            "peak_vram_allocated": 40 * 1024**3,
            "peak_vram_reserved": 50 * 1024**3,
            "throughput_tokens_per_second": 160000.0,
            "non_train_overhead_fraction": 0.2,
        },
        hardware_summary={
            "device_type": "cuda",
            "raw_device_name": "NVIDIA H100 PCIe",
            "gpu_class": "h100",
            "total_device_vram_bytes": 80 * 1024**3,
            "hardware_profile_id": "h100_80gb",
        },
        training_surface_record={
            "runtime": {
                "mixed_precision": "bf16",
            }
        },
        compute_accounting={
            "train_flops_per_token": 2.5e7,
        },
    )

    assert utilization == {
        "peak_vram_allocated_fraction": pytest.approx(0.5),
        "peak_vram_reserved_fraction": pytest.approx(0.625),
        "non_train_overhead_fraction": 0.2,
        "achieved_train_tflops_per_second": pytest.approx(4.0),
        "theoretical_peak_tflops_per_second": None,
        "compute_utilization_fraction": None,
        "theoretical_hbm_bandwidth_gbps": None,
        "roofline_knee_flops_per_byte": None,
        "peak_compute_basis": None,
    }
