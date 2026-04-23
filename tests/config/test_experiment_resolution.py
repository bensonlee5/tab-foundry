from __future__ import annotations

from pathlib import Path

from hydra.errors import MissingConfigException
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from tab_foundry.config_inspection import resolve_config_payload
from tab_foundry.model.spec import model_build_spec_from_mappings


def _compose(*overrides: str):
    cfg_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def _resolved_model_spec(cfg: object):
    model_cfg = OmegaConf.to_container(getattr(cfg, "model"), resolve=True)
    assert isinstance(model_cfg, dict)
    return model_build_spec_from_mappings(task=str(getattr(cfg, "task")), primary=model_cfg)


def test_cls_workstation_task_resolution() -> None:
    cfg = _compose("experiment=cls_workstation")
    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_sandwich"
    assert cfg.model.stage is None
    assert int(cfg.model.feature_group_size) == 1
    assert bool(cfg.preprocessing.impute_missing) is True
    assert float(cfg.preprocessing.all_nan_fill) == 0.0
    assert str(cfg.optimizer.name) == "muon"
    assert bool(cfg.optimizer.require_requested) is True
    assert bool(cfg.runtime.activation_checkpointing) is False


def test_cls_workstation_sandwich_resolution() -> None:
    cfg = _compose("experiment=cls_workstation_sandwich")
    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_sandwich"
    assert cfg.model.stage is None
    assert int(cfg.model.d_icl) == 60
    assert str(cfg.model.input_normalization) == "train_zscore_clip"
    assert bool(cfg.model.sandwich_packed_attention) is True
    assert int(cfg.model.head_hidden_dim) == 96
    assert int(cfg.model.sandwich_latents) == 24
    assert int(cfg.model.sandwich_layers) == 2
    assert int(cfg.model.sandwich_heads) == 4
    assert int(cfg.model.sandwich_summary_tokens_per_axis) == 3
    assert int(cfg.model.sandwich_self_attention_per_cross) == 4
    assert int(cfg.model.sandwich_pre_row_attention_layers) == 1
    assert int(cfg.model.sandwich_pre_column_attention_layers) == 1
    assert int(cfg.model.sandwich_pre_column_inducing_tokens) == 16
    assert str(cfg.data.surface_label) == "tf_rd_010_dagzoo_medium_control"
    assert str(cfg.data.corpus_ref) == "tf_rd_010_dagzoo_medium_control_curated_v5"
    assert str(cfg.optimizer.name) == "muon"
    assert bool(cfg.optimizer.require_requested) is True
    assert str(cfg.runtime.device) == "cuda"
    assert str(cfg.runtime.loader_task_batch_cache_mode) == "bounded_streaming"
    assert str(cfg.runtime.num_workers) == "auto"
    assert bool(cfg.runtime.loader_pin_memory) is True
    assert str(cfg.runtime.loader_prefetch_factor) == "auto"
    assert bool(cfg.runtime.non_blocking_device_transfer) is True
    assert bool(cfg.runtime.compile_model) is True
    assert bool(cfg.runtime.compile_dynamic) is True
    assert str(cfg.runtime.compile_backend) == "eager"
    assert str(cfg.runtime.compile_shape_dispatch_mode) == "signature_family"
    assert int(cfg.runtime.compile_shape_dispatch_max_families) == 16
    assert int(cfg.runtime.signature_family_run_length) == 4
    assert str(cfg.runtime.output_dir) == "outputs/cls_workstation_sandwich"
    assert bool(cfg.runtime.trace_activations) is False
    assert str(cfg.logging.run_name) == "cls-workstation-sandwich"


def test_cls_workstation_sandwich_followon_arch_swaps_resolve() -> None:
    routed_cfg = _compose("experiment=cls_workstation_sandwich", "model.arch=routed_sandwich")
    routed_spec = _resolved_model_spec(routed_cfg)

    assert str(routed_cfg.model.sandwich_summary_tokens_per_axis) == "3"
    assert routed_spec.arch == "routed_sandwich"
    assert "sandwich_summary_tokens_per_axis" not in routed_spec.to_dict()

    grid_cfg = _compose("experiment=cls_workstation_sandwich", "model.arch=grid_sandwich")
    grid_spec = _resolved_model_spec(grid_cfg)

    assert str(grid_cfg.model.sandwich_summary_tokens_per_axis) == "3"
    assert grid_spec.arch == "grid_sandwich"
    assert "sandwich_summary_tokens_per_axis" not in grid_spec.to_dict()
    assert "sandwich_latents" not in grid_spec.to_dict()
    assert "sandwich_self_attention_per_cross" not in grid_spec.to_dict()


def test_cls_workstation_routed_sandwich_resolution() -> None:
    cfg = _compose("experiment=cls_workstation_routed_sandwich")

    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "routed_sandwich"
    assert str(cfg.data.corpus_ref) == "tf_rd_010_dagzoo_medium_control_curated_v6"
    assert int(cfg.model.d_icl) == 144
    assert int(cfg.model.head_hidden_dim) == 96
    assert int(cfg.model.sandwich_layers) == 4
    assert int(cfg.model.sandwich_heads) == 1
    assert cfg.model.sandwich_summary_tokens_per_axis is None
    assert int(cfg.model.routed_row_summary_tokens) == 4
    assert int(cfg.model.routed_column_summary_tokens) == 2
    assert int(cfg.model.routed_evidence_tokens) == 16
    assert int(cfg.runtime.max_steps) == 5000
    assert int(cfg.schedule.stages[0].steps) == 5000
    assert str(cfg.runtime.output_dir) == "outputs/cls_workstation_routed_sandwich"
    assert str(cfg.logging.run_name) == "cls-workstation-routed-sandwich"


def test_cls_workstation_grid_sandwich_resolution() -> None:
    cfg = _compose("experiment=cls_workstation_grid_sandwich")

    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "grid_sandwich"
    assert str(cfg.data.corpus_ref) == "tf_rd_010_dagzoo_medium_control_curated_v6"
    assert int(cfg.model.d_icl) == 144
    assert int(cfg.model.head_hidden_dim) == 96
    assert int(cfg.model.sandwich_layers) == 4
    assert int(cfg.model.sandwich_heads) == 1
    assert cfg.model.sandwich_latents is None
    assert cfg.model.sandwich_summary_tokens_per_axis is None
    assert cfg.model.sandwich_self_attention_per_cross is None
    assert int(cfg.model.sandwich_pre_row_attention_layers) == 1
    assert int(cfg.model.sandwich_pre_column_attention_layers) == 1
    assert str(cfg.model.grid_residual_mode) == "prenorm"
    assert str(cfg.model.grid_attention_mode) == "standard"
    assert str(cfg.model.grid_ffn_mode) == "swiglu"
    assert int(cfg.model.grid_recurrence_steps) == 8
    assert int(cfg.model.grid_recurrence_unique_layers) == 2
    assert cfg.model.classification_logit_softcap is None
    assert bool(cfg.model.attention_qk_norm) is False
    assert float(cfg.training.classification_z_loss_coeff) == 0.0
    assert bool(cfg.runtime.activation_checkpointing) is False
    assert int(cfg.runtime.max_steps) == 5000
    assert int(cfg.schedule.stages[0].steps) == 5000
    assert str(cfg.runtime.output_dir) == "outputs/cls_workstation_grid_sandwich"
    assert str(cfg.logging.run_name) == "cls-workstation-grid-sandwich"


def test_resolve_config_rejects_negative_classification_z_loss_coeff() -> None:
    with pytest.raises(ValueError, match="classification_z_loss_coeff"):
        _ = resolve_config_payload(("training.classification_z_loss_coeff=-0.1",))


@pytest.mark.parametrize(
    ("experiment", "field_name", "expected_value"),
    (
        (
            "cls_workstation_grid_sandwich_tf_rd_026_01_hyper_connection_lite",
            "grid_residual_mode",
            "hyper_connection_lite",
        ),
        (
            "cls_workstation_grid_sandwich_tf_rd_026_02_differential_attention",
            "grid_attention_mode",
            "differential",
        ),
        (
            "cls_workstation_grid_sandwich_tf_rd_026_03_swiglu_ffn",
            "grid_ffn_mode",
            "swiglu",
        ),
        (
            "cls_workstation_grid_sandwich_tf_rd_026_04_recurrent_grid_8",
            "grid_recurrence_steps",
            8,
        ),
    ),
)
def test_tf_rd_026_grid_sandwich_experiment_rows_resolve(
    experiment: str,
    field_name: str,
    expected_value: object,
) -> None:
    cfg = _compose(f"experiment={experiment}")
    spec = _resolved_model_spec(cfg)

    assert spec.arch == "grid_sandwich"
    assert getattr(spec, field_name) == expected_value
    assert str(cfg.logging.group) == "tf_rd_026_grid_sandwich_broad_ml_v1"
    assert str(cfg.runtime.output_dir).startswith(
        "outputs/staged_ladder/research/tf_rd_026_grid_sandwich_broad_ml_v1/"
    )


def test_tf_rd_026_grid_sandwich_combo_row_resolves() -> None:
    cfg = _compose("experiment=cls_workstation_grid_sandwich_tf_rd_026_05_hc_swiglu_combo")
    spec = _resolved_model_spec(cfg)

    assert spec.grid_residual_mode == "hyper_connection_lite"
    assert spec.grid_ffn_mode == "swiglu"


def test_cls_workstation_sandwich_legacy_resolution() -> None:
    cfg = _compose("experiment=cls_workstation_sandwich_legacy_v1")

    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_sandwich"
    assert bool(cfg.model.sandwich_packed_attention) is False
    assert int(cfg.model.sandwich_summary_tokens_per_axis) == 4
    assert str(cfg.runtime.device) == "auto"
    assert str(cfg.runtime.loader_task_batch_cache_mode) == "off"
    assert bool(cfg.runtime.compile_model) is False
    assert str(cfg.runtime.output_dir) == "outputs/cls_workstation_sandwich_legacy_v1"
    assert str(cfg.logging.run_name) == "cls-workstation-sandwich-legacy-v1"


def test_default_config_follows_grid_architecture_anchor() -> None:
    cfg = _compose()

    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "grid_sandwich"
    assert str(cfg.data.corpus_ref) == "tf_rd_010_dagzoo_medium_control_curated_v6"
    assert int(cfg.model.d_icl) == 144
    assert int(cfg.model.sandwich_layers) == 4
    assert int(cfg.model.sandwich_heads) == 1
    assert int(cfg.model.head_hidden_dim) == 96
    assert bool(cfg.model.sandwich_packed_attention) is True
    assert str(cfg.optimizer.name) == "muon"
    assert str(cfg.runtime.device) == "cuda"
    assert str(cfg.runtime.loader_task_batch_cache_mode) == "bounded_streaming"
    assert bool(cfg.runtime.compile_model) is True
    assert str(cfg.runtime.compile_shape_dispatch_mode) == "signature_family"


def test_generic_sandwich_compose_accepts_pre_perceiver_override_without_plus() -> None:
    cfg = _compose(
        "model.arch=tabfoundry_sandwich",
        "model.sandwich_pre_row_attention_layers=2",
        "model.sandwich_pre_column_attention_layers=0",
        "model.sandwich_pre_column_inducing_tokens=8",
    )

    assert str(cfg.model.arch) == "tabfoundry_sandwich"
    assert int(cfg.model.sandwich_pre_row_attention_layers) == 2
    assert int(cfg.model.sandwich_pre_column_attention_layers) == 0
    assert int(cfg.model.sandwich_pre_column_inducing_tokens) == 8


def test_cls_smoke_optimizer_resolution() -> None:
    cfg = _compose("experiment=cls_smoke")
    assert str(cfg.model.arch) == "tabfoundry_sandwich"
    assert cfg.model.stage is None
    assert str(cfg.optimizer.name) == "muon"
    assert bool(cfg.optimizer.require_requested) is True
    assert int(cfg.model.feature_group_size) == 1
    assert cfg.logging.history_jsonl_path is None


def test_runtime_smoke_override_resolution() -> None:
    cfg = _compose("experiment=cls_workstation_sandwich_legacy_v1", "runtime=smoke")
    assert str(cfg.runtime.mixed_precision) == "no"
    assert bool(cfg.runtime.loader_pin_memory) is False
    assert bool(cfg.runtime.loader_persistent_workers) is False
    assert cfg.runtime.loader_prefetch_factor is None
    assert str(cfg.runtime.loader_task_batch_cache_mode) == "off"
    assert bool(cfg.runtime.non_blocking_device_transfer) is False
    assert cfg.runtime.checkpoint_every is None
    assert bool(cfg.runtime.compile_model) is False
    assert bool(cfg.runtime.compile_dynamic) is False
    assert str(cfg.runtime.compile_backend) == "inductor"
    assert str(cfg.runtime.compile_mode) == "max-autotune-no-cudagraphs"
    assert str(cfg.runtime.compile_shape_dispatch_mode) == "off"
    assert int(cfg.runtime.compile_shape_dispatch_max_families) == 16
    assert int(cfg.runtime.signature_family_run_length) == 1
    assert bool(cfg.runtime.activation_checkpointing) is False


def test_runtime_workstation_resolution() -> None:
    cfg = _compose("experiment=cls_workstation_sandwich_legacy_v1", "runtime=workstation")
    assert str(cfg.runtime.mixed_precision) == "bf16"
    assert int(cfg.runtime.num_workers) == 0
    assert bool(cfg.runtime.loader_pin_memory) is False
    assert bool(cfg.runtime.loader_persistent_workers) is False
    assert cfg.runtime.loader_prefetch_factor is None
    assert str(cfg.runtime.loader_task_batch_cache_mode) == "off"
    assert bool(cfg.runtime.non_blocking_device_transfer) is False
    assert bool(cfg.runtime.compile_model) is False
    assert bool(cfg.runtime.compile_dynamic) is False
    assert str(cfg.runtime.compile_backend) == "inductor"
    assert str(cfg.runtime.compile_mode) == "max-autotune-no-cudagraphs"
    assert str(cfg.runtime.compile_shape_dispatch_mode) == "off"
    assert int(cfg.runtime.compile_shape_dispatch_max_families) == 16
    assert int(cfg.runtime.signature_family_run_length) == 1


def test_runtime_tf_rd_022_policy_resolution() -> None:
    cfg = _compose("experiment=cls_workstation_sandwich_legacy_v1", "runtime=tf_rd_022_policy")
    assert str(cfg.runtime.mixed_precision) == "bf16"
    assert bool(cfg.runtime.loader_pin_memory) is False
    assert bool(cfg.runtime.loader_persistent_workers) is False
    assert cfg.runtime.loader_prefetch_factor is None
    assert str(cfg.runtime.loader_task_batch_cache_mode) == "off"
    assert bool(cfg.runtime.non_blocking_device_transfer) is False
    assert float(cfg.runtime.grad_clip) == 0.0
    assert int(cfg.runtime.grad_accum_steps) == 4
    assert bool(cfg.runtime.compile_model) is False
    assert bool(cfg.runtime.compile_dynamic) is False
    assert str(cfg.runtime.compile_backend) == "inductor"
    assert str(cfg.runtime.compile_mode) == "max-autotune-no-cudagraphs"
    assert str(cfg.runtime.compile_shape_dispatch_mode) == "off"
    assert int(cfg.runtime.compile_shape_dispatch_max_families) == 16
    assert int(cfg.runtime.signature_family_run_length) == 1
    assert bool(cfg.runtime.trace_activations) is False
    assert bool(cfg.runtime.activation_checkpointing) is True
    assert int(cfg.runtime.max_steps) == 2500


def test_cls_benchmark_sandwich_tf_rd_022_policy_train_speed_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_train_speed_v1")
    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_sandwich"
    assert str(cfg.data.source) == "manifest"
    assert str(cfg.data.surface_label) == "tf_rd_010_dagzoo_medium_control"
    assert str(cfg.data.corpus_ref) == "tf_rd_010_dagzoo_medium_control_curated_v5"
    assert "legacy_prior" not in cfg
    assert str(cfg.runtime.device) == "cuda"
    assert str(cfg.runtime.mixed_precision) == "bf16"
    assert bool(cfg.model.sandwich_packed_attention) is True
    assert bool(cfg.runtime.compile_model) is True
    assert bool(cfg.runtime.compile_dynamic) is True
    assert str(cfg.runtime.compile_backend) == "eager"
    assert str(cfg.runtime.compile_shape_dispatch_mode) == "signature_family"
    assert int(cfg.runtime.num_workers) == 2
    assert bool(cfg.runtime.loader_pin_memory) is True
    assert bool(cfg.runtime.loader_persistent_workers) is True
    assert int(cfg.runtime.loader_prefetch_factor) == 2
    assert bool(cfg.runtime.non_blocking_device_transfer) is True
    assert str(cfg.runtime.loader_task_batch_cache_mode) == "bounded_streaming"
    assert str(cfg.runtime.output_dir) == (
        "outputs/cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_train_speed_v1"
    )
    assert str(cfg.logging.run_name) == (
        "cls-benchmark-sandwich-classification-evolution-tf-rd-022-policy-train-speed-v1"
    )


def test_cls_benchmark_sandwich_tf_rd_022_policy_train_speed_workers_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_train_speed_workers_v1")
    assert str(cfg.runtime.device) == "cuda"
    assert bool(cfg.runtime.compile_model) is True
    assert str(cfg.runtime.compile_backend) == "eager"
    assert int(cfg.runtime.num_workers) == 2
    assert bool(cfg.runtime.loader_pin_memory) is True
    assert bool(cfg.runtime.loader_persistent_workers) is False
    assert str(cfg.runtime.loader_prefetch_factor) == "auto"
    assert bool(cfg.runtime.non_blocking_device_transfer) is True
    assert str(cfg.runtime.output_dir) == (
        "outputs/cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_train_speed_workers_v1"
    )
    assert str(cfg.logging.run_name) == (
        "cls-benchmark-sandwich-classification-evolution-tf-rd-022-policy-train-speed-workers-v1"
    )


def test_cls_benchmark_sandwich_tf_rd_022_policy_train_speed_loader_overlap_resolution() -> None:
    cfg = _compose(
        "experiment=cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_train_speed_loader_overlap_v1"
    )
    assert str(cfg.runtime.device) == "cuda"
    assert bool(cfg.runtime.compile_model) is True
    assert int(cfg.runtime.num_workers) == 2
    assert bool(cfg.runtime.loader_pin_memory) is True
    assert bool(cfg.runtime.loader_persistent_workers) is True
    assert int(cfg.runtime.loader_prefetch_factor) == 2
    assert bool(cfg.runtime.non_blocking_device_transfer) is True
    assert str(cfg.runtime.output_dir) == (
        "outputs/cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_train_speed_loader_overlap_v1"
    )
    assert str(cfg.logging.run_name) == (
        "cls-benchmark-sandwich-classification-evolution-tf-rd-022-policy-train-speed-loader-overlap-v1"
    )


def test_cls_benchmark_sandwich_tf_rd_022_policy_train_speed_transfer_resolution() -> None:
    cfg = _compose(
        "experiment=cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_train_speed_transfer_v1"
    )
    assert str(cfg.runtime.device) == "cuda"
    assert bool(cfg.runtime.compile_model) is True
    assert int(cfg.runtime.num_workers) == 2
    assert bool(cfg.runtime.loader_pin_memory) is True
    assert bool(cfg.runtime.loader_persistent_workers) is False
    assert str(cfg.runtime.loader_prefetch_factor) == "auto"
    assert bool(cfg.runtime.non_blocking_device_transfer) is True
    assert str(cfg.runtime.output_dir) == (
        "outputs/cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_train_speed_transfer_v1"
    )
    assert str(cfg.logging.run_name) == (
        "cls-benchmark-sandwich-classification-evolution-tf-rd-022-policy-train-speed-transfer-v1"
    )


def test_cls_benchmark_sandwich_speedrun_cached_packed_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_sandwich_speedrun_cached_packed_v1")

    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_sandwich"
    assert bool(cfg.model.sandwich_packed_attention) is True
    assert str(cfg.runtime.loader_task_batch_cache_mode) == "bounded_streaming"
    assert bool(cfg.runtime.loader_task_batch_cache) is False
    assert str(cfg.runtime.num_workers) == "auto"
    assert bool(cfg.runtime.loader_pin_memory) is True
    assert bool(cfg.runtime.loader_persistent_workers) is False
    assert str(cfg.runtime.loader_prefetch_factor) == "auto"
    assert bool(cfg.runtime.non_blocking_device_transfer) is True
    assert str(cfg.runtime.compile_shape_dispatch_mode) == "signature_family"
    assert int(cfg.runtime.compile_shape_dispatch_max_families) == 16
    assert int(cfg.runtime.signature_family_run_length) == 1
    assert int(cfg.runtime.signature_family_optimizer_step_block_length) == 1
    assert str(cfg.runtime.output_dir) == "outputs/cls_benchmark_sandwich_speedrun_cached_packed_v1"
    assert str(cfg.logging.run_name) == "cls-benchmark-sandwich-speedrun-cached-packed-v1"


def test_cls_benchmark_sandwich_speedrun_default_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_sandwich_speedrun_default_v1")

    assert bool(cfg.model.sandwich_packed_attention) is True
    assert str(cfg.optimizer.name) == "muon"
    assert str(cfg.runtime.loader_task_batch_cache_mode) == "bounded_streaming"
    assert str(cfg.runtime.compile_shape_dispatch_mode) == "signature_family"
    assert int(cfg.runtime.signature_family_run_length) == 1
    assert int(cfg.runtime.signature_family_optimizer_step_block_length) == 1
    assert str(cfg.runtime.output_dir) == "outputs/cls_benchmark_sandwich_speedrun_default_v1"
    assert str(cfg.logging.run_name) == "cls-benchmark-sandwich-speedrun-default-v1"


def test_cls_benchmark_sandwich_speedrun_cached_packed_family_block_2step_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_sandwich_speedrun_cached_packed_family_block_2step_v1")

    assert bool(cfg.model.sandwich_packed_attention) is True
    assert str(cfg.optimizer.name) == "schedulefree_adamw"
    assert int(cfg.runtime.signature_family_optimizer_step_block_length) == 2
    assert str(cfg.runtime.output_dir) == (
        "outputs/cls_benchmark_sandwich_speedrun_cached_packed_family_block_2step_v1"
    )
    assert str(cfg.logging.run_name) == (
        "cls-benchmark-sandwich-speedrun-cached-packed-family-block-2step-v1"
    )


def test_cls_benchmark_sandwich_speedrun_cached_packed_family_block_4step_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_sandwich_speedrun_cached_packed_family_block_4step_v1")

    assert bool(cfg.model.sandwich_packed_attention) is True
    assert str(cfg.optimizer.name) == "schedulefree_adamw"
    assert int(cfg.runtime.signature_family_optimizer_step_block_length) == 4
    assert str(cfg.runtime.output_dir) == (
        "outputs/cls_benchmark_sandwich_speedrun_cached_packed_family_block_4step_v1"
    )
    assert str(cfg.logging.run_name) == (
        "cls-benchmark-sandwich-speedrun-cached-packed-family-block-4step-v1"
    )


def test_cls_benchmark_sandwich_tf_rd_022_policy_compile_resolution() -> None:
    base_cfg = _compose("experiment=cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1")
    cfg = _compose("experiment=cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_v1")

    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_sandwich"
    assert str(cfg.data.source) == str(base_cfg.data.source)
    assert str(cfg.data.surface_label) == str(base_cfg.data.surface_label)
    assert str(cfg.data.corpus_ref) == str(base_cfg.data.corpus_ref)
    assert str(cfg.runtime.device) == str(base_cfg.runtime.device)
    assert str(cfg.runtime.mixed_precision) == str(base_cfg.runtime.mixed_precision)
    assert float(cfg.runtime.grad_clip) == float(base_cfg.runtime.grad_clip)
    assert int(cfg.runtime.grad_accum_steps) == int(base_cfg.runtime.grad_accum_steps)
    assert bool(cfg.runtime.activation_checkpointing) is bool(base_cfg.runtime.activation_checkpointing)
    assert bool(cfg.runtime.trace_activations) is bool(base_cfg.runtime.trace_activations)
    assert bool(base_cfg.runtime.compile_model) is True
    assert bool(cfg.runtime.compile_model) is True
    assert bool(base_cfg.runtime.compile_dynamic) is True
    assert bool(cfg.runtime.compile_dynamic) is False
    assert str(cfg.runtime.compile_backend) == "inductor"
    assert str(cfg.runtime.compile_mode) == "max-autotune-no-cudagraphs"
    assert str(base_cfg.runtime.compile_shape_dispatch_mode) == "signature_family"
    assert str(cfg.runtime.compile_shape_dispatch_mode) == "off"
    assert int(cfg.runtime.max_steps) == int(base_cfg.runtime.max_steps)
    assert int(cfg.training.task_batch_size) == int(base_cfg.training.task_batch_size)
    assert str(cfg.runtime.output_dir) == (
        "outputs/cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_v1"
    )
    assert str(cfg.logging.run_name) == (
        "cls-benchmark-sandwich-classification-evolution-tf-rd-022-policy-compile-v1"
    )
    assert str(cfg.logging.history_jsonl_path) == (
        "outputs/cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_v1/"
        "train_history.jsonl"
    )


def test_cls_benchmark_sandwich_tf_rd_022_policy_compile_eager_dynamic_resolution() -> None:
    base_cfg = _compose("experiment=cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1")
    cfg = _compose(
        "experiment=cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1"
    )

    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_sandwich"
    assert str(cfg.data.source) == str(base_cfg.data.source)
    assert str(cfg.data.surface_label) == str(base_cfg.data.surface_label)
    assert str(cfg.data.corpus_ref) == str(base_cfg.data.corpus_ref)
    assert str(cfg.runtime.device) == str(base_cfg.runtime.device)
    assert str(cfg.runtime.mixed_precision) == str(base_cfg.runtime.mixed_precision)
    assert float(cfg.runtime.grad_clip) == float(base_cfg.runtime.grad_clip)
    assert int(cfg.runtime.grad_accum_steps) == int(base_cfg.runtime.grad_accum_steps)
    assert bool(cfg.runtime.activation_checkpointing) is bool(base_cfg.runtime.activation_checkpointing)
    assert bool(cfg.runtime.trace_activations) is bool(base_cfg.runtime.trace_activations)
    assert bool(base_cfg.runtime.compile_model) is True
    assert bool(cfg.runtime.compile_model) is True
    assert bool(base_cfg.runtime.compile_dynamic) is True
    assert bool(cfg.runtime.compile_dynamic) is True
    assert str(cfg.runtime.compile_backend) == "eager"
    assert str(cfg.runtime.compile_mode) == str(base_cfg.runtime.compile_mode)
    assert str(cfg.runtime.compile_shape_dispatch_mode) == "signature_family"
    assert int(cfg.runtime.max_steps) == int(base_cfg.runtime.max_steps)
    assert int(cfg.training.task_batch_size) == int(base_cfg.training.task_batch_size)
    assert str(cfg.runtime.output_dir) == (
        "outputs/cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1"
    )
    assert str(cfg.logging.run_name) == (
        "cls-benchmark-sandwich-classification-evolution-tf-rd-022-policy-compile-eager-dynamic-v1"
    )
    assert str(cfg.logging.history_jsonl_path) == (
        "outputs/cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1/"
        "train_history.jsonl"
    )


def test_cls_smoke_adamw_override_resolution() -> None:
    cfg = _compose("experiment=cls_smoke", "optimizer=adamw")
    assert str(cfg.optimizer.name) == "adamw"
    assert bool(cfg.optimizer.require_requested) is False


def test_cls_benchmark_linear_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_linear")
    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_sandwich"
    assert cfg.model.stage is None
    assert int(cfg.model.feature_group_size) == 1
    assert str(cfg.optimizer.name) == "adamw"
    assert bool(cfg.optimizer.require_requested) is False
    assert int(cfg.runtime.eval_every) == 25
    assert int(cfg.runtime.checkpoint_every) == 25
    assert int(cfg.runtime.max_steps) == 400
    assert int(cfg.training.task_batch_size) == 1
    assert float(cfg.runtime.target_train_seconds) == 330.0
    assert str(cfg.logging.history_jsonl_path) == "outputs/cls_benchmark_linear/train_history.jsonl"
    stage = cfg.schedule.stages[0]
    assert str(stage["lr_schedule"]) == "linear"
    assert float(stage["warmup_ratio"]) == 0.05


def test_cls_benchmark_linear_simple_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_linear_simple")
    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_simple"
    assert cfg.model.stage is None
    assert int(cfg.model.d_icl) == 96
    assert str(cfg.model.input_normalization) == "train_zscore_clip"
    assert int(cfg.model.many_class_base) == 2
    assert int(cfg.model.tficl_n_heads) == 4
    assert int(cfg.model.tficl_n_layers) == 3
    assert int(cfg.model.head_hidden_dim) == 192
    assert str(cfg.logging.history_jsonl_path) == "outputs/cls_benchmark_linear_simple/train_history.jsonl"


def test_cls_benchmark_linear_simple_prior_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_linear_simple_prior")
    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_simple"
    assert cfg.model.stage is None
    assert int(cfg.model.d_icl) == 96
    assert str(cfg.model.input_normalization) == "train_zscore_clip"
    assert int(cfg.model.many_class_base) == 2
    assert int(cfg.runtime.max_steps) == 2500
    assert int(cfg.runtime.eval_every) == 25
    assert int(cfg.runtime.checkpoint_every) == 25
    assert str(cfg.optimizer.name) == "schedulefree_adamw"
    assert bool(cfg.optimizer.require_requested) is True
    assert int(cfg.training.task_batch_size) == 1
    assert float(cfg.optimizer.weight_decay) == 0.0
    assert list(cfg.optimizer.betas) == [0.9, 0.999]
    assert float(cfg.optimizer.min_lr) == 4.0e-3
    assert bool(cfg.optimizer.muon_per_parameter_lr) is False
    assert str(cfg.logging.history_jsonl_path) == "outputs/cls_benchmark_linear_simple_prior/train_history.jsonl"


def test_cls_benchmark_sandwich_prior_is_retired() -> None:
    with pytest.raises(MissingConfigException):
        _compose("experiment=cls_benchmark_sandwich_prior")


def test_cls_benchmark_sandwich_hybrid_prior_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_sandwich_hybrid_prior")
    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_sandwich"
    assert cfg.model.stage is None
    assert int(cfg.model.d_icl) == 60
    assert str(cfg.model.input_normalization) == "train_zscore_clip"
    assert int(cfg.model.many_class_base) == 2
    assert int(cfg.model.head_hidden_dim) == 96
    assert int(cfg.model.sandwich_latents) == 24
    assert int(cfg.model.sandwich_layers) == 2
    assert int(cfg.model.sandwich_heads) == 4
    assert int(cfg.model.sandwich_ff_expansion) == 2
    assert int(cfg.model.sandwich_summary_tokens_per_axis) == 4
    assert int(cfg.model.sandwich_self_attention_per_cross) == 4
    assert int(cfg.model.sandwich_pre_row_attention_layers) == 1
    assert int(cfg.model.sandwich_pre_column_attention_layers) == 1
    assert int(cfg.model.sandwich_pre_column_inducing_tokens) == 16
    assert int(cfg.runtime.max_steps) == 2500
    assert int(cfg.runtime.eval_every) == 25
    assert int(cfg.runtime.checkpoint_every) == 25
    assert bool(cfg.runtime.trace_activations) is False
    assert str(cfg.optimizer.name) == "schedulefree_adamw"
    assert bool(cfg.optimizer.require_requested) is True
    assert str(cfg.runtime.output_dir) == "outputs/cls_benchmark_sandwich_hybrid_prior"
    assert str(cfg.logging.run_name) == "cls-benchmark-sandwich-hybrid-prior"
    assert (
        str(cfg.logging.history_jsonl_path)
        == "outputs/cls_benchmark_sandwich_hybrid_prior/train_history.jsonl"
    )


def test_cls_benchmark_sandwich_classification_evolution_v1_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_sandwich_classification_evolution_v1")
    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_sandwich"
    assert cfg.model.stage is None
    assert int(cfg.model.d_icl) == 60
    assert str(cfg.model.input_normalization) == "train_zscore_clip"
    assert str(cfg.model.feature_type_conditioning) == "film"
    assert int(cfg.model.many_class_base) == 10
    assert int(cfg.model.head_hidden_dim) == 96
    assert int(cfg.model.sandwich_latents) == 24
    assert int(cfg.model.sandwich_layers) == 2
    assert int(cfg.model.sandwich_heads) == 4
    assert int(cfg.model.sandwich_ff_expansion) == 2
    assert int(cfg.model.sandwich_summary_tokens_per_axis) == 3
    assert int(cfg.model.sandwich_self_attention_per_cross) == 4
    assert int(cfg.model.sandwich_pre_row_attention_layers) == 1
    assert int(cfg.model.sandwich_pre_column_attention_layers) == 1
    assert int(cfg.model.sandwich_pre_column_inducing_tokens) == 16
    assert str(cfg.training.loss_surface) == "classification"
    assert int(cfg.runtime.max_steps) == 2500
    assert int(cfg.runtime.eval_every) == 25
    assert int(cfg.runtime.checkpoint_every) == 25
    assert bool(cfg.runtime.trace_activations) is False
    assert str(cfg.runtime.output_dir) == "outputs/cls_benchmark_sandwich_classification_evolution_v1"
    assert str(cfg.logging.run_name) == "cls-benchmark-sandwich-classification-evolution-v1"
    assert (
        str(cfg.logging.history_jsonl_path)
        == "outputs/cls_benchmark_sandwich_classification_evolution_v1/train_history.jsonl"
    )


def test_cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1")
    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_sandwich"
    assert str(cfg.data.source) == "manifest"
    assert str(cfg.data.surface_label) == "tf_rd_010_dagzoo_medium_control"
    assert str(cfg.data.corpus_ref) == "tf_rd_010_dagzoo_medium_control_curated_v5"
    assert "legacy_prior" not in cfg
    assert str(cfg.runtime.device) == "cuda"
    assert int(cfg.model.d_icl) == 60
    assert int(cfg.model.head_hidden_dim) == 96
    assert int(cfg.model.sandwich_summary_tokens_per_axis) == 3
    assert int(cfg.training.task_batch_size) == 16
    assert int(cfg.training.prior_dump_batch_size) == 64
    assert str(cfg.training.loss_surface) == "classification"
    assert bool(cfg.training.apply_schedule) is True
    assert bool(cfg.model.sandwich_packed_attention) is True
    assert str(cfg.optimizer.name) == "muon"
    assert bool(cfg.optimizer.require_requested) is True
    assert float(cfg.optimizer.weight_decay) == 0.01
    assert list(cfg.optimizer.betas) == [0.9, 0.95]
    assert float(cfg.optimizer.momentum) == 0.95
    assert str(cfg.runtime.mixed_precision) == "bf16"
    assert float(cfg.runtime.grad_clip) == 0.0
    assert int(cfg.runtime.grad_accum_steps) == 4
    assert bool(cfg.runtime.trace_activations) is False
    assert bool(cfg.runtime.activation_checkpointing) is True
    assert str(cfg.runtime.loader_task_batch_cache_mode) == "bounded_streaming"
    assert str(cfg.runtime.num_workers) == "auto"
    assert bool(cfg.runtime.loader_pin_memory) is True
    assert str(cfg.runtime.loader_prefetch_factor) == "auto"
    assert bool(cfg.runtime.non_blocking_device_transfer) is True
    assert bool(cfg.runtime.compile_model) is True
    assert bool(cfg.runtime.compile_dynamic) is True
    assert str(cfg.runtime.compile_backend) == "eager"
    assert str(cfg.runtime.compile_shape_dispatch_mode) == "signature_family"
    assert int(cfg.runtime.compile_shape_dispatch_max_families) == 16
    assert int(cfg.runtime.signature_family_run_length) == 4
    assert int(cfg.runtime.eval_every) == 25
    assert int(cfg.runtime.checkpoint_every) == 25
    assert int(cfg.runtime.max_steps) == 2500
    stage = cfg.schedule.stages[0]
    assert str(stage["lr_schedule"]) == "linear"
    assert float(stage["warmup_ratio"]) == 0.10
    assert float(stage["lr_max"]) == 1.0e-3
    assert float(cfg.optimizer.min_lr) == 1.0e-6
    assert str(cfg.runtime.output_dir) == (
        "outputs/cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1"
    )
    assert (
        str(cfg.logging.history_jsonl_path)
        == "outputs/cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1/train_history.jsonl"
    )


def test_cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1")

    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_sandwich"
    assert str(cfg.data.source) == "manifest"
    assert str(cfg.data.surface_label) == "tf_rd_010_dagzoo_medium_control"
    assert str(cfg.data.corpus_ref) == "tf_rd_010_dagzoo_medium_control_curated_v6"
    assert int(cfg.model.d_icl) == 60
    assert int(cfg.model.sandwich_layers) == 2
    assert int(cfg.model.sandwich_heads) == 1
    assert bool(cfg.model.sandwich_packed_attention) is True
    assert str(cfg.optimizer.name) == "muon"
    assert bool(cfg.optimizer.require_requested) is True
    assert float(cfg.optimizer.weight_decay) == 0.01
    assert list(cfg.optimizer.betas) == [0.9, 0.95]
    assert float(cfg.optimizer.momentum) == 0.95
    assert float(cfg.optimizer.min_lr) == 1.0e-6
    assert bool(cfg.optimizer.muon_per_parameter_lr) is True
    assert bool(cfg.optimizer.muon_partition_non2d) is True
    assert str(cfg.runtime.mixed_precision) == "bf16"
    assert bool(cfg.runtime.compile_model) is True
    assert str(cfg.runtime.compile_backend) == "eager"
    assert bool(cfg.runtime.compile_dynamic) is True
    assert str(cfg.runtime.loader_task_batch_cache_mode) == "bounded_streaming"
    assert int(cfg.runtime.grad_accum_steps) == 4
    assert int(cfg.runtime.max_steps) == 2500
    assert str(cfg.runtime.output_dir) == (
        "outputs/cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1"
    )
    assert str(cfg.logging.run_name) == (
        "cls-benchmark-sandwich-classification-evolution-tf-rd-009-muon-medium-v1"
    )
    assert (
        str(cfg.logging.history_jsonl_path)
        == "outputs/cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1/train_history.jsonl"
    )


def test_cls_benchmark_staged_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_staged")
    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_staged"
    assert str(cfg.model.stage) == "nano_exact"
    assert int(cfg.model.d_icl) == 96
    assert str(cfg.model.input_normalization) == "train_zscore_clip"
    assert int(cfg.model.many_class_base) == 2
    assert int(cfg.model.tficl_n_heads) == 4
    assert int(cfg.model.tficl_n_layers) == 3
    assert int(cfg.model.head_hidden_dim) == 192
    assert bool(cfg.runtime.trace_activations) is True
    assert bool(cfg.runtime.activation_checkpointing) is False
    assert str(cfg.logging.history_jsonl_path) == "outputs/cls_benchmark_staged/train_history.jsonl"


def test_cls_benchmark_staged_corpus_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_staged_corpus")
    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_staged"
    assert str(cfg.model.stage) == "nano_exact"
    assert str(cfg.training.surface_label) == "linear_warmup_decay"
    assert bool(cfg.training.apply_schedule) is False
    assert int(cfg.training.task_batch_size) == 1
    assert bool(cfg.runtime.activation_checkpointing) is False
    assert str(cfg.runtime.output_dir) == "outputs/cls_benchmark_staged_corpus"
    assert str(cfg.logging.run_name) == "cls-benchmark-staged-corpus"
    assert str(cfg.logging.history_jsonl_path) == "outputs/cls_benchmark_staged_corpus/train_history.jsonl"


def test_cls_benchmark_staged_explore_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_staged_explore")
    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_staged"
    assert str(cfg.model.stage) == "nano_exact"
    assert int(cfg.model.d_icl) == 96
    assert str(cfg.model.input_normalization) == "train_zscore_clip"
    assert int(cfg.model.many_class_base) == 2
    assert int(cfg.model.tficl_n_heads) == 4
    assert int(cfg.model.tficl_n_layers) == 3
    assert int(cfg.model.head_hidden_dim) == 192
    assert int(cfg.runtime.num_workers) == 2
    assert bool(cfg.runtime.trace_activations) is False
    assert int(cfg.runtime.eval_every) == 100
    assert cfg.runtime.checkpoint_every is None
    assert int(cfg.runtime.val_batches) == 8
    assert str(cfg.runtime.output_dir) == "outputs/cls_benchmark_staged_explore"
    assert str(cfg.logging.run_name) == "cls-benchmark-staged-explore"
    assert str(cfg.logging.history_jsonl_path) == "outputs/cls_benchmark_staged_explore/train_history.jsonl"


def test_cls_benchmark_staged_prior_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_staged_prior")
    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_staged"
    assert str(cfg.model.stage) == "nano_exact"
    assert str(cfg.model.stage_label) == "dpnb_row_cls_cls2_linear_warmup_decay"
    assert cfg.model.module_overrides == {
        "table_block_style": "prenorm",
        "allow_test_self_attention": False,
        "row_pool": "row_cls",
    }
    assert int(cfg.model.d_icl) == 96
    assert str(cfg.model.input_normalization) == "train_zscore_clip"
    assert int(cfg.model.many_class_base) == 2
    assert int(cfg.model.tfrow_n_heads) == 8
    assert int(cfg.model.tfrow_n_layers) == 3
    assert int(cfg.model.tfrow_cls_tokens) == 2
    assert str(cfg.model.tfrow_norm) == "layernorm"
    assert int(cfg.runtime.max_steps) == 2500
    assert int(cfg.runtime.eval_every) == 25
    assert int(cfg.runtime.checkpoint_every) == 25
    assert bool(cfg.runtime.trace_activations) is True
    assert str(cfg.optimizer.name) == "schedulefree_adamw"
    assert bool(cfg.optimizer.require_requested) is True
    assert float(cfg.optimizer.weight_decay) == 0.0
    assert list(cfg.optimizer.betas) == [0.9, 0.999]
    assert float(cfg.optimizer.min_lr) == 4.0e-4
    assert bool(cfg.optimizer.muon_per_parameter_lr) is False
    assert str(cfg.model.norm_type) == "layernorm"
    assert str(cfg.training.surface_label) == "prior_linear_warmup_decay"
    assert bool(cfg.training.apply_schedule) is True
    assert int(cfg.training.task_batch_size) == 1
    assert str(cfg.legacy_prior.non_finite_policy) == "skip"
    assert int(cfg.legacy_prior.batch_size) == 32
    assert str(cfg.legacy_prior.lr_scale_rule) == "none"
    assert int(cfg.legacy_prior.batch_reference_size) == 32
    stage = cfg.schedule.stages[0]
    assert str(stage["name"]) == "prior_dump"
    assert int(stage["steps"]) == 2500
    assert float(stage["lr_max"]) == 4.0e-3
    assert str(stage["lr_schedule"]) == "linear"
    assert float(stage["warmup_ratio"]) == 0.05
    assert str(cfg.logging.run_name) == "dpnb_row_cls_cls2_linear_warmup_decay"
    assert str(cfg.logging.history_jsonl_path) == "outputs/cls_benchmark_staged_prior/train_history.jsonl"


def test_cls_benchmark_staged_prior_explore_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_staged_prior_explore")
    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_staged"
    assert str(cfg.model.stage) == "nano_exact"
    assert str(cfg.model.stage_label) == "dpnb_row_cls_cls2_linear_warmup_decay"
    assert cfg.model.module_overrides == {
        "table_block_style": "prenorm",
        "allow_test_self_attention": False,
        "row_pool": "row_cls",
    }
    assert int(cfg.model.d_icl) == 96
    assert str(cfg.model.input_normalization) == "train_zscore_clip"
    assert int(cfg.model.many_class_base) == 2
    assert int(cfg.model.tfrow_n_heads) == 8
    assert int(cfg.model.tfrow_n_layers) == 3
    assert int(cfg.model.tfrow_cls_tokens) == 2
    assert str(cfg.model.tfrow_norm) == "layernorm"
    assert int(cfg.runtime.max_steps) == 2500
    assert int(cfg.runtime.eval_every) == 250
    assert int(cfg.runtime.checkpoint_every) == 250
    assert bool(cfg.runtime.trace_activations) is False
    assert str(cfg.optimizer.name) == "schedulefree_adamw"
    assert bool(cfg.optimizer.require_requested) is True
    assert float(cfg.optimizer.weight_decay) == 0.0
    assert list(cfg.optimizer.betas) == [0.9, 0.999]
    assert float(cfg.optimizer.min_lr) == 4.0e-4
    assert bool(cfg.optimizer.muon_per_parameter_lr) is False
    assert str(cfg.model.norm_type) == "layernorm"
    assert str(cfg.training.surface_label) == "prior_linear_warmup_decay"
    assert bool(cfg.training.apply_schedule) is True
    assert int(cfg.training.task_batch_size) == 1
    assert str(cfg.legacy_prior.non_finite_policy) == "skip"
    assert int(cfg.legacy_prior.batch_size) == 32
    assert str(cfg.legacy_prior.lr_scale_rule) == "none"
    assert int(cfg.legacy_prior.batch_reference_size) == 32
    stage = cfg.schedule.stages[0]
    assert str(stage["name"]) == "prior_dump"
    assert int(stage["steps"]) == 2500
    assert float(stage["lr_max"]) == 4.0e-3
    assert str(stage["lr_schedule"]) == "linear"
    assert float(stage["warmup_ratio"]) == 0.05
    assert str(cfg.logging.run_name) == "dpnb_row_cls_cls2_linear_warmup_decay_explore"
    assert str(cfg.runtime.output_dir) == "outputs/cls_benchmark_staged_prior_explore"
    assert (
        str(cfg.logging.history_jsonl_path)
        == "outputs/cls_benchmark_staged_prior_explore/train_history.jsonl"
    )


def test_cls_benchmark_staged_prior_cuda_scale_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_staged_prior_cuda_scale")
    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_staged"
    assert str(cfg.model.stage) == "nano_exact"
    assert str(cfg.model.stage_label) == "dpnb_cuda_large_anchor"
    assert cfg.model.module_overrides == {
        "table_block_style": "prenorm",
        "allow_test_self_attention": False,
        "row_pool": "row_cls",
    }
    assert int(cfg.model.d_col) == 128
    assert int(cfg.model.d_icl) == 512
    assert str(cfg.model.input_normalization) == "train_zscore_clip"
    assert int(cfg.model.tfrow_n_heads) == 8
    assert int(cfg.model.tfrow_n_layers) == 3
    assert int(cfg.model.tfrow_cls_tokens) == 2
    assert str(cfg.model.tfrow_norm) == "layernorm"
    assert int(cfg.model.tficl_n_heads) == 8
    assert int(cfg.model.tficl_n_layers) == 12
    assert int(cfg.model.head_hidden_dim) == 1024
    assert int(cfg.runtime.max_steps) == 2500
    assert int(cfg.runtime.eval_every) == 25
    assert int(cfg.runtime.checkpoint_every) == 25
    assert bool(cfg.runtime.trace_activations) is True
    assert str(cfg.optimizer.name) == "schedulefree_adamw"
    assert bool(cfg.optimizer.require_requested) is True
    assert float(cfg.optimizer.weight_decay) == 0.0
    assert list(cfg.optimizer.betas) == [0.9, 0.999]
    assert float(cfg.optimizer.min_lr) == 4.0e-4
    assert bool(cfg.optimizer.muon_per_parameter_lr) is False
    assert str(cfg.training.surface_label) == "prior_linear_warmup_decay"
    assert bool(cfg.training.apply_schedule) is True
    assert str(cfg.legacy_prior.non_finite_policy) == "skip"
    assert int(cfg.legacy_prior.batch_size) == 64
    assert str(cfg.legacy_prior.lr_scale_rule) == "sqrt"
    assert int(cfg.legacy_prior.batch_reference_size) == 32
    stage = cfg.schedule.stages[0]
    assert str(stage["name"]) == "prior_dump"
    assert int(stage["steps"]) == 2500
    assert float(stage["lr_max"]) == 4.0e-3
    assert str(stage["lr_schedule"]) == "linear"
    assert float(stage["warmup_ratio"]) == 0.05
    assert str(cfg.logging.run_name) == "dpnb_cuda_large_anchor"
    assert str(cfg.runtime.output_dir) == "outputs/cls_benchmark_staged_prior_cuda_scale"
    assert (
        str(cfg.logging.history_jsonl_path)
        == "outputs/cls_benchmark_staged_prior_cuda_scale/train_history.jsonl"
    )


def test_cls_benchmark_staged_prior_cuda_scale_explore_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_staged_prior_cuda_scale_explore")
    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_staged"
    assert str(cfg.model.stage) == "nano_exact"
    assert str(cfg.model.stage_label) == "dpnb_cuda_large_anchor"
    assert cfg.model.module_overrides == {
        "table_block_style": "prenorm",
        "allow_test_self_attention": False,
        "row_pool": "row_cls",
    }
    assert int(cfg.model.d_col) == 128
    assert int(cfg.model.d_icl) == 512
    assert str(cfg.model.input_normalization) == "train_zscore_clip"
    assert int(cfg.model.tfrow_n_heads) == 8
    assert int(cfg.model.tfrow_n_layers) == 3
    assert int(cfg.model.tfrow_cls_tokens) == 2
    assert str(cfg.model.tfrow_norm) == "layernorm"
    assert int(cfg.model.tficl_n_heads) == 8
    assert int(cfg.model.tficl_n_layers) == 12
    assert int(cfg.model.head_hidden_dim) == 1024
    assert int(cfg.runtime.max_steps) == 2500
    assert int(cfg.runtime.eval_every) == 250
    assert int(cfg.runtime.checkpoint_every) == 250
    assert bool(cfg.runtime.trace_activations) is False
    assert str(cfg.optimizer.name) == "schedulefree_adamw"
    assert bool(cfg.optimizer.require_requested) is True
    assert float(cfg.optimizer.weight_decay) == 0.0
    assert list(cfg.optimizer.betas) == [0.9, 0.999]
    assert float(cfg.optimizer.min_lr) == 4.0e-4
    assert bool(cfg.optimizer.muon_per_parameter_lr) is False
    assert str(cfg.training.surface_label) == "prior_linear_warmup_decay"
    assert bool(cfg.training.apply_schedule) is True
    assert str(cfg.legacy_prior.non_finite_policy) == "skip"
    assert int(cfg.legacy_prior.batch_size) == 64
    assert str(cfg.legacy_prior.lr_scale_rule) == "sqrt"
    assert int(cfg.legacy_prior.batch_reference_size) == 32
    stage = cfg.schedule.stages[0]
    assert str(stage["name"]) == "prior_dump"
    assert int(stage["steps"]) == 2500
    assert float(stage["lr_max"]) == 4.0e-3
    assert str(stage["lr_schedule"]) == "linear"
    assert float(stage["warmup_ratio"]) == 0.05
    assert str(cfg.logging.run_name) == "dpnb_cuda_large_anchor_explore"
    assert str(cfg.runtime.output_dir) == "outputs/cls_benchmark_staged_prior_cuda_scale_explore"
    assert (
        str(cfg.logging.history_jsonl_path)
        == "outputs/cls_benchmark_staged_prior_cuda_scale_explore/train_history.jsonl"
    )


def test_cls_benchmark_sandwich_tf_rd_022_policy_rational_resolution() -> None:
    cfg = _compose(
        "experiment=cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_rational_v1"
    )

    assert str(cfg.task) == "classification"
    assert str(cfg.model.arch) == "tabfoundry_sandwich"
    assert str(cfg.model.sandwich_activation) == "rational"
    assert str(cfg.model.sandwich_block_norm) == "none"
    assert str(cfg.model.norm_type) == "layernorm"
    assert bool(cfg.model.sandwich_packed_attention) is True
    assert str(cfg.optimizer.name) == "muon"
    assert str(cfg.logging.run_name) == (
        "cls-benchmark-sandwich-classification-evolution-tf-rd-022-policy-rational-v1"
    )
    assert str(cfg.runtime.output_dir) == (
        "outputs/cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_rational_v1"
    )
