"""Schema models for the benchmark run registry."""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
)


REGISTRY_SCHEMA = "tab-foundry-benchmark-runs-v2"
REGISTRY_VERSION = 2
DEFAULT_BUDGET_CLASS = "short-run"
ALLOWED_DECISIONS = ("keep", "reject", "defer")

_TOP_LEVEL_KEYS = {"schema", "version", "runs"}
_TAB_FOUNDRY_METRIC_KEYS = {
    "best_step",
    "best_training_time",
    "best_bpc",
    "best_bpf",
    "best_roc_auc",
    "best_log_loss",
    "best_brier_score",
    "best_crps",
    "best_avg_pinball_loss",
    "best_picp_90",
    "final_step",
    "final_training_time",
    "final_bpc",
    "final_bpf",
    "final_roc_auc",
    "final_log_loss",
    "final_brier_score",
    "final_crps",
    "final_avg_pinball_loss",
    "final_picp_90",
}

_RegistryPayloadT = TypeVar("_RegistryPayloadT", bound="_RegistryPayloadModel")


class _RegistryPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)

    @field_validator("*")
    @classmethod
    def _normalize_string(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class _BenchmarkRunModelPayload(_RegistryPayloadModel):
    arch: StrictStr
    stage: StrictStr | None = None
    stage_label: StrictStr | None = None
    benchmark_profile: StrictStr | None = None
    d_icl: StrictInt
    tficl_n_heads: StrictInt
    tficl_n_layers: StrictInt
    head_hidden_dim: StrictInt
    input_normalization: StrictStr
    feature_group_size: StrictInt | None = None
    many_class_base: StrictInt
    module_selection: dict[StrictStr, Any] | None = None
    module_hyperparameters: dict[StrictStr, Any] | None = None
    task_contract: dict[StrictStr, Any] | None = None
    architecture: dict[StrictStr, Any] | None = None
    build_spec: dict[StrictStr, Any] | None = None


class _BenchmarkBundlePayload(_RegistryPayloadModel):
    name: StrictStr
    version: StrictInt
    source_path: StrictStr
    task_count: StrictInt
    task_ids: list[StrictInt] = Field(min_length=1)


class _BenchmarkArtifactsPayload(_RegistryPayloadModel):
    run_dir: StrictStr
    benchmark_dir: StrictStr
    prior_dir: StrictStr | None = None
    history_path: StrictStr
    best_checkpoint_path: StrictStr
    comparison_summary_path: StrictStr
    comparison_curve_path: StrictStr
    benchmark_run_record_path: StrictStr | None = None
    training_surface_record_path: StrictStr | None = None
    accounting_artifact_path: StrictStr | None = None


class _WandbIdentityPayload(_RegistryPayloadModel):
    entity: StrictStr | None = None
    project: StrictStr
    run_id: StrictStr
    run_name: StrictStr


class _RemoteArtifactsPayload(_RegistryPayloadModel):
    best_checkpoint_wandb_artifact: StrictStr


class _TabFoundryMetricsPayload(_RegistryPayloadModel):
    best_step: FiniteFloat | None = None
    best_training_time: FiniteFloat | None = None
    best_bpc: FiniteFloat | None = None
    best_bpf: FiniteFloat | None = None
    best_roc_auc: FiniteFloat | None = None
    best_log_loss: FiniteFloat | None = None
    best_brier_score: FiniteFloat | None = None
    best_crps: FiniteFloat | None = None
    best_avg_pinball_loss: FiniteFloat | None = None
    best_picp_90: FiniteFloat | None = None
    final_step: FiniteFloat | None = None
    final_training_time: FiniteFloat | None = None
    final_bpc: FiniteFloat | None = None
    final_bpf: FiniteFloat | None = None
    final_roc_auc: FiniteFloat | None = None
    final_log_loss: FiniteFloat | None = None
    final_brier_score: FiniteFloat | None = None
    final_crps: FiniteFloat | None = None
    final_avg_pinball_loss: FiniteFloat | None = None
    final_picp_90: FiniteFloat | None = None


class _TrainingDiagnosticsPayload(_RegistryPayloadModel):
    best_val_loss: FiniteFloat | None = None
    final_val_loss: FiniteFloat | None = None
    best_val_step: FiniteFloat | None = None
    post_warmup_train_loss_var: FiniteFloat | None = None
    mean_grad_norm: FiniteFloat | None = None
    max_grad_norm: FiniteFloat | None = None
    final_grad_norm: FiniteFloat | None = None
    train_elapsed_seconds: FiniteFloat | None = None
    wall_elapsed_seconds: FiniteFloat | None = None


class _RuntimeSummaryPayload(_RegistryPayloadModel):
    peak_vram_allocated: StrictInt | None = None
    peak_vram_reserved: StrictInt | None = None
    throughput_examples_per_second: FiniteFloat | None = None
    throughput_tokens_per_second: FiniteFloat | None = None
    non_train_overhead_seconds: FiniteFloat | None = None
    end_to_end_wall_seconds: FiniteFloat | None = None
    loader_setup_seconds: FiniteFloat | None = None
    loader_effective_num_workers: StrictInt | None = None
    loader_effective_prefetch_factor: StrictInt | None = None
    loader_task_batch_cache_mode: StrictStr | None = None
    compile_shape_dispatch_mode: StrictStr | None = None
    compile_shape_dispatch_max_families: StrictInt | None = None
    compile_shape_dispatch: dict[StrictStr, Any] | None = None


class _BenchmarkTimingPayload(_RegistryPayloadModel):
    wall_elapsed_seconds: FiniteFloat | None = None
    mean_checkpoint_elapsed_seconds: FiniteFloat | None = None
    max_checkpoint_elapsed_seconds: FiniteFloat | None = None
    attempted_checkpoint_count: StrictInt | None = None
    successful_checkpoint_count: StrictInt | None = None
    failed_checkpoint_count: StrictInt | None = None
    requested_device: StrictStr | None = None
    resolved_device: StrictStr | None = None
    host_fingerprint: StrictStr | None = None


class _HardwareSummaryPayload(_RegistryPayloadModel):
    device_type: StrictStr | None = None
    raw_device_name: StrictStr | None = None
    gpu_class: StrictStr | None = None
    total_device_vram_bytes: StrictInt | None = None
    vram_class_gb: StrictInt | None = None
    hardware_profile_id: StrictStr | None = None


class _InferenceTimingPayload(_RegistryPayloadModel):
    fixture_id: StrictStr
    requested_device: StrictStr | None = None
    resolved_device: StrictStr | None = None
    device_type: StrictStr | None = None
    raw_device_name: StrictStr | None = None
    gpu_class: StrictStr | None = None
    vram_class_gb: StrictInt | None = None
    hardware_profile_id: StrictStr | None = None
    warmup_iterations: StrictInt
    measured_iterations: StrictInt
    n_train: StrictInt
    n_test: StrictInt
    n_features: StrictInt
    num_classes: StrictInt
    mean_ms: FiniteFloat | None = None
    p50_ms: FiniteFloat | None = None
    p95_ms: FiniteFloat | None = None
    max_ms: FiniteFloat | None = None
    total_measured_seconds: FiniteFloat | None = None


class _RegimeBudgetPayload(_RegistryPayloadModel):
    tokens_per_step: FiniteFloat | None = None
    tokens_seen: StrictInt | None = None
    token_budget: StrictInt | None = None
    unique_task_budget: StrictInt | None = None
    objective_metric: StrictStr | None = None
    curriculum_id: StrictStr | None = None
    curriculum_mix: dict[StrictStr, Any] | None = None
    scm_complexity_summary: dict[StrictStr, Any] | None = None


class _ModelSizePayload(_RegistryPayloadModel):
    total_params: StrictInt
    trainable_params: StrictInt


class _ParameterAccountingPartitionPayload(_RegistryPayloadModel):
    embedding_params: StrictInt | None = None
    embedding_like_params: StrictInt | None = None
    non_embedding_params: StrictInt


class _ParameterAccountingPayload(_RegistryPayloadModel):
    schema_: StrictStr | None = Field(default=None, alias="schema")
    method: StrictStr
    total_params: StrictInt
    trainable_params: StrictInt
    strict: _ParameterAccountingPartitionPayload
    expanded: _ParameterAccountingPartitionPayload
    canonical_non_embedding_params: StrictInt
    module_breakdown: list[dict[StrictStr, Any]] | None = None
    parameter_breakdown: list[dict[StrictStr, Any]] | None = None


class _ComputeAccountingPayload(_RegistryPayloadModel):
    schema_: StrictStr | None = Field(default=None, alias="schema")
    method: StrictStr
    training_multiplier: FiniteFloat
    forward_flops_per_token: FiniteFloat
    train_flops_per_token: FiniteFloat
    train_flops_per_step: FiniteFloat | None = None
    total_train_flops: FiniteFloat | None = None
    tokens_seen: StrictInt | None = None
    tokens_per_step: FiniteFloat | None = None
    training_shape_summary: dict[StrictStr, Any] | None = None
    signature_breakdown: list[dict[StrictStr, Any]] | None = None
    module_forward_flop_totals: list[dict[StrictStr, Any]] | None = None


class _ComparisonPayload(_RegistryPayloadModel):
    reference_run_id: StrictStr
    final_bpc_delta: FiniteFloat | None = None
    final_bpf_delta: FiniteFloat | None = None
    best_roc_auc_delta: FiniteFloat | None = None
    final_roc_auc_delta: FiniteFloat | None = None
    final_log_loss_delta: FiniteFloat | None = None
    final_brier_score_delta: FiniteFloat | None = None
    final_crps_delta: FiniteFloat | None = None
    final_avg_pinball_loss_delta: FiniteFloat | None = None
    final_picp_90_delta: FiniteFloat | None = None
    best_training_time_delta: FiniteFloat | None = None
    final_training_time_delta: FiniteFloat | None = None


class _ComparisonsPayload(_RegistryPayloadModel):
    vs_parent: _ComparisonPayload | None = None
    vs_anchor: _ComparisonPayload | None = None


class _LineagePayload(_RegistryPayloadModel):
    parent_run_id: StrictStr | None = None
    anchor_run_id: StrictStr | None = None
    control_baseline_id: StrictStr | None = None


class _SurfaceLabelsPayload(_RegistryPayloadModel):
    model: StrictStr
    data: StrictStr
    preprocessing: StrictStr
    training: StrictStr | None = None


class _SweepPayload(_RegistryPayloadModel):
    sweep_id: StrictStr
    delta_id: StrictStr
    parent_sweep_id: StrictStr | None = None
    queue_order: StrictInt | None = None
    run_kind: Literal["primary", "followup"]


class _BenchmarkRunRecordPayload(_RegistryPayloadModel):
    manifest_path: StrictStr
    seed_set: list[StrictInt] = Field(min_length=1)
    model: _BenchmarkRunModelPayload
    benchmark_bundle: _BenchmarkBundlePayload
    artifacts: _BenchmarkArtifactsPayload
    wandb: _WandbIdentityPayload | None = None
    tab_foundry_metrics: _TabFoundryMetricsPayload
    training_diagnostics: _TrainingDiagnosticsPayload
    runtime_summary: _RuntimeSummaryPayload | None = None
    benchmark_timing: _BenchmarkTimingPayload | None = None
    hardware_summary: _HardwareSummaryPayload | None = None
    inference_timing: _InferenceTimingPayload | None = None
    regime_budget: _RegimeBudgetPayload | None = None
    model_size: _ModelSizePayload
    parameter_accounting: _ParameterAccountingPayload | None = None
    compute_accounting: _ComputeAccountingPayload | None = None
    surface_labels: _SurfaceLabelsPayload | None = None
    sweep: _SweepPayload | None = None
    generated_at_utc: StrictStr


class _BenchmarkRunEntryPayload(_RegistryPayloadModel):
    run_id: StrictStr
    track: StrictStr
    experiment: StrictStr
    config_profile: StrictStr
    budget_class: StrictStr
    model: _BenchmarkRunModelPayload
    lineage: _LineagePayload
    manifest_path: StrictStr
    seed_set: list[StrictInt] = Field(min_length=1)
    benchmark_bundle: _BenchmarkBundlePayload
    artifacts: _BenchmarkArtifactsPayload
    wandb: _WandbIdentityPayload | None = None
    remote_artifacts: _RemoteArtifactsPayload | None = None
    tab_foundry_metrics: _TabFoundryMetricsPayload
    training_diagnostics: _TrainingDiagnosticsPayload
    runtime_summary: _RuntimeSummaryPayload | None = None
    benchmark_timing: _BenchmarkTimingPayload | None = None
    hardware_summary: _HardwareSummaryPayload | None = None
    inference_timing: _InferenceTimingPayload | None = None
    regime_budget: _RegimeBudgetPayload | None = None
    model_size: _ModelSizePayload
    parameter_accounting: _ParameterAccountingPayload | None = None
    compute_accounting: _ComputeAccountingPayload | None = None
    surface_labels: _SurfaceLabelsPayload | None = None
    sweep: _SweepPayload | None = None
    comparisons: _ComparisonsPayload
    decision: Literal["keep", "reject", "defer"]
    conclusion: StrictStr
    registered_at_utc: StrictStr


def _validate_payload_model(
    payload_model: type[_RegistryPayloadT],
    payload: Any,
    *,
    context: str,
) -> _RegistryPayloadT:
    try:
        return payload_model.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeError(f"{context} is invalid: {exc}") from exc
