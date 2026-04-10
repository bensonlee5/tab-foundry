"""Tracked study-config loading for robust prior pilots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from tab_foundry.bench.openml_benchmark import default_benchmark_manifest_path
from tab_foundry.repo_paths import repo_root, resolve_repo_relative_path


ROBUST_PRIOR_STUDY_SCHEMA = "tab-foundry-robust-prior-v1"


@dataclass(frozen=True, slots=True)
class RobustPriorGuardrails:
    """Guardrail thresholds for candidate feasibility and selection."""

    class_entropy_floor: float = 0.35
    min_class_prior_headroom: float = 0.02
    depth_ratio_tolerance: float = 0.08
    diversity_min_distance: float = 0.08

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RobustPriorStudyConfig:
    """Resolved study config for one robust-prior pilot."""

    schema: str
    study_id: str
    description: str
    output_root: str
    anchor_checkpoint_path: str
    base_experiment: str = "cls_benchmark_staged_corpus"
    control_corpus_ref: str | None = None
    benchmark_manifest_path: str | None = None
    search_space_id: str = "robust_prior_search_space_v1"
    dagzoo_base_config_ref: str = "configs/default.yaml"
    outer_rounds: int = 3
    trials_per_round: int = 32
    probe_datasets_per_trial: int = 6
    topk_training_candidates: int = 8
    round_train_datasets: int = 8192
    train_steps_per_round: int = 150
    matched_control: bool = True
    benchmark_device: str = "cpu"
    benchmark_checkpoint_selection: str = "all"
    logging_use_wandb: bool = False
    exploration_rate: float = 0.20
    entropy_floor_ratio: float = 0.65
    materialize_worker_threads: int | None = None
    guardrails: RobustPriorGuardrails = field(default_factory=RobustPriorGuardrails)
    training_overrides: dict[str, Any] = field(default_factory=dict)

    def output_root_path(self, *, root: Path | None = None) -> Path:
        return resolve_repo_relative_path(self.output_root, root=root or repo_root())

    def anchor_checkpoint(self) -> Path:
        return resolve_repo_relative_path(self.anchor_checkpoint_path)

    def benchmark_manifest(self) -> Path:
        if self.benchmark_manifest_path is None:
            return default_benchmark_manifest_path()
        return resolve_repo_relative_path(self.benchmark_manifest_path)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["guardrails"] = self.guardrails.as_dict()
        return payload


def default_robust_prior_studies_root(*, root: Path | None = None) -> Path:
    """Return the repo-tracked robust-prior study root."""

    return (root or repo_root()).expanduser().resolve() / "reference" / "robust_prior"


def default_robust_prior_study_path(
    study_id: str,
    *,
    studies_root: Path | None = None,
) -> Path:
    """Return the default YAML path for one named robust-prior study."""

    resolved_root = (
        studies_root.expanduser().resolve()
        if studies_root is not None
        else default_robust_prior_studies_root()
    )
    return resolved_root / f"{str(study_id).strip()}.yaml"


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"robust prior config field {key!r} must be a non-empty string")
    return str(value)


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"robust prior config field {key!r} must be a non-empty string")
    return str(value)


def _optional_int(payload: Mapping[str, Any], key: str, *, default: int | None = None) -> int | None:
    value = payload.get(key, default)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"robust prior config field {key!r} must be an integer") from exc


def _optional_float(payload: Mapping[str, Any], key: str, *, default: float | None = None) -> float | None:
    value = payload.get(key, default)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"robust prior config field {key!r} must be a float") from exc


def _optional_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    raw = payload.get(key)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"robust prior config field {key!r} must be a mapping")
    return {str(item_key): item_value for item_key, item_value in raw.items()}


def load_robust_prior_study_config(
    *,
    study_id: str | None = None,
    study_path: Path | None = None,
    studies_root: Path | None = None,
) -> RobustPriorStudyConfig:
    """Load and validate one robust-prior YAML config."""

    if study_id is None and study_path is None:
        raise RuntimeError("load_robust_prior_study_config requires study_id or study_path")
    resolved_path = (
        study_path.expanduser().resolve()
        if study_path is not None
        else default_robust_prior_study_path(str(study_id), studies_root=studies_root)
    )
    try:
        raw_payload = OmegaConf.to_container(OmegaConf.load(resolved_path), resolve=True)
    except OSError as exc:
        raise RuntimeError(f"failed to load robust prior config: {resolved_path}") from exc
    if not isinstance(raw_payload, Mapping):
        raise RuntimeError(f"robust prior config must be a mapping: {resolved_path}")
    payload = {str(key): value for key, value in raw_payload.items()}
    schema = _required_str(payload, "schema")
    if schema != ROBUST_PRIOR_STUDY_SCHEMA:
        raise RuntimeError(
            f"robust prior schema mismatch: expected={ROBUST_PRIOR_STUDY_SCHEMA!r}, actual={schema!r}"
        )
    guardrails_payload = _optional_mapping(payload, "guardrails")
    guardrails = RobustPriorGuardrails(
        class_entropy_floor=float(guardrails_payload.get("class_entropy_floor", 0.35)),
        min_class_prior_headroom=float(guardrails_payload.get("min_class_prior_headroom", 0.02)),
        depth_ratio_tolerance=float(guardrails_payload.get("depth_ratio_tolerance", 0.08)),
        diversity_min_distance=float(guardrails_payload.get("diversity_min_distance", 0.08)),
    )
    config = RobustPriorStudyConfig(
        schema=schema,
        study_id=_required_str(payload, "study_id"),
        description=_required_str(payload, "description"),
        output_root=_required_str(payload, "output_root"),
        anchor_checkpoint_path=_required_str(payload, "anchor_checkpoint_path"),
        base_experiment=str(payload.get("base_experiment", "cls_benchmark_staged_corpus")),
        control_corpus_ref=_optional_str(payload, "control_corpus_ref"),
        benchmark_manifest_path=_optional_str(payload, "benchmark_manifest_path"),
        search_space_id=str(payload.get("search_space_id", "robust_prior_search_space_v1")),
        dagzoo_base_config_ref=str(payload.get("dagzoo_base_config_ref", "configs/default.yaml")),
        outer_rounds=int(payload.get("outer_rounds", 3)),
        trials_per_round=int(payload.get("trials_per_round", 32)),
        probe_datasets_per_trial=int(payload.get("probe_datasets_per_trial", 6)),
        topk_training_candidates=int(payload.get("topk_training_candidates", 8)),
        round_train_datasets=int(payload.get("round_train_datasets", 8192)),
        train_steps_per_round=int(payload.get("train_steps_per_round", 150)),
        matched_control=bool(payload.get("matched_control", True)),
        benchmark_device=str(payload.get("benchmark_device", "cpu")),
        benchmark_checkpoint_selection=str(payload.get("benchmark_checkpoint_selection", "all")),
        logging_use_wandb=bool(payload.get("logging_use_wandb", False)),
        exploration_rate=float(payload.get("exploration_rate", 0.20)),
        entropy_floor_ratio=float(payload.get("entropy_floor_ratio", 0.65)),
        materialize_worker_threads=_optional_int(payload, "materialize_worker_threads"),
        guardrails=guardrails,
        training_overrides=_optional_mapping(payload, "training_overrides"),
    )
    if config.outer_rounds <= 0:
        raise RuntimeError("robust prior outer_rounds must be >= 1")
    if config.trials_per_round <= 0:
        raise RuntimeError("robust prior trials_per_round must be >= 1")
    if config.probe_datasets_per_trial <= 0:
        raise RuntimeError("robust prior probe_datasets_per_trial must be >= 1")
    if config.topk_training_candidates <= 0:
        raise RuntimeError("robust prior topk_training_candidates must be >= 1")
    if config.round_train_datasets <= 0:
        raise RuntimeError("robust prior round_train_datasets must be >= 1")
    if config.train_steps_per_round <= 0:
        raise RuntimeError("robust prior train_steps_per_round must be >= 1")
    if not 0.0 <= config.exploration_rate <= 1.0:
        raise RuntimeError("robust prior exploration_rate must be in [0, 1]")
    if not 0.0 <= config.entropy_floor_ratio <= 1.0:
        raise RuntimeError("robust prior entropy_floor_ratio must be in [0, 1]")
    return config
