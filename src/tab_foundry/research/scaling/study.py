"""Study-config loading for paper-faithful scaling analyses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from tab_foundry.repo_paths import repo_root, resolve_repo_relative_path


SCALING_STUDY_SCHEMA = "tab-foundry-scaling-study-v1"


@dataclass(frozen=True, slots=True)
class ScalingStudySweepRef:
    """One contributing sweep for a scaling study."""

    name: str
    sweep_id: str
    family: str


@dataclass(frozen=True, slots=True)
class ScalingStudyConfig:
    """Resolved scaling-study configuration."""

    schema: str
    study_id: str
    phase: int
    output_root: str
    phase1_reference_sweep_id: str | None
    sweeps: tuple[ScalingStudySweepRef, ...]
    geometry_row_labels: tuple[str, ...]
    step_ladder: tuple[int, ...]
    batch_grad_accum_ladder: tuple[int, ...]
    canonical_loss_axes: dict[str, str]
    canonical_variables: dict[str, str]
    slice_selection: dict[str, str]

    def output_root_path(self, *, root: Path | None = None) -> Path:
        return resolve_repo_relative_path(self.output_root, root=root or repo_root())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_scaling_studies_root(*, root: Path | None = None) -> Path:
    """Return the repo-tracked scaling-study config root."""

    return (root or repo_root()).expanduser().resolve() / "reference" / "scaling_studies"


def default_scaling_study_path(
    study_id: str,
    *,
    studies_root: Path | None = None,
) -> Path:
    """Return the default YAML path for one named study."""

    resolved_root = (
        studies_root.expanduser().resolve()
        if studies_root is not None
        else default_scaling_studies_root()
    )
    return resolved_root / f"{str(study_id).strip()}.yaml"


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"scaling study config field {key!r} must be a non-empty string")
    return str(value)


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"scaling study config field {key!r} must be an integer")
    return int(value)


def _required_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"scaling study config field {key!r} must be a mapping")
    return {str(k): v for k, v in value.items()}


def _required_list(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise RuntimeError(f"scaling study config field {key!r} must be a list")
    return list(value)


def load_scaling_study_config(
    *,
    study_id: str | None = None,
    study_path: Path | None = None,
    studies_root: Path | None = None,
) -> ScalingStudyConfig:
    """Load and validate one scaling-study YAML config."""

    resolved_path = (
        study_path.expanduser().resolve()
        if study_path is not None
        else default_scaling_study_path(
            str(study_id),
            studies_root=studies_root,
        )
    )
    if study_id is None and study_path is None:
        raise RuntimeError("load_scaling_study_config requires study_id or study_path")
    try:
        raw_payload = OmegaConf.to_container(OmegaConf.load(resolved_path), resolve=True)
    except OSError as exc:
        raise RuntimeError(f"failed to load scaling study config: {resolved_path}") from exc
    if not isinstance(raw_payload, Mapping):
        raise RuntimeError(f"scaling study config must be a mapping: path={resolved_path}")
    payload = {str(key): value for key, value in raw_payload.items()}
    schema = _required_str(payload, "schema")
    if schema != SCALING_STUDY_SCHEMA:
        raise RuntimeError(
            f"scaling study schema mismatch: expected={SCALING_STUDY_SCHEMA!r}, actual={schema!r}"
        )
    raw_sweeps = _required_list(payload, "sweeps")
    sweeps: list[ScalingStudySweepRef] = []
    for index, item in enumerate(raw_sweeps):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"scaling study sweeps[{index}] must be a mapping")
        item_payload = {str(key): value for key, value in item.items()}
        sweeps.append(
            ScalingStudySweepRef(
                name=_required_str(item_payload, "name"),
                sweep_id=_required_str(item_payload, "sweep_id"),
                family=_required_str(item_payload, "family"),
            )
        )
    geometry_row_labels = tuple(str(value) for value in _required_list(payload, "geometry_row_labels"))
    step_ladder = tuple(int(value) for value in _required_list(payload, "step_ladder"))
    batch_grad_accum_ladder = tuple(
        int(value) for value in _required_list(payload, "batch_grad_accum_ladder")
    )
    phase1_reference_sweep_id = payload.get("phase1_reference_sweep_id")
    if phase1_reference_sweep_id is not None:
        if not isinstance(phase1_reference_sweep_id, str) or not phase1_reference_sweep_id.strip():
            raise RuntimeError("phase1_reference_sweep_id must be a non-empty string when present")
        resolved_phase1_reference_sweep_id: str | None = str(phase1_reference_sweep_id)
    else:
        resolved_phase1_reference_sweep_id = None
    return ScalingStudyConfig(
        schema=schema,
        study_id=_required_str(payload, "study_id"),
        phase=_required_int(payload, "phase"),
        output_root=_required_str(payload, "output_root"),
        phase1_reference_sweep_id=resolved_phase1_reference_sweep_id,
        sweeps=tuple(sweeps),
        geometry_row_labels=geometry_row_labels,
        step_ladder=step_ladder,
        batch_grad_accum_ladder=batch_grad_accum_ladder,
        canonical_loss_axes=_required_mapping(payload, "canonical_loss_axes"),
        canonical_variables=_required_mapping(payload, "canonical_variables"),
        slice_selection=_required_mapping(payload, "slice_selection"),
    )
