"""dagzoo CLI workflow helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from .dagzoo_handoff import DagzooHandoffInfo, load_dagzoo_handoff_info
from .manifest import ManifestSummary, build_manifest


@dataclass(slots=True, frozen=True)
class DagzooGenerateManifestConfig:
    """Typed input for the dagzoo generate -> manifest workflow."""

    dagzoo_root: Path
    dagzoo_config: Path
    handoff_root: Path
    out_manifest: Path
    num_datasets: int = 10
    seed: int | None = None
    rows: str | None = None
    device: str | None = None
    hardware_policy: str = "none"
    diagnostics: bool = False
    diagnostics_out_dir: Path | None = None
    missing_rate: float | None = None
    missing_mechanism: str | None = None
    missing_mar_observed_fraction: float | None = None
    missing_mar_logit_scale: float | None = None
    missing_mnar_logit_scale: float | None = None
    train_ratio: float = 0.90
    val_ratio: float = 0.05
    filter_policy: str = "include_all"
    missing_value_policy: str = "allow_any"


@dataclass(slots=True, frozen=True)
class DagzooGenerateManifestResult:
    """Result of one dagzoo generate -> manifest workflow run."""

    handoff: DagzooHandoffInfo
    summary: ManifestSummary


def _resolve_from_root(root: Path, raw_path: Path) -> Path:
    expanded = raw_path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def build_dagzoo_generate_argv(config: DagzooGenerateManifestConfig) -> list[str]:
    """Build the dagzoo CLI argv for one generate run."""

    dagzoo_root = config.dagzoo_root.expanduser().resolve()
    dagzoo_config = _resolve_from_root(dagzoo_root, config.dagzoo_config)
    handoff_root = _resolve_from_root(dagzoo_root, config.handoff_root)
    argv = [
        "uv",
        "run",
        "dagzoo",
        "generate",
        "--config",
        str(dagzoo_config),
        "--handoff-root",
        str(handoff_root),
        "--num-datasets",
        str(int(config.num_datasets)),
        "--hardware-policy",
        str(config.hardware_policy),
    ]
    if config.seed is not None:
        argv.extend(["--seed", str(int(config.seed))])
    if config.rows is not None:
        argv.extend(["--rows", str(config.rows)])
    if config.device is not None:
        argv.extend(["--device", str(config.device)])
    if config.diagnostics:
        argv.append("--diagnostics")
    if config.diagnostics_out_dir is not None:
        argv.extend(
            [
                "--diagnostics-out-dir",
                str(_resolve_from_root(dagzoo_root, config.diagnostics_out_dir)),
            ]
        )
    if config.missing_rate is not None:
        argv.extend(["--missing-rate", str(float(config.missing_rate))])
    if config.missing_mechanism is not None:
        argv.extend(["--missing-mechanism", str(config.missing_mechanism)])
    if config.missing_mar_observed_fraction is not None:
        argv.extend(
            [
                "--missing-mar-observed-fraction",
                str(float(config.missing_mar_observed_fraction)),
            ]
        )
    if config.missing_mar_logit_scale is not None:
        argv.extend(["--missing-mar-logit-scale", str(float(config.missing_mar_logit_scale))])
    if config.missing_mnar_logit_scale is not None:
        argv.extend(["--missing-mnar-logit-scale", str(float(config.missing_mnar_logit_scale))])
    return argv


def run_dagzoo_generate_manifest(config: DagzooGenerateManifestConfig) -> DagzooGenerateManifestResult:
    """Run dagzoo generate through the CLI and materialize one tab-foundry manifest."""

    dagzoo_root = config.dagzoo_root.expanduser().resolve()
    if not dagzoo_root.exists():
        raise RuntimeError(f"dagzoo root does not exist: {dagzoo_root}")
    if not dagzoo_root.is_dir():
        raise RuntimeError(f"dagzoo root must be a directory: {dagzoo_root}")
    dagzoo_config = _resolve_from_root(dagzoo_root, config.dagzoo_config)
    if not dagzoo_config.exists():
        raise RuntimeError(f"dagzoo config does not exist: {dagzoo_config}")

    argv = build_dagzoo_generate_argv(config)
    subprocess.run(argv, cwd=dagzoo_root, check=True)

    handoff_root = _resolve_from_root(dagzoo_root, config.handoff_root)
    handoff = load_dagzoo_handoff_info(handoff_root / "handoff_manifest.json")
    if not handoff.generated_dir.exists() or not handoff.generated_dir.is_dir():
        raise RuntimeError(
            f"dagzoo handoff generated directory does not exist: {handoff.generated_dir}"
        )

    summary = build_manifest(
        data_roots=[handoff.generated_dir],
        out_path=config.out_manifest.expanduser().resolve(),
        train_ratio=float(config.train_ratio),
        val_ratio=float(config.val_ratio),
        filter_policy=str(config.filter_policy),
        missing_value_policy=str(config.missing_value_policy),
        dagzoo_handoff_manifest_path=handoff.handoff_manifest_path,
    )
    return DagzooGenerateManifestResult(handoff=handoff, summary=summary)
