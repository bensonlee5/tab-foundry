"""dagzoo CLI workflow helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, cast

from tab_realdata_hub.manifest import ManifestSummary, build_manifest

from tab_realdata_hub.dagzoo_handoff import DagzooHandoffInfo, load_dagzoo_handoff_info


@dataclass(slots=True, frozen=True)
class DagzooGenerateConfig:
    """Typed input for one dagzoo generate CLI invocation."""

    dagzoo_root: Path
    dagzoo_config: Path
    handoff_root: Path
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
    worker_threads: int | None = None
    set_overrides: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class DagzooGenerateManifestConfig(DagzooGenerateConfig):
    """Typed input for the dagzoo generate -> manifest workflow."""

    out_manifest: Path = Path("manifest.parquet")
    train_ratio: float = 0.90
    val_ratio: float = 0.05
    filter_policy: str = "include_all"
    missing_value_policy: str = "allow_any"


@dataclass(slots=True, frozen=True)
class DagzooGenerateManifestResult:
    """Result of one dagzoo generate -> manifest workflow run."""

    handoff: DagzooHandoffInfo
    summary: ManifestSummary
    filter_result: DagzooFilterResult | None = None


@dataclass(slots=True, frozen=True)
class DagzooFilterConfig:
    """Typed input for one dagzoo filter CLI invocation."""

    dagzoo_root: Path
    input_dir: Path
    filter_out_dir: Path
    curated_out_dir: Path | None = None
    worker_threads: int | None = None
    set_overrides: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class DagzooFilterResult:
    """Result of one dagzoo filter CLI invocation."""

    manifest_path: Path
    summary_path: Path
    total_datasets: int
    accepted_datasets: int
    rejected_datasets: int
    elapsed_seconds: float | None
    datasets_per_minute: float | None
    curated_out_dir: Path | None = None
    curated_accepted_datasets: int = 0


def _resolve_from_root(root: Path, raw_path: Path) -> Path:
    expanded = raw_path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def _read_json_mapping(path: Path, *, context: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{context} must decode to a JSON object: {path}")
    return {str(key): value for key, value in cast(Mapping[str, Any], payload).items()}


def _append_dagzoo_set_override(argv: list[str], *, key: str, value: object) -> None:
    argv.extend(["--set", f"{key}={value}"])


def _dagzoo_python_executable(dagzoo_root: Path) -> Path:
    executable = dagzoo_root.expanduser().resolve() / ".venv" / "bin" / "python"
    if not executable.exists():
        raise RuntimeError(
            "dagzoo venv interpreter does not exist; expected "
            f"{executable}. Bootstrap ../dagzoo/.venv before materializing corpora."
        )
    if not executable.is_file():
        raise RuntimeError(f"dagzoo venv interpreter must be a file: {executable}")
    return executable


_THREAD_BUDGET_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _dagzoo_subprocess_env(*, worker_threads: int | None) -> dict[str, str] | None:
    if worker_threads is None:
        return None
    resolved_threads = int(worker_threads)
    env = dict(os.environ)
    for env_var in _THREAD_BUDGET_ENV_VARS:
        env[env_var] = str(resolved_threads)
    return env


def build_dagzoo_generate_argv(config: DagzooGenerateConfig) -> list[str]:
    """Build the dagzoo CLI argv for one generate run."""

    dagzoo_root = config.dagzoo_root.expanduser().resolve()
    dagzoo_python = _dagzoo_python_executable(dagzoo_root)
    dagzoo_config = _resolve_from_root(dagzoo_root, config.dagzoo_config)
    handoff_root = _resolve_from_root(dagzoo_root, config.handoff_root)
    argv = [
        str(dagzoo_python),
        "-m",
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
    # dagzoo's explicit missingness flags were removed in favor of config-level
    # overrides, so preserve the tab-foundry recipe contract through --set.
    if config.missing_rate is not None:
        _append_dagzoo_set_override(
            argv,
            key="dataset.missing_rate",
            value=float(config.missing_rate),
        )
    if config.missing_mechanism is not None:
        _append_dagzoo_set_override(
            argv,
            key="dataset.missing_mechanism",
            value=str(config.missing_mechanism),
        )
    if config.missing_mar_observed_fraction is not None:
        _append_dagzoo_set_override(
            argv,
            key="dataset.missing_mar_observed_fraction",
            value=float(config.missing_mar_observed_fraction),
        )
    if config.missing_mar_logit_scale is not None:
        _append_dagzoo_set_override(
            argv,
            key="dataset.missing_mar_logit_scale",
            value=float(config.missing_mar_logit_scale),
        )
    if config.missing_mnar_logit_scale is not None:
        _append_dagzoo_set_override(
            argv,
            key="dataset.missing_mnar_logit_scale",
            value=float(config.missing_mnar_logit_scale),
        )
    for override in config.set_overrides:
        argv.extend(["--set", str(cast(str, override))])
    return argv


def build_dagzoo_filter_argv(config: DagzooFilterConfig) -> list[str]:
    """Build the dagzoo CLI argv for one filter run."""

    dagzoo_root = config.dagzoo_root.expanduser().resolve()
    dagzoo_python = _dagzoo_python_executable(dagzoo_root)
    input_dir = _resolve_from_root(dagzoo_root, config.input_dir)
    filter_out_dir = _resolve_from_root(dagzoo_root, config.filter_out_dir)
    argv = [
        str(dagzoo_python),
        "-m",
        "dagzoo",
        "filter",
        "--in",
        str(input_dir),
        "--out",
        str(filter_out_dir),
    ]
    if config.curated_out_dir is not None:
        argv.extend(
            [
                "--curated-out",
                str(_resolve_from_root(dagzoo_root, config.curated_out_dir)),
            ]
        )
    for override in config.set_overrides:
        argv.extend(["--set", str(cast(str, override))])
    # Dagzoo's structural filter no longer accepts filter.n_jobs as a config
    # override, so keep the worker budget in subprocess env vars only.
    return argv


def run_dagzoo_generate(config: DagzooGenerateConfig) -> DagzooHandoffInfo:
    """Run dagzoo generate through the CLI and return validated handoff metadata."""

    dagzoo_root = config.dagzoo_root.expanduser().resolve()
    if not dagzoo_root.exists():
        raise RuntimeError(f"dagzoo root does not exist: {dagzoo_root}")
    if not dagzoo_root.is_dir():
        raise RuntimeError(f"dagzoo root must be a directory: {dagzoo_root}")
    dagzoo_config = _resolve_from_root(dagzoo_root, config.dagzoo_config)
    if not dagzoo_config.exists():
        raise RuntimeError(f"dagzoo config does not exist: {dagzoo_config}")

    argv = build_dagzoo_generate_argv(config)
    subprocess.run(
        argv,
        cwd=dagzoo_root,
        check=True,
        env=_dagzoo_subprocess_env(worker_threads=config.worker_threads),
    )

    handoff_root = _resolve_from_root(dagzoo_root, config.handoff_root)
    handoff = load_dagzoo_handoff_info(handoff_root / "handoff_manifest.json")
    if not handoff.generated_dir.exists() or not handoff.generated_dir.is_dir():
        raise RuntimeError(
            f"dagzoo handoff generated directory does not exist: {handoff.generated_dir}"
        )
    return handoff


def run_dagzoo_filter(config: DagzooFilterConfig) -> DagzooFilterResult:
    """Run dagzoo filter through the CLI and return parsed filter metadata."""

    dagzoo_root = config.dagzoo_root.expanduser().resolve()
    if not dagzoo_root.exists():
        raise RuntimeError(f"dagzoo root does not exist: {dagzoo_root}")
    if not dagzoo_root.is_dir():
        raise RuntimeError(f"dagzoo root must be a directory: {dagzoo_root}")
    input_dir = _resolve_from_root(dagzoo_root, config.input_dir)
    if not input_dir.exists():
        raise RuntimeError(f"dagzoo filter input does not exist: {input_dir}")
    argv = build_dagzoo_filter_argv(config)
    subprocess.run(
        argv,
        cwd=dagzoo_root,
        check=True,
        env=_dagzoo_subprocess_env(worker_threads=config.worker_threads),
    )

    filter_out_dir = _resolve_from_root(dagzoo_root, config.filter_out_dir)
    manifest_path = filter_out_dir / "filter_manifest.ndjson"
    summary_path = filter_out_dir / "filter_summary.json"
    if not manifest_path.exists():
        raise RuntimeError(f"dagzoo filter manifest not found: {manifest_path}")
    if not summary_path.exists():
        raise RuntimeError(f"dagzoo filter summary not found: {summary_path}")
    summary = _read_json_mapping(summary_path, context="dagzoo filter summary")
    curated_out_dir = None
    raw_curated_out_dir = summary.get("curated_out_dir")
    if isinstance(raw_curated_out_dir, str) and raw_curated_out_dir.strip():
        curated_out_dir = Path(raw_curated_out_dir).expanduser().resolve()
    elif config.curated_out_dir is not None:
        curated_out_dir = _resolve_from_root(dagzoo_root, config.curated_out_dir)
    return DagzooFilterResult(
        manifest_path=manifest_path.resolve(),
        summary_path=summary_path.resolve(),
        total_datasets=int(summary.get("total_datasets", 0)),
        accepted_datasets=int(summary.get("accepted_datasets", 0)),
        rejected_datasets=int(summary.get("rejected_datasets", 0)),
        elapsed_seconds=(
            None if summary.get("elapsed_seconds") is None else float(summary["elapsed_seconds"])
        ),
        datasets_per_minute=(
            None
            if summary.get("datasets_per_minute") is None
            else float(summary["datasets_per_minute"])
        ),
        curated_out_dir=curated_out_dir,
        curated_accepted_datasets=int(summary.get("curated_accepted_datasets", 0)),
    )


def _manifest_data_root(*, handoff: DagzooHandoffInfo, filter_policy: str) -> Path:
    normalized_filter_policy = str(filter_policy).strip()
    if normalized_filter_policy == "accepted_only":
        if handoff.curated_dir is None:
            raise RuntimeError(
                "filter_policy='accepted_only' requires a curated dagzoo corpus. "
                "Run `dagzoo filter --in "
                f"{handoff.generated_dir} --out <filter_dir> --curated-out <curated_dir>` first."
            )
        return handoff.curated_dir
    return handoff.generated_dir


def run_dagzoo_generate_manifest(config: DagzooGenerateManifestConfig) -> DagzooGenerateManifestResult:
    """Run dagzoo generate through the CLI and materialize one tab-foundry manifest."""

    handoff = run_dagzoo_generate(config)
    filter_result = None
    if str(config.filter_policy).strip() == "accepted_only":
        filter_result = run_dagzoo_filter(
            DagzooFilterConfig(
                dagzoo_root=config.dagzoo_root,
                input_dir=handoff.generated_dir,
                filter_out_dir=handoff.handoff_manifest_path.parent / "filter",
                curated_out_dir=handoff.curated_dir or (handoff.handoff_manifest_path.parent / "curated"),
                worker_threads=config.worker_threads,
            )
        )
        if filter_result.curated_out_dir is None:
            raise RuntimeError("dagzoo filter did not report a curated output directory")
        data_root = filter_result.curated_out_dir
    else:
        data_root = _manifest_data_root(
            handoff=handoff,
            filter_policy=str(config.filter_policy),
        )

    summary = build_manifest(
        data_roots=[data_root],
        out_path=config.out_manifest.expanduser().resolve(),
        train_ratio=float(config.train_ratio),
        val_ratio=float(config.val_ratio),
        filter_policy=str(config.filter_policy),
        missing_value_policy=str(config.missing_value_policy),
        dagzoo_handoff_manifest_path=(
            None
            if str(config.filter_policy).strip() == "accepted_only"
            else handoff.handoff_manifest_path
        ),
    )
    return DagzooGenerateManifestResult(
        handoff=handoff,
        summary=summary,
        filter_result=filter_result,
    )
