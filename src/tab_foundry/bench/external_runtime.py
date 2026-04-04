"""Shared external benchmark runtime helpers for comparison execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tab_foundry.bench.comparison_contract import BenchmarkComparisonConfig


def nanotabpfn_python(root: Path) -> Path:
    return root.expanduser().resolve() / ".venv" / "bin" / "python"


def nanotabpfn_prior_dump(root: Path, override: Path | None) -> Path:
    return (override or (root / "300k_150x5_2.h5")).expanduser().resolve()


def tabiclv2_python(root: Path) -> Path:
    return root.expanduser().resolve() / ".venv" / "bin" / "python"


def resolved_tab_realdata_hub_root(config: BenchmarkComparisonConfig) -> Path | None:
    if config.tab_realdata_hub_root is None:
        return None
    return config.tab_realdata_hub_root.expanduser().resolve()


def nanotabpfn_helper_command(
    *,
    config: BenchmarkComparisonConfig,
    benchmark_manifest: Path,
    out_path: Path,
    allow_missing_values: bool,
    helper_script_path: Path,
    src_root: Path,
) -> list[str]:
    nanotab_root = config.nanotabpfn_root.expanduser().resolve()
    command = [
        str(nanotabpfn_python(nanotab_root)),
        str(helper_script_path),
        "--tab-foundry-src",
        str(src_root),
        "--benchmark-manifest",
        str(benchmark_manifest),
        "--prior-dump",
        str(nanotabpfn_prior_dump(nanotab_root, config.nanotab_prior_dump)),
        "--out-path",
        str(out_path),
        "--device",
        str(config.device),
        "--steps",
        str(int(config.nanotabpfn_steps)),
        "--eval-every",
        str(int(config.nanotabpfn_eval_every)),
        "--seeds",
        str(int(config.nanotabpfn_seeds)),
        "--batch-size",
        str(int(config.nanotabpfn_batch_size)),
        "--lr",
        str(float(config.nanotabpfn_lr)),
    ]
    hub_root = resolved_tab_realdata_hub_root(config)
    if hub_root is not None:
        command.extend(["--tab-realdata-hub-root", str(hub_root)])
    if allow_missing_values:
        command.append("--allow-missing-values")
    return command


def validate_nanotabpfn_environment(config: BenchmarkComparisonConfig) -> tuple[Path, Path]:
    nanotabpfn_root = config.nanotabpfn_root.expanduser().resolve()
    python_path = nanotabpfn_python(nanotabpfn_root)
    prior_dump = nanotabpfn_prior_dump(nanotabpfn_root, config.nanotab_prior_dump)
    if not nanotabpfn_root.exists():
        raise RuntimeError(f"nanoTabPFN root does not exist: {nanotabpfn_root}")
    if not python_path.exists():
        raise RuntimeError(
            "missing nanoTabPFN interpreter at "
            f"{python_path}; run `tab-foundry bench env bootstrap` first"
        )
    if not prior_dump.exists():
        raise RuntimeError(f"nanoTabPFN prior dump does not exist: {prior_dump}")
    return nanotabpfn_root, prior_dump


def resolve_reuse_curve_path(config: BenchmarkComparisonConfig) -> Path | None:
    if config.reuse_nanotabpfn_curve_path is None:
        return None
    return config.reuse_nanotabpfn_curve_path.expanduser().resolve()


def resolve_reuse_nanotabpfn_error(config: BenchmarkComparisonConfig) -> dict[str, Any] | None:
    if config.reuse_nanotabpfn_error is None:
        return None
    return dict(config.reuse_nanotabpfn_error)


def validate_tabiclv2_environment(config: BenchmarkComparisonConfig) -> tuple[Path, Path]:
    tabicl_root = config.tabicl_root.expanduser().resolve()
    python_path = tabiclv2_python(tabicl_root)
    if not tabicl_root.exists():
        raise RuntimeError(f"TabICLv2 root does not exist: {tabicl_root}")
    if not python_path.exists():
        raise RuntimeError(
            "missing TabICLv2 interpreter at "
            f"{python_path}; run `tab-foundry bench env bootstrap` first"
        )
    return tabicl_root, python_path


def tabiclv2_checkpoint_version(
    *,
    task_type: str,
    config: BenchmarkComparisonConfig,
) -> str:
    checkpoint_version = (
        config.tabicl_classifier_checkpoint_version
        if task_type == "supervised_classification"
        else config.tabicl_regressor_checkpoint_version
    )
    resolved = str(checkpoint_version).strip()
    if not resolved:
        raise RuntimeError(f"missing TabICLv2 checkpoint version for task_type={task_type!r}")
    return resolved


def nanotabpfn_execution_metadata(
    *,
    requested_device: str,
    resolved_device: str,
    host_fingerprint: str,
    nanotabpfn_root: Path | None,
    nanotabpfn_python_path: Path | None,
    prior_dump: Path | None,
    tab_realdata_hub_root: Path | None,
    steps: int,
    eval_every: int,
    seeds: int,
    batch_size: int,
    lr: float,
    reuse_curve_path: Path | None,
) -> dict[str, Any]:
    return {
        "root": None if nanotabpfn_root is None else str(nanotabpfn_root.expanduser().resolve()),
        "python": None if nanotabpfn_python_path is None else str(nanotabpfn_python_path.expanduser().resolve()),
        "num_seeds": int(seeds),
        "device": str(requested_device),
        "resolved_device": str(resolved_device),
        "benchmark_host_fingerprint": str(host_fingerprint),
        "prior_dump_path": None if prior_dump is None else str(prior_dump.expanduser().resolve()),
        "tab_realdata_hub_root": None if tab_realdata_hub_root is None else str(tab_realdata_hub_root.expanduser().resolve()),
        "steps": int(steps),
        "eval_every": int(eval_every),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "curve_source_mode": "reused" if reuse_curve_path is not None else "fresh",
        "reused_curve_path": None if reuse_curve_path is None else str(reuse_curve_path.expanduser().resolve()),
    }


def tabiclv2_helper_command(
    *,
    config: BenchmarkComparisonConfig,
    benchmark_manifest: Path,
    out_path: Path,
    task_type: str,
    allow_missing_values: bool,
    helper_script_path: Path,
    src_root: Path,
) -> list[str]:
    tabicl_root = config.tabicl_root.expanduser().resolve()
    command = [
        str(tabiclv2_python(tabicl_root)),
        str(helper_script_path),
        "--tab-foundry-src",
        str(src_root),
        "--benchmark-manifest",
        str(benchmark_manifest),
        "--out-path",
        str(out_path),
        "--task-type",
        str(task_type),
        "--checkpoint-version",
        tabiclv2_checkpoint_version(task_type=task_type, config=config),
        "--device",
        str(config.device),
    ]
    hub_root = resolved_tab_realdata_hub_root(config)
    if hub_root is not None:
        command.extend(["--tab-realdata-hub-root", str(hub_root)])
    if allow_missing_values:
        command.append("--allow-missing-values")
    return command


def tabiclv2_execution_metadata(
    *,
    requested_device: str,
    resolved_device: str,
    host_fingerprint: str,
    tabicl_root: Path,
    tabicl_python_path: Path,
    checkpoint_version: str,
    tab_realdata_hub_root: Path | None,
) -> dict[str, Any]:
    return {
        "root": str(tabicl_root.expanduser().resolve()),
        "python": str(tabicl_python_path.expanduser().resolve()),
        "checkpoint_version": str(checkpoint_version),
        "device": str(requested_device),
        "resolved_device": str(resolved_device),
        "benchmark_host_fingerprint": str(host_fingerprint),
        "tab_realdata_hub_root": None if tab_realdata_hub_root is None else str(tab_realdata_hub_root.expanduser().resolve()),
    }


def required_reuse_metadata_string(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"reuse_nanotabpfn_metadata.{key} must be a non-empty string")
    return str(value).strip()


def optional_reuse_metadata_path(metadata: Mapping[str, Any], key: str) -> Path | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"reuse_nanotabpfn_metadata.{key} must be a non-empty string when provided")
    return Path(str(value)).expanduser().resolve()


def reused_nanotabpfn_execution_metadata(
    *,
    metadata: Mapping[str, Any],
    reuse_curve_path: Path,
) -> dict[str, Any]:
    nanotabpfn_root = optional_reuse_metadata_path(metadata, "root")
    nanotabpfn_python_path = optional_reuse_metadata_path(metadata, "python")
    prior_dump = optional_reuse_metadata_path(metadata, "prior_dump_path")
    seeds = metadata.get("num_seeds", metadata.get("seeds"))
    if not isinstance(seeds, int) or isinstance(seeds, bool):
        raise RuntimeError("reuse_nanotabpfn_metadata.num_seeds must be an integer")
    steps = metadata.get("steps")
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise RuntimeError("reuse_nanotabpfn_metadata.steps must be an integer")
    eval_every = metadata.get("eval_every")
    if not isinstance(eval_every, int) or isinstance(eval_every, bool):
        raise RuntimeError("reuse_nanotabpfn_metadata.eval_every must be an integer")
    batch_size = metadata.get("batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise RuntimeError("reuse_nanotabpfn_metadata.batch_size must be an integer")
    lr = metadata.get("lr")
    if not isinstance(lr, (int, float)) or isinstance(lr, bool):
        raise RuntimeError("reuse_nanotabpfn_metadata.lr must be numeric")
    return nanotabpfn_execution_metadata(
        requested_device=required_reuse_metadata_string(metadata, "device"),
        resolved_device=required_reuse_metadata_string(metadata, "resolved_device"),
        host_fingerprint=required_reuse_metadata_string(metadata, "benchmark_host_fingerprint"),
        nanotabpfn_root=nanotabpfn_root,
        nanotabpfn_python_path=nanotabpfn_python_path,
        prior_dump=prior_dump,
        tab_realdata_hub_root=optional_reuse_metadata_path(metadata, "tab_realdata_hub_root"),
        steps=int(steps),
        eval_every=int(eval_every),
        seeds=int(seeds),
        batch_size=int(batch_size),
        lr=float(lr),
        reuse_curve_path=reuse_curve_path,
    )
