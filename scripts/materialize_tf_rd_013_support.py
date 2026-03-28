#!/usr/bin/env python3
"""Materialize TF-RD-013 dagzoo and curated-comparator support artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tab_foundry.bench.nanotabpfn.bundle import (  # noqa: E402
    benchmark_bundle_summary,
    load_benchmark_bundle_for_execution,
)
from tab_foundry.bench.nanotabpfn.datasets import (  # noqa: E402
    PreparedOpenMLBenchmarkTask,
    prepare_openml_benchmark_task,
)
from tab_foundry.data.corpus_materialization import (  # noqa: E402
    materialize_corpus_recipe,
)
from tab_foundry.data.corpus_reporting import (  # noqa: E402
    corpus_compare_payload,
)
from tab_foundry.data.inspection import compare_jsonlike_payloads  # noqa: E402
from tab_foundry.data.manifest import build_manifest  # noqa: E402


DEFAULT_DAGZOO_ROOT = REPO_ROOT.parent / "dagzoo"
DEFAULT_DAGZOO_CONFIG_REF = "configs/default.yaml"
DEFAULT_MANIFEST_CONFIG_REF = "configs/data/default.yaml"
DEFAULT_BENCHMARK_BUNDLE_REF = "src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json"
DEFAULT_NUM_DATASETS = 8192
SIZE_LADDER_CONTROL_NUM_DATASETS = 10
DEFAULT_SEED = 1
DEFAULT_DEVICE = "cpu"
DEFAULT_HARDWARE_POLICY = "none"
DEFAULT_COMPARATOR_SPLIT_SEED = 0
DEFAULT_COMPARATOR_TEST_SIZE = 0.20

FIRST_MATERIALIZATION_ISSUE_NUMBER = 120
FIRST_EXECUTION_ISSUE_NUMBER = 122
SHAPE_AWARE_ISSUE_NUMBER = 127
EPIC_ISSUE_NUMBER = 96
DOWNSTREAM_TRAINING_SURFACE_ISSUE_NUMBER = 107
FILTERING_POLICY_ISSUE_NUMBER = 124
SIZE_LADDER_ISSUE_NUMBER = 132

LOCAL_OUTPUT_ROOT = REPO_ROOT / "outputs" / "staged_ladder_support" / "tf_rd_013"
SUPPORT_ROOT = (
    REPO_ROOT
    / "reference"
    / "system_delta_sweeps"
    / "tf_rd_013_data_source_contract_v1"
    / "support"
)

SHAPE_AWARE_SWEEP_ID = "tf_rd_013_shape_aware_dagzoo_v1"
SHAPE_AWARE_LOCAL_OUTPUT_ROOT = (
    REPO_ROOT / "outputs" / "staged_ladder_support" / SHAPE_AWARE_SWEEP_ID
)
SHAPE_AWARE_SUPPORT_ROOT = (
    REPO_ROOT
    / "reference"
    / "system_delta_sweeps"
    / SHAPE_AWARE_SWEEP_ID
    / "support"
)
SHAPE_AWARE_DAGZOO_SURFACE_ROOT_NAME = "dagzoo_shape_aware_multi_invocation"

SIZE_LADDER_SWEEP_ID = "tf_rd_013_dagzoo_size_ladder_v1"
SIZE_LADDER_LOCAL_OUTPUT_ROOT = (
    REPO_ROOT / "outputs" / "staged_ladder_support" / SIZE_LADDER_SWEEP_ID
)
SIZE_LADDER_SUPPORT_ROOT = (
    REPO_ROOT
    / "reference"
    / "system_delta_sweeps"
    / SIZE_LADDER_SWEEP_ID
    / "support"
)

ANCHOR_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "default.parquet"
CURRENT_CORPUS_HANDOFF_ROOT = (
    REPO_ROOT / "outputs" / "current_corpus" / "default_generated_source"
)
ANCHOR_MANIFEST_REQUIRED_COLUMNS = ("train_path", "test_path", "metadata_path")

CURRENT_CORPUS_RECIPE_ID = "tf_rd_013_current_corpus_default_v1"
SIZE_LADDER_RECIPE_IDS: tuple[str, ...] = (
    CURRENT_CORPUS_RECIPE_ID,
    "tf_rd_013_dagzoo_shape_aware_size_small_v1",
    "tf_rd_013_dagzoo_shape_aware_size_medium_v1",
    "tf_rd_013_dagzoo_shape_aware_size_large_v1",
)


def _load_tool_roots_module():
    spec = importlib.util.spec_from_file_location(
        "tool_roots_script",
        SCRIPT_DIR / "audit" / "tool_roots.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load scripts/audit/tool_roots.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool_roots = _load_tool_roots_module()


def _resolve_tab_foundry_roots() -> tuple[Path, Path]:
    resolved_tool_roots = tool_roots.resolve_tool_roots(cwd=REPO_ROOT)
    candidate_roots = [resolved_tool_roots.tool_root]
    if resolved_tool_roots.primary_root not in candidate_roots:
        candidate_roots.append(resolved_tool_roots.primary_root)
    for root in candidate_roots:
        executable = root / ".venv" / "bin" / "tab-foundry"
        if executable.is_file():
            return root, executable
    attempted = ", ".join(str(root / ".venv" / "bin" / "tab-foundry") for root in candidate_roots)
    raise RuntimeError(f"tab-foundry CLI not found. Checked {attempted}")


TAB_FOUNDRY_ROOT, TAB_FOUNDRY_BIN = _resolve_tab_foundry_roots()


@dataclass(frozen=True)
class MaterializationPaths:
    local_output_root: Path
    generated_source_root: Path
    generated_dir: Path
    generated_manifest_path: Path
    handoff_manifest_path: Path
    curated_realdata_root: Path
    curated_openml_baseline_root: Path
    curated_openml_baseline_data_root: Path
    curated_openml_baseline_manifest_path: Path


@dataclass(frozen=True)
class ShapeAwareInvocationSpec:
    invocation_id: str
    config_ref: str
    num_datasets: int
    seed: int = DEFAULT_SEED
    device: str = DEFAULT_DEVICE
    hardware_policy: str = "none"


@dataclass(frozen=True)
class ShapeAwareInvocationPaths:
    invocation_id: str
    invocation_root: Path
    generated_dir: Path
    handoff_manifest_path: Path


@dataclass(frozen=True)
class MultiInvocationSurfaceSpec:
    surface_name: str
    invocations: tuple[ShapeAwareInvocationSpec, ...]


@dataclass(frozen=True)
class MultiInvocationSurfacePaths:
    surface_name: str
    surface_root: Path
    manifest_path: Path
    invocation_paths: tuple[ShapeAwareInvocationPaths, ...]


@dataclass(frozen=True)
class ShapeAwareMaterializationPaths:
    local_output_root: Path
    dagzoo_surface_root: Path
    dagzoo_manifest_path: Path
    invocation_paths: tuple[ShapeAwareInvocationPaths, ...]
    curated_realdata_root: Path
    curated_openml_baseline_root: Path
    curated_openml_baseline_data_root: Path
    curated_openml_baseline_manifest_path: Path


@dataclass(frozen=True)
class SizeLadderMaterializationPaths:
    local_output_root: Path
    dagzoo_surfaces: tuple[MultiInvocationSurfacePaths, ...]


SHAPE_AWARE_INVOCATIONS: tuple[ShapeAwareInvocationSpec, ...] = (
    ShapeAwareInvocationSpec(
        invocation_id="benchmark_cpu",
        config_ref="configs/benchmark_cpu.yaml",
        num_datasets=2048,
    ),
    ShapeAwareInvocationSpec(
        invocation_id="default_medium",
        config_ref="configs/default.yaml",
        num_datasets=4096,
    ),
    ShapeAwareInvocationSpec(
        invocation_id="large_shape",
        config_ref="configs/benchmark_cuda_h100_large_shape.yaml",
        num_datasets=128,
    ),
)

SHAPE_AWARE_SURFACE_SPEC = MultiInvocationSurfaceSpec(
    surface_name=SHAPE_AWARE_DAGZOO_SURFACE_ROOT_NAME,
    invocations=SHAPE_AWARE_INVOCATIONS,
)

SIZE_LADDER_SURFACE_SPECS: tuple[MultiInvocationSurfaceSpec, ...] = (
    MultiInvocationSurfaceSpec(
        surface_name="dagzoo_shape_aware_size_small",
        invocations=(
            ShapeAwareInvocationSpec(
                invocation_id="benchmark_cpu",
                config_ref="configs/benchmark_cpu.yaml",
                num_datasets=4,
            ),
            ShapeAwareInvocationSpec(
                invocation_id="default_medium",
                config_ref="configs/default.yaml",
                num_datasets=14,
            ),
            ShapeAwareInvocationSpec(
                invocation_id="large_shape",
                config_ref="configs/benchmark_cuda_h100_large_shape.yaml",
                num_datasets=2,
            ),
        ),
    ),
    MultiInvocationSurfaceSpec(
        surface_name="dagzoo_shape_aware_size_medium",
        invocations=(
            ShapeAwareInvocationSpec(
                invocation_id="benchmark_cpu",
                config_ref="configs/benchmark_cpu.yaml",
                num_datasets=8,
            ),
            ShapeAwareInvocationSpec(
                invocation_id="default_medium",
                config_ref="configs/default.yaml",
                num_datasets=28,
            ),
            ShapeAwareInvocationSpec(
                invocation_id="large_shape",
                config_ref="configs/benchmark_cuda_h100_large_shape.yaml",
                num_datasets=4,
            ),
        ),
    ),
    MultiInvocationSurfaceSpec(
        surface_name="dagzoo_shape_aware_size_large",
        invocations=(
            ShapeAwareInvocationSpec(
                invocation_id="benchmark_cpu",
                config_ref="configs/benchmark_cpu.yaml",
                num_datasets=16,
            ),
            ShapeAwareInvocationSpec(
                invocation_id="default_medium",
                config_ref="configs/default.yaml",
                num_datasets=56,
            ),
            ShapeAwareInvocationSpec(
                invocation_id="large_shape",
                config_ref="configs/benchmark_cuda_h100_large_shape.yaml",
                num_datasets=8,
            ),
        ),
    ),
)


def _issue_urls(*, materialization_issue: int, execution_issue: int) -> dict[str, str]:
    return {
        "materialization_issue": (
            f"https://github.com/bensonlee5/tab-foundry/issues/{materialization_issue}"
        ),
        "execution_issue": f"https://github.com/bensonlee5/tab-foundry/issues/{execution_issue}",
        "subsequent_filtering_policy_issue": (
            f"https://github.com/bensonlee5/tab-foundry/issues/{FILTERING_POLICY_ISSUE_NUMBER}"
        ),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if resolved == REPO_ROOT:
        return "."
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        pass
    try:
        return Path(os.path.relpath(resolved, REPO_ROOT)).as_posix()
    except ValueError:
        pass
    home = Path.home().resolve()
    try:
        return f"~/{resolved.relative_to(home).as_posix()}"
    except ValueError:
        return str(resolved)


def _sanitize_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_paths(item) for item in value]
    if isinstance(value, str):
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return _portable_path(candidate)
    return value


def _render_command(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd) if sys.platform == "win32" else " ".join(cmd)


def _anchor_manifest_bootstrap_command() -> str:
    return "\n".join(
        [
            f"{_portable_path(TAB_FOUNDRY_BIN)} dev data generate-manifest \\",
            f"  --dagzoo-root {_portable_path(DEFAULT_DAGZOO_ROOT)} \\",
            f"  --dagzoo-config {DEFAULT_DAGZOO_CONFIG_REF} \\",
            f"  --handoff-root {_portable_path(CURRENT_CORPUS_HANDOFF_ROOT)} \\",
            f"  --out-manifest {_portable_path(ANCHOR_MANIFEST_PATH)} \\",
            f"  --num-datasets {DEFAULT_NUM_DATASETS} \\",
            f"  --seed {DEFAULT_SEED} \\",
            f"  --device {DEFAULT_DEVICE} \\",
            f"  --hardware-policy {DEFAULT_HARDWARE_POLICY}",
        ]
    )


def _anchor_manifest_bootstrap_error(reason: str) -> RuntimeError:
    command = _anchor_manifest_bootstrap_command()
    return RuntimeError(
        f"{reason}\n"
        f"TF-RD-013 support materialization requires a fresh current-corpus manifest at "
        f"{_portable_path(ANCHOR_MANIFEST_PATH)}.\n"
        "Stale absolute-path local snapshots are unsupported for this flow.\n\n"
        "From the repo root, generate it with:\n"
        f"{command}\n\n"
        "Then rerun `scripts/materialize_tf_rd_013_support.py --variant size-ladder --force`."
    )


def _resolve_manifest_record_path(*, manifest_path: Path, raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (manifest_path.parent / candidate).resolve()


def _assert_anchor_manifest_ready(manifest_path: Path) -> None:
    if not manifest_path.exists():
        raise _anchor_manifest_bootstrap_error(
            f"anchor manifest does not exist: {manifest_path}"
        )
    try:
        table = pq.read_table(manifest_path, columns=list(ANCHOR_MANIFEST_REQUIRED_COLUMNS))
    except Exception as exc:  # pragma: no cover - defensive bootstrap guidance
        raise _anchor_manifest_bootstrap_error(
            f"anchor manifest could not be inspected: {manifest_path}: {exc}"
        ) from exc

    missing_examples: list[str] = []
    for row_index, record in enumerate(table.to_pylist()):
        for column in ANCHOR_MANIFEST_REQUIRED_COLUMNS:
            raw_value = record.get(column)
            if not isinstance(raw_value, str) or not raw_value.strip():
                missing_examples.append(f"row {row_index} has invalid {column}={raw_value!r}")
            else:
                resolved = _resolve_manifest_record_path(
                    manifest_path=manifest_path,
                    raw_path=raw_value,
                )
                if not resolved.exists():
                    missing_examples.append(
                        f"row {row_index} {column} -> {resolved}"
                    )
            if len(missing_examples) >= 5:
                break
        if len(missing_examples) >= 5:
            break

    if missing_examples:
        details = "\n".join(f"  - {example}" for example in missing_examples)
        raise _anchor_manifest_bootstrap_error(
            "anchor manifest exists but its backing files are missing or invalid:\n"
            f"{details}"
        )


def _run_checked(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    rendered = _render_command(cmd)
    print(f"+ (cd {cwd} && {rendered})")
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _manifest_inspect(manifest_path: Path) -> dict[str, Any]:
    completed = _run_checked(
        [str(TAB_FOUNDRY_BIN), "data", "manifest-inspect", "--manifest", str(manifest_path), "--json"],
        cwd=REPO_ROOT,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError(f"manifest inspect output must be an object for {manifest_path}")
    return _sanitize_paths(payload)


def _ensure_absent(path: Path, *, force: bool) -> None:
    if not path.exists():
        return
    if not force:
        raise RuntimeError(f"path already exists, re-run with --force to replace it: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _materialization_paths(local_output_root: Path) -> MaterializationPaths:
    generated_source_root = local_output_root / "generated_source"
    curated_realdata_root = local_output_root / "curated_realdata"
    curated_openml_baseline_root = curated_realdata_root / "openml_baseline"
    return MaterializationPaths(
        local_output_root=local_output_root,
        generated_source_root=generated_source_root,
        generated_dir=generated_source_root / "generated",
        generated_manifest_path=generated_source_root / "manifest.parquet",
        handoff_manifest_path=generated_source_root / "handoff_manifest.json",
        curated_realdata_root=curated_realdata_root,
        curated_openml_baseline_root=curated_openml_baseline_root,
        curated_openml_baseline_data_root=curated_openml_baseline_root / "packed_shards",
        curated_openml_baseline_manifest_path=curated_openml_baseline_root / "manifest.parquet",
    )


def _multi_invocation_surface_paths(
    local_output_root: Path,
    surface_spec: MultiInvocationSurfaceSpec,
) -> MultiInvocationSurfacePaths:
    surface_root = local_output_root / surface_spec.surface_name
    invocation_root = surface_root / "invocations"
    invocation_paths = tuple(
        ShapeAwareInvocationPaths(
            invocation_id=spec.invocation_id,
            invocation_root=invocation_root / spec.invocation_id,
            generated_dir=invocation_root / spec.invocation_id / "generated",
            handoff_manifest_path=invocation_root / spec.invocation_id / "handoff_manifest.json",
        )
        for spec in surface_spec.invocations
    )
    return MultiInvocationSurfacePaths(
        surface_name=surface_spec.surface_name,
        surface_root=surface_root,
        manifest_path=surface_root / "manifest.parquet",
        invocation_paths=invocation_paths,
    )


def _shape_aware_materialization_paths(
    local_output_root: Path,
) -> ShapeAwareMaterializationPaths:
    dagzoo_surface = _multi_invocation_surface_paths(
        local_output_root,
        SHAPE_AWARE_SURFACE_SPEC,
    )
    curated_realdata_root = local_output_root / "curated_realdata"
    curated_openml_baseline_root = curated_realdata_root / "openml_baseline"
    return ShapeAwareMaterializationPaths(
        local_output_root=local_output_root,
        dagzoo_surface_root=dagzoo_surface.surface_root,
        dagzoo_manifest_path=dagzoo_surface.manifest_path,
        invocation_paths=dagzoo_surface.invocation_paths,
        curated_realdata_root=curated_realdata_root,
        curated_openml_baseline_root=curated_openml_baseline_root,
        curated_openml_baseline_data_root=curated_openml_baseline_root / "packed_shards",
        curated_openml_baseline_manifest_path=curated_openml_baseline_root / "manifest.parquet",
    )


def _size_ladder_materialization_paths(
    local_output_root: Path,
) -> SizeLadderMaterializationPaths:
    return SizeLadderMaterializationPaths(
        local_output_root=local_output_root,
        dagzoo_surfaces=tuple(
            _multi_invocation_surface_paths(local_output_root, surface_spec)
            for surface_spec in SIZE_LADDER_SURFACE_SPECS
        ),
    )


def _dagzoo_commands(paths: MaterializationPaths) -> list[str]:
    return [
        (
            'cd "$DAGZOO_ROOT" && uv run dagzoo generate '
            f'--config {DEFAULT_DAGZOO_CONFIG_REF} '
            f'--handoff-root "$TAB_FOUNDRY_ROOT/{_portable_path(paths.generated_source_root)}" '
            f"--num-datasets {DEFAULT_NUM_DATASETS} "
            f"--seed {DEFAULT_SEED} "
            f"--device {DEFAULT_DEVICE} "
            "--hardware-policy none"
        ),
        (
            'cd "$TAB_FOUNDRY_ROOT" && ./.venv/bin/tab-foundry dev data build-manifest '
            f"--data-root {_portable_path(paths.generated_dir)} "
            f"--out-manifest {_portable_path(paths.generated_manifest_path)}"
        ),
    ]


def _multi_invocation_generate_command(
    spec: ShapeAwareInvocationSpec,
    paths: ShapeAwareInvocationPaths,
) -> str:
    return (
        'cd "$DAGZOO_ROOT" && uv run dagzoo generate '
        f"--config {spec.config_ref} "
        f'--handoff-root "$TAB_FOUNDRY_ROOT/{_portable_path(paths.invocation_root)}" '
        f"--num-datasets {spec.num_datasets} "
        f"--seed {spec.seed} "
        f"--device {spec.device} "
        f"--hardware-policy {spec.hardware_policy}"
    )


def _multi_invocation_build_manifest_command(
    *,
    invocation_paths: tuple[ShapeAwareInvocationPaths, ...],
    manifest_path: Path,
) -> str:
    roots_rendered = ", ".join(
        _portable_path(invocation_paths.generated_dir)
        for invocation_paths in invocation_paths
    )
    return (
        "build_manifest(data_roots=["
        f"{roots_rendered}"
        f"], out_path={_portable_path(manifest_path)})"
    )


def _curated_commands(
    paths: MaterializationPaths | ShapeAwareMaterializationPaths,
    *,
    variant: str,
) -> list[str]:
    variant_flag = "" if variant == "historical" else f" --variant {variant}"
    return [
        (
            'cd "$TAB_FOUNDRY_ROOT" && ./.venv/bin/python '
            f"scripts/materialize_tf_rd_013_support.py{variant_flag} --force"
        ),
        (
            'cd "$TAB_FOUNDRY_ROOT" && ./.venv/bin/tab-foundry dev data build-manifest '
            f"--data-root {_portable_path(paths.curated_openml_baseline_data_root)} "
            f"--out-manifest {_portable_path(paths.curated_openml_baseline_manifest_path)}"
        ),
    ]


def _comparison_entry(
    *,
    comparison_id: str,
    left_id: str,
    right_id: str,
    manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    left = manifests[left_id]
    right = manifests[right_id]
    left_inspection = {key: value for key, value in left["inspection"].items() if key != "manifest_path"}
    right_inspection = {key: value for key, value in right["inspection"].items() if key != "manifest_path"}
    differences = compare_jsonlike_payloads(left_inspection, right_inspection)
    return {
        "comparison_id": comparison_id,
        "status": "available",
        "left_manifest_id": left_id,
        "right_manifest_id": right_id,
        "left_manifest_path": left["manifest_path"],
        "right_manifest_path": right["manifest_path"],
        "difference_count": len(differences),
        "differences": differences,
    }


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "dataset"


def _build_split_table(rows: list[tuple[int, np.ndarray, np.ndarray]]) -> pa.Table:
    dataset_indices: list[int] = []
    row_indices: list[int] = []
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    for dataset_index, x, y in rows:
        for row_index in range(int(x.shape[0])):
            dataset_indices.append(int(dataset_index))
            row_indices.append(int(row_index))
            x_rows.append(np.asarray(x[row_index], dtype=np.float32).tolist())
            y_rows.append(int(y[row_index]))

    return pa.table(
        {
            "dataset_index": pa.array(dataset_indices, type=pa.int64()),
            "row_index": pa.array(row_indices, type=pa.int64()),
            "x": pa.array(x_rows, type=pa.list_(pa.float32())),
            "y": pa.array(y_rows, type=pa.int64()),
        }
    )


def _write_packed_shard(
    shard_dir: Path,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(_build_split_table([(0, x_train, y_train)]), shard_dir / "train.parquet")
    pq.write_table(_build_split_table([(0, x_test, y_test)]), shard_dir / "test.parquet")

    payload = {
        "dataset_index": 0,
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_features": int(x_train.shape[1]),
        "feature_types": ["floating"] * int(x_train.shape[1]),
        "metadata": metadata,
    }
    with (shard_dir / "metadata.ndjson").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _split_prepared_task(
    prepared: PreparedOpenMLBenchmarkTask,
    *,
    split_seed: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    split_kwargs: dict[str, Any] = {
        "test_size": float(test_size),
        "random_state": int(split_seed),
    }
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            prepared.x,
            prepared.y,
            stratify=prepared.y,
            **split_kwargs,
        )
        return x_train, x_test, y_train, y_test, "stratified"
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            prepared.x,
            prepared.y,
            stratify=None,
            **split_kwargs,
        )
        return x_train, x_test, y_train, y_test, "unstratified_fallback"


def _materialize_curated_openml_baseline(
    *,
    paths: MaterializationPaths | ShapeAwareMaterializationPaths,
    bundle_path: Path,
) -> dict[str, Any]:
    bundle, allow_missing_values = load_benchmark_bundle_for_execution(bundle_path)
    selection = bundle["selection"]
    task_ids = [int(task_id) for task_id in bundle["task_ids"]]
    task_summaries: list[dict[str, Any]] = []

    paths.curated_openml_baseline_data_root.mkdir(parents=True, exist_ok=True)
    for task_order, task_id in enumerate(task_ids, start=1):
        prepared = prepare_openml_benchmark_task(
            int(task_id),
            new_instances=int(selection["new_instances"]),
            task_type=str(selection["task_type"]),
        )
        x_train, x_test, y_train, y_test, split_mode = _split_prepared_task(
            prepared,
            split_seed=DEFAULT_COMPARATOR_SPLIT_SEED,
            test_size=DEFAULT_COMPARATOR_TEST_SIZE,
        )
        metadata = {
            "config": {"dataset": {"task": "classification"}},
            "filter": {"mode": "deferred", "status": "not_run"},
            "seed": DEFAULT_COMPARATOR_SPLIT_SEED,
            "n_features": int(prepared.x.shape[1]),
            "n_classes": int(np.unique(prepared.y).size),
            "source_platform": "openml",
            "benchmark_bundle": {
                "name": str(bundle["name"]),
                "source_path": DEFAULT_BENCHMARK_BUNDLE_REF,
                "task_id": int(task_id),
                "allow_missing_values": bool(allow_missing_values),
            },
            "openml": {
                "task_id": int(task_id),
                "dataset_name": str(prepared.dataset_name),
            },
            "split_policy": {
                "name": "deterministic_holdout",
                "test_size": DEFAULT_COMPARATOR_TEST_SIZE,
                "seed": DEFAULT_COMPARATOR_SPLIT_SEED,
                "mode": split_mode,
            },
        }
        shard_dir = paths.curated_openml_baseline_data_root / f"shard_{task_order:05d}_{_slugify(prepared.dataset_name)}"
        _write_packed_shard(
            shard_dir,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            metadata=metadata,
        )
        task_summaries.append(
            {
                "task_id": int(task_id),
                "dataset_name": str(prepared.dataset_name),
                "n_rows": int(prepared.x.shape[0]),
                "n_features": int(prepared.x.shape[1]),
                "n_classes": int(np.unique(prepared.y).size),
                "n_train": int(x_train.shape[0]),
                "n_test": int(x_test.shape[0]),
                "split_mode": split_mode,
                "shard_dir": _portable_path(shard_dir),
            }
        )

    curated_manifest_cmd = [
        str(TAB_FOUNDRY_BIN),
        "dev",
        "data",
        "build-manifest",
        "--data-root",
        str(paths.curated_openml_baseline_data_root),
        "--out-manifest",
        str(paths.curated_openml_baseline_manifest_path),
    ]
    curated_manifest_completed = _run_checked(curated_manifest_cmd, cwd=REPO_ROOT)

    return {
        "bundle_summary": benchmark_bundle_summary(bundle, source_path=bundle_path),
        "task_summaries": task_summaries,
        "allow_missing_values": bool(allow_missing_values),
        "build_manifest_completed": curated_manifest_completed,
    }


def _assert_common_inputs(*, dagzoo_root: Path) -> Path:
    if not dagzoo_root.exists():
        raise RuntimeError(f"dagzoo root does not exist: {dagzoo_root}")
    if not TAB_FOUNDRY_BIN.exists():
        raise RuntimeError(f"tab-foundry CLI not found at {TAB_FOUNDRY_BIN}")
    _assert_anchor_manifest_ready(ANCHOR_MANIFEST_PATH)
    benchmark_bundle_path = REPO_ROOT / DEFAULT_BENCHMARK_BUNDLE_REF
    if not benchmark_bundle_path.exists():
        raise RuntimeError(f"benchmark bundle does not exist: {benchmark_bundle_path}")
    return benchmark_bundle_path


def _historical_materialization_summary(
    *,
    paths: MaterializationPaths,
    manifests: dict[str, dict[str, Any]],
    handoff_payload: dict[str, Any],
    dagzoo_commands: list[str],
    curated_commands: list[str],
    generate_completed: subprocess.CompletedProcess[str],
    generated_manifest_completed: subprocess.CompletedProcess[str],
    curated_result: dict[str, Any],
) -> dict[str, Any]:
    issue_urls = _issue_urls(
        materialization_issue=FIRST_MATERIALIZATION_ISSUE_NUMBER,
        execution_issue=FIRST_EXECUTION_ISSUE_NUMBER,
    )
    dagzoo_provenance = {
        "corpus_variant": "dagzoo_generated_source",
        "comparator_role": "promoted_anchor_candidate",
        "commands": dagzoo_commands,
        "config_refs": [DEFAULT_DAGZOO_CONFIG_REF],
        "curated_root_lineage": [],
        "materialization_issue": FIRST_MATERIALIZATION_ISSUE_NUMBER,
    }
    curated_openml_provenance = {
        "surface_variant": "curated_realdata_openml_baseline",
        "comparator_role": "evidence_only_openml_baseline",
        "commands": curated_commands,
        "benchmark_bundle": curated_result["bundle_summary"],
        "task_ids": [int(task["task_id"]) for task in curated_result["task_summaries"]],
        "dataset_names": [str(task["dataset_name"]) for task in curated_result["task_summaries"]],
        "task_summaries": curated_result["task_summaries"],
        "approved_external_augmentations": [],
        "split_policy": {
            "name": "deterministic_holdout",
            "test_size": DEFAULT_COMPARATOR_TEST_SIZE,
            "seed": DEFAULT_COMPARATOR_SPLIT_SEED,
            "preferred_mode": "stratified",
            "fallback_mode": "unstratified_fallback",
        },
        "execution_issue": FIRST_EXECUTION_ISSUE_NUMBER,
    }
    return {
        "schema": "tab-foundry-tf-rd-013-materialization-summary-v4",
        "generated_at_utc": _utc_now(),
        "issues": {
            "materialization_issue": FIRST_MATERIALIZATION_ISSUE_NUMBER,
            "materialization_issue_url": issue_urls["materialization_issue"],
            "execution_issue": FIRST_EXECUTION_ISSUE_NUMBER,
            "execution_issue_url": issue_urls["execution_issue"],
            "subsequent_filtering_policy_issue": FILTERING_POLICY_ISSUE_NUMBER,
            "subsequent_filtering_policy_issue_url": issue_urls["subsequent_filtering_policy_issue"],
        },
        "env_hints": {
            "dagzoo_root": "../dagzoo",
            "tab_foundry_root": ".",
            "anchor_bootstrap_command": _anchor_manifest_bootstrap_command(),
        },
        "config_refs": {
            "dagzoo": [DEFAULT_DAGZOO_CONFIG_REF],
            "tab_foundry": [DEFAULT_MANIFEST_CONFIG_REF],
            "benchmark_bundle": [DEFAULT_BENCHMARK_BUNDLE_REF],
        },
        "artifacts": {
            "local_output_root": _portable_path(paths.local_output_root),
            "generated_source_root": _portable_path(paths.generated_source_root),
            "generated_dir": _portable_path(paths.generated_dir),
            "generated_source_manifest_path": _portable_path(paths.generated_manifest_path),
            "generated_source_manifest_sha256": _sha256_path(paths.generated_manifest_path),
            "handoff_manifest_path": _portable_path(paths.handoff_manifest_path),
            "handoff_manifest_sha256": _sha256_path(paths.handoff_manifest_path),
            "curated_openml_baseline_root": _portable_path(paths.curated_openml_baseline_root),
            "curated_openml_baseline_data_root": _portable_path(paths.curated_openml_baseline_data_root),
            "curated_openml_baseline_manifest_path": _portable_path(paths.curated_openml_baseline_manifest_path),
            "curated_openml_baseline_manifest_sha256": _sha256_path(paths.curated_openml_baseline_manifest_path),
        },
        "steps": [
            {
                "step_id": "dagzoo_generate",
                "cwd": "$DAGZOO_ROOT",
                "command": dagzoo_commands[0],
                "status": "completed",
                "returncode": generate_completed.returncode,
            },
            {
                "step_id": "build_manifest_generated_source",
                "cwd": "$TAB_FOUNDRY_ROOT",
                "command": dagzoo_commands[1],
                "status": "completed",
                "returncode": generated_manifest_completed.returncode,
            },
            {
                "step_id": "prepare_openml_baseline_shards",
                "cwd": "$TAB_FOUNDRY_ROOT",
                "command": curated_commands[0],
                "status": "completed",
                "returncode": 0,
            },
            {
                "step_id": "build_manifest_curated_realdata_openml_baseline",
                "cwd": "$TAB_FOUNDRY_ROOT",
                "command": curated_commands[1],
                "status": "completed",
                "returncode": int(curated_result["build_manifest_completed"].returncode),
            },
        ],
        "handoff": {
            "identity": handoff_payload.get("identity"),
            "artifacts_relative": handoff_payload.get("artifacts_relative"),
            "defaults": handoff_payload.get("defaults"),
            "hardware": handoff_payload.get("hardware"),
            "summary": handoff_payload.get("summary"),
            "throughput": handoff_payload.get("throughput"),
            "generate_invocation": handoff_payload.get("generate_invocation"),
        },
        "surfaces": {
            "dagzoo_generated_source": {
                "state": "materialized_local_only",
                "manifest_path": manifests["dagzoo_generated_source"]["manifest_path"],
                "manifest_sha256": manifests["dagzoo_generated_source"]["manifest_sha256"],
                "dagzoo_provenance": dagzoo_provenance,
            },
            "curated_realdata_openml_baseline": {
                "state": "materialized_local_only",
                "manifest_path": manifests["curated_realdata_openml_baseline"]["manifest_path"],
                "manifest_sha256": manifests["curated_realdata_openml_baseline"]["manifest_sha256"],
                "curated_realdata_provenance": curated_openml_provenance,
            },
        },
    }


def _historical_manifest_characteristics_summary(
    *,
    manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "tab-foundry-tf-rd-013-manifest-characteristics-summary-v4",
        "generated_at_utc": _utc_now(),
        "issues": {
            "materialization_issue": FIRST_MATERIALIZATION_ISSUE_NUMBER,
            "execution_issue": FIRST_EXECUTION_ISSUE_NUMBER,
            "subsequent_filtering_policy_issue": FILTERING_POLICY_ISSUE_NUMBER,
        },
        "manifests": manifests,
        "comparisons": {
            "anchor_vs_generated_source": _comparison_entry(
                comparison_id="anchor_vs_generated_source",
                left_id="anchor_manifest_default",
                right_id="dagzoo_generated_source",
                manifests=manifests,
            ),
            "anchor_vs_curated_realdata_openml_baseline": _comparison_entry(
                comparison_id="anchor_vs_curated_realdata_openml_baseline",
                left_id="anchor_manifest_default",
                right_id="curated_realdata_openml_baseline",
                manifests=manifests,
            ),
            "dagzoo_generated_source_vs_curated_realdata_openml_baseline": _comparison_entry(
                comparison_id="dagzoo_generated_source_vs_curated_realdata_openml_baseline",
                left_id="dagzoo_generated_source",
                right_id="curated_realdata_openml_baseline",
                manifests=manifests,
            ),
        },
    }


def _materialize_historical_support(*, dagzoo_root: Path, force: bool) -> int:
    benchmark_bundle_path = _assert_common_inputs(dagzoo_root=dagzoo_root)
    dagzoo_config_path = dagzoo_root / DEFAULT_DAGZOO_CONFIG_REF
    if not dagzoo_config_path.exists():
        raise RuntimeError(f"dagzoo config does not exist: {dagzoo_config_path}")

    paths = _materialization_paths(LOCAL_OUTPUT_ROOT)
    _ensure_absent(paths.generated_source_root, force=force)
    _ensure_absent(paths.curated_openml_baseline_root, force=force)
    paths.generated_source_root.mkdir(parents=True, exist_ok=True)
    paths.curated_openml_baseline_root.mkdir(parents=True, exist_ok=True)

    generate_cmd = [
        "uv",
        "run",
        "dagzoo",
        "generate",
        "--config",
        str(dagzoo_config_path),
        "--handoff-root",
        str(paths.generated_source_root),
        "--num-datasets",
        str(DEFAULT_NUM_DATASETS),
        "--seed",
        str(DEFAULT_SEED),
        "--device",
        DEFAULT_DEVICE,
        "--hardware-policy",
        "none",
    ]
    generated_manifest_cmd = [
        str(TAB_FOUNDRY_BIN),
        "dev",
        "data",
        "build-manifest",
        "--data-root",
        str(paths.generated_dir),
        "--out-manifest",
        str(paths.generated_manifest_path),
    ]

    generate_completed = _run_checked(generate_cmd, cwd=dagzoo_root)
    generated_manifest_completed = _run_checked(generated_manifest_cmd, cwd=REPO_ROOT)
    curated_result = _materialize_curated_openml_baseline(
        paths=paths,
        bundle_path=benchmark_bundle_path,
    )

    handoff_payload = _sanitize_paths(_load_json(paths.handoff_manifest_path))
    dagzoo_commands = _dagzoo_commands(paths)
    curated_commands = _curated_commands(paths, variant="historical")
    anchor_inspection = _manifest_inspect(ANCHOR_MANIFEST_PATH)
    generated_inspection = _manifest_inspect(paths.generated_manifest_path)
    curated_inspection = _manifest_inspect(paths.curated_openml_baseline_manifest_path)

    manifests = {
        "anchor_manifest_default": {
            "status": "available",
            "manifest_path": _portable_path(ANCHOR_MANIFEST_PATH),
            "manifest_sha256": _sha256_path(ANCHOR_MANIFEST_PATH),
            "inspection": anchor_inspection,
        },
        "dagzoo_generated_source": {
            "status": "available",
            "manifest_path": _portable_path(paths.generated_manifest_path),
            "manifest_sha256": _sha256_path(paths.generated_manifest_path),
            "inspection": generated_inspection,
        },
        "curated_realdata_openml_baseline": {
            "status": "available",
            "manifest_path": _portable_path(paths.curated_openml_baseline_manifest_path),
            "manifest_sha256": _sha256_path(paths.curated_openml_baseline_manifest_path),
            "inspection": curated_inspection,
        },
    }

    materialization_summary = _historical_materialization_summary(
        paths=paths,
        manifests=manifests,
        handoff_payload=handoff_payload,
        dagzoo_commands=dagzoo_commands,
        curated_commands=curated_commands,
        generate_completed=generate_completed,
        generated_manifest_completed=generated_manifest_completed,
        curated_result=curated_result,
    )
    manifest_characteristics_summary = _historical_manifest_characteristics_summary(
        manifests=manifests,
    )

    _write_json(SUPPORT_ROOT / "materialization_summary.json", materialization_summary)
    _write_json(
        SUPPORT_ROOT / "manifest_characteristics_summary.json",
        manifest_characteristics_summary,
    )
    print("Wrote support summaries:")
    print(f"  {SUPPORT_ROOT / 'materialization_summary.json'}")
    print(f"  {SUPPORT_ROOT / 'manifest_characteristics_summary.json'}")
    return 0


def _multi_invocation_invocation_summary(
    *,
    spec: ShapeAwareInvocationSpec,
    paths: ShapeAwareInvocationPaths,
    handoff_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "invocation_id": spec.invocation_id,
        "command": _multi_invocation_generate_command(spec, paths),
        "config_ref": spec.config_ref,
        "num_datasets": spec.num_datasets,
        "seed": spec.seed,
        "device": spec.device,
        "hardware_policy": spec.hardware_policy,
        "invocation_root": _portable_path(paths.invocation_root),
        "generated_dir": _portable_path(paths.generated_dir),
        "handoff_manifest_path": _portable_path(paths.handoff_manifest_path),
        "handoff_manifest_sha256": _sha256_path(paths.handoff_manifest_path),
        "identity": handoff_payload.get("identity"),
        "artifacts_relative": handoff_payload.get("artifacts_relative"),
        "defaults": handoff_payload.get("defaults"),
        "hardware": handoff_payload.get("hardware"),
        "summary": handoff_payload.get("summary"),
        "throughput": handoff_payload.get("throughput"),
        "generate_invocation": handoff_payload.get("generate_invocation"),
    }


def _shape_aware_materialization_summary(
    *,
    paths: ShapeAwareMaterializationPaths,
    manifests: dict[str, dict[str, Any]],
    invocation_summaries: list[dict[str, Any]],
    generate_steps: list[dict[str, Any]],
    curated_commands: list[str],
    curated_result: dict[str, Any],
    combined_summary: Any,
) -> dict[str, Any]:
    issue_urls = _issue_urls(
        materialization_issue=SHAPE_AWARE_ISSUE_NUMBER,
        execution_issue=SHAPE_AWARE_ISSUE_NUMBER,
    )
    dagzoo_commands = [str(invocation["command"]) for invocation in invocation_summaries]
    dagzoo_commands.append(
        _multi_invocation_build_manifest_command(
            invocation_paths=paths.invocation_paths,
            manifest_path=paths.dagzoo_manifest_path,
        )
    )
    dagzoo_provenance = {
        "corpus_variant": "dagzoo_shape_aware_multi_invocation",
        "comparator_role": "promoted_anchor_candidate",
        "commands": dagzoo_commands,
        "config_refs": [spec.config_ref for spec in SHAPE_AWARE_INVOCATIONS],
        "curated_root_lineage": [],
        "materialization_issue": SHAPE_AWARE_ISSUE_NUMBER,
        "invocations": [
            {
                "invocation_id": invocation["invocation_id"],
                "config_ref": invocation["config_ref"],
                "num_datasets": invocation["num_datasets"],
                "seed": invocation["seed"],
                "device": invocation["device"],
                "hardware_policy": invocation["hardware_policy"],
                "command": invocation["command"],
                "generated_dir": invocation["generated_dir"],
                "handoff_manifest_path": invocation["handoff_manifest_path"],
                "handoff_manifest_sha256": invocation["handoff_manifest_sha256"],
                "identity": invocation["identity"],
                "summary": invocation["summary"],
            }
            for invocation in invocation_summaries
        ],
    }
    curated_openml_provenance = {
        "surface_variant": "curated_realdata_openml_baseline",
        "comparator_role": "evidence_only_openml_baseline",
        "commands": curated_commands,
        "benchmark_bundle": curated_result["bundle_summary"],
        "task_ids": [int(task["task_id"]) for task in curated_result["task_summaries"]],
        "dataset_names": [str(task["dataset_name"]) for task in curated_result["task_summaries"]],
        "task_summaries": curated_result["task_summaries"],
        "approved_external_augmentations": [],
        "split_policy": {
            "name": "deterministic_holdout",
            "test_size": DEFAULT_COMPARATOR_TEST_SIZE,
            "seed": DEFAULT_COMPARATOR_SPLIT_SEED,
            "preferred_mode": "stratified",
            "fallback_mode": "unstratified_fallback",
        },
        "execution_issue": SHAPE_AWARE_ISSUE_NUMBER,
    }
    return {
        "schema": "tab-foundry-tf-rd-013-shape-aware-materialization-summary-v1",
        "generated_at_utc": _utc_now(),
        "issues": {
            "materialization_issue": SHAPE_AWARE_ISSUE_NUMBER,
            "materialization_issue_url": issue_urls["materialization_issue"],
            "execution_issue": SHAPE_AWARE_ISSUE_NUMBER,
            "execution_issue_url": issue_urls["execution_issue"],
            "epic_issue": EPIC_ISSUE_NUMBER,
            "epic_issue_url": f"https://github.com/bensonlee5/tab-foundry/issues/{EPIC_ISSUE_NUMBER}",
            "downstream_training_surface_issue": DOWNSTREAM_TRAINING_SURFACE_ISSUE_NUMBER,
            "downstream_training_surface_issue_url": (
                "https://github.com/bensonlee5/tab-foundry/issues/"
                f"{DOWNSTREAM_TRAINING_SURFACE_ISSUE_NUMBER}"
            ),
            "subsequent_filtering_policy_issue": FILTERING_POLICY_ISSUE_NUMBER,
            "subsequent_filtering_policy_issue_url": issue_urls["subsequent_filtering_policy_issue"],
        },
        "env_hints": {
            "dagzoo_root": "../dagzoo",
            "tab_foundry_root": ".",
        },
        "config_refs": {
            "dagzoo": [spec.config_ref for spec in SHAPE_AWARE_INVOCATIONS],
            "tab_foundry": [DEFAULT_MANIFEST_CONFIG_REF],
            "benchmark_bundle": [DEFAULT_BENCHMARK_BUNDLE_REF],
        },
        "artifacts": {
            "local_output_root": _portable_path(paths.local_output_root),
            "dagzoo_shape_aware_root": _portable_path(paths.dagzoo_surface_root),
            "dagzoo_shape_aware_manifest_path": _portable_path(paths.dagzoo_manifest_path),
            "dagzoo_shape_aware_manifest_sha256": _sha256_path(paths.dagzoo_manifest_path),
            "curated_openml_baseline_root": _portable_path(paths.curated_openml_baseline_root),
            "curated_openml_baseline_data_root": _portable_path(paths.curated_openml_baseline_data_root),
            "curated_openml_baseline_manifest_path": _portable_path(paths.curated_openml_baseline_manifest_path),
            "curated_openml_baseline_manifest_sha256": _sha256_path(paths.curated_openml_baseline_manifest_path),
        },
        "shape_program": {
            "kind": "config_ladder",
            "assembly_policy": "build_manifest(data_roots=[...]) without dagzoo_handoff_manifest_path",
            "notes": [
                "The broader dagzoo follow-up uses three explicit config-backed invocations rather than a single default-config generate run.",
                "The combined manifest intentionally omits single-handoff manifest metadata so the multi-invocation surface stays schema-stable.",
                "Keep filtering policy out of this follow-up; issue 124 remains a later decision only if the broader read shows a predictability problem.",
            ],
            "invocation_ids": [spec.invocation_id for spec in SHAPE_AWARE_INVOCATIONS],
            "input_roots": [_portable_path(invocation.generated_dir) for invocation in paths.invocation_paths],
        },
        "steps": (
            generate_steps
            + [
                {
                    "step_id": "build_manifest_dagzoo_shape_aware_multi_invocation",
                    "cwd": "$TAB_FOUNDRY_ROOT",
                    "command": _multi_invocation_build_manifest_command(
                        invocation_paths=paths.invocation_paths,
                        manifest_path=paths.dagzoo_manifest_path,
                    ),
                    "status": "completed",
                    "returncode": 0,
                },
                {
                    "step_id": "prepare_openml_baseline_shards",
                    "cwd": "$TAB_FOUNDRY_ROOT",
                    "command": curated_commands[0],
                    "status": "completed",
                    "returncode": 0,
                },
                {
                    "step_id": "build_manifest_curated_realdata_openml_baseline",
                    "cwd": "$TAB_FOUNDRY_ROOT",
                    "command": curated_commands[1],
                    "status": "completed",
                    "returncode": int(curated_result["build_manifest_completed"].returncode),
                },
            ]
        ),
        "assembly": {
            "combined_manifest_path": _portable_path(paths.dagzoo_manifest_path),
            "combined_manifest_sha256": _sha256_path(paths.dagzoo_manifest_path),
            "invocation_count": len(invocation_summaries),
            "input_roots": [_portable_path(invocation.generated_dir) for invocation in paths.invocation_paths],
            "persisted_summary": {
                "discovered_records": int(combined_summary.discovered_records),
                "excluded_records": int(combined_summary.excluded_records),
                "excluded_for_missing_values": int(combined_summary.excluded_for_missing_values),
                "filter_policy": str(combined_summary.filter_policy),
                "missing_value_policy": str(combined_summary.missing_value_policy),
                "total_records": int(combined_summary.total_records),
                "train_records": int(combined_summary.train_records),
                "val_records": int(combined_summary.val_records),
                "test_records": int(combined_summary.test_records),
                "filter_status_counts": dict(combined_summary.filter_status_counts),
                "missing_value_status_counts": dict(combined_summary.missing_value_status_counts),
            },
            "notes": [
                "Each invocation keeps its own handoff manifest and identity record.",
                "The follow-up sweep trains against one merged manifest that combines all generated roots.",
            ],
        },
        "surfaces": {
            "dagzoo_shape_aware_multi_invocation": {
                "state": "materialized_local_only",
                "manifest_path": manifests["dagzoo_shape_aware_multi_invocation"]["manifest_path"],
                "manifest_sha256": manifests["dagzoo_shape_aware_multi_invocation"]["manifest_sha256"],
                "dagzoo_provenance": dagzoo_provenance,
                "invocation_handoffs": invocation_summaries,
            },
            "curated_realdata_openml_baseline": {
                "state": "materialized_local_only",
                "manifest_path": manifests["curated_realdata_openml_baseline"]["manifest_path"],
                "manifest_sha256": manifests["curated_realdata_openml_baseline"]["manifest_sha256"],
                "curated_realdata_provenance": curated_openml_provenance,
            },
        },
    }


def _shape_aware_manifest_characteristics_summary(
    *,
    manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "tab-foundry-tf-rd-013-shape-aware-manifest-characteristics-summary-v1",
        "generated_at_utc": _utc_now(),
        "issues": {
            "materialization_issue": SHAPE_AWARE_ISSUE_NUMBER,
            "execution_issue": SHAPE_AWARE_ISSUE_NUMBER,
            "epic_issue": EPIC_ISSUE_NUMBER,
            "downstream_training_surface_issue": DOWNSTREAM_TRAINING_SURFACE_ISSUE_NUMBER,
            "subsequent_filtering_policy_issue": FILTERING_POLICY_ISSUE_NUMBER,
        },
        "manifests": manifests,
        "comparisons": {
            "anchor_vs_dagzoo_shape_aware_multi_invocation": _comparison_entry(
                comparison_id="anchor_vs_dagzoo_shape_aware_multi_invocation",
                left_id="anchor_manifest_default",
                right_id="dagzoo_shape_aware_multi_invocation",
                manifests=manifests,
            ),
            "anchor_vs_curated_realdata_openml_baseline": _comparison_entry(
                comparison_id="anchor_vs_curated_realdata_openml_baseline",
                left_id="anchor_manifest_default",
                right_id="curated_realdata_openml_baseline",
                manifests=manifests,
            ),
            "dagzoo_shape_aware_multi_invocation_vs_curated_realdata_openml_baseline": _comparison_entry(
                comparison_id="dagzoo_shape_aware_multi_invocation_vs_curated_realdata_openml_baseline",
                left_id="dagzoo_shape_aware_multi_invocation",
                right_id="curated_realdata_openml_baseline",
                manifests=manifests,
            ),
        },
    }


def _size_ladder_materialization_summary(
    *,
    corpus_records: list[dict[str, Any]],
) -> dict[str, Any]:
    issue_urls = _issue_urls(
        materialization_issue=SIZE_LADDER_ISSUE_NUMBER,
        execution_issue=SIZE_LADDER_ISSUE_NUMBER,
    )
    surfaces: dict[str, dict[str, Any]] = {}
    recipes: list[dict[str, Any]] = []
    config_refs: set[str] = set()
    for record in corpus_records:
        manifest = record["manifest"]
        dagzoo_provenance = record["dagzoo_provenance"]
        invocations = dagzoo_provenance.get("invocations", [])
        config_refs.update(
            str(invocation["config_ref"])
            for invocation in invocations
            if isinstance(invocation, dict) and invocation.get("config_ref") is not None
        )
        recipe = record["recipe"]
        requested_dataset_count = sum(
            int(invocation["num_datasets"])
            for invocation in recipe.get("invocations", [])
            if isinstance(invocation, dict)
        )
        recipes.append(
            {
                "recipe_id": record["recipe_id"],
                "corpus_ref": record["corpus_ref"],
                "surface_label": record["surface_label"],
                "requested_dataset_count": int(requested_dataset_count),
                "invocations": [
                    {
                        "invocation_id": invocation["invocation_id"],
                        "config_ref": invocation["config_ref"],
                        "num_datasets": invocation["num_datasets"],
                    }
                    for invocation in recipe.get("invocations", [])
                    if isinstance(invocation, dict)
                ],
            }
        )
        surfaces[str(record["surface_label"])] = {
            "state": "materialized_local_only",
            "recipe_id": record["recipe_id"],
            "corpus_id": record["corpus_id"],
            "corpus_ref": record["corpus_ref"],
            "manifest_path": manifest["manifest_path"],
            "manifest_sha256": manifest["manifest_sha256"],
            "dagzoo_provenance": dagzoo_provenance,
        }

    return {
        "schema": "tab-foundry-tf-rd-013-size-ladder-materialization-summary-v1",
        "generated_at_utc": _utc_now(),
        "issues": {
            "materialization_issue": SIZE_LADDER_ISSUE_NUMBER,
            "materialization_issue_url": issue_urls["materialization_issue"],
            "execution_issue": SIZE_LADDER_ISSUE_NUMBER,
            "execution_issue_url": issue_urls["execution_issue"],
            "epic_issue": EPIC_ISSUE_NUMBER,
            "epic_issue_url": f"https://github.com/bensonlee5/tab-foundry/issues/{EPIC_ISSUE_NUMBER}",
            "downstream_training_surface_issue": DOWNSTREAM_TRAINING_SURFACE_ISSUE_NUMBER,
            "downstream_training_surface_issue_url": (
                "https://github.com/bensonlee5/tab-foundry/issues/"
                f"{DOWNSTREAM_TRAINING_SURFACE_ISSUE_NUMBER}"
            ),
            "subsequent_filtering_policy_issue": FILTERING_POLICY_ISSUE_NUMBER,
            "subsequent_filtering_policy_issue_url": issue_urls["subsequent_filtering_policy_issue"],
        },
        "env_hints": {
            "dagzoo_root": "../dagzoo",
            "tab_foundry_root": ".",
        },
        "config_refs": {
            "dagzoo": sorted(config_refs),
            "tab_foundry": ["tab-foundry data corpus materialize"],
        },
        "artifacts": {
            "local_output_root": _portable_path(REPO_ROOT / "outputs" / "corpora"),
        },
        "size_ladder_program": {
            "kind": "shape_aware_size_ladder",
            "notes": [
                "The size ladder now materializes through first-class corpus recipes under outputs/corpora/ instead of sweep-local dagzoo orchestration.",
                "Every ladder surface is generated on CPU with hardware-policy none so the comparison stays about corpus content rather than generation hardware.",
                f"Row 1 is a TF-RD-008-scale fresh current-corpus control recipe with {SIZE_LADDER_CONTROL_NUM_DATASETS} generated datasets rather than a replay of the stale 2026-02-22 absolute-path snapshot.",
                "Rows 2-4 remain the `small`, `medium`, and `large` shrunken shape-aware ladders, now sized to 20, 40, and 80 generated datasets under the same max_steps=2500 stop contract.",
            ],
            "recipes": recipes,
        },
        "surfaces": surfaces,
    }


def _size_ladder_manifest_characteristics_summary(
    *,
    corpus_records: list[dict[str, Any]],
) -> dict[str, Any]:
    manifests = {
        str(record["surface_label"]): {
            "status": "available",
            "recipe_id": record["recipe_id"],
            "corpus_id": record["corpus_id"],
            "corpus_ref": record["corpus_ref"],
            "manifest_path": record["manifest"]["manifest_path"],
            "manifest_sha256": record["manifest"]["manifest_sha256"],
            "inspection": record["manifest"]["inspection"],
        }
        for record in corpus_records
    }
    return {
        "schema": "tab-foundry-tf-rd-013-size-ladder-manifest-characteristics-summary-v1",
        "generated_at_utc": _utc_now(),
        "issues": {
            "materialization_issue": SIZE_LADDER_ISSUE_NUMBER,
            "execution_issue": SIZE_LADDER_ISSUE_NUMBER,
            "epic_issue": EPIC_ISSUE_NUMBER,
            "downstream_training_surface_issue": DOWNSTREAM_TRAINING_SURFACE_ISSUE_NUMBER,
            "subsequent_filtering_policy_issue": FILTERING_POLICY_ISSUE_NUMBER,
        },
        "manifests": manifests,
        "comparisons": {
            "anchor_vs_dagzoo_shape_aware_size_small": corpus_compare_payload(
                left=CURRENT_CORPUS_RECIPE_ID,
                right="tf_rd_013_dagzoo_shape_aware_size_small_v1",
            ),
            "anchor_vs_dagzoo_shape_aware_size_medium": corpus_compare_payload(
                left=CURRENT_CORPUS_RECIPE_ID,
                right="tf_rd_013_dagzoo_shape_aware_size_medium_v1",
            ),
            "anchor_vs_dagzoo_shape_aware_size_large": corpus_compare_payload(
                left=CURRENT_CORPUS_RECIPE_ID,
                right="tf_rd_013_dagzoo_shape_aware_size_large_v1",
            ),
            "dagzoo_shape_aware_size_small_vs_dagzoo_shape_aware_size_medium": corpus_compare_payload(
                left="tf_rd_013_dagzoo_shape_aware_size_small_v1",
                right="tf_rd_013_dagzoo_shape_aware_size_medium_v1",
            ),
            "dagzoo_shape_aware_size_medium_vs_dagzoo_shape_aware_size_large": corpus_compare_payload(
                left="tf_rd_013_dagzoo_shape_aware_size_medium_v1",
                right="tf_rd_013_dagzoo_shape_aware_size_large_v1",
            ),
            "dagzoo_shape_aware_size_small_vs_dagzoo_shape_aware_size_large": corpus_compare_payload(
                left="tf_rd_013_dagzoo_shape_aware_size_small_v1",
                right="tf_rd_013_dagzoo_shape_aware_size_large_v1",
            ),
        },
    }


def _materialize_shape_aware_support(*, dagzoo_root: Path, force: bool) -> int:
    benchmark_bundle_path = _assert_common_inputs(dagzoo_root=dagzoo_root)
    for spec in SHAPE_AWARE_INVOCATIONS:
        dagzoo_config_path = dagzoo_root / spec.config_ref
        if not dagzoo_config_path.exists():
            raise RuntimeError(f"dagzoo config does not exist: {dagzoo_config_path}")

    paths = _shape_aware_materialization_paths(SHAPE_AWARE_LOCAL_OUTPUT_ROOT)
    _ensure_absent(paths.dagzoo_surface_root, force=force)
    _ensure_absent(paths.curated_openml_baseline_root, force=force)
    paths.dagzoo_surface_root.mkdir(parents=True, exist_ok=True)
    paths.curated_openml_baseline_root.mkdir(parents=True, exist_ok=True)

    invocation_summaries: list[dict[str, Any]] = []
    generate_steps: list[dict[str, Any]] = []
    for spec, invocation_paths in zip(
        SHAPE_AWARE_INVOCATIONS,
        paths.invocation_paths,
        strict=True,
    ):
        generate_cmd = [
            "uv",
            "run",
            "dagzoo",
            "generate",
            "--config",
            str((dagzoo_root / spec.config_ref).resolve()),
            "--handoff-root",
            str(invocation_paths.invocation_root),
            "--num-datasets",
            str(spec.num_datasets),
            "--seed",
            str(spec.seed),
            "--device",
            spec.device,
            "--hardware-policy",
            spec.hardware_policy,
        ]
        generate_completed = _run_checked(generate_cmd, cwd=dagzoo_root)
        handoff_payload = _sanitize_paths(_load_json(invocation_paths.handoff_manifest_path))
        invocation_summaries.append(
            _multi_invocation_invocation_summary(
                spec=spec,
                paths=invocation_paths,
                handoff_payload=handoff_payload,
            )
        )
        generate_steps.append(
            {
                "step_id": f"dagzoo_generate_{spec.invocation_id}",
                "cwd": "$DAGZOO_ROOT",
                "command": _multi_invocation_generate_command(spec, invocation_paths),
                "status": "completed",
                "returncode": generate_completed.returncode,
            }
        )

    combined_summary = build_manifest(
        data_roots=[invocation.generated_dir for invocation in paths.invocation_paths],
        out_path=paths.dagzoo_manifest_path,
    )
    curated_result = _materialize_curated_openml_baseline(
        paths=paths,
        bundle_path=benchmark_bundle_path,
    )
    curated_commands = _curated_commands(paths, variant="shape-aware")

    anchor_inspection = _manifest_inspect(ANCHOR_MANIFEST_PATH)
    generated_inspection = _manifest_inspect(paths.dagzoo_manifest_path)
    curated_inspection = _manifest_inspect(paths.curated_openml_baseline_manifest_path)
    manifests = {
        "anchor_manifest_default": {
            "status": "available",
            "manifest_path": _portable_path(ANCHOR_MANIFEST_PATH),
            "manifest_sha256": _sha256_path(ANCHOR_MANIFEST_PATH),
            "inspection": anchor_inspection,
        },
        "dagzoo_shape_aware_multi_invocation": {
            "status": "available",
            "manifest_path": _portable_path(paths.dagzoo_manifest_path),
            "manifest_sha256": _sha256_path(paths.dagzoo_manifest_path),
            "inspection": generated_inspection,
        },
        "curated_realdata_openml_baseline": {
            "status": "available",
            "manifest_path": _portable_path(paths.curated_openml_baseline_manifest_path),
            "manifest_sha256": _sha256_path(paths.curated_openml_baseline_manifest_path),
            "inspection": curated_inspection,
        },
    }
    materialization_summary = _shape_aware_materialization_summary(
        paths=paths,
        manifests=manifests,
        invocation_summaries=invocation_summaries,
        generate_steps=generate_steps,
        curated_commands=curated_commands,
        curated_result=curated_result,
        combined_summary=combined_summary,
    )
    manifest_characteristics_summary = _shape_aware_manifest_characteristics_summary(
        manifests=manifests,
    )

    _write_json(
        SHAPE_AWARE_SUPPORT_ROOT / "materialization_summary.json",
        materialization_summary,
    )
    _write_json(
        SHAPE_AWARE_SUPPORT_ROOT / "manifest_characteristics_summary.json",
        manifest_characteristics_summary,
    )
    print("Wrote support summaries:")
    print(f"  {SHAPE_AWARE_SUPPORT_ROOT / 'materialization_summary.json'}")
    print(f"  {SHAPE_AWARE_SUPPORT_ROOT / 'manifest_characteristics_summary.json'}")
    return 0


def _materialize_size_ladder_support(*, dagzoo_root: Path, force: bool) -> int:
    if not dagzoo_root.exists():
        raise RuntimeError(f"dagzoo root does not exist: {dagzoo_root}")
    if not TAB_FOUNDRY_BIN.exists():
        raise RuntimeError(f"tab-foundry CLI not found at {TAB_FOUNDRY_BIN}")

    corpus_records = [
        materialize_corpus_recipe(
            recipe_id=recipe_id,
            dagzoo_root=dagzoo_root,
            force=force,
        )
        for recipe_id in SIZE_LADDER_RECIPE_IDS
    ]
    materialization_summary = _size_ladder_materialization_summary(
        corpus_records=corpus_records,
    )
    manifest_characteristics_summary = _size_ladder_manifest_characteristics_summary(
        corpus_records=corpus_records,
    )

    _write_json(
        SIZE_LADDER_SUPPORT_ROOT / "materialization_summary.json",
        materialization_summary,
    )
    _write_json(
        SIZE_LADDER_SUPPORT_ROOT / "manifest_characteristics_summary.json",
        manifest_characteristics_summary,
    )
    print("Wrote support summaries:")
    print(f"  {SIZE_LADDER_SUPPORT_ROOT / 'materialization_summary.json'}")
    print(f"  {SIZE_LADDER_SUPPORT_ROOT / 'manifest_characteristics_summary.json'}")
    return 0


def materialize_support(*, variant: str, dagzoo_root: Path, force: bool) -> int:
    if variant == "historical":
        return _materialize_historical_support(dagzoo_root=dagzoo_root, force=force)
    if variant == "shape-aware":
        return _materialize_shape_aware_support(dagzoo_root=dagzoo_root, force=force)
    if variant == "size-ladder":
        return _materialize_size_ladder_support(dagzoo_root=dagzoo_root, force=force)
    raise RuntimeError(f"unsupported materialization variant: {variant}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dagzoo-root", default=str(DEFAULT_DAGZOO_ROOT), help="Local dagzoo checkout root")
    parser.add_argument("--force", action="store_true", help="Replace existing local TF-RD-013 outputs")
    parser.add_argument(
        "--variant",
        choices=("historical", "shape-aware", "size-ladder"),
        default="historical",
        help="Support bundle variant to materialize",
    )
    args = parser.parse_args(argv)

    dagzoo_root = Path(str(args.dagzoo_root)).expanduser().resolve()
    return materialize_support(
        variant=str(args.variant),
        dagzoo_root=dagzoo_root,
        force=bool(args.force),
    )


if __name__ == "__main__":
    raise SystemExit(main())
