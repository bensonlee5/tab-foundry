#!/usr/bin/env python3
"""Materialize unfiltered TF-RD-013 dagzoo support artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAGZOO_ROOT = REPO_ROOT.parent / "dagzoo"
DEFAULT_DAGZOO_CONFIG_REF = "configs/default.yaml"
DEFAULT_MANIFEST_CONFIG_REF = "configs/data/default.yaml"
DEFAULT_NUM_DATASETS = 200
DEFAULT_SEED = 1
DEFAULT_DEVICE = "cpu"
MATERIALIZATION_ISSUE_NUMBER = 120
FILTERING_POLICY_ISSUE_NUMBER = 124
LOCAL_OUTPUT_ROOT = REPO_ROOT / "outputs" / "staged_ladder_support" / "tf_rd_013"
SUPPORT_ROOT = (
    REPO_ROOT
    / "reference"
    / "system_delta_sweeps"
    / "tf_rd_013_data_source_contract_v1"
    / "support"
)
ANCHOR_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "default.parquet"
TAB_FOUNDRY_BIN = REPO_ROOT / ".venv" / "bin" / "tab-foundry"
ISSUE_URLS = {
    "materialization_issue": f"https://github.com/bensonlee5/tab-foundry/issues/{MATERIALIZATION_ISSUE_NUMBER}",
    "subsequent_filtering_policy_issue": (
        f"https://github.com/bensonlee5/tab-foundry/issues/{FILTERING_POLICY_ISSUE_NUMBER}"
    ),
}


@dataclass(frozen=True)
class MaterializationPaths:
    local_output_root: Path
    generated_source_root: Path
    generated_dir: Path
    generated_manifest_path: Path
    handoff_manifest_path: Path


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
    return MaterializationPaths(
        local_output_root=local_output_root,
        generated_source_root=generated_source_root,
        generated_dir=generated_source_root / "generated",
        generated_manifest_path=generated_source_root / "manifest.parquet",
        handoff_manifest_path=generated_source_root / "handoff_manifest.json",
    )


def _queue_commands(paths: MaterializationPaths) -> list[str]:
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
            'cd "$TAB_FOUNDRY_ROOT" && ./.venv/bin/tab-foundry data build-manifest '
            f"--data-root {_portable_path(paths.generated_dir)} "
            f"--out-manifest {_portable_path(paths.generated_manifest_path)}"
        ),
    ]


def _compare_payloads(left: Any, right: Any, *, prefix: str = "") -> dict[str, dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: dict[str, dict[str, Any]] = {}
        for key in sorted(set(left.keys()) | set(right.keys())):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            differences.update(_compare_payloads(left.get(key), right.get(key), prefix=next_prefix))
        return differences
    if left == right:
        return {}
    return {prefix: {"left": left, "right": right}}


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
    differences = _compare_payloads(left_inspection, right_inspection)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dagzoo-root", default=str(DEFAULT_DAGZOO_ROOT), help="Local dagzoo checkout root")
    parser.add_argument("--force", action="store_true", help="Replace existing local TF-RD-013 outputs")
    args = parser.parse_args(argv)

    dagzoo_root = Path(str(args.dagzoo_root)).expanduser().resolve()
    if not dagzoo_root.exists():
        raise RuntimeError(f"dagzoo root does not exist: {dagzoo_root}")
    if not TAB_FOUNDRY_BIN.exists():
        raise RuntimeError(f"tab-foundry CLI not found at {TAB_FOUNDRY_BIN}")
    dagzoo_config_path = dagzoo_root / DEFAULT_DAGZOO_CONFIG_REF
    if not dagzoo_config_path.exists():
        raise RuntimeError(f"dagzoo config does not exist: {dagzoo_config_path}")
    if not ANCHOR_MANIFEST_PATH.exists():
        raise RuntimeError(f"anchor manifest does not exist: {ANCHOR_MANIFEST_PATH}")

    paths = _materialization_paths(LOCAL_OUTPUT_ROOT)
    _ensure_absent(paths.generated_source_root, force=bool(args.force))
    paths.generated_source_root.mkdir(parents=True, exist_ok=True)

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
        "data",
        "build-manifest",
        "--data-root",
        str(paths.generated_dir),
        "--out-manifest",
        str(paths.generated_manifest_path),
    ]

    generate_completed = _run_checked(generate_cmd, cwd=dagzoo_root)
    generated_manifest_completed = _run_checked(generated_manifest_cmd, cwd=REPO_ROOT)

    handoff_payload = _sanitize_paths(_load_json(paths.handoff_manifest_path))
    commands = _queue_commands(paths)
    anchor_inspection = _manifest_inspect(ANCHOR_MANIFEST_PATH)
    generated_inspection = _manifest_inspect(paths.generated_manifest_path)

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
    }

    dagzoo_provenance = {
        "corpus_variant": "dagzoo_generated_source",
        "comparator_role": "promoted_anchor_candidate",
        "commands": commands,
        "config_refs": [DEFAULT_DAGZOO_CONFIG_REF],
        "curated_root_lineage": [],
        "materialization_issue": MATERIALIZATION_ISSUE_NUMBER,
    }

    materialization_summary = {
        "schema": "tab-foundry-tf-rd-013-materialization-summary-v3",
        "generated_at_utc": _utc_now(),
        "issues": {
            "materialization_issue": MATERIALIZATION_ISSUE_NUMBER,
            "materialization_issue_url": ISSUE_URLS["materialization_issue"],
            "subsequent_filtering_policy_issue": FILTERING_POLICY_ISSUE_NUMBER,
            "subsequent_filtering_policy_issue_url": ISSUE_URLS["subsequent_filtering_policy_issue"],
        },
        "env_hints": {
            "dagzoo_root": "../dagzoo",
            "tab_foundry_root": ".",
        },
        "config_refs": {
            "dagzoo": [DEFAULT_DAGZOO_CONFIG_REF],
            "tab_foundry": [DEFAULT_MANIFEST_CONFIG_REF],
        },
        "artifacts": {
            "local_output_root": _portable_path(paths.local_output_root),
            "generated_source_root": _portable_path(paths.generated_source_root),
            "generated_dir": _portable_path(paths.generated_dir),
            "generated_source_manifest_path": _portable_path(paths.generated_manifest_path),
            "generated_source_manifest_sha256": _sha256_path(paths.generated_manifest_path),
            "handoff_manifest_path": _portable_path(paths.handoff_manifest_path),
            "handoff_manifest_sha256": _sha256_path(paths.handoff_manifest_path),
        },
        "steps": [
            {
                "step_id": "dagzoo_generate",
                "cwd": "$DAGZOO_ROOT",
                "command": commands[0],
                "status": "completed",
                "returncode": generate_completed.returncode,
            },
            {
                "step_id": "build_manifest_generated_source",
                "cwd": "$TAB_FOUNDRY_ROOT",
                "command": commands[1],
                "status": "completed",
                "returncode": generated_manifest_completed.returncode,
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
            }
        },
    }

    manifest_characteristics_summary = {
        "schema": "tab-foundry-tf-rd-013-manifest-characteristics-summary-v3",
        "generated_at_utc": _utc_now(),
        "issues": {
            "materialization_issue": MATERIALIZATION_ISSUE_NUMBER,
            "subsequent_filtering_policy_issue": FILTERING_POLICY_ISSUE_NUMBER,
        },
        "manifests": manifests,
        "comparisons": {
            "anchor_vs_generated_source": _comparison_entry(
                comparison_id="anchor_vs_generated_source",
                left_id="anchor_manifest_default",
                right_id="dagzoo_generated_source",
                manifests=manifests,
            )
        },
    }

    _write_json(SUPPORT_ROOT / "materialization_summary.json", materialization_summary)
    _write_json(SUPPORT_ROOT / "manifest_characteristics_summary.json", manifest_characteristics_summary)

    print("Wrote support summaries:")
    print(f"  {SUPPORT_ROOT / 'materialization_summary.json'}")
    print(f"  {SUPPORT_ROOT / 'manifest_characteristics_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
