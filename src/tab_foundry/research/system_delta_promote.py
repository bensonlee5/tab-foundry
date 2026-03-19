"""Promote a completed system-delta run to the canonical sweep anchor."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import sys
from pathlib import Path
from typing import Any, cast

from tab_foundry.bench.benchmark_run_registry import load_benchmark_run_registry
from tab_foundry.research import system_delta


_OBJECTIVE_RE = re.compile(
    r"Optimize for attributable evidence against the locked anchor\s+`[^`]+`, not for rapid base\s+promotion\.",
    re.MULTILINE,
)


@dataclass(frozen=True)
class PromotionPaths:
    index_path: Path
    catalog_path: Path
    sweeps_root: Path
    registry_path: Path
    program_path: Path

    @classmethod
    def default(cls) -> "PromotionPaths":
        return cls(
            index_path=system_delta.default_sweep_index_path(),
            catalog_path=system_delta.default_catalog_path(),
            sweeps_root=system_delta.default_sweeps_root(),
            registry_path=system_delta.default_registry_path(),
            program_path=system_delta.repo_root() / "program.md",
        )


def _render_sweep_matrix(*, sweep_id: str, paths: PromotionPaths) -> None:
    _ = system_delta.render_and_write_system_delta_matrix(
        sweep_id=sweep_id,
        registry_path=paths.registry_path,
        index_path=paths.index_path,
        catalog_path=paths.catalog_path,
        sweeps_root=paths.sweeps_root,
    )


def _read_yaml(path: Path, *, context: str) -> dict[str, Any]:
    return system_delta._load_yaml_mapping(path, context=context)


def _replace_prefixed_line(text: str, *, prefix: str, replacement: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(prefix)}.*$")
    if pattern.search(text) is None:
        raise RuntimeError(f"missing program line with prefix {prefix!r}")
    return pattern.sub(replacement, text, count=1)


def _update_program_contract(*, sweep_id: str, anchor_run_id: str, paths: PromotionPaths) -> None:
    registry = load_benchmark_run_registry(paths.registry_path)
    runs = cast(dict[str, Any], registry["runs"])
    entry = cast(dict[str, Any], runs[anchor_run_id])
    artifacts = cast(dict[str, Any], entry["artifacts"])
    sweep = system_delta.load_system_delta_sweep(
        sweep_id,
        index_path=paths.index_path,
        sweeps_root=paths.sweeps_root,
    )

    program_text = paths.program_path.read_text(encoding="utf-8")
    objective = (
        "Optimize for attributable evidence against the locked anchor\n"
        f"`{anchor_run_id}`, not for rapid base\n"
        "promotion."
    )
    if _OBJECTIVE_RE.search(program_text) is None:
        raise RuntimeError("program.md objective anchor block is missing or has drifted")
    program_text = _OBJECTIVE_RE.sub(objective, program_text, count=1)

    replacements = {
        "- active sweep id: ": f"- active sweep id: `{sweep_id}`",
        "- anchor run id: ": f"- anchor run id: `{anchor_run_id}`",
        "- anchor prior run: ": f"- anchor prior run: `{artifacts['run_dir']}`",
        "- anchor benchmark: ": f"- anchor benchmark: `{artifacts['benchmark_dir']}`",
        "- canonical benchmark bundle: ": f"- canonical benchmark bundle: `{sweep['benchmark_bundle_path']}`",
        "- canonical control baseline id: ": f"- canonical control baseline id: `{sweep['control_baseline_id']}`",
        "- canonical registry: ": "- canonical registry: `src/tab_foundry/bench/benchmark_run_registry_v1.json`",
        "- delta catalog: ": "- delta catalog: `reference/system_delta_catalog.yaml`",
        "- sweep index: ": "- sweep index: `reference/system_delta_sweeps/index.yaml`",
        "- canonical sweep queue: ": f"- canonical sweep queue: `reference/system_delta_sweeps/{sweep_id}/queue.yaml`",
        "- canonical sweep matrix: ": f"- canonical sweep matrix: `reference/system_delta_sweeps/{sweep_id}/matrix.md`",
        "- active queue alias: ": "- active queue alias: `reference/system_delta_queue.yaml`",
        "- active matrix alias: ": "- active matrix alias: `reference/system_delta_matrix.md`",
    }
    for prefix, replacement in replacements.items():
        program_text = _replace_prefixed_line(program_text, prefix=prefix, replacement=replacement)

    system_delta._write_text(paths.program_path, program_text)


def resolve_run_id_for_order(*, sweep_id: str, order: int, paths: PromotionPaths | None = None) -> str:
    resolved_paths = PromotionPaths.default() if paths is None else paths
    queue = system_delta.load_system_delta_queue_instance(
        sweep_id,
        index_path=resolved_paths.index_path,
        sweeps_root=resolved_paths.sweeps_root,
    )
    for row in cast(list[dict[str, Any]], queue["rows"]):
        if int(row["order"]) != int(order):
            continue
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise RuntimeError(
                f"sweep {sweep_id!r} row {order} does not have a completed run_id to promote"
            )
        return run_id
    raise RuntimeError(f"sweep {sweep_id!r} does not contain queue order {order}")


def promote_anchor(
    *,
    sweep_id: str,
    anchor_run_id: str,
    set_active: bool = False,
    paths: PromotionPaths | None = None,
) -> dict[str, str]:
    resolved_paths = PromotionPaths.default() if paths is None else paths
    normalized_sweep_id = str(sweep_id).strip()
    normalized_anchor_run_id = str(anchor_run_id).strip()
    if not normalized_sweep_id:
        raise RuntimeError("sweep_id must be non-empty")
    if not normalized_anchor_run_id:
        raise RuntimeError("anchor_run_id must be non-empty")

    _ = system_delta._anchor_context_from_registry_run(
        anchor_run_id=normalized_anchor_run_id,
        registry_path=resolved_paths.registry_path,
    )

    sweep_path = system_delta.sweep_metadata_path(
        normalized_sweep_id,
        sweeps_root=resolved_paths.sweeps_root,
    )
    sweep = _read_yaml(sweep_path, context=f"sweep {normalized_sweep_id!r}")
    sweep["anchor_run_id"] = normalized_anchor_run_id
    sweep["anchor_context"] = system_delta._anchor_context_from_registry_run(
        anchor_run_id=normalized_anchor_run_id,
        registry_path=resolved_paths.registry_path,
    )
    system_delta._write_yaml(sweep_path, sweep)

    index = system_delta.load_system_delta_index(resolved_paths.index_path)
    sweeps = cast(dict[str, Any], index["sweeps"])
    if normalized_sweep_id not in sweeps:
        raise RuntimeError(f"unknown sweep_id: {normalized_sweep_id}")
    cast(dict[str, Any], sweeps[normalized_sweep_id])["anchor_run_id"] = normalized_anchor_run_id
    if set_active:
        index["active_sweep_id"] = normalized_sweep_id
    system_delta._write_yaml(resolved_paths.index_path, index)

    _render_sweep_matrix(sweep_id=normalized_sweep_id, paths=resolved_paths)

    active_sweep_id = str(index["active_sweep_id"])
    if active_sweep_id == normalized_sweep_id:
        _ = system_delta.sync_active_sweep_aliases(
            sweep_id=normalized_sweep_id,
            index_path=resolved_paths.index_path,
            catalog_path=resolved_paths.catalog_path,
            registry_path=resolved_paths.registry_path,
            sweeps_root=resolved_paths.sweeps_root,
        )
        _update_program_contract(
            sweep_id=normalized_sweep_id,
            anchor_run_id=normalized_anchor_run_id,
            paths=resolved_paths,
        )

    return {
        "sweep_id": normalized_sweep_id,
        "anchor_run_id": normalized_anchor_run_id,
        "sweep_path": str(sweep_path.resolve()),
        "index_path": str(resolved_paths.index_path.resolve()),
        "matrix_path": str(
            system_delta.sweep_matrix_path(
                normalized_sweep_id,
                sweeps_root=resolved_paths.sweeps_root,
            ).resolve()
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote a completed system-delta run to the sweep anchor")
    parser.add_argument("--sweep-id", required=True, help="Sweep id whose anchor should be updated")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id", help="Benchmark registry run id to promote")
    target.add_argument("--order", type=int, help="Queue order whose run_id should be promoted")
    parser.add_argument("--set-active", action="store_true", help="Also mark this sweep as the active sweep")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = PromotionPaths.default()
    run_id = (
        str(args.run_id)
        if args.run_id is not None
        else resolve_run_id_for_order(sweep_id=str(args.sweep_id), order=int(args.order), paths=paths)
    )
    result = promote_anchor(
        sweep_id=str(args.sweep_id),
        anchor_run_id=run_id,
        set_active=bool(args.set_active),
        paths=paths,
    )
    print(
        "Promotion complete.",
        f"sweep_id={result['sweep_id']}",
        f"anchor_run_id={result['anchor_run_id']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
