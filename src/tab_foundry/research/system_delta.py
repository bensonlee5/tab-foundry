"""Queue and matrix helpers for the anchor-only system-delta sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence, cast

from omegaconf import OmegaConf

from tab_foundry.bench.benchmark_run_registry import (
    load_benchmark_run_registry,
    resolve_registry_path_value,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_queue_path() -> Path:
    return repo_root() / "reference" / "system_delta_queue.yaml"


def default_matrix_path() -> Path:
    return repo_root() / "reference" / "system_delta_matrix.md"


def default_registry_path() -> Path:
    return repo_root() / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"


def load_system_delta_queue(path: Path | None = None) -> dict[str, Any]:
    payload = OmegaConf.to_container(
        OmegaConf.load(path or default_queue_path()),
        resolve=True,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("system delta queue must decode to a mapping")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("system delta queue must include a non-empty rows list")
    return cast(dict[str, Any], payload)


def ordered_rows(queue: dict[str, Any]) -> list[dict[str, Any]]:
    rows = cast(list[dict[str, Any]], queue["rows"])
    return sorted(rows, key=lambda row: (int(row["order"]), str(row["delta_id"])))


def next_ready_row(queue: dict[str, Any]) -> dict[str, Any] | None:
    for row in ordered_rows(queue):
        if str(row.get("status", "")).strip().lower() == "ready":
            return row
    return None


def _metric_summary(run: dict[str, Any], anchor: dict[str, Any]) -> dict[str, str]:
    metrics = cast(dict[str, Any], run["tab_foundry_metrics"])
    anchor_metrics = cast(dict[str, Any], anchor["tab_foundry_metrics"])
    best = float(metrics["best_roc_auc"])
    final = float(metrics["final_roc_auc"])
    best_time = float(metrics["best_training_time"])
    final_time = float(metrics["final_training_time"])
    anchor_best = float(anchor_metrics["best_roc_auc"])
    anchor_final = float(anchor_metrics["final_roc_auc"])
    anchor_best_time = float(anchor_metrics["best_training_time"])
    anchor_final_time = float(anchor_metrics["final_training_time"])
    drift = final - best
    anchor_drift = anchor_final - anchor_best
    return {
        "best_roc_auc": f"{best:.4f}",
        "final_roc_auc": f"{final:.4f}",
        "final_minus_best": f"{drift:+.4f}",
        "delta_best_roc_auc": f"{best - anchor_best:+.4f}",
        "delta_final_roc_auc": f"{final - anchor_final:+.4f}",
        "delta_drift": f"{drift - anchor_drift:+.4f}",
        "delta_training_time": f"{final_time - anchor_final_time:+.1f}s",
        "final_training_time": f"{final_time:.1f}s",
        "best_training_time": f"{best_time:.1f}s",
        "delta_best_training_time": f"{best_time - anchor_best_time:+.1f}s",
    }


def _result_card_path(delta_id: str) -> Path:
    return repo_root() / "outputs" / "staged_ladder" / "research" / delta_id / "result_card.md"


def validate_system_delta_queue(
    queue: dict[str, Any],
    *,
    registry_path: Path | None = None,
) -> list[str]:
    issues: list[str] = []
    registry = load_benchmark_run_registry(registry_path or default_registry_path())
    runs = cast(dict[str, dict[str, Any]], registry["runs"])
    for row in ordered_rows(queue):
        status = str(row.get("status", "")).strip().lower()
        if status != "completed":
            continue
        delta_id = str(row["delta_id"])
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            issues.append(f"{delta_id}: completed rows must include run_id")
            continue
        run = runs.get(run_id)
        if run is None:
            issues.append(f"{delta_id}: run_id {run_id!r} is missing from the benchmark registry")
            continue
        result_card_path = _result_card_path(delta_id)
        if not result_card_path.exists():
            issues.append(f"{delta_id}: missing result card at {result_card_path}")
        training_surface_record_path = cast(dict[str, Any], run["artifacts"]).get(
            "training_surface_record_path"
        )
        if not isinstance(training_surface_record_path, str) or not training_surface_record_path.strip():
            issues.append(f"{delta_id}: run {run_id!r} is missing artifacts.training_surface_record_path")
        else:
            resolved = resolve_registry_path_value(training_surface_record_path)
            if not resolved.exists():
                issues.append(
                    f"{delta_id}: training surface artifact does not exist at {resolved}"
                )
    return issues


def render_system_delta_matrix(
    queue: dict[str, Any],
    *,
    registry_path: Path | None = None,
) -> str:
    registry = load_benchmark_run_registry(registry_path or default_registry_path())
    runs = cast(dict[str, dict[str, Any]], registry["runs"])
    anchor_run_id = str(queue["anchor_run_id"])
    anchor = runs.get(anchor_run_id)
    if anchor is None:
        raise RuntimeError(f"anchor_run_id {anchor_run_id!r} is missing from the benchmark registry")
    anchor_metrics = cast(dict[str, Any], anchor["tab_foundry_metrics"])
    upstream = cast(dict[str, Any], queue["upstream_reference"])
    anchor_surface = cast(dict[str, Any], queue["anchor_surface"])

    lines: list[str] = []
    lines.append("# System Delta Matrix")
    lines.append("")
    lines.append("This file is rendered from `reference/system_delta_queue.yaml` plus the canonical benchmark registry.")
    lines.append("")
    lines.append("## Locked Surface")
    lines.append("")
    lines.append(f"- Anchor run id: `{anchor_run_id}`")
    lines.append(f"- Benchmark bundle: `{queue['benchmark_bundle_path']}`")
    lines.append(f"- Control baseline id: `{queue['control_baseline_id']}`")
    lines.append(f"- Comparison policy: `{queue['comparison_policy']}`")
    lines.append(
        f"- Anchor metrics: best ROC AUC `{float(anchor_metrics['best_roc_auc']):.4f}`, "
        f"final ROC AUC `{float(anchor_metrics['final_roc_auc']):.4f}`, "
        f"final training time `{float(anchor_metrics['final_training_time']):.1f}s`"
    )
    lines.append("")
    lines.append("## Anchor Comparison")
    lines.append("")
    lines.append(f"Upstream reference: `{upstream['name']}` from `{upstream['model_source']}`.")
    lines.append("")
    lines.append("| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |")
    lines.append("| --- | --- | --- | --- |")
    for dimension_row in cast(list[dict[str, Any]], anchor_surface["dimension_table"]):
        lines.append(
            f"| {dimension_row['dimension']} | {dimension_row['upstream']} | "
            f"{dimension_row['anchor']} | {dimension_row['interpretation']} |"
        )
    lines.append("")
    lines.append("## Queue Summary")
    lines.append("")
    lines.append(
        "| Order | Delta | Family | Binary | Status | Legacy entanglement | Effective change | Next action |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for queue_row in ordered_rows(queue):
        lines.append(
            f"| {queue_row['order']} | `{queue_row['delta_id']}` | {queue_row['family']} | "
            f"{'yes' if queue_row.get('binary_applicable', False) else 'no'} | {queue_row['status']} | "
            f"{queue_row.get('entangled_legacy_stage', 'n/a')} | {queue_row['description']} | "
            f"{queue_row['next_action']} |"
        )
    lines.append("")
    lines.append("## Detailed Rows")
    lines.append("")
    for queue_row in ordered_rows(queue):
        delta_id = str(queue_row["delta_id"])
        run_id = queue_row.get("run_id")
        run = runs.get(run_id) if isinstance(run_id, str) else None
        lines.append(f"### {queue_row['order']}. `{delta_id}`")
        lines.append("")
        lines.append(f"- Dimension family: `{queue_row['dimension_family']}`")
        lines.append(f"- Status: `{queue_row['status']}`")
        lines.append(f"- Binary applicable: `{queue_row.get('binary_applicable', False)}`")
        lines.append(
            f"- Legacy cumulative entanglement: `{queue_row.get('entangled_legacy_stage', 'n/a')}`"
        )
        lines.append(f"- Description: {queue_row['description']}")
        lines.append(f"- Rationale: {queue_row['rationale']}")
        lines.append(f"- Hypothesis: {queue_row['hypothesis']}")
        lines.append(f"- Upstream delta: {queue_row['upstream_delta']}")
        lines.append(f"- Anchor delta: {queue_row['anchor_delta']}")
        lines.append(f"- Expected effect: {queue_row['expected_effect']}")
        lines.append(
            f"- Effective labels: model=`{queue_row['model']['stage_label']}`, "
            f"data=`{queue_row['data']['surface_label']}`, "
            f"preprocessing=`{queue_row['preprocessing']['surface_label']}`"
        )
        if queue_row["dimension_family"] == "model":
            lines.append(f"- Model overrides: `{queue_row['model'].get('module_overrides', {})}`")
        elif queue_row["dimension_family"] == "data":
            lines.append(f"- Data overrides: `{queue_row['data'].get('surface_overrides', {})}`")
        else:
            lines.append(
                f"- Preprocessing overrides: `{queue_row['preprocessing'].get('overrides', {})}`"
            )
        lines.append("- Parameter adequacy plan:")
        for plan_item in cast(list[str], queue_row.get("parameter_adequacy_plan", [])):
            lines.append(f"  - {plan_item}")
        if cast(list[str], queue_row.get("adequacy_knobs", [])):
            lines.append("- Adequacy knobs to dimension explicitly:")
            for adequacy_knob in cast(list[str], queue_row["adequacy_knobs"]):
                lines.append(f"  - {adequacy_knob}")
        lines.append(
            f"- Interpretation status: `{queue_row.get('interpretation_status', 'pending')}`"
        )
        lines.append(f"- Decision: `{queue_row.get('decision')}`")
        if cast(list[str], queue_row.get("confounders", [])):
            lines.append("- Confounders:")
            for confounder in cast(list[str], queue_row["confounders"]):
                lines.append(f"  - {confounder}")
        if cast(list[str], queue_row.get("notes", [])):
            lines.append("- Notes:")
            for note in cast(list[str], queue_row["notes"]):
                lines.append(f"  - {note}")
        lines.append(f"- Follow-up run ids: `{queue_row.get('followup_run_ids', [])}`")
        lines.append(f"- Result card path: `{_result_card_path(delta_id)}`")
        if run is None:
            lines.append("- Benchmark metrics: pending")
        else:
            metrics = _metric_summary(run, anchor)
            lines.append(
                f"- Registered run: `{run_id}` with best ROC AUC `{metrics['best_roc_auc']}`, "
                f"final ROC AUC `{metrics['final_roc_auc']}`, "
                f"final-minus-best `{metrics['final_minus_best']}`, "
                f"delta final ROC AUC `{metrics['delta_final_roc_auc']}`, "
                f"delta drift `{metrics['delta_drift']}`, "
                f"delta final training time `{metrics['delta_training_time']}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_matrix(path: Path, contents: str) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(contents, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the anchor-only system delta queue")
    parser.add_argument(
        "--queue-path",
        default=str(default_queue_path()),
        help="Path to reference/system_delta_queue.yaml",
    )
    parser.add_argument(
        "--registry-path",
        default=str(default_registry_path()),
        help="Path to benchmark_run_registry_v1.json",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List queue rows in order")
    subparsers.add_parser("next", help="Print the next ready row")
    render_parser = subparsers.add_parser("render", help="Render the markdown matrix")
    render_parser.add_argument(
        "--out-path",
        default=str(default_matrix_path()),
        help="Output markdown path",
    )
    subparsers.add_parser("validate", help="Validate completed rows")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    queue = load_system_delta_queue(Path(str(args.queue_path)))
    registry_path = Path(str(args.registry_path))
    if args.command == "list":
        for row in ordered_rows(queue):
            print(
                f"{int(row['order']):02d}  {row['status']:<18}  {row['dimension_family']:<13}  {row['delta_id']}"
            )
        return 0
    if args.command == "next":
        next_row = next_ready_row(queue)
        if next_row is None:
            print("No ready rows.")
            return 0
        print(OmegaConf.to_yaml(next_row, resolve=True).strip())
        return 0
    if args.command == "render":
        contents = render_system_delta_matrix(queue, registry_path=registry_path)
        _write_matrix(Path(str(args.out_path)), contents)
        print(f"Rendered system delta matrix to {Path(str(args.out_path)).expanduser().resolve()}")
        return 0
    if args.command == "validate":
        issues = validate_system_delta_queue(queue, registry_path=registry_path)
        if not issues:
            print("System delta queue validation passed.")
            return 0
        for issue in issues:
            print(issue)
        return 1
    raise RuntimeError(f"Unsupported command: {args.command!r}")
