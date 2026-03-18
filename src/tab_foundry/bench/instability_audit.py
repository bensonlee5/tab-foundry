"""Audit existing queue runs for scalar instability signals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from tab_foundry.bench.artifacts import load_history, write_json
from tab_foundry.bench.benchmark_run_registry import (
    default_benchmark_run_registry_path,
    load_benchmark_run_registry,
    resolve_registry_path_value,
)
from tab_foundry.training.instability import history_loss_summary


DEFAULT_SWEEP_ID = "binary_md_v1"
DEFAULT_ANCHOR_RUN_ID = "01_nano_exact_md_prior_parity_fix_binary_medium_v1"

_RUN_ID_PATTERN = re.compile(r"^- (?:primary )?run id: `([^`]+)`$", re.MULTILINE)
_DECISION_PATTERN = re.compile(r"^- decision recommendation: `([^`]+)`$", re.MULTILINE)
_NEXT_ACTION_PATTERN = re.compile(r"^- recommended next action: `([^`]+)`$", re.MULTILINE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return payload


def _result_card_index(research_root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for result_card in sorted(research_root.glob("**/result_card.md")):
        text = result_card.read_text(encoding="utf-8")
        run_id_match = _RUN_ID_PATTERN.search(text)
        if run_id_match is None:
            continue
        run_id = run_id_match.group(1)
        decision_match = _DECISION_PATTERN.search(text)
        next_action_match = _NEXT_ACTION_PATTERN.search(text)
        index[run_id] = {
            "path": str(result_card.resolve()),
            "decision_recommendation": None if decision_match is None else decision_match.group(1),
            "next_action": None if next_action_match is None else next_action_match.group(1),
        }
    return index


def _finite_history_values(history: list[dict[str, Any]], key: str) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for record in history:
        raw = record.get(key)
        if raw is None:
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append((int(record["step"]), value))
    return values


def _severity(max_grad_norm: float | None) -> str:
    if max_grad_norm is None:
        return "unknown"
    if max_grad_norm >= 1.0e3:
        return "extreme"
    if max_grad_norm >= 1.0e2:
        return "high"
    if max_grad_norm >= 10.0:
        return "elevated"
    return "moderate"


def _run_entry(
    *,
    run_id: str,
    history_path: Path,
    comparison_summary_path: Path | None,
    result_card_payload: Mapping[str, Any] | None,
    is_anchor: bool,
) -> dict[str, Any]:
    history = load_history(history_path)
    grad_norms = _finite_history_values(history, "grad_norm")
    train_loss_deltas = _finite_history_values(history, "train_loss_delta")
    loss_summary = history_loss_summary(history)
    peak_grad = None if not grad_norms else max(grad_norms, key=lambda item: item[1])
    peak_loss_delta = None if not train_loss_deltas else max(train_loss_deltas, key=lambda item: abs(item[1]))
    comparison_summary = None if comparison_summary_path is None else _load_json(comparison_summary_path)
    benchmark_payload = None
    if comparison_summary is not None:
        tab_foundry = comparison_summary.get("tab_foundry")
        if isinstance(tab_foundry, Mapping):
            benchmark_payload = {
                "best_roc_auc": tab_foundry.get("best_roc_auc"),
                "final_roc_auc": tab_foundry.get("final_roc_auc"),
                "final_log_loss": tab_foundry.get("final_log_loss"),
                "final_brier_score": tab_foundry.get("final_brier_score"),
                "best_crps": tab_foundry.get("best_crps"),
                "final_crps": tab_foundry.get("final_crps"),
                "final_avg_pinball_loss": tab_foundry.get("final_avg_pinball_loss"),
                "final_picp_90": tab_foundry.get("final_picp_90"),
                "best_to_final_roc_auc_delta": tab_foundry.get("best_to_final_roc_auc_delta"),
                "best_to_final_crps_delta": tab_foundry.get("best_to_final_crps_delta"),
            }

    max_grad_norm = None if peak_grad is None else float(peak_grad[1])
    instability_score = 0.0 if max_grad_norm is None else float(max_grad_norm)
    if loss_summary["max_abs_train_loss_delta"] is not None:
        instability_score += 10.0 * float(loss_summary["max_abs_train_loss_delta"])

    return {
        "run_id": run_id,
        "is_anchor": bool(is_anchor),
        "history_path": str(history_path.resolve()),
        "comparison_summary_path": None
        if comparison_summary_path is None or not comparison_summary_path.exists()
        else str(comparison_summary_path.resolve()),
        "result_card_path": None if result_card_payload is None else result_card_payload.get("path"),
        "decision_recommendation": None
        if result_card_payload is None
        else result_card_payload.get("decision_recommendation"),
        "next_action": None if result_card_payload is None else result_card_payload.get("next_action"),
        "history": {
            "step_count": int(len(history)),
            "max_grad_norm": max_grad_norm,
            "peak_grad_step": None if peak_grad is None else int(peak_grad[0]),
            "mean_grad_norm": None
            if not grad_norms
            else float(sum(value for _, value in grad_norms) / float(len(grad_norms))),
            "final_grad_norm": None if not grad_norms else float(grad_norms[-1][1]),
            "max_abs_train_loss_delta": loss_summary["max_abs_train_loss_delta"],
            "peak_loss_delta_step": None if peak_loss_delta is None else int(peak_loss_delta[0]),
            "train_loss_variance": loss_summary["train_loss_variance"],
        },
        "benchmark": benchmark_payload,
        "instability_score": float(instability_score),
        "severity": _severity(max_grad_norm),
    }


def _discover_runs(
    *,
    staged_ladder_root: Path,
    sweep_id: str,
    anchor_run_id: str,
    registry_path: Path,
) -> list[dict[str, Any]]:
    research_root = staged_ladder_root / "research"
    result_cards = _result_card_index(research_root)
    entries: list[dict[str, Any]] = []

    for history_path in sorted(staged_ladder_root.glob(f"sd_{sweep_id}_*/train/train_history.jsonl")):
        run_root = history_path.parent.parent
        run_id = run_root.name
        entries.append(
            _run_entry(
                run_id=run_id,
                history_path=history_path,
                comparison_summary_path=run_root / "benchmark" / "comparison_summary.json",
                result_card_payload=result_cards.get(run_id),
                is_anchor=False,
            )
        )

    registry = load_benchmark_run_registry(registry_path)
    anchor_entry = None
    if isinstance(registry.get("runs"), Mapping):
        anchor_entry = registry["runs"].get(anchor_run_id)
    if isinstance(anchor_entry, Mapping):
        raw_artifacts = anchor_entry.get("artifacts")
        if isinstance(raw_artifacts, Mapping):
            raw_run_dir = raw_artifacts.get("run_dir")
            raw_benchmark_summary = raw_artifacts.get("comparison_summary_path")
            if isinstance(raw_run_dir, str) and raw_run_dir.strip():
                run_dir = resolve_registry_path_value(raw_run_dir)
                history_path = run_dir / "train_history.jsonl"
                comparison_summary_path = (
                    resolve_registry_path_value(raw_benchmark_summary)
                    if isinstance(raw_benchmark_summary, str) and raw_benchmark_summary.strip()
                    else None
                )
                if history_path.exists():
                    entries.insert(
                        0,
                        _run_entry(
                            run_id=anchor_run_id,
                            history_path=history_path,
                            comparison_summary_path=comparison_summary_path,
                            result_card_payload=result_cards.get(anchor_run_id),
                            is_anchor=True,
                        ),
                    )
    return entries


def build_instability_audit(
    *,
    staged_ladder_root: Path,
    sweep_id: str = DEFAULT_SWEEP_ID,
    anchor_run_id: str = DEFAULT_ANCHOR_RUN_ID,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Build one audit payload from existing queue-run outputs."""

    resolved_root = staged_ladder_root.expanduser().resolve()
    resolved_registry_path = (registry_path or default_benchmark_run_registry_path()).expanduser().resolve()
    entries = _discover_runs(
        staged_ladder_root=resolved_root,
        sweep_id=sweep_id,
        anchor_run_id=anchor_run_id,
        registry_path=resolved_registry_path,
    )
    ranked = sorted(
        [entry for entry in entries if not entry["is_anchor"]],
        key=lambda entry: (
            -float(entry["instability_score"]),
            str(entry["run_id"]),
        ),
    )
    recommendations: list[dict[str, Any]] = []
    anchor_entry = next((entry for entry in entries if entry["is_anchor"]), None)
    if anchor_entry is not None:
        recommendations.append(
            {
                "run_id": str(anchor_entry["run_id"]),
                "reason": "Anchor reference rerun with module telemetry to compare against the first-pass queue outputs.",
            }
        )
    for entry in ranked[:3]:
        history = entry["history"]
        recommendations.append(
            {
                "run_id": str(entry["run_id"]),
                "reason": (
                    "Highest scalar instability signal from the first pass: "
                    f"max_grad_norm={history['max_grad_norm']}, "
                    f"max_abs_train_loss_delta={history['max_abs_train_loss_delta']}."
                ),
            }
        )
    return {
        "schema": "tab-foundry-instability-audit-v1",
        "generated_at_utc": _utc_now(),
        "sweep_id": str(sweep_id),
        "anchor_run_id": str(anchor_run_id),
        "staged_ladder_root": str(resolved_root),
        "registry_path": str(resolved_registry_path),
        "run_count": int(len(entries)),
        "recommendations": recommendations,
        "runs": [anchor_entry] + ranked if anchor_entry is not None else ranked,
    }


def _markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Instability Audit",
        "",
        f"- Generated at: `{payload['generated_at_utc']}`",
        f"- Sweep id: `{payload['sweep_id']}`",
        f"- Anchor run id: `{payload['anchor_run_id']}`",
        f"- Runs scanned: `{payload['run_count']}`",
        "",
        "## Default Rerun Recommendations",
        "",
    ]
    recommendations = payload.get("recommendations")
    if isinstance(recommendations, list) and recommendations:
        for index, recommendation in enumerate(recommendations, start=1):
            lines.append(
                f"{index}. `{recommendation['run_id']}`: {recommendation['reason']}"
            )
    else:
        lines.append("1. No completed runs were discovered.")

    lines.extend(["", "## Ranked Runs", ""])
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        lines.append("- No run artifacts were discovered.")
        return "\n".join(lines) + "\n"

    for entry in runs:
        history = entry["history"]
        benchmark = entry.get("benchmark")
        lines.append(f"### `{entry['run_id']}`")
        lines.append(
            f"- role: `{'anchor' if entry['is_anchor'] else 'candidate'}` | severity: `{entry['severity']}`"
        )
        lines.append(
            f"- instability: max grad `{history['max_grad_norm']}` at step `{history['peak_grad_step']}`, "
            f"max abs loss delta `{history['max_abs_train_loss_delta']}` at step `{history['peak_loss_delta_step']}`"
        )
        lines.append(
            f"- loss variance: `{history['train_loss_variance']}` | final grad `{history['final_grad_norm']}`"
        )
        if isinstance(benchmark, Mapping):
            metric_parts: list[str] = []
            for label, key in (
                ("best ROC AUC", "best_roc_auc"),
                ("final ROC AUC", "final_roc_auc"),
                ("final log loss", "final_log_loss"),
                ("final Brier score", "final_brier_score"),
                ("best CRPS", "best_crps"),
                ("final CRPS", "final_crps"),
                ("final avg pinball loss", "final_avg_pinball_loss"),
                ("final PICP 90", "final_picp_90"),
                ("best-to-final ROC delta", "best_to_final_roc_auc_delta"),
                ("best-to-final CRPS delta", "best_to_final_crps_delta"),
            ):
                if benchmark.get(key) is not None:
                    metric_parts.append(f"{label} `{benchmark.get(key)}`")
            if metric_parts:
                lines.append(f"- benchmark: {', '.join(metric_parts)}")
        if entry.get("decision_recommendation") is not None:
            lines.append(
                f"- result card decision: `{entry['decision_recommendation']}`"
            )
        if entry.get("next_action") is not None:
            lines.append(f"- result card next action: `{entry['next_action']}`")
        if entry.get("comparison_summary_path") is not None:
            lines.append(f"- comparison summary: `{entry['comparison_summary_path']}`")
        if entry.get("result_card_path") is not None:
            lines.append(f"- result card: `{entry['result_card_path']}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_instability_audit(
    payload: Mapping[str, Any],
    *,
    out_root: Path,
    sweep_id: str,
) -> dict[str, str]:
    """Write JSON and Markdown audit reports."""

    resolved_out_root = out_root.expanduser().resolve()
    resolved_out_root.mkdir(parents=True, exist_ok=True)
    json_path = resolved_out_root / f"{sweep_id}_instability_audit.json"
    markdown_path = resolved_out_root / f"{sweep_id}_instability_audit.md"
    write_json(json_path, payload)
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit existing system-delta runs for instability.")
    parser.add_argument(
        "--staged-ladder-root",
        default="outputs/staged_ladder",
        help="Root directory containing staged ladder outputs.",
    )
    parser.add_argument(
        "--reports-root",
        default=None,
        help="Optional report output directory. Defaults to <staged-ladder-root>/reports.",
    )
    parser.add_argument(
        "--sweep-id",
        default=DEFAULT_SWEEP_ID,
        help="Sweep identifier used to match sd_<sweep_id>_* run directories.",
    )
    parser.add_argument(
        "--anchor-run-id",
        default=DEFAULT_ANCHOR_RUN_ID,
        help="Registry run id for the reference anchor row.",
    )
    parser.add_argument(
        "--registry-path",
        default=str(default_benchmark_run_registry_path()),
        help="Benchmark run registry JSON used to resolve the anchor run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    staged_ladder_root = Path(str(args.staged_ladder_root))
    payload = build_instability_audit(
        staged_ladder_root=staged_ladder_root,
        sweep_id=str(args.sweep_id),
        anchor_run_id=str(args.anchor_run_id),
        registry_path=Path(str(args.registry_path)),
    )
    report_paths = write_instability_audit(
        payload,
        out_root=(
            staged_ladder_root / "reports"
            if args.reports_root is None
            else Path(str(args.reports_root))
        ),
        sweep_id=str(args.sweep_id),
    )
    print("Instability audit complete:")
    print(f"  json={report_paths['json']}")
    print(f"  markdown={report_paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
