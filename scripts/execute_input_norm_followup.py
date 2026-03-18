from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import DictConfig, OmegaConf
import yaml

from tab_foundry.bench.benchmark_run_registry import load_benchmark_run_registry, register_benchmark_run
from tab_foundry.bench.compare import NanoTabPFNBenchmarkConfig, run_nanotabpfn_benchmark
from tab_foundry.bench.prior_train import train_tabfoundry_simple_prior
from tab_foundry.config import compose_config
from tab_foundry.research import system_delta as system_delta_module
from tab_foundry.training.instability import diagnostics_summary


DEFAULT_SWEEP_ID = "input_norm_followup"
DEFAULT_PRIOR_DUMP = Path("/workspace/nanoTabPFN/300k_150x5_2.h5")
DEFAULT_NANOTABPFN_ROOT = Path("/workspace/nanoTabPFN")
DEFAULT_DEVICE = "cuda"


def _repo_root() -> Path:
    return system_delta_module.repo_root()


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected YAML mapping at {path}")
    return cast(dict[str, Any], payload)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    system_delta_module._write_yaml(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(cast(dict[str, Any], payload))
    return records


def _row_id_for_order(sweep_id: str, order: int, delta_ref: str, existing_run_id: str | None) -> str:
    base = f"sd_{sweep_id}_{order:02d}_{delta_ref}"
    if existing_run_id is None:
        return f"{base}_v1"
    match = re.fullmatch(rf"{re.escape(base)}_v(\d+)", existing_run_id)
    if match is None:
        return f"{base}_v1"
    return f"{base}_v{int(match.group(1)) + 1}"


def _ensure_nanotabpfn_python(*, nanotabpfn_root: Path, fallback_python: Path) -> Path:
    nanotab_python = nanotabpfn_root / ".venv" / "bin" / "python"
    nanotab_python.parent.mkdir(parents=True, exist_ok=True)
    if nanotab_python.exists() and not nanotab_python.is_symlink():
        return nanotab_python
    if nanotab_python.exists() or nanotab_python.is_symlink():
        nanotab_python.unlink()
    nanotab_python.write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(str(fallback_python.resolve()))} \"$@\"\n",
        encoding="utf-8",
    )
    nanotab_python.chmod(0o755)
    return nanotab_python


def _sorted_rows(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = queue.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("queue rows must be a list")
    return sorted(cast(list[dict[str, Any]], rows), key=lambda row: int(row["order"]))


def _completed_train_artifacts_exist(run_dir: Path) -> bool:
    required_paths = (
        run_dir / "train_history.jsonl",
        run_dir / "gradient_history.jsonl",
        run_dir / "training_surface_record.json",
        run_dir / "checkpoints" / "latest.pt",
    )
    return all(path.exists() for path in required_paths)


def _materialized_row_map(sweep_id: str) -> dict[str, dict[str, Any]]:
    materialized = system_delta_module.load_system_delta_queue(sweep_id=sweep_id)
    rows = cast(list[dict[str, Any]], materialized["rows"])
    return {str(row["delta_id"]): row for row in rows}


def _apply_mapping(cfg: DictConfig, prefix: str, payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        OmegaConf.update(cfg, f"{prefix}.{key}", value, merge=True)


def _compose_cfg(*, row: Mapping[str, Any], run_dir: Path, device: str) -> DictConfig:
    cfg = compose_config(["experiment=cls_benchmark_staged_prior", "logging.use_wandb=false"])
    cfg.runtime.output_dir = str(run_dir.resolve())
    cfg.runtime.device = str(device)
    _apply_mapping(cfg, "model", cast(Mapping[str, Any], row.get("model", {})))
    _apply_mapping(cfg, "data", cast(Mapping[str, Any], row.get("data", {})))
    _apply_mapping(cfg, "preprocessing", cast(Mapping[str, Any], row.get("preprocessing", {})))

    training_payload = cast(Mapping[str, Any], row.get("training", {}))
    for key in (
        "surface_label",
        "prior_dump_non_finite_policy",
        "prior_dump_batch_size",
        "prior_dump_lr_scale_rule",
        "prior_dump_batch_reference_size",
    ):
        if key in training_payload:
            OmegaConf.update(cfg, f"training.{key}", training_payload[key], merge=True)

    overrides = cast(Mapping[str, Any], training_payload.get("overrides", {}))
    if "apply_schedule" in overrides:
        OmegaConf.update(cfg, "training.apply_schedule", overrides["apply_schedule"], merge=True)
    for key in ("optimizer", "runtime", "schedule"):
        override_payload = overrides.get(key)
        if isinstance(override_payload, dict):
            _apply_mapping(cfg, key, cast(Mapping[str, Any], override_payload))
    return cfg


def _queue_metrics(summary: Mapping[str, Any], *, run_dir: Path) -> dict[str, Any]:
    tab_foundry = cast(dict[str, Any], summary["tab_foundry"])
    nanotabpfn = cast(dict[str, Any], summary["nanotabpfn"])
    gradient_records = _read_jsonl(run_dir / "gradient_history.jsonl")
    training_surface_record = _read_json(run_dir / "training_surface_record.json")
    diagnostics = diagnostics_summary(gradient_records, training_surface_record=training_surface_record)
    return {
        "best_roc_auc": float(tab_foundry["best_roc_auc"]),
        "best_step": int(float(tab_foundry["best_step"])),
        "final_roc_auc": float(tab_foundry["final_roc_auc"]),
        "drift": float(tab_foundry["best_to_final_roc_auc_delta"]),
        "nanotabpfn_best": float(nanotabpfn["best_roc_auc"]),
        "nanotabpfn_final": float(nanotabpfn["final_roc_auc"]),
        "max_grad_norm": float(cast(dict[str, Any], tab_foundry["training_diagnostics"])["max_grad_norm"]),
        "clipped_step_fraction": float(cast(dict[str, Any], diagnostics["grad_clip"])["clipped_step_fraction"]),
    }


def _decision_for_row(order: int) -> tuple[str, str]:
    if order == 1:
        return (
            "keep",
            "CUDA rerun establishes the same-machine anchor for input_norm_followup and supersedes the earlier v1 queue reference.",
        )
    return (
        "defer",
        "CUDA rerun recorded the canonical benchmark comparison, but the row remains deferred pending full sweep interpretation.",
    )


def _recommended_next_action(order: int, fallback: str) -> str:
    if order == 9:
        return "Review rows 1-9 together before redefining the default normalization or opening a new follow-up sweep."
    return fallback


def _result_card_text(
    *,
    row: Mapping[str, Any],
    run_id: str,
    anchor_run_id: str,
    summary: Mapping[str, Any],
    queue_metrics: Mapping[str, Any],
    decision: str,
) -> str:
    tab_foundry = cast(dict[str, Any], summary["tab_foundry"])
    recommended = "accept_signal" if int(row["order"]) == 1 else "needs_followup"
    return "\n".join(
        [
            "# Result Card",
            "",
            "## What changed",
            "",
            f"- `delta_id`: `{row['delta_id']}`",
            f"- `run_id`: `{run_id}`",
            f"- `anchor_run_id`: `{anchor_run_id}`",
            f"- `description`: {row['description']}",
            f"- `anchor_delta`: {row['anchor_delta']}",
            "",
            "## Measured metrics versus the anchor",
            "",
            f"- Best ROC AUC: `{float(tab_foundry['best_roc_auc']):.4f}` at step `{int(float(tab_foundry['best_step']))}`",
            f"- Final ROC AUC: `{float(tab_foundry['final_roc_auc']):.4f}`",
            f"- Final minus best: `{float(tab_foundry['best_to_final_roc_auc_delta']):+.4f}`",
            f"- nanoTabPFN best ROC AUC: `{float(cast(dict[str, Any], summary['nanotabpfn'])['best_roc_auc']):.4f}`",
            f"- nanoTabPFN final ROC AUC: `{float(cast(dict[str, Any], summary['nanotabpfn'])['final_roc_auc']):.4f}`",
            f"- max_grad_norm: `{float(queue_metrics['max_grad_norm']):.4f}`",
            f"- clipped_step_fraction: `{float(queue_metrics['clipped_step_fraction']):.4f}`",
            "",
            "## Was the change actually isolated?",
            "",
            "- The run used the queue row as the only source of model/data/preprocessing/training overrides.",
            "- Bundle, control baseline, experiment family, and schedule budget stayed on the locked sweep surface.",
            "",
            "## Hyperparameter adequacy",
            "",
            "- The row used the queue’s declared adequacy plan and preserved the locked 2500-step short-run budget.",
            "- No extra tuning beyond the queue row was introduced during this rerun.",
            "",
            "## Why this may have helped or hurt",
            "",
            f"- Decision recorded in the registry: `{decision}`.",
            "- Final ROC AUC relative to the locked anchor should be interpreted from the registered comparison deltas after the full queue completes.",
            "",
            "## Remaining confounders",
            "",
            "- This auto-generated card is intentionally conservative; detailed row-by-row interpretation still belongs in the sweep review.",
            "",
            "## Recommended next action",
            "",
            f"- `{recommended}`",
            "",
        ]
    )


def _research_card_text(*, row: Mapping[str, Any], sweep_id: str, anchor_run_id: str) -> str:
    plan = cast(list[str], row.get("parameter_adequacy_plan", []))
    plan_lines = "\n".join(f"- {item}" for item in plan) if plan else "- No extra adequacy plan recorded."
    return "\n".join(
        [
            "# Research Card",
            "",
            "## Delta",
            "",
            f"- `delta_id`: `{row['delta_id']}`",
            f"- `sweep_id`: `{sweep_id}`",
            f"- `dimension_family`: `{row['dimension_family']}`",
            f"- `family`: `{row['family']}`",
            f"- `anchor_run_id`: `{anchor_run_id}`",
            "- `comparison_policy`: `anchor_only`",
            "",
            "## What Changes",
            "",
            f"- {row['description']}",
            f"- Anchor delta: {row['anchor_delta']}",
            "",
            "## Why This Row Is Informative",
            "",
            f"- {row['rationale']}",
            f"- Hypothesis: {row['hypothesis']}",
            "",
            "## Adequacy Plan",
            "",
            plan_lines,
            "",
        ]
    )


def _campaign_payload(
    *,
    queue_row: Mapping[str, Any],
    materialized_row: Mapping[str, Any],
    sweep_id: str,
    anchor_run_id: str,
) -> dict[str, Any]:
    changed_settings: dict[str, Any] = {
        "model": cast(dict[str, Any], queue_row.get("model", {})),
        "data": cast(dict[str, Any], queue_row.get("data", {})),
        "preprocessing": cast(dict[str, Any], queue_row.get("preprocessing", {})),
        "training": cast(dict[str, Any], queue_row.get("training", {})),
    }
    return {
        "sweep_id": sweep_id,
        "delta_id": materialized_row["delta_id"],
        "dimension_family": materialized_row["dimension_family"],
        "family": materialized_row["family"],
        "comparison_policy": "anchor_only",
        "anchor_run_id": anchor_run_id,
        "locked_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "locked_control_baseline_id": "cls_benchmark_linear_v2",
        "training_experiment": "cls_benchmark_staged_prior",
        "preserved_settings": {
            "queue_ref": f"reference/system_delta_sweeps/{sweep_id}/queue.yaml",
            "runtime.device": DEFAULT_DEVICE,
            "logging.use_wandb": False,
        },
        "changed_settings": changed_settings,
        "adequacy_knobs": cast(list[str], materialized_row.get("adequacy_knobs", [])),
        "decision_hypothesis": "needs_followup" if int(queue_row["order"]) != 1 else "accept_signal",
    }


def _write_research_package(
    *,
    delta_root: Path,
    materialized_row: Mapping[str, Any],
    queue_row: Mapping[str, Any],
    sweep_id: str,
    anchor_run_id: str,
) -> None:
    delta_root.mkdir(parents=True, exist_ok=True)
    (delta_root / "research_card.md").write_text(
        _research_card_text(row=materialized_row, sweep_id=sweep_id, anchor_run_id=anchor_run_id),
        encoding="utf-8",
    )
    (delta_root / "campaign.yaml").write_text(
        yaml.safe_dump(
            _campaign_payload(
                queue_row=queue_row,
                materialized_row=materialized_row,
                sweep_id=sweep_id,
                anchor_run_id=anchor_run_id,
            ),
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )


def _append_note(notes: list[str], note: str) -> list[str]:
    updated = list(notes)
    if note not in updated:
        updated.append(note)
    return updated


def _update_queue_row(
    *,
    queue_row: dict[str, Any],
    run_id: str,
    queue_metrics: Mapping[str, Any],
    decision: str,
    conclusion: str,
) -> None:
    original_run_id = queue_row.get("run_id")
    queue_row["status"] = "completed"
    queue_row["run_id"] = run_id
    queue_row["followup_run_ids"] = []
    queue_row["decision"] = decision
    queue_row["interpretation_status"] = "completed"
    queue_row["benchmark_metrics"] = dict(queue_metrics)
    queue_row["confounders"] = []
    queue_row["next_action"] = _recommended_next_action(int(queue_row["order"]), str(queue_row.get("next_action", "")))
    notes = cast(list[str], queue_row.get("notes", []))
    if isinstance(original_run_id, str) and original_run_id.strip() and original_run_id != run_id:
        notes = _append_note(
            notes,
            f"Supersedes historical queue run `{original_run_id}`; that registry entry is retained as history only.",
        )
    notes = _append_note(notes, f"Canonical CUDA rerun registered as `{run_id}`.")
    notes = _append_note(notes, conclusion)
    queue_row["notes"] = notes


def _update_anchor_metadata(*, sweep_id: str, anchor_run_id: str) -> None:
    registry = load_benchmark_run_registry(system_delta_module.default_registry_path())
    entry = cast(dict[str, Any], cast(dict[str, Any], registry["runs"])[anchor_run_id])

    sweep_path = system_delta_module.sweep_metadata_path(sweep_id)
    sweep = _read_yaml(sweep_path)
    sweep["anchor_run_id"] = anchor_run_id
    sweep["anchor_context"] = {
        "run_id": anchor_run_id,
        "experiment": entry["experiment"],
        "config_profile": entry["config_profile"],
        "model": {
            "arch": cast(dict[str, Any], entry["model"]).get("arch"),
            "stage": cast(dict[str, Any], entry["model"]).get("stage"),
            "stage_label": cast(dict[str, Any], entry["model"]).get("stage_label"),
            "module_selection": cast(dict[str, Any], entry["model"]).get("module_selection"),
        },
        "surface_labels": entry.get("surface_labels"),
    }
    notes = cast(list[str], cast(dict[str, Any], sweep["anchor_surface"]).get("notes", []))
    cast(dict[str, Any], sweep["anchor_surface"])["notes"] = [
        note.replace("sd_input_norm_followup_01_dpnb_input_norm_anchor_replay_v1", anchor_run_id)
        for note in notes
    ]
    _write_yaml(sweep_path, sweep)

    index_path = system_delta_module.default_sweep_index_path()
    index = _read_yaml(index_path)
    sweeps = cast(dict[str, Any], index["sweeps"])
    cast(dict[str, Any], sweeps[sweep_id])["anchor_run_id"] = anchor_run_id
    _write_yaml(index_path, index)

    program_path = _repo_root() / "program.md"
    program_text = program_path.read_text(encoding="utf-8")
    program_text = program_text.replace("sd_input_norm_followup_01_dpnb_input_norm_anchor_replay_v1", anchor_run_id)
    program_text = re.sub(
        r"- anchor prior run: `outputs/staged_ladder/research/input_norm_followup/dpnb_input_norm_anchor_replay[^`]*`",
        f"- anchor prior run: `outputs/staged_ladder/research/{sweep_id}/dpnb_input_norm_anchor_replay/{anchor_run_id}/train`",
        program_text,
    )
    program_text = re.sub(
        r"- anchor benchmark: `outputs/staged_ladder/research/input_norm_followup/dpnb_input_norm_anchor_replay[^`]*`",
        f"- anchor benchmark: `outputs/staged_ladder/research/{sweep_id}/dpnb_input_norm_anchor_replay/{anchor_run_id}/benchmark`",
        program_text,
    )
    system_delta_module._write_text(program_path, program_text)


def _sync_aliases(sweep_id: str) -> None:
    queue = system_delta_module.load_system_delta_queue(
        sweep_id=sweep_id,
        index_path=system_delta_module.default_sweep_index_path(),
        catalog_path=system_delta_module.default_catalog_path(),
    )
    matrix_contents = system_delta_module.render_system_delta_matrix(
        queue,
        registry_path=system_delta_module.default_registry_path(),
    )
    system_delta_module._write_text(system_delta_module.sweep_matrix_path(sweep_id), matrix_contents)
    system_delta_module.sync_active_sweep_aliases(
        sweep_id=sweep_id,
        index_path=system_delta_module.default_sweep_index_path(),
        catalog_path=system_delta_module.default_catalog_path(),
        registry_path=system_delta_module.default_registry_path(),
    )


def _run_row(
    *,
    sweep_id: str,
    queue_row: dict[str, Any],
    materialized_row: Mapping[str, Any],
    anchor_run_id: str,
    prior_dump: Path,
    nanotabpfn_root: Path,
    device: str,
    fallback_python: Path,
) -> str:
    existing_run_id = queue_row.get("run_id")
    run_id = _row_id_for_order(
        sweep_id,
        int(queue_row["order"]),
        str(queue_row["delta_ref"]),
        str(existing_run_id) if isinstance(existing_run_id, str) else None,
    )
    delta_root = _repo_root() / "outputs" / "staged_ladder" / "research" / sweep_id / str(queue_row["delta_ref"])
    run_root = delta_root / run_id
    train_dir = run_root / "train"
    benchmark_dir = run_root / "benchmark"

    effective_anchor_run_id = run_id if int(queue_row["order"]) == 1 else anchor_run_id

    _write_research_package(
        delta_root=delta_root,
        materialized_row=materialized_row,
        queue_row=queue_row,
        sweep_id=sweep_id,
        anchor_run_id=effective_anchor_run_id,
    )
    if _completed_train_artifacts_exist(train_dir):
        print(
            f"[row {int(queue_row['order']):02d}] reusing existing train artifacts",
            f"run_id={run_id}",
            f"output_dir={train_dir}",
            flush=True,
        )
    else:
        cfg = _compose_cfg(row=queue_row, run_dir=train_dir, device=device)
        train_result = train_tabfoundry_simple_prior(cfg, prior_dump_path=prior_dump)
        print(
            f"[row {int(queue_row['order']):02d}] train complete",
            f"run_id={run_id}",
            f"output_dir={train_result.output_dir}",
            flush=True,
        )

    _ = _ensure_nanotabpfn_python(nanotabpfn_root=nanotabpfn_root, fallback_python=fallback_python)
    summary = run_nanotabpfn_benchmark(
        NanoTabPFNBenchmarkConfig(
            tab_foundry_run_dir=train_dir,
            out_root=benchmark_dir,
            nanotabpfn_root=nanotabpfn_root,
            nanotab_prior_dump=prior_dump,
            device=device,
            control_baseline_id="cls_benchmark_linear_v2",
            control_baseline_registry=_repo_root() / "src" / "tab_foundry" / "bench" / "control_baselines_v1.json",
            benchmark_bundle_path=_repo_root() / "src" / "tab_foundry" / "bench" / "nanotabpfn_openml_binary_medium_v1.json",
        )
    )
    queue_metrics = _queue_metrics(summary, run_dir=train_dir)
    decision, conclusion = _decision_for_row(int(queue_row["order"]))
    _ = register_benchmark_run(
        run_id=run_id,
        track="system_delta_binary_medium_v1",
        experiment="cls_benchmark_staged_prior",
        config_profile="cls_benchmark_staged_prior",
        budget_class="short-run",
        run_dir=train_dir,
        comparison_summary_path=benchmark_dir / "comparison_summary.json",
        decision=decision,
        conclusion=conclusion,
        parent_run_id=None if int(queue_row["order"]) == 1 else anchor_run_id,
        anchor_run_id=None if int(queue_row["order"]) == 1 else anchor_run_id,
        prior_dir=None,
        control_baseline_id="cls_benchmark_linear_v2",
        sweep_id=sweep_id,
        delta_id=str(queue_row["delta_ref"]),
        parent_sweep_id="stability_followup",
        queue_order=int(queue_row["order"]),
        run_kind="primary",
        registry_path=system_delta_module.default_registry_path(),
    )
    (delta_root / "result_card.md").write_text(
        _result_card_text(
            row=materialized_row,
            run_id=run_id,
            anchor_run_id=anchor_run_id,
            summary=summary,
            queue_metrics=queue_metrics,
            decision=decision,
        ),
        encoding="utf-8",
    )
    _update_queue_row(
        queue_row=queue_row,
        run_id=run_id,
        queue_metrics=queue_metrics,
        decision=decision,
        conclusion=conclusion,
    )
    print(
        f"[row {int(queue_row['order']):02d}] benchmark+registry complete",
        f"run_id={run_id}",
        f"final_roc_auc={float(cast(dict[str, Any], summary['tab_foundry'])['final_roc_auc']):.4f}",
        flush=True,
    )
    return run_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute the input_norm_followup system-delta queue")
    parser.add_argument("--sweep-id", default=DEFAULT_SWEEP_ID, help="Sweep id to execute")
    parser.add_argument("--prior-dump", default=str(DEFAULT_PRIOR_DUMP), help="Path to the nanoTabPFN prior dump")
    parser.add_argument("--nanotabpfn-root", default=str(DEFAULT_NANOTABPFN_ROOT), help="Path to the nanoTabPFN checkout")
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=("cpu", "cuda", "mps", "auto"))
    parser.add_argument(
        "--tab-foundry-python",
        default=str(_repo_root() / ".venv" / "bin" / "python"),
        help="Interpreter to expose under nanoTabPFN/.venv/bin/python",
    )
    parser.add_argument("--start-order", type=int, default=1, help="Optional starting queue order")
    parser.add_argument("--stop-after-order", type=int, default=None, help="Optional last queue order to execute")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sweep_id = str(args.sweep_id)
    prior_dump = Path(str(args.prior_dump)).expanduser().resolve()
    nanotabpfn_root = Path(str(args.nanotabpfn_root)).expanduser().resolve()
    fallback_python = Path(str(args.tab_foundry_python)).expanduser().resolve()
    if not prior_dump.exists():
        raise RuntimeError(f"prior dump does not exist: {prior_dump}")
    if not fallback_python.exists():
        raise RuntimeError(f"tab-foundry interpreter does not exist: {fallback_python}")

    queue_path = system_delta_module.sweep_queue_path(sweep_id)
    queue = _read_yaml(queue_path)
    materialized_rows = _materialized_row_map(sweep_id)
    anchor_run_id = str(_read_yaml(system_delta_module.sweep_metadata_path(sweep_id))["anchor_run_id"])
    stop_after_order = None if args.stop_after_order is None else int(args.stop_after_order)

    for queue_row in _sorted_rows(queue):
        order = int(queue_row["order"])
        if order < int(args.start_order):
            continue
        if stop_after_order is not None and order > stop_after_order:
            break
        materialized_row = materialized_rows[str(queue_row["delta_ref"])]
        run_id = _run_row(
            sweep_id=sweep_id,
            queue_row=queue_row,
            materialized_row=materialized_row,
            anchor_run_id=anchor_run_id,
            prior_dump=prior_dump,
            nanotabpfn_root=nanotabpfn_root,
            device=str(args.device),
            fallback_python=fallback_python,
        )
        _write_yaml(queue_path, queue)
        if order == 1:
            anchor_run_id = run_id
            _update_anchor_metadata(sweep_id=sweep_id, anchor_run_id=anchor_run_id)
        _sync_aliases(sweep_id)

    print("Queue execution complete.", f"sweep_id={sweep_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
