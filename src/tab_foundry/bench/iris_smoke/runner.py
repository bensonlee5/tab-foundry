"""Execution helpers for the Iris smoke harness."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable


def run_iris_smoke(
    config: Any,
    *,
    resolve_device_fn: Callable[[str], str],
    write_iris_tasks_fn: Callable[..., Path],
    build_manifest_fn: Callable[..., Any],
    train_fn: Callable[[Any], Any],
    evaluate_checkpoint_fn: Callable[[Any], Any],
    evaluate_iris_checkpoint_fn: Callable[..., Any],
    ensure_finite_metrics_fn: Callable[..., Any],
    plot_loss_curve_fn: Callable[..., Any],
    checkpoint_snapshots_from_history_fn: Callable[..., Any],
    write_json_fn: Callable[[Path, dict[str, Any]], Any],
    write_summary_markdown_fn: Callable[[Path, dict[str, Any]], Path],
    iris_benchmark_payload_fn: Callable[[Any], dict[str, Any]],
    build_cls_smoke_train_config_fn: Callable[..., Any],
    build_cls_smoke_eval_config_fn: Callable[..., Any],
    build_manifest_payload_fn: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    """Execute the end-to-end Iris smoke harness."""

    out_root = config.out_root.expanduser().resolve()
    resolved_device = resolve_device_fn(config.device)
    if config.initial_num_tasks <= 0:
        raise ValueError(f"initial_num_tasks must be > 0, got {config.initial_num_tasks}")
    if config.max_num_tasks < config.initial_num_tasks:
        raise ValueError(
            "max_num_tasks must be >= initial_num_tasks, "
            f"got max_num_tasks={config.max_num_tasks}, initial_num_tasks={config.initial_num_tasks}"
        )
    if config.checkpoint_every <= 0:
        raise ValueError(f"checkpoint_every must be > 0, got {config.checkpoint_every}")
    if config.iris_benchmark_seeds <= 0:
        raise ValueError(
            f"iris_benchmark_seeds must be > 0, got {config.iris_benchmark_seeds}"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    generated_dir = out_root / "generated"
    manifest_path = out_root / "manifest.parquet"
    train_output_dir = out_root / "train_outputs"
    history_path = train_output_dir / "train_history.jsonl"
    loss_curve_path = train_output_dir / "loss_curve.png"
    telemetry_path = out_root / "telemetry.json"
    summary_path = out_root / "summary.md"

    timings_seconds: dict[str, float] = {}
    attempted_task_counts: list[int] = []
    telemetry: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "success": False,
        "config": {
            "device": resolved_device,
            "seed": int(config.seed),
            "requested_num_tasks": int(config.initial_num_tasks),
            "max_num_tasks": int(config.max_num_tasks),
            "final_num_tasks": None,
            "task_count_attempts": attempted_task_counts,
            "test_size": float(config.test_size),
            "checkpoint_every": int(config.checkpoint_every),
            "iris_benchmark_seeds": int(config.iris_benchmark_seeds),
        },
        "artifacts": {
            "generated_dir": str(generated_dir),
            "manifest_path": str(manifest_path),
            "train_output_dir": str(train_output_dir),
            "train_history_jsonl": str(history_path),
            "loss_curve_png": str(loss_curve_path),
            "telemetry_json": str(telemetry_path),
            "summary_md": str(summary_path),
        },
        "timings_seconds": timings_seconds,
    }

    total_start = time.perf_counter()
    manifest_summary: Any = None
    try:
        num_tasks = int(config.initial_num_tasks)
        timings_seconds["generate_iris_tasks"] = 0.0
        timings_seconds["build_manifest"] = 0.0
        while True:
            attempted_task_counts.append(num_tasks)

            stage_start = time.perf_counter()
            write_iris_tasks_fn(
                generated_dir,
                num_tasks=num_tasks,
                seed=int(config.seed),
                test_size=float(config.test_size),
            )
            timings_seconds["generate_iris_tasks"] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            manifest_summary = build_manifest_fn(
                data_roots=[generated_dir],
                out_path=manifest_path,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                filter_policy=config.filter_policy,
            )
            timings_seconds["build_manifest"] += time.perf_counter() - stage_start

            if (
                manifest_summary.train_records > 0
                and manifest_summary.val_records > 0
                and manifest_summary.test_records > 0
            ):
                break
            if num_tasks >= config.max_num_tasks:
                raise RuntimeError(
                    "iris smoke could not populate all manifest splits up to "
                    f"max_num_tasks={config.max_num_tasks}: "
                    f"train={manifest_summary.train_records}, "
                    f"val={manifest_summary.val_records}, "
                    f"test={manifest_summary.test_records}"
                )
            num_tasks = min(int(config.max_num_tasks), int(num_tasks * 2))

        telemetry["config"]["final_num_tasks"] = num_tasks

        stage_start = time.perf_counter()
        train_result = train_fn(
            build_cls_smoke_train_config_fn(
                manifest_path=manifest_path,
                output_dir=train_output_dir,
                history_path=history_path,
                device=resolved_device,
                checkpoint_every=config.checkpoint_every,
                schedule_stages=[
                    {"name": "stage1", "steps": 4, "lr_max": 8.0e-4},
                    {"name": "stage2", "steps": 2, "lr_max": 1.0e-4},
                ],
                clear_row_caps=True,
            )
        )
        timings_seconds["train"] = time.perf_counter() - stage_start
        if train_result.best_checkpoint is None:
            raise RuntimeError("training did not produce a best checkpoint")

        stage_start = time.perf_counter()
        eval_result = evaluate_checkpoint_fn(
            build_cls_smoke_eval_config_fn(
                manifest_path=manifest_path,
                checkpoint_path=train_result.best_checkpoint,
                device=resolved_device,
                clear_row_caps=True,
            )
        )
        timings_seconds["eval"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        iris_summary = evaluate_iris_checkpoint_fn(
            train_result.best_checkpoint,
            device=resolved_device,
            seeds=int(config.iris_benchmark_seeds),
        )
        timings_seconds["iris_benchmark"] = time.perf_counter() - stage_start

        ensure_finite_metrics_fn(train_result.metrics, context="train")
        ensure_finite_metrics_fn(eval_result.metrics, context="eval")

        plot_loss_curve_fn(history_path, loss_curve_path, title="tab-foundry iris smoke loss curve")
        checkpoint_snapshots = checkpoint_snapshots_from_history_fn(
            history_path,
            train_output_dir / "checkpoints",
        )

        telemetry["success"] = True
        telemetry["manifest"] = build_manifest_payload_fn(manifest_summary)
        telemetry["checkpoint_snapshots"] = checkpoint_snapshots
        telemetry["train_metrics"] = {
            "global_step": int(train_result.global_step),
            **{key: float(value) for key, value in train_result.metrics.items()},
        }
        telemetry["eval_metrics"] = {
            key: float(value) for key, value in eval_result.metrics.items()
        }
        telemetry["iris_benchmark"] = iris_benchmark_payload_fn(iris_summary)
        telemetry["artifacts"].update(
            {
                "best_checkpoint": str(train_result.best_checkpoint),
                "latest_checkpoint": (
                    str(train_result.latest_checkpoint) if train_result.latest_checkpoint else None
                ),
            }
        )
        write_summary_markdown_fn(summary_path, telemetry)
        return telemetry
    except Exception as exc:
        telemetry["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        timings_seconds["total"] = time.perf_counter() - total_start
        write_json_fn(telemetry_path, telemetry)
