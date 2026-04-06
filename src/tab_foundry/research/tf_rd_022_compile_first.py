"""Helpers for the TF-RD-022 compile-first training slice."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
import torch

from tab_foundry.config import compose_config
from tab_foundry.training.trainer import train


TF_RD_022_COMPILE_FIRST_EXPERIMENT = (
    "cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_v1"
)


def _compose_tf_rd_022_compile_cfg(
    *,
    output_dir: Path | str | None = None,
    max_steps: int | None = None,
    eval_every: int | None = None,
    checkpoint_every: int | None = None,
    compile_model: bool | None = None,
    compile_dynamic: bool | None = None,
    compile_backend: str | None = None,
    compile_mode: str | None = None,
    output_dir_suffix: str | None = None,
    run_name_suffix: str | None = None,
) -> DictConfig:
    resolved_output_dir = (
        Path(str(output_dir)).expanduser().resolve()
        if output_dir is not None
        else None
    )
    overrides = [f"experiment={TF_RD_022_COMPILE_FIRST_EXPERIMENT}"]
    if max_steps is not None:
        overrides.append(f"runtime.max_steps={int(max_steps)}")
    if eval_every is not None:
        overrides.append(f"runtime.eval_every={int(eval_every)}")
    if checkpoint_every is not None:
        overrides.append(f"runtime.checkpoint_every={int(checkpoint_every)}")
    if resolved_output_dir is not None:
        overrides.append(f"runtime.output_dir={resolved_output_dir}")
    if compile_model is not None:
        overrides.append(f"runtime.compile_model={str(bool(compile_model)).lower()}")
    if compile_dynamic is not None:
        overrides.append(f"runtime.compile_dynamic={str(bool(compile_dynamic)).lower()}")
    if compile_backend is not None:
        overrides.append(f"runtime.compile_backend={str(compile_backend)}")
    if compile_mode is not None:
        overrides.append(f"runtime.compile_mode={str(compile_mode)}")
    cfg = compose_config(overrides)
    if resolved_output_dir is None and output_dir_suffix:
        base_output_dir = Path(str(cfg.runtime.output_dir)).expanduser()
        cfg.runtime.output_dir = str(base_output_dir.with_name(f"{base_output_dir.name}{output_dir_suffix}"))
    if run_name_suffix:
        cfg.logging.run_name = f"{cfg.logging.run_name}{run_name_suffix}"
    return cfg


def tf_rd_022_compile_first_cfg() -> DictConfig:
    """Resolve the canonical TF-RD-022 compile-first experiment config."""

    return _compose_tf_rd_022_compile_cfg()


def tf_rd_022_compile_profile_cfg(
    *,
    output_dir: Path | str | None = None,
    max_steps: int = 24,
) -> DictConfig:
    """Resolve the short profiler screen config for the compile-first experiment."""

    return _compose_tf_rd_022_compile_cfg(
        output_dir=output_dir,
        max_steps=max_steps,
        eval_every=max_steps,
        checkpoint_every=max_steps,
        output_dir_suffix="_profile_short",
        run_name_suffix="-profile-short",
    )


def run_tf_rd_022_compile_profile(
    output_dir: Path | str,
    *,
    max_steps: int = 24,
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
    repeat: int = 1,
) -> dict[str, Any]:
    """Run the short same-host torch profiler screen for the compile-first slice."""

    run_output_dir = Path(str(output_dir)).expanduser().resolve()
    cfg = tf_rd_022_compile_profile_cfg(output_dir=run_output_dir, max_steps=max_steps)
    trace_dir = run_output_dir / "torch_profiler"
    trace_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    sort_by = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    profile_schedule = torch.profiler.schedule(
        wait=int(wait),
        warmup=int(warmup),
        active=int(active),
        repeat=int(repeat),
    )
    trace_handler = torch.profiler.tensorboard_trace_handler(str(trace_dir))
    with torch.profiler.profile(
        activities=activities,
        schedule=profile_schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        result = train(cfg, profiler=profiler)
        key_averages = profiler.key_averages()
    summary_path = trace_dir / "key_averages.txt"
    summary_path.write_text(
        key_averages.table(sort_by=sort_by, row_limit=64),
        encoding="utf-8",
    )
    metadata = {
        "experiment": TF_RD_022_COMPILE_FIRST_EXPERIMENT,
        "run_output_dir": str(result.output_dir),
        "trace_dir": str(trace_dir),
        "summary_path": str(summary_path),
        "activities": [str(activity) for activity in activities],
        "max_steps": int(max_steps),
        "schedule": {
            "wait": int(wait),
            "warmup": int(warmup),
            "active": int(active),
            "repeat": int(repeat),
        },
    }
    metadata_path = trace_dir / "profile_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata
