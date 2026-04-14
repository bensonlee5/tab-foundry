"""Training CLI group."""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch

from tab_foundry.config import compose_config
import tab_foundry.cli.train_prior as train_prior_cli
from tab_foundry.training.trainer import train as run_training
from tab_foundry.cli.click_utils import GROUP_KWARGS


def _run_training_command(*, overrides: tuple[str, ...]) -> int:
    cfg = compose_config(list(overrides))
    result = run_training(cfg)
    print(
        "Training complete:",
        f"output_dir={result.output_dir}",
        f"best={result.best_checkpoint}",
        f"latest={result.latest_checkpoint}",
        f"step={result.global_step}",
        f"metrics={result.metrics}",
    )
    return 0


def _run_training_profile_command(
    *,
    overrides: tuple[str, ...],
    max_steps: int,
    wait: int,
    warmup: int,
    active: int,
    repeat: int,
) -> int:
    cfg = compose_config(list(overrides))
    cfg.runtime.max_steps = int(max_steps)
    cfg.runtime.eval_every = int(max_steps)
    cfg.runtime.checkpoint_every = int(max_steps)
    cfg.runtime.profile_step_timing = True
    output_dir = Path(str(cfg.runtime.output_dir)).expanduser().resolve()
    trace_dir = output_dir / "torch_profiler"
    trace_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    sort_by = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=int(wait),
            warmup=int(warmup),
            active=int(active),
            repeat=int(repeat),
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        result = run_training(cfg, profiler=profiler)
        key_averages = profiler.key_averages()
    summary_path = trace_dir / "key_averages.txt"
    summary_path.write_text(
        key_averages.table(sort_by=sort_by, row_limit=64),
        encoding="utf-8",
    )
    metadata_path = trace_dir / "profile_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "summary_path": str(summary_path),
                "trace_dir": str(trace_dir),
                "activities": [str(activity) for activity in activities],
                "max_steps": int(max_steps),
                "schedule": {
                    "wait": int(wait),
                    "warmup": int(warmup),
                    "active": int(active),
                    "repeat": int(repeat),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        "Training profile complete:",
        f"output_dir={result.output_dir}",
        f"trace_dir={trace_dir}",
        f"summary={summary_path}",
        f"step={result.global_step}",
        f"metrics={result.metrics}",
    )
    return 0


@click.group(name="train", help="Training workflows", **GROUP_KWARGS)
def GROUP() -> None:
    """Training workflows."""


@click.command(name="run", help="Train from Hydra config")
@click.argument("overrides", nargs=-1, type=str)
def RUN_COMMAND(overrides: tuple[str, ...]) -> int:
    return _run_training_command(overrides=overrides)


@click.command(name="profile", help="Train a short run with torch profiler traces")
@click.option("--max-steps", default=24, show_default=True, type=int)
@click.option("--wait", default=1, show_default=True, type=int)
@click.option("--warmup", default=1, show_default=True, type=int)
@click.option("--active", default=3, show_default=True, type=int)
@click.option("--repeat", default=1, show_default=True, type=int)
@click.argument("overrides", nargs=-1, type=str)
def PROFILE_COMMAND(
    overrides: tuple[str, ...],
    max_steps: int,
    wait: int,
    warmup: int,
    active: int,
    repeat: int,
) -> int:
    return _run_training_profile_command(
        overrides=overrides,
        max_steps=max_steps,
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
    )


@click.group(name="legacy-prior", help="Legacy exact-prior training workflows", **GROUP_KWARGS)
def _legacy_prior_group() -> None:
    """Legacy exact-prior training workflows."""


_legacy_prior_group.add_command(train_prior_cli.COMMAND)
_legacy_prior_group.add_command(train_prior_cli.STAGED_COMMAND)


GROUP.add_command(RUN_COMMAND)
GROUP.add_command(PROFILE_COMMAND)
GROUP.add_command(_legacy_prior_group)
