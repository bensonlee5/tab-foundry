"""Wandb helpers for exact prior-dump training."""

from __future__ import annotations


def update_prior_wandb_summary(
    run,
    *,
    output_dir,
    global_step: int,
    telemetry_payload,
    prior_wandb_summary_payload_fn,
    update_wandb_summary_fn,
) -> None:
    update_wandb_summary_fn(
        run,
        prior_wandb_summary_payload_fn(
            output_dir=output_dir,
            global_step=global_step,
            telemetry_payload=telemetry_payload,
        ),
    )
