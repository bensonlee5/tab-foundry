"""Library helper for external nanoTabPFN benchmark execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import numpy as np


def run_nanotabpfn_helper(
    *,
    tab_foundry_src: Path,
    dataset_cache: Path,
    prior_dump: Path,
    out_path: Path,
    device: str = "auto",
    steps: int = 2500,
    eval_every: int = 250,
    seeds: int = 2,
    batch_size: int = 32,
    lr: float = 4.0e-3,
    allow_missing_values: bool = False,
    helper_root: Path | None = None,
) -> int:
    """Train and evaluate nanoTabPFN on cached benchmark datasets."""

    src_root = tab_foundry_src.expanduser().resolve()
    nanotabpfn_root = Path.cwd().resolve() if helper_root is None else helper_root.expanduser().resolve()
    if str(nanotabpfn_root) not in sys.path:
        sys.path.insert(0, str(nanotabpfn_root))
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from model import NanoTabPFNModel  # type: ignore[attr-defined]
    from tab_foundry.bench.artifacts import write_jsonl
    from tab_foundry.bench.nanotabpfn import (
        dataset_brier_score_metrics,
        dataset_log_loss_metrics,
        dataset_roc_auc_metrics,
        evaluate_classifier,
        load_dataset_cache,
    )
    from train import PriorDumpDataLoader, get_default_device, set_randomness_seed, train

    resolved_device = get_default_device() if str(device).strip().lower() == "auto" else str(device)
    dataset_cache_path = dataset_cache.expanduser().resolve()
    prior_dump_path = prior_dump.expanduser().resolve()
    out_path = out_path.expanduser().resolve()
    if not prior_dump_path.exists():
        raise RuntimeError(f"nanoTabPFN prior dump does not exist: {prior_dump_path}")
    datasets = load_dataset_cache(dataset_cache_path)

    records: list[dict[str, object]] = []
    num_outputs = max(int(np.unique(np.asarray(y)).size) for _name, (_x, y) in datasets.items())
    for seed in range(int(seeds)):
        set_randomness_seed(seed)
        prior = PriorDumpDataLoader(
            str(prior_dump_path),
            num_steps=int(steps),
            batch_size=int(batch_size),
            device=resolved_device,
        )
        model = NanoTabPFNModel(
            embedding_size=96,
            num_attention_heads=4,
            mlp_hidden_size=192,
            num_layers=3,
            num_outputs=num_outputs,
        )
        model_instance: Any = model
        model_instance, history = train(
            model_instance,
            prior,
            lr=float(lr),
            device=resolved_device,
            steps_per_eval=int(eval_every),
            eval_func=lambda classifier: evaluate_classifier(
                classifier,
                datasets,
                allow_missing_values=bool(allow_missing_values),
            ),
        )
        _ = model_instance
        for index, (training_time, metrics) in enumerate(history, start=1):
            records.append(
                {
                    "seed": int(seed),
                    "step": int(index * int(eval_every)),
                    "training_time": float(training_time),
                    "roc_auc": float(metrics["ROC AUC"]),
                    "log_loss": float(metrics["Log Loss"]),
                    "brier_score": float(metrics["Brier Score"]),
                    "dataset_roc_auc": dataset_roc_auc_metrics(metrics),
                    "dataset_log_loss": dataset_log_loss_metrics(metrics),
                    "dataset_brier_score": dataset_brier_score_metrics(metrics),
                }
            )

    write_jsonl(out_path, records)
    return 0
