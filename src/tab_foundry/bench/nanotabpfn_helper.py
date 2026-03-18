"""External nanoTabPFN benchmark helper entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from typing import Sequence
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate nanoTabPFN on cached benchmark datasets")
    parser.add_argument("--tab-foundry-src", required=True, help="tab-foundry src directory for shared helpers")
    parser.add_argument("--dataset-cache", required=True, help="Path to cached benchmark datasets (.npz)")
    parser.add_argument("--prior-dump", required=True, help="Path to nanoTabPFN prior dump (.h5)")
    parser.add_argument("--out-path", required=True, help="Output JSONL path")
    parser.add_argument("--device", default="auto", help="Device override")
    parser.add_argument("--steps", type=int, default=2500, help="Training steps")
    parser.add_argument("--eval-every", type=int, default=250, help="Evaluation cadence in steps")
    parser.add_argument("--seeds", type=int, default=2, help="Number of random seeds")
    parser.add_argument("--batch-size", type=int, default=32, help="Prior batch size")
    parser.add_argument("--lr", type=float, default=4.0e-3, help="Learning rate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    src_root = Path(str(args.tab_foundry_src)).expanduser().resolve()
    nanotabpfn_root = Path.cwd().resolve()
    if str(nanotabpfn_root) not in sys.path:
        sys.path.insert(0, str(nanotabpfn_root))
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from model import NanoTabPFNModel  # type: ignore[attr-defined]
    from tab_foundry.bench.artifacts import write_jsonl
    from tab_foundry.bench.nanotabpfn import (
        dataset_log_loss_metrics,
        dataset_roc_auc_metrics,
        evaluate_classifier,
        load_dataset_cache,
    )
    from train import PriorDumpDataLoader, get_default_device, set_randomness_seed, train

    device = get_default_device() if str(args.device).strip().lower() == "auto" else str(args.device)
    datasets = load_dataset_cache(Path(str(args.dataset_cache)).expanduser().resolve())
    prior_dump = Path(str(args.prior_dump)).expanduser().resolve()
    if not prior_dump.exists():
        raise RuntimeError(f"nanoTabPFN prior dump does not exist: {prior_dump}")

    records: list[dict[str, object]] = []
    for seed in range(int(args.seeds)):
        set_randomness_seed(seed)
        prior = PriorDumpDataLoader(
            str(prior_dump),
            num_steps=int(args.steps),
            batch_size=int(args.batch_size),
            device=device,
        )
        model = NanoTabPFNModel(
            embedding_size=96,
            num_attention_heads=4,
            mlp_hidden_size=192,
            num_layers=3,
            num_outputs=2,
        )
        model_instance: Any = model
        model_instance, history = train(
            model_instance,
            prior,
            lr=float(args.lr),
            device=device,
            steps_per_eval=int(args.eval_every),
            eval_func=lambda classifier: evaluate_classifier(classifier, datasets),
        )
        _ = model_instance
        for index, (training_time, metrics) in enumerate(history, start=1):
            records.append(
                {
                    "seed": int(seed),
                    "step": int(index * int(args.eval_every)),
                    "training_time": float(training_time),
                    "roc_auc": float(metrics["ROC AUC"]),
                    "log_loss": float(metrics["Log Loss"]),
                    "dataset_roc_auc": dataset_roc_auc_metrics(metrics),
                    "dataset_log_loss": dataset_log_loss_metrics(metrics),
                }
            )

    write_jsonl(Path(str(args.out_path)).expanduser().resolve(), records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
