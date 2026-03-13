"""Build-preprocessor-state CLI command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tab_foundry.data.dataset import fit_manifest_preprocessor_state


def _run(args: argparse.Namespace) -> int:
    state = fit_manifest_preprocessor_state(
        Path(str(args.manifest_path)),
        dataset_id=str(args.dataset_id),
        task=str(args.task),
    )
    out_path = Path(str(args.out_path)).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(state.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        "Preprocessor state built:",
        f"path={out_path}",
        f"dataset_id={args.dataset_id}",
        f"task={args.task}",
    )
    return 0


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "build-preprocessor-state",
        help="Fit one v3 preprocessor_state.json from a manifest dataset train split",
    )
    parser.add_argument("--manifest-path", required=True, help="Input manifest parquet path")
    parser.add_argument("--dataset-id", required=True, help="Manifest dataset_id to fit against")
    parser.add_argument(
        "--task",
        required=True,
        choices=("classification", "regression"),
        help="Task for the selected dataset record",
    )
    parser.add_argument("--out-path", required=True, help="Output preprocessor_state.json path")
    parser.set_defaults(func=_run)
