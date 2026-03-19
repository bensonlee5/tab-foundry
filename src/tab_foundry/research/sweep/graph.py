"""Render sweep architecture graphs with torchview."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence, cast

from omegaconf import OmegaConf
import torch

from tab_foundry.bench.benchmark_run_registry import (
    load_benchmark_run_registry,
    resolve_registry_path_value,
)
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.spec import (
    ModelBuildSpec,
    checkpoint_model_build_spec_from_mappings,
    model_build_spec_from_mappings,
)

from .materialize import load_system_delta_queue, ordered_rows
from .paths_io import (
    _write_text,
    default_catalog_path,
    default_registry_path,
    default_sweep_index_path,
    default_sweeps_root,
    repo_root,
)
from .runner import compose_cfg
from .validation import ensure_mapping, ensure_non_empty_string


_SAFE_FILENAME_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class GraphPaths:
    index_path: Path
    catalog_path: Path
    sweeps_root: Path
    registry_path: Path

    @classmethod
    def default(cls) -> "GraphPaths":
        return cls(
            index_path=default_sweep_index_path(),
            catalog_path=default_catalog_path(),
            sweeps_root=default_sweeps_root(),
            registry_path=default_registry_path(),
        )


@dataclass(frozen=True)
class GraphTarget:
    kind: str
    title: str
    filename: str
    model_spec: ModelBuildSpec
    metadata: dict[str, Any]


class _ForwardBatchedWrapper(torch.nn.Module):
    """Expose the repo's batched forward path as a simple tensor-only module."""

    def __init__(self, model: torch.nn.Module, *, train_test_split_index: int) -> None:
        super().__init__()
        self.model = model
        self.train_test_split_index = int(train_test_split_index)

    def forward(self, x_all: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        forward_batched = getattr(self.model, "forward_batched", None)
        if not callable(forward_batched):
            raise RuntimeError("selected model does not expose forward_batched()")
        return cast(
            torch.Tensor,
            forward_batched(
                x_all=x_all,
                y_train=y_train,
                train_test_split_index=self.train_test_split_index,
            ),
        )


def _sanitize_filename_component(value: str) -> str:
    sanitized = _SAFE_FILENAME_CHARS_RE.sub("_", value.strip())
    return sanitized.strip("._") or "graph"


def _load_json_mapping(path: Path, *, context: str) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must decode to a JSON object: {path.expanduser().resolve()}")
    return cast(dict[str, Any], payload)


def _queue_row_run_dir(queue_row: Mapping[str, Any]) -> Path:
    delta_id = ensure_non_empty_string(
        queue_row.get("delta_id", queue_row.get("delta_ref")),
        context="queue row delta_id",
    )
    return repo_root() / "outputs" / ".graph_spec_resolution" / delta_id / "train"


def resolve_queue_row_model_spec(queue_row: Mapping[str, Any]) -> ModelBuildSpec:
    cfg = compose_cfg(
        row=queue_row,
        run_dir=_queue_row_run_dir(queue_row),
        device="cpu",
    )
    task = str(getattr(cfg, "task", "classification")).strip().lower()
    raw_model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(raw_model_cfg, dict):
        raise RuntimeError("resolved queue-row cfg.model must be a mapping")
    normalized_model_cfg = {str(key): value for key, value in raw_model_cfg.items()}
    return model_build_spec_from_mappings(task=task, primary=normalized_model_cfg)


def _training_surface_record_model_spec(training_surface_record_path: Path) -> ModelBuildSpec:
    payload = _load_json_mapping(
        training_surface_record_path,
        context="training surface record",
    )
    model_payload = ensure_mapping(payload.get("model"), context="training surface record model")
    build_spec_payload = ensure_mapping(
        model_payload.get("build_spec"),
        context="training surface record model.build_spec",
    )
    normalized_build_spec = {str(key): value for key, value in build_spec_payload.items()}
    task = str(normalized_build_spec.get("task", "classification")).strip().lower()
    return model_build_spec_from_mappings(task=task, primary=normalized_build_spec)


def _checkpoint_model_spec_from_path(checkpoint_path: Path) -> ModelBuildSpec:
    payload = torch.load(
        checkpoint_path.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload must be a mapping: {checkpoint_path.expanduser().resolve()}")
    raw_cfg = payload.get("config")
    checkpoint_cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
    task = str(checkpoint_cfg.get("task", "classification")).strip().lower()
    raw_model_cfg = checkpoint_cfg.get("model")
    model_cfg = raw_model_cfg if isinstance(raw_model_cfg, dict) else {}
    raw_state_dict = payload.get("model")
    state_dict = raw_state_dict if isinstance(raw_state_dict, dict) else None
    return checkpoint_model_build_spec_from_mappings(
        task=task,
        primary=model_cfg,
        state_dict=state_dict,
    )


def resolve_anchor_model_spec(
    *,
    queue: Mapping[str, Any],
    registry_path: Path | None = None,
) -> tuple[ModelBuildSpec, dict[str, Any]]:
    anchor_run_id = ensure_non_empty_string(queue.get("anchor_run_id"), context="anchor_run_id")
    for row in ordered_rows(queue):
        row_run_id = row.get("run_id")
        if isinstance(row_run_id, str) and row_run_id == anchor_run_id:
            return resolve_queue_row_model_spec(row), {
                "source": "queue_row",
                "run_id": anchor_run_id,
                "order": int(row["order"]),
                "delta_id": str(row["delta_id"]),
            }

    registry = load_benchmark_run_registry(registry_path or default_registry_path())
    runs = ensure_mapping(registry.get("runs"), context="benchmark registry runs")
    raw_run = runs.get(anchor_run_id)
    if not isinstance(raw_run, dict):
        raise RuntimeError(f"anchor_run_id {anchor_run_id!r} is missing from the benchmark registry")
    run = cast(dict[str, Any], raw_run)
    artifacts = ensure_mapping(run.get("artifacts"), context=f"benchmark registry run {anchor_run_id}.artifacts")

    raw_training_surface_path = artifacts.get("training_surface_record_path")
    if isinstance(raw_training_surface_path, str) and raw_training_surface_path.strip():
        training_surface_path = resolve_registry_path_value(raw_training_surface_path)
        if training_surface_path.exists():
            return _training_surface_record_model_spec(training_surface_path), {
                "source": "training_surface_record",
                "run_id": anchor_run_id,
                "training_surface_record_path": str(training_surface_path),
            }

    raw_checkpoint_path = artifacts.get("best_checkpoint_path")
    if isinstance(raw_checkpoint_path, str) and raw_checkpoint_path.strip():
        checkpoint_path = resolve_registry_path_value(raw_checkpoint_path)
        if checkpoint_path.exists():
            return _checkpoint_model_spec_from_path(checkpoint_path), {
                "source": "checkpoint",
                "run_id": anchor_run_id,
                "checkpoint_path": str(checkpoint_path),
            }

    raise RuntimeError(
        "unable to resolve anchor model spec for "
        f"{anchor_run_id!r}: no matching completed sweep row, readable "
        "`training_surface_record.json`, or readable best checkpoint config"
    )


def _select_rows(
    *,
    queue: Mapping[str, Any],
    all_rows: bool,
    orders: Sequence[int],
    delta_refs: Sequence[str],
) -> list[dict[str, Any]]:
    if all_rows and (orders or delta_refs):
        raise RuntimeError("cannot combine --all-rows with --order or --delta-ref")

    queue_rows = [cast(dict[str, Any], row) for row in ordered_rows(queue)]
    if all_rows:
        return queue_rows

    order_set = {int(value) for value in orders}
    delta_set = {str(value).strip() for value in delta_refs if str(value).strip()}
    selected: list[dict[str, Any]] = []
    seen_orders: set[int] = set()
    for row in queue_rows:
        order = int(row["order"])
        delta_id = str(row["delta_id"])
        if order not in order_set and delta_id not in delta_set:
            continue
        if order in seen_orders:
            continue
        selected.append(row)
        seen_orders.add(order)

    unknown_orders = sorted(order_set.difference({int(row["order"]) for row in selected}))
    if unknown_orders:
        raise RuntimeError(f"unknown sweep order(s): {unknown_orders}")
    unknown_delta_refs = sorted(delta_set.difference({str(row['delta_id']) for row in selected}))
    if unknown_delta_refs:
        raise RuntimeError(f"unknown sweep delta_ref(s): {unknown_delta_refs}")
    return selected


def _build_targets(
    *,
    queue: Mapping[str, Any],
    anchor: bool,
    all_rows: bool,
    orders: Sequence[int],
    delta_refs: Sequence[str],
    registry_path: Path,
) -> list[GraphTarget]:
    if not anchor and not all_rows and not orders and not delta_refs:
        raise RuntimeError("select at least one target with --anchor, --all-rows, --order, or --delta-ref")

    targets: list[GraphTarget] = []
    if anchor:
        model_spec, metadata = resolve_anchor_model_spec(queue=queue, registry_path=registry_path)
        run_id = str(metadata["run_id"])
        targets.append(
            GraphTarget(
                kind="anchor",
                title=f"anchor:{run_id}",
                filename=f"anchor__{_sanitize_filename_component(run_id)}.svg",
                model_spec=model_spec,
                metadata=metadata,
            )
        )

    for row in _select_rows(
        queue=queue,
        all_rows=all_rows,
        orders=orders,
        delta_refs=delta_refs,
    ):
        order = int(row["order"])
        delta_id = str(row["delta_id"])
        targets.append(
            GraphTarget(
                kind="row",
                title=f"row:{order:02d}:{delta_id}",
                filename=f"row_{order:02d}__{_sanitize_filename_component(delta_id)}.svg",
                model_spec=resolve_queue_row_model_spec(row),
                metadata={
                    "source": "queue_row",
                    "order": order,
                    "delta_id": delta_id,
                    "run_id": row.get("run_id"),
                    "status": str(row.get("status", "")),
                },
            )
        )
    return targets


def _synthetic_inputs(spec: ModelBuildSpec) -> tuple[torch.Tensor, torch.Tensor, int]:
    train_rows = 3
    test_rows = 2
    feature_count = 4
    total_rows = train_rows + test_rows
    x_all = torch.arange(total_rows * feature_count, dtype=torch.float32).reshape(1, total_rows, feature_count)
    x_all = (x_all / float(feature_count)) - 1.0
    max_classes = 2 if spec.arch == "tabfoundry_simple" else max(2, min(int(spec.many_class_base), 4))
    y_train = torch.arange(train_rows, dtype=torch.int64).remainder(max_classes).reshape(1, train_rows)
    return x_all, y_train, train_rows


def _import_draw_graph() -> Any:
    try:
        from torchview import draw_graph
    except Exception as exc:  # pragma: no cover - exercised when dependency is missing in user env
        raise RuntimeError(
            "torchview is not installed. Run `uv sync` in this repo before using "
            "`tab-foundry research sweep graph`."
        ) from exc
    return draw_graph


def _require_graphviz_dot() -> None:
    if shutil.which("dot") is not None:
        return
    raise RuntimeError(
        "Graphviz `dot` is required to render SVG architecture graphs. Install Graphviz "
        "(for example `brew install graphviz` or `sudo apt-get install graphviz`) and "
        "ensure `dot` is on PATH."
    )


def render_graph_target(target: GraphTarget, *, out_dir: Path) -> Path:
    _require_graphviz_dot()
    draw_graph = _import_draw_graph()

    model = build_model_from_spec(target.model_spec)
    model.to(torch.device("cpu"))
    model.eval()
    surface = getattr(model, "surface", None)
    if surface is not None and getattr(surface, "head", None) == "many_class":
        raise RuntimeError(
            "tab-foundry research sweep graph currently supports direct-head models only; "
            f"got staged head='many_class' for target {target.title!r}"
        )

    x_all, y_train, train_test_split_index = _synthetic_inputs(target.model_spec)
    wrapper = _ForwardBatchedWrapper(model, train_test_split_index=train_test_split_index)
    wrapper.eval()
    with torch.no_grad():
        graph = draw_graph(
            wrapper,
            input_data=[x_all, y_train],
            graph_name=target.title,
            mode="eval",
            expand_nested=True,
        )
    visual_graph = getattr(graph, "visual_graph", None)
    pipe = getattr(visual_graph, "pipe", None)
    if not callable(pipe):
        raise RuntimeError("torchview draw_graph() did not return a graphviz-backed visual graph")
    rendered = pipe(format="svg")
    rendered_bytes = rendered if isinstance(rendered, (bytes, bytearray)) else str(rendered).encode("utf-8")
    out_path = out_dir.expanduser().resolve() / target.filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(rendered_bytes)
    return out_path


def _default_out_dir(*, sweep_id: str) -> Path:
    return repo_root() / "outputs" / "staged_ladder" / "research" / sweep_id / "architecture_graphs"


def _index_contents(
    *,
    sweep_id: str,
    targets: Sequence[GraphTarget],
    graph_paths: Sequence[Path],
    out_dir: Path,
) -> str:
    lines = [
        "# Sweep Architecture Graphs",
        "",
        f"- Sweep id: `{sweep_id}`",
        f"- Output directory: `{out_dir}`",
        f"- Target count: `{len(targets)}`",
        "",
    ]
    for target, graph_path in zip(targets, graph_paths, strict=True):
        lines.append(f"## {target.kind.title()} `{target.title}`")
        lines.append("")
        lines.append(f"- Graph path: `{graph_path}`")
        lines.append(f"- Source: `{target.metadata.get('source', 'unknown')}`")
        if target.kind == "anchor":
            lines.append(f"- Anchor run id: `{target.metadata.get('run_id')}`")
        else:
            lines.append(f"- Order: `{target.metadata.get('order')}`")
            lines.append(f"- Delta id: `{target.metadata.get('delta_id')}`")
            run_id = target.metadata.get("run_id")
            lines.append(f"- Queue run id: `{run_id if run_id is not None else 'none'}`")
        lines.append(f"- Model arch: `{target.model_spec.arch}`")
        lines.append(f"- Model stage: `{target.model_spec.stage}`")
        lines.append(f"- Model stage label: `{target.model_spec.stage_label}`")
        lines.append("")
    return "\n".join(lines)


def render_sweep_graphs(
    *,
    sweep_id: str | None = None,
    anchor: bool = False,
    all_rows: bool = False,
    orders: Sequence[int] | None = None,
    delta_refs: Sequence[str] | None = None,
    out_dir: Path | None = None,
    paths: GraphPaths | None = None,
) -> dict[str, Any]:
    resolved_paths = GraphPaths.default() if paths is None else paths
    queue = load_system_delta_queue(
        sweep_id=sweep_id,
        index_path=resolved_paths.index_path,
        catalog_path=resolved_paths.catalog_path,
        sweeps_root=resolved_paths.sweeps_root,
    )
    resolved_sweep_id = ensure_non_empty_string(queue.get("sweep_id"), context="sweep_id")
    targets = _build_targets(
        queue=queue,
        anchor=anchor,
        all_rows=all_rows,
        orders=[] if orders is None else [int(value) for value in orders],
        delta_refs=[] if delta_refs is None else [str(value) for value in delta_refs],
        registry_path=resolved_paths.registry_path,
    )
    resolved_out_dir = (out_dir or _default_out_dir(sweep_id=resolved_sweep_id)).expanduser().resolve()
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    graph_paths = [render_graph_target(target, out_dir=resolved_out_dir) for target in targets]
    index_path = resolved_out_dir / "index.md"
    _write_text(
        index_path,
        _index_contents(
            sweep_id=resolved_sweep_id,
            targets=targets,
            graph_paths=graph_paths,
            out_dir=resolved_out_dir,
        ),
    )
    return {
        "sweep_id": resolved_sweep_id,
        "out_dir": str(resolved_out_dir),
        "index_path": str(index_path),
        "graphs": [
            {
                "kind": target.kind,
                "title": target.title,
                "path": str(path),
                "source": target.metadata.get("source"),
                "stage": target.model_spec.stage,
                "stage_label": target.model_spec.stage_label,
            }
            for target, path in zip(targets, graph_paths, strict=True)
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render torchview architecture graphs for sweep targets")
    parser.add_argument("--sweep-id", default=None, help="Sweep id to inspect; defaults to the active sweep")
    parser.add_argument("--anchor", action="store_true", help="Render the selected sweep anchor graph")
    parser.add_argument("--all-rows", action="store_true", help="Render graphs for every row in the sweep")
    parser.add_argument("--order", type=int, action="append", default=[], help="Specific queue order to render")
    parser.add_argument(
        "--delta-ref",
        action="append",
        default=[],
        help="Specific delta_ref / materialized delta_id to render; repeatable",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory; defaults to outputs/staged_ladder/research/<sweep_id>/architecture_graphs",
    )
    parser.add_argument(
        "--catalog-path",
        default=str(default_catalog_path()),
        help="Path to reference/system_delta_catalog.yaml",
    )
    parser.add_argument(
        "--index-path",
        default=str(default_sweep_index_path()),
        help="Path to reference/system_delta_sweeps/index.yaml",
    )
    parser.add_argument(
        "--registry-path",
        default=str(default_registry_path()),
        help="Path to benchmark_run_registry_v1.json",
    )
    parser.add_argument(
        "--sweeps-root",
        default=str(default_sweeps_root()),
        help="Path to reference/system_delta_sweeps/",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = render_sweep_graphs(
        sweep_id=None if args.sweep_id is None else str(args.sweep_id),
        anchor=bool(args.anchor),
        all_rows=bool(args.all_rows),
        orders=[int(value) for value in args.order],
        delta_refs=[str(value) for value in args.delta_ref],
        out_dir=None if args.out_dir is None else Path(str(args.out_dir)),
        paths=GraphPaths(
            index_path=Path(str(args.index_path)).expanduser().resolve(),
            catalog_path=Path(str(args.catalog_path)).expanduser().resolve(),
            sweeps_root=Path(str(args.sweeps_root)).expanduser().resolve(),
            registry_path=Path(str(args.registry_path)).expanduser().resolve(),
        ),
    )
    print(
        "Sweep graph render complete.",
        f"sweep_id={result['sweep_id']}",
        f"graphs={len(result['graphs'])}",
        f"index={result['index_path']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
