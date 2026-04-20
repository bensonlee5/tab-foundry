"""Render sweep architecture graphs with torchview."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence, cast

import torch

from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.inspection import synthetic_forward_batch
from tab_foundry.model.spec import ModelBuildSpec, SANDWICH_FAMILY_MODEL_ARCHES

from tab_foundry.research.lane_contract import resolve_training_surface_context

from .queue_loading import (
    load_system_delta_queue,
    ordered_rows,
)
from .paths_io import (
    default_catalog_path,
    default_registry_path,
    default_sweep_index_path,
    default_sweeps_root,
    repo_root,
    write_text,
)
from . import surface_resolution as surface_resolution_module


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

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        train_test_split_index: int,
        feature_types: list[str] | list[list[str]] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.train_test_split_index = int(train_test_split_index)
        self.feature_types = feature_types

    def forward(self, x_all: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        forward_batched = getattr(self.model, "forward_batched", None)
        if not callable(forward_batched):
            raise RuntimeError("selected model does not expose forward_batched()")
        batched_kwargs: dict[str, Any] = {
            "x_all": x_all,
            "y_train": y_train,
            "train_test_split_index": self.train_test_split_index,
        }
        if str(getattr(self.model, "arch", "")).strip().lower() in SANDWICH_FAMILY_MODEL_ARCHES:
            batched_kwargs["feature_types"] = self.feature_types
        return cast(
            torch.Tensor,
            forward_batched(**batched_kwargs),
        )


def _sanitize_filename_component(value: str) -> str:
    sanitized = _SAFE_FILENAME_CHARS_RE.sub("_", value.strip())
    return sanitized.strip("._") or "graph"


def _require_non_empty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string")
    return str(value)


def resolve_queue_row_model_spec(
    queue_row: Mapping[str, Any],
    *,
    training_experiment: str,
) -> ModelBuildSpec:
    return surface_resolution_module.resolve_queue_row_model_spec(
        queue_row,
        training_experiment=training_experiment,
    )


def resolve_anchor_originating_queue_row(
    *,
    queue: Mapping[str, Any],
    registry_path: Path | None = None,
    index_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    return surface_resolution_module.resolve_anchor_originating_queue_row(
        queue=queue,
        registry_path=registry_path,
        index_path=index_path,
        sweeps_root=sweeps_root,
    )


def resolve_anchor_model_spec(
    *,
    queue: Mapping[str, Any],
    registry_path: Path | None = None,
    index_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> tuple[ModelBuildSpec, dict[str, Any]]:
    return surface_resolution_module.resolve_anchor_model_spec(
        queue=queue,
        registry_path=registry_path,
        index_path=index_path,
        sweeps_root=sweeps_root,
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
    index_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> list[GraphTarget]:
    if not anchor and not all_rows and not orders and not delta_refs:
        raise RuntimeError("select at least one target with --anchor, --all-rows, --order, or --delta-ref")

    training_experiment = resolve_training_surface_context(queue).training_experiment
    targets: list[GraphTarget] = []
    if anchor:
        model_spec, metadata = resolve_anchor_model_spec(
            queue=queue,
            registry_path=registry_path,
            index_path=index_path,
            sweeps_root=sweeps_root,
        )
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
                model_spec=resolve_queue_row_model_spec(
                    row,
                    training_experiment=training_experiment,
                ),
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

    forward_batch = synthetic_forward_batch(target.model_spec)
    wrapper = _ForwardBatchedWrapper(
        model,
        train_test_split_index=forward_batch.train_test_split_index,
        feature_types=forward_batch.task_batch.metadata.get("feature_types"),
    )
    wrapper.eval()
    with torch.no_grad():
        graph = draw_graph(
            wrapper,
            input_data=[forward_batch.x_all, forward_batch.y_train_batched],
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
    resolved_sweep_id = _require_non_empty_string(queue.get("sweep_id"), context="sweep_id")
    targets = _build_targets(
        queue=queue,
        anchor=anchor,
        all_rows=all_rows,
        orders=[] if orders is None else [int(value) for value in orders],
        delta_refs=[] if delta_refs is None else [str(value) for value in delta_refs],
        registry_path=resolved_paths.registry_path,
        index_path=resolved_paths.index_path,
        sweeps_root=resolved_paths.sweeps_root,
    )
    resolved_out_dir = (out_dir or _default_out_dir(sweep_id=resolved_sweep_id)).expanduser().resolve()
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    graph_paths = [render_graph_target(target, out_dir=resolved_out_dir) for target in targets]
    index_path = resolved_out_dir / "index.md"
    write_text(
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
