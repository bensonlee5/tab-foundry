"""Concrete sweep corpus materialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.data.corpus_materialization import materialize_corpus_refs_batch

from .configuration import _effective_queue_corpus_ref, _resolved_repo_root
from .queue_loading import load_system_delta_queue, write_resolved_system_delta_queue


def materialize_sweep_corpora(
    *,
    dagzoo_root: Path,
    sweep_id: str | None = None,
    force: bool = False,
    materialize_processes: int | None = None,
    materialize_worker_threads: int | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    try:
        _ = write_resolved_system_delta_queue(
            sweep_id=sweep_id,
            index_path=index_path,
            catalog_path=catalog_path,
            sweeps_root=sweeps_root,
        )
    except FileNotFoundError:
        pass
    queue = load_system_delta_queue(
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    resolved_repo_root = _resolved_repo_root(catalog_path=catalog_path, sweeps_root=sweeps_root)
    resolved_sweep_id = str(queue["sweep_id"])
    requested_corpus_refs: list[str] = []
    seen_corpus_refs: set[str] = set()
    raw_rows = queue.get("rows")
    if not isinstance(raw_rows, list):
        raise RuntimeError("materialized system delta queue must include rows")
    ordered_rows_for_corpora = sorted(
        [cast(Mapping[str, Any], row) for row in raw_rows if isinstance(row, Mapping)],
        key=lambda row: (
            int(row.get("order", 0)),
            str(row.get("delta_id", row.get("delta_ref", ""))),
        ),
    )
    for row in ordered_rows_for_corpora:
        raw_data = row.get("data")
        data_payload = cast(Mapping[str, Any], raw_data) if isinstance(raw_data, Mapping) else {}
        normalized_corpus_ref = _effective_queue_corpus_ref(data_payload)
        if normalized_corpus_ref is None:
            continue
        if normalized_corpus_ref in seen_corpus_refs:
            continue
        seen_corpus_refs.add(normalized_corpus_ref)
        requested_corpus_refs.append(normalized_corpus_ref)
    records = materialize_corpus_refs_batch(
        corpus_refs=requested_corpus_refs,
        dagzoo_root=dagzoo_root,
        force=force,
        materialize_processes=materialize_processes,
        materialize_worker_threads=materialize_worker_threads,
        repo_root=resolved_repo_root,
        sweep_id=resolved_sweep_id,
        sweeps_root=sweeps_root,
    )
    return {
        "sweep_id": resolved_sweep_id,
        "recipe_count": len(records),
        "requested_corpus_refs": requested_corpus_refs,
        "requested_recipe_ids": [
            corpus_ref.split("/", 1)[0]
            for corpus_ref in requested_corpus_refs
        ],
        "corpus_refs": [str(record["corpus_ref"]) for record in records],
        "records": records,
    }
