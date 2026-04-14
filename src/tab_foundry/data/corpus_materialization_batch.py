"""Batch-oriented corpus materialization helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from .corpus_loading import _ensure_non_empty_string, _repo_root
from .corpus_lookup import _load_reusable_corpus_record
from .corpus_materialization_recipe import (
    _load_reusable_recipe_record_for_request,
    materialize_corpus_recipe,
)
from .corpus_materialization_shared import (
    _SUBPROCESS_POLL_INTERVAL_SECONDS,
    _resolve_materialize_processes,
)


@dataclass(slots=True)
class _BatchedRecipeRequest:
    recipe_id: str
    requires_recipe_record: bool = False
    exact_refs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _PendingRecipeWorkerMaterialization:
    recipe_id: str
    dagzoo_root: Path
    force: bool
    repo_root: Path
    requested_exact_ref: str | None
    requires_recipe_record: bool
    sweep_id: str | None
    sweeps_root: Path | None


@dataclass(slots=True)
class _ActiveRecipeProcess:
    process: subprocess.Popen[str]
    pending: _PendingRecipeWorkerMaterialization
    materialize_processes: int
    result_path: Path


def _recipe_worker_argv(
    *,
    recipe_id: str,
    dagzoo_root: Path,
    force: bool,
    repo_root: Path,
    result_path: Path,
    materialize_processes: int,
    materialize_worker_threads: int | None,
    compact_workers: int | None,
    compact_shard_workers: int | None,
    manifest_workers: int | None,
    sweep_id: str | None,
    sweeps_root: Path | None,
) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "tab_foundry.data.corpus_materialization_recipe_worker",
        "--recipe-id",
        str(recipe_id),
        "--dagzoo-root",
        str(dagzoo_root.expanduser().resolve()),
        "--repo-root",
        str(repo_root.expanduser().resolve()),
        "--result-path",
        str(result_path.expanduser().resolve()),
        "--materialize-processes",
        str(int(materialize_processes)),
    ]
    if materialize_worker_threads is not None:
        argv.extend(
            [
                "--materialize-worker-threads",
                str(int(materialize_worker_threads)),
            ]
        )
    if compact_workers is not None:
        argv.extend(["--compact-workers", str(int(compact_workers))])
    if compact_shard_workers is not None:
        argv.extend(
            [
                "--compact-shard-workers",
                str(int(compact_shard_workers)),
            ]
        )
    if manifest_workers is not None:
        argv.extend(["--manifest-workers", str(int(manifest_workers))])
    if force:
        argv.append("--force")
    if sweep_id is not None:
        argv.extend(["--sweep-id", str(sweep_id)])
    if sweeps_root is not None:
        argv.extend(["--sweeps-root", str(sweeps_root.expanduser().resolve())])
    return argv


def _terminate_active_recipe_subprocesses(
    active_processes: Mapping[int, _ActiveRecipeProcess],
) -> None:
    for active_process in active_processes.values():
        process = active_process.process
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5.0
    for active_process in active_processes.values():
        process = active_process.process
        if process.poll() is not None:
            continue
        remaining = deadline - time.monotonic()
        try:
            process.wait(timeout=max(0.0, remaining))
        except subprocess.TimeoutExpired:
            process.kill()
    for active_process in active_processes.values():
        process = active_process.process
        if process.poll() is None:
            process.wait()


def _materialize_process_slot_allocations(
    *,
    total_processes: int,
    worker_count: int,
) -> list[int]:
    if worker_count <= 0:
        return []
    resolved_total = max(1, int(total_processes))
    resolved_workers = max(1, int(worker_count))
    base, remainder = divmod(resolved_total, resolved_workers)
    return [
        base + (1 if index < remainder else 0)
        for index in range(resolved_workers)
    ]


def _load_completed_recipe_worker_record(
    active_process: _ActiveRecipeProcess,
) -> dict[str, Any]:
    try:
        process = active_process.process
        returncode = process.returncode
        if returncode is None:
            returncode = process.poll()
        if returncode is None:
            raise RuntimeError(
                "recipe materialization subprocess ended without a return code: "
                f"recipe_id={active_process.pending.recipe_id!r}"
            )
        if int(returncode) != 0:
            raise RuntimeError(
                "recipe materialization subprocess failed: "
                f"recipe_id={active_process.pending.recipe_id!r} "
                f"returncode={int(returncode)} "
                f"argv={process.args!r}"
            )
        if not active_process.result_path.exists():
            raise RuntimeError(
                "recipe materialization subprocess produced no JSON record: "
                f"recipe_id={active_process.pending.recipe_id!r} "
                f"argv={process.args!r}"
            )
        payload = json.loads(active_process.result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                "recipe materialization subprocess JSON payload must decode to an object: "
                f"recipe_id={active_process.pending.recipe_id!r} "
                f"argv={process.args!r}"
            )
        return {str(key): value for key, value in payload.items()}
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "recipe materialization subprocess emitted invalid JSON: "
            f"recipe_id={active_process.pending.recipe_id!r} "
            f"argv={process.args!r}"
        ) from exc
    finally:
        active_process.result_path.unlink(missing_ok=True)


def _materialize_pending_recipes_with_subprocess_fanout(
    *,
    pending_requests: Sequence[_PendingRecipeWorkerMaterialization],
    materialize_processes: int | None,
    materialize_worker_threads: int | None,
    compact_workers: int | None,
    compact_shard_workers: int | None,
    manifest_workers: int | None,
    prioritized_recipe_ids: Sequence[str],
    on_recipe_materialized: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if not pending_requests:
        return []

    total_process_budget = _resolve_materialize_processes(materialize_processes)
    max_workers = min(total_process_budget, len(pending_requests))
    available_process_slots = deque(
        _materialize_process_slot_allocations(
            total_processes=total_process_budget,
            worker_count=max_workers,
        )
    )
    prioritized_ids = {str(recipe_id) for recipe_id in prioritized_recipe_ids}
    launch_queue = deque(
        [
            pending
            for pending in pending_requests
            if pending.recipe_id in prioritized_ids
        ]
        + [
            pending
            for pending in pending_requests
            if pending.recipe_id not in prioritized_ids
        ]
    )
    active_processes: dict[int, _ActiveRecipeProcess] = {}
    finalized_records: list[dict[str, Any]] = []

    try:
        while launch_queue or active_processes:
            while launch_queue and available_process_slots:
                pending = launch_queue.popleft()
                allocated_processes = available_process_slots.popleft()
                result_fd, result_path_raw = tempfile.mkstemp(
                    prefix="tab-foundry-corpus-materialization-",
                    suffix=".json",
                )
                os.close(result_fd)
                result_path = Path(result_path_raw)
                try:
                    process = subprocess.Popen(
                        _recipe_worker_argv(
                            recipe_id=pending.recipe_id,
                            dagzoo_root=pending.dagzoo_root,
                            force=pending.force,
                            repo_root=pending.repo_root,
                            result_path=result_path,
                            materialize_processes=allocated_processes,
                            materialize_worker_threads=materialize_worker_threads,
                            compact_workers=compact_workers,
                            compact_shard_workers=compact_shard_workers,
                            manifest_workers=manifest_workers,
                            sweep_id=pending.sweep_id,
                            sweeps_root=pending.sweeps_root,
                        ),
                        cwd=pending.repo_root,
                        text=True,
                    )
                except Exception:
                    result_path.unlink(missing_ok=True)
                    raise
                active_processes[int(process.pid)] = _ActiveRecipeProcess(
                    process=process,
                    pending=pending,
                    materialize_processes=allocated_processes,
                    result_path=result_path,
                )

            completed_pid: int | None = None
            completed_active_process: _ActiveRecipeProcess | None = None
            while completed_active_process is None:
                for pid, active_process in list(active_processes.items()):
                    if active_process.process.poll() is None:
                        continue
                    completed_pid = pid
                    completed_active_process = active_process
                    break
                if completed_active_process is None:
                    time.sleep(_SUBPROCESS_POLL_INTERVAL_SECONDS)

            assert completed_pid is not None
            del active_processes[completed_pid]
            assert completed_active_process is not None
            available_process_slots.appendleft(completed_active_process.materialize_processes)
            record = _load_completed_recipe_worker_record(completed_active_process)
            finalized_records.append(record)
            if on_recipe_materialized is not None:
                on_recipe_materialized(record)
        return finalized_records
    finally:
        if active_processes:
            _terminate_active_recipe_subprocesses(active_processes)
        for active_process in active_processes.values():
            active_process.result_path.unlink(missing_ok=True)


def materialize_corpus_ref(
    *,
    corpus_ref: str,
    dagzoo_root: Path,
    force: bool = False,
    materialize_processes: int | None = None,
    materialize_worker_threads: int | None = None,
    compact_workers: int | None = None,
    compact_shard_workers: int | None = None,
    manifest_workers: int | None = None,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    normalized_corpus_ref = _ensure_non_empty_string(corpus_ref, context="corpus_ref")
    recipe_id, separator, _corpus_id = normalized_corpus_ref.partition("/")
    if not separator:
        return materialize_corpus_recipe(
            recipe_id=recipe_id,
            dagzoo_root=dagzoo_root,
            force=force,
            materialize_processes=materialize_processes,
            materialize_worker_threads=materialize_worker_threads,
            compact_workers=compact_workers,
            compact_shard_workers=compact_shard_workers,
            manifest_workers=manifest_workers,
            repo_root=repo_root,
            sweep_id=sweep_id,
            sweeps_root=sweeps_root,
        )

    if not force:
        existing_record = _load_reusable_corpus_record(
            normalized_corpus_ref,
            repo_root=repo_root,
            sweep_id=sweep_id,
            sweeps_root=sweeps_root,
        )
        if existing_record is not None:
            return existing_record

    record = materialize_corpus_recipe(
        recipe_id=recipe_id,
        dagzoo_root=dagzoo_root,
        force=force,
        materialize_processes=materialize_processes,
        materialize_worker_threads=materialize_worker_threads,
        compact_workers=compact_workers,
        compact_shard_workers=compact_shard_workers,
        manifest_workers=manifest_workers,
        repo_root=repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    materialized_corpus_ref = _ensure_non_empty_string(
        record.get("corpus_ref"),
        context="materialized corpus record corpus_ref",
    )
    if materialized_corpus_ref != normalized_corpus_ref:
        raise RuntimeError(
            f"requested corpus_ref {normalized_corpus_ref!r} is pinned to an exact corpus id, "
            f"but materializing recipe {recipe_id!r} produced {materialized_corpus_ref!r}"
        )
    return record


def materialize_corpus_refs_batch(
    *,
    corpus_refs: Sequence[str],
    dagzoo_root: Path,
    force: bool = False,
    materialize_processes: int | None = None,
    materialize_worker_threads: int | None = None,
    compact_workers: int | None = None,
    compact_shard_workers: int | None = None,
    manifest_workers: int | None = None,
    prioritized_recipe_ids: Sequence[str] = (),
    on_corpus_materialized: Callable[[dict[str, Any]], None] | None = None,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> list[dict[str, Any]]:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    resolved_dagzoo_root = dagzoo_root.expanduser().resolve()
    normalized_refs = [
        _ensure_non_empty_string(corpus_ref, context="corpus_ref") for corpus_ref in corpus_refs
    ]
    grouped_requests: dict[str, _BatchedRecipeRequest] = {}
    ordered_recipe_ids: list[str] = []
    records_by_recipe_id: dict[str, dict[str, Any]] = {}
    records_by_exact_ref: dict[str, dict[str, Any]] = {}
    notified_recipe_ids: set[str] = set()
    pending_requests: list[_PendingRecipeWorkerMaterialization] = []

    def _dispatch_corpus_materialized(record: dict[str, Any]) -> None:
        if on_corpus_materialized is None:
            return
        recipe_id = _ensure_non_empty_string(
            record.get("recipe_id"),
            context="materialized corpus record recipe_id",
        )
        if recipe_id in notified_recipe_ids:
            return
        on_corpus_materialized(record)
        notified_recipe_ids.add(recipe_id)

    for normalized_ref in normalized_refs:
        recipe_id, separator, _corpus_id = normalized_ref.partition("/")
        request = grouped_requests.get(recipe_id)
        if request is None:
            request = _BatchedRecipeRequest(recipe_id=recipe_id)
            grouped_requests[recipe_id] = request
            ordered_recipe_ids.append(recipe_id)
        if separator:
            if normalized_ref not in request.exact_refs:
                request.exact_refs.append(normalized_ref)
            continue
        request.requires_recipe_record = True

    for recipe_id in ordered_recipe_ids:
        request = grouped_requests[recipe_id]
        cached_exact_records: dict[str, dict[str, Any]] = {}
        unresolved_exact_refs: list[str] = []
        for exact_ref in request.exact_refs:
            cached_record = None
            if not force:
                cached_record = _load_reusable_corpus_record(
                    exact_ref,
                    repo_root=resolved_repo_root,
                    sweep_id=sweep_id,
                    sweeps_root=sweeps_root,
                )
            if cached_record is None:
                unresolved_exact_refs.append(exact_ref)
                continue
            records_by_exact_ref[exact_ref] = cached_record
            cached_exact_records[exact_ref] = cached_record

        distinct_unresolved_exact_refs = list(dict.fromkeys(unresolved_exact_refs))
        if len(distinct_unresolved_exact_refs) > 1:
            raise RuntimeError(
                "batch requested multiple pinned corpus ids for recipe "
                f"{recipe_id!r}: {distinct_unresolved_exact_refs!r}. "
                "Materialize those exact corpus refs separately."
            )
        unresolved_exact_ref = (
            distinct_unresolved_exact_refs[0] if distinct_unresolved_exact_refs else None
        )
        if unresolved_exact_ref is not None:
            pending_requests.append(
                _PendingRecipeWorkerMaterialization(
                    recipe_id=recipe_id,
                    dagzoo_root=resolved_dagzoo_root,
                    force=force,
                    repo_root=resolved_repo_root,
                    requested_exact_ref=unresolved_exact_ref,
                    requires_recipe_record=request.requires_recipe_record,
                    sweep_id=sweep_id,
                    sweeps_root=sweeps_root,
                )
            )
            continue

        recipe_record = None
        if request.requires_recipe_record:
            recipe_record = _load_reusable_recipe_record_for_request(
                recipe_id=recipe_id,
                force=force,
                repo_root=resolved_repo_root,
                sweep_id=sweep_id,
                sweeps_root=sweeps_root,
            )
            if recipe_record is None:
                pending_requests.append(
                    _PendingRecipeWorkerMaterialization(
                        recipe_id=recipe_id,
                        dagzoo_root=resolved_dagzoo_root,
                        force=force,
                        repo_root=resolved_repo_root,
                        requested_exact_ref=None,
                        requires_recipe_record=True,
                        sweep_id=sweep_id,
                        sweeps_root=sweeps_root,
                    )
                )
                continue
            records_by_recipe_id[recipe_id] = recipe_record

        notification_record = recipe_record
        if notification_record is None and cached_exact_records:
            notification_record = records_by_exact_ref[request.exact_refs[0]]
        if notification_record is not None:
            _dispatch_corpus_materialized(notification_record)

    if pending_requests:
        pending_by_recipe_id = {pending.recipe_id: pending for pending in pending_requests}

        def _on_recipe_materialized(record: dict[str, Any]) -> None:
            recipe_id = _ensure_non_empty_string(
                record.get("recipe_id"),
                context="recipe worker record recipe_id",
            )
            pending = pending_by_recipe_id[recipe_id]
            materialized_corpus_ref = _ensure_non_empty_string(
                record.get("corpus_ref"),
                context="recipe worker record corpus_ref",
            )
            if (
                pending.requested_exact_ref is not None
                and materialized_corpus_ref != pending.requested_exact_ref
            ):
                raise RuntimeError(
                    f"requested corpus_ref {pending.requested_exact_ref!r} is pinned to an exact corpus id, "
                    f"but materializing recipe {recipe_id!r} produced {materialized_corpus_ref!r}"
                )
            if pending.requires_recipe_record:
                records_by_recipe_id[recipe_id] = record
            if pending.requested_exact_ref is not None:
                records_by_exact_ref[pending.requested_exact_ref] = record
            _dispatch_corpus_materialized(record)

        _ = _materialize_pending_recipes_with_subprocess_fanout(
            pending_requests=pending_requests,
            materialize_processes=materialize_processes,
            materialize_worker_threads=materialize_worker_threads,
            compact_workers=compact_workers,
            compact_shard_workers=compact_shard_workers,
            manifest_workers=manifest_workers,
            prioritized_recipe_ids=prioritized_recipe_ids,
            on_recipe_materialized=_on_recipe_materialized,
        )

    resolved_records: list[dict[str, Any]] = []
    for normalized_ref in normalized_refs:
        recipe_id, separator, _corpus_id = normalized_ref.partition("/")
        if separator:
            resolved_records.append(records_by_exact_ref[normalized_ref])
            continue
        resolved_records.append(records_by_recipe_id[recipe_id])
    return resolved_records
