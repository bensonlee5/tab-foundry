"""Manifest builder for cauchy-generator packed parquet shard outputs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5, sha1
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(slots=True)
class ManifestSummary:
    """Build summary."""

    out_path: Path
    total_records: int
    train_records: int
    val_records: int
    test_records: int


def _stable_split(key: str, train_ratio: float, val_ratio: float) -> str:
    """Split by deterministic hash."""

    token = int(md5(key.encode("utf-8")).hexdigest(), 16) % 10_000
    p = token / 10_000.0
    if p < train_ratio:
        return "train"
    if p < train_ratio + val_ratio:
        return "val"
    return "test"


def _root_id(root: Path) -> str:
    token = root.expanduser().resolve().as_posix().encode("utf-8")
    return sha1(token).hexdigest()[:12]


def _dataset_id(
    *,
    root_id: str,
    shard_id: int,
    dataset_index: int,
    meta: dict[str, Any],
) -> str:
    """Stable dataset ID with root-level uniqueness."""

    if shard_id >= 0 and dataset_index >= 0:
        return f"root_{root_id}/shard_{shard_id:05d}/dataset_{dataset_index:06d}"

    # Fallback for non-standard directory names.
    stable_payload = {
        "seed": int(meta.get("seed", -1)),
        "n_features": int(meta.get("n_features", -1)),
        "n_classes": int(meta["n_classes"]) if meta.get("n_classes") is not None else -1,
        "curriculum_stage": (
            int(meta.get("curriculum", {}).get("stage"))
            if meta.get("curriculum", {}).get("stage") is not None
            else -1
        ),
    }
    token = json.dumps(stable_payload, sort_keys=True, separators=(",", ":"))
    return f"root_{root_id}/dataset_{md5(token.encode('utf-8')).hexdigest()[:16]}"


def _manifest_relative_path(path: Path, *, manifest_dir: Path) -> str:
    """Serialize data path relative to manifest directory when possible."""

    absolute = path.expanduser().resolve()
    try:
        return os.path.relpath(absolute, start=manifest_dir)
    except ValueError:
        # Cross-volume on Windows may fail; retain absolute path fallback.
        return absolute.as_posix()


def _infer_task(meta: dict[str, Any]) -> str:
    config_task = (
        meta.get("config", {})
        .get("dataset", {})
        .get("task")
    )
    if config_task in {"classification", "regression"}:
        return str(config_task)
    n_classes = meta.get("n_classes")
    return "classification" if n_classes is not None else "regression"


def _extract_shard_id(shard_dir: Path) -> int:
    name = shard_dir.name
    if not name.startswith("shard_"):
        return -1
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return -1


def _iter_shard_dirs(root: Path) -> list[Path]:
    """Return sorted shard directories."""

    shard_dirs: list[Path] = []
    for shard_dir in sorted(root.rglob("shard_*")):
        if not shard_dir.is_dir():
            continue
        shard_dirs.append(shard_dir)
    return shard_dirs


def _read_metadata_records(metadata_path: Path) -> list[tuple[int, int, dict[str, Any]]]:
    """Read metadata.ndjson records and include byte offsets for random access."""

    records: list[tuple[int, int, dict[str, Any]]] = []
    with metadata_path.open("rb") as handle:
        while True:
            offset = int(handle.tell())
            line = handle.readline()
            if not line:
                break

            size = int(len(line))
            stripped = line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped.decode("utf-8"))
            except Exception as exc:  # pragma: no cover - defensive parse context
                raise RuntimeError(
                    f"failed to parse NDJSON metadata record in {metadata_path} at byte offset {offset}"
                ) from exc
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"metadata record must be a JSON object: path={metadata_path}, offset={offset}"
                )
            records.append((offset, size, payload))
    return records


def build_manifest(
    data_roots: list[Path],
    out_path: Path,
    *,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
) -> ManifestSummary:
    """Scan parquet roots and persist manifest parquet."""

    if not data_roots:
        raise ValueError("data_roots must not be empty")
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("invalid split ratios")

    out_path = out_path.expanduser().resolve()
    manifest_dir = out_path.parent
    roots = sorted({root.expanduser().resolve() for root in data_roots})

    records: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        source_root_id = _root_id(root)
        for shard_dir in _iter_shard_dirs(root):
            train_path = shard_dir / "train.parquet"
            test_path = shard_dir / "test.parquet"
            metadata_path = shard_dir / "metadata.ndjson"
            if not (train_path.exists() and test_path.exists() and metadata_path.exists()):
                continue

            shard_id = _extract_shard_id(shard_dir)
            for offset, size, record in _read_metadata_records(metadata_path):
                if "dataset_index" not in record:
                    raise RuntimeError(
                        f"metadata record missing dataset_index: path={metadata_path}, offset={offset}"
                    )
                dataset_index = int(record["dataset_index"])
                meta_raw = record.get("metadata")
                if not isinstance(meta_raw, dict):
                    raise RuntimeError(
                        "metadata record missing object payload at key 'metadata': "
                        f"path={metadata_path}, dataset_index={dataset_index}"
                    )
                meta = meta_raw

                dsid = _dataset_id(
                    root_id=source_root_id,
                    shard_id=shard_id,
                    dataset_index=dataset_index,
                    meta=meta,
                )
                split = _stable_split(dsid, train_ratio, val_ratio)

                records.append(
                    {
                        "dataset_id": dsid,
                        "source_root_id": source_root_id,
                        "split": split,
                        "task": _infer_task(meta),
                        "shard_id": shard_id,
                        "dataset_index": dataset_index,
                        "train_path": _manifest_relative_path(train_path, manifest_dir=manifest_dir),
                        "test_path": _manifest_relative_path(test_path, manifest_dir=manifest_dir),
                        "metadata_path": _manifest_relative_path(metadata_path, manifest_dir=manifest_dir),
                        "metadata_offset_bytes": offset,
                        "metadata_size_bytes": size,
                        "n_train": int(record.get("n_train", -1)),
                        "n_test": int(record.get("n_test", -1)),
                        "n_features": int(record.get("n_features", meta.get("n_features", -1))),
                        "n_classes": (
                            int(meta["n_classes"]) if meta.get("n_classes") is not None else None
                        ),
                        "seed": int(meta.get("seed", -1)),
                        "curriculum_stage": (
                            int(meta.get("curriculum", {}).get("stage"))
                            if meta.get("curriculum", {}).get("stage") is not None
                            else None
                        ),
                    }
                )

    if not records:
        raise RuntimeError("no datasets discovered while building manifest")

    records.sort(
        key=lambda record: (
            str(record["source_root_id"]),
            int(record["shard_id"]),
            int(record["dataset_index"]),
            str(record["dataset_id"]),
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(records)
    pq.write_table(table, out_path, compression="zstd")

    train_records = sum(1 for record in records if record["split"] == "train")
    val_records = sum(1 for record in records if record["split"] == "val")
    test_records = sum(1 for record in records if record["split"] == "test")
    return ManifestSummary(
        out_path=out_path,
        total_records=len(records),
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
    )
