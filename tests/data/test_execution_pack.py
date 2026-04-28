from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tab_foundry.data.execution_pack import materialize_exact_shape_execution_pack


def _write_source_manifest(path: Path) -> Path:
    rows: list[dict[str, object]] = []
    dataset_index = 0
    for n_features in (6, 32):
        for mechanism in ("mcar", "mar"):
            for n_classes in (2, 3):
                for item_index in range(4):
                    dataset_index += 1
                    invocation = f"{mechanism}_r0128_f{n_features:03d}_c{n_classes:02d}"
                    rows.append(
                        {
                            "dataset_id": f"d{dataset_index}",
                            "source_root_id": "source",
                            "source_shard_relpath": "shard_00000",
                            "split": "train",
                            "task": "classification",
                            "shard_id": 0,
                            "dataset_index": dataset_index,
                            "train_path": f"invocations/{invocation}/curated/shard_00000/train.parquet",
                            "test_path": f"invocations/{invocation}/curated/shard_00000/test.parquet",
                            "catalog_path": f"invocations/{invocation}/curated/shard_00000/dataset_catalog.parquet",
                            "catalog_dataset_index": dataset_index,
                            "catalog_record_sha256": "0" * 64,
                            "n_train": 96,
                            "n_test": 32,
                            "n_features": n_features,
                            "n_classes": n_classes,
                            "filter_mode": "deferred",
                            "filter_status": "accepted",
                            "filter_accepted": True,
                            "missing_value_policy": "allow_any",
                            "missing_value_status": "contains_nan_or_inf",
                        }
                    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_materialize_exact_shape_execution_pack_preserves_exact_signatures_and_no_reuse(
    tmp_path: Path,
) -> None:
    source_manifest = _write_source_manifest(tmp_path / "source" / "manifest.parquet")
    out_manifest = tmp_path / "pack" / "manifest.parquet"

    summary = materialize_exact_shape_execution_pack(
        source_manifest_path=source_manifest,
        out_manifest_path=out_manifest,
        max_steps=4,
        grad_accum_steps=2,
        task_batch_size=2,
        signature_family_optimizer_step_block_length=1,
        total_train_tasks=16,
    )

    rows = pq.read_table(out_manifest).to_pylist()
    assert summary.selected_records == 16
    assert len({row["dataset_id"] for row in rows}) == 16
    assert {row["execution_pack_signature"] for row in rows} == {
        "96x32x6x2",
        "96x32x6x3",
        "96x32x32x2",
        "96x32x32x3",
    }
    for batch_start in range(0, len(rows), 2):
        batch = rows[batch_start : batch_start + 2]
        signatures = {row["execution_pack_signature"] for row in batch}
        assert len(signatures) == 1


def test_materialize_exact_shape_execution_pack_orders_family_blocks_and_balances_regimes(
    tmp_path: Path,
) -> None:
    source_manifest = _write_source_manifest(tmp_path / "source" / "manifest.parquet")
    out_manifest = tmp_path / "pack" / "manifest.parquet"

    summary = materialize_exact_shape_execution_pack(
        source_manifest_path=source_manifest,
        out_manifest_path=out_manifest,
        max_steps=4,
        grad_accum_steps=2,
        task_batch_size=2,
        signature_family_optimizer_step_block_length=1,
        total_train_tasks=16,
    )

    rows = pq.read_table(out_manifest).to_pylist()
    block_families = [
        {row["execution_pack_signature_family"] for row in rows[0:4]},
        {row["execution_pack_signature_family"] for row in rows[4:8]},
        {row["execution_pack_signature_family"] for row in rows[8:12]},
        {row["execution_pack_signature_family"] for row in rows[12:16]},
    ]
    assert block_families == [{"96x32x6"}, {"96x32x32"}, {"96x32x6"}, {"96x32x32"}]
    assert summary.regime_counts == {"mar": 8, "mcar": 8}
    assert summary.class_counts == {"2": 8, "3": 8}
    assert summary.feature_counts == {"6": 8, "32": 8}
