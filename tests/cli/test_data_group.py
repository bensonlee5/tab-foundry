from __future__ import annotations

import tab_foundry.cli.groups.data as data_group


def test_run_corpus_inspect_prints_materialization_summary(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        data_group,
        "load_corpus_record",
        lambda corpus_ref: {
            "corpus_ref": corpus_ref,
            "recipe_id": "current_recipe",
            "surface_label": "current_surface",
            "manifest": {
                "manifest_path": "/tmp/manifest.parquet",
                "inspection": {
                    "total_records": 12,
                    "split_counts": {"test": 1, "train": 10, "val": 1},
                },
            },
            "dagzoo_provenance_summary": {
                "materialization_timing": {
                    "materialization_mode": "fresh",
                    "materialization_reused": False,
                    "materialize_processes": 4,
                    "materialize_worker_threads": 2,
                    "compact_workers": 3,
                    "compact_shard_workers": 1,
                    "manifest_workers": 8,
                    "invocation_count": 2,
                    "cumulative_round_count": 3,
                    "cumulative_generated_datasets": 12,
                    "cumulative_accepted_datasets": 10,
                    "cumulative_rejected_datasets": 2,
                    "cumulative_curated_accepted_datasets": 10,
                    "cumulative_source_shard_count": 5,
                    "cumulative_output_shard_count": 3,
                    "invocation_fanout_elapsed_seconds": 5.0,
                    "cumulative_generate_elapsed_seconds": 9.0,
                    "cumulative_filter_elapsed_seconds": 4.0,
                    "cumulative_copy_elapsed_seconds": 2.0,
                    "staged_compaction_elapsed_seconds": 1.5,
                    "manifest_build_elapsed_seconds": 0.5,
                    "promotion_elapsed_seconds": 0.25,
                    "recipe_elapsed_seconds": 12.0,
                    "staged_compaction_status": "already_compacted",
                    "staged_compaction_reused": True,
                }
            },
        },
    )

    exit_code = data_group._run_corpus_inspect(
        corpus_ref="current_recipe/current_recipe__123456789abc",
        json_mode=False,
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Materialization:" in output
    assert "Workers: processes=4, worker_threads=2, compact_workers=3" in output
    assert "compact_shard_workers=1" in output
    assert "manifest_workers=8" in output
    assert "Totals: invocations=2, rounds=3, generated=12" in output
    assert "accepted=10" in output
    assert "source_shards=5" in output
    assert "Elapsed: fanout=5.00s" in output
    assert "Status: mode=fresh, reused=False" in output
    assert "staged_compaction=already_compacted" in output
