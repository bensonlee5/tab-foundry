from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
import pytest

from tab_foundry.config import compose_config
import tab_foundry.data.corpus_loading as corpus_loading_module
import tab_foundry.data.corpus_lookup as corpus_lookup_module
import tab_foundry.data.corpus_materialization_invocation as corpus_materialization_invocation_module
from tab_foundry.data.corpus_materialization import materialize_corpus_recipe
from tab_foundry.training.surface import build_training_surface_record

from tests.data.test_corpus import (
    _fake_run_dagzoo_generate,
    _write_legacy_unscoped_corpus_record,
    _write_recipe_registry,
)


def _write_manifest(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "dataset_id": "root_a/shard_0/dataset_000001",
                "source_root_id": "root_a",
                "source_shard_relpath": "shard_0",
                "split": "train",
                "task": "classification",
                "dataset_index": 1,
                "train_path": "train.parquet",
                "test_path": "test.parquet",
                "catalog_path": "metadata.ndjson",
                "catalog_offset_bytes": 0,
                "catalog_size_bytes": 16,
                "catalog_sha256": "0" * 64,
                "n_train": 24,
                "n_test": 8,
                "n_features": 6,
                "n_classes": 2,
                "seed": 1,
                "filter_mode": "curated",
                "filter_status": "accepted",
                "filter_accepted": True,
                "missing_value_policy": "forbid_any",
                "missing_value_status": "clean",
            },
            {
                "dataset_id": "root_a/shard_0/dataset_000002",
                "source_root_id": "root_a",
                "source_shard_relpath": "shard_0",
                "split": "val",
                "task": "classification",
                "dataset_index": 2,
                "train_path": "train.parquet",
                "test_path": "test.parquet",
                "catalog_path": "metadata.ndjson",
                "catalog_offset_bytes": 16,
                "catalog_size_bytes": 16,
                "catalog_sha256": "1" * 64,
                "n_train": 30,
                "n_test": 10,
                "n_features": 8,
                "n_classes": 2,
                "seed": 2,
                "filter_mode": "curated",
                "filter_status": "accepted",
                "filter_accepted": True,
                "missing_value_policy": "forbid_any",
                "missing_value_status": "clean",
            },
        ]
    )
    pq.write_table(table, path)
    return path


def _write_legacy_manifest(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "dataset_id": "root_a/shard_0/dataset_000001",
                "source_root_id": "root_a",
                "source_shard_relpath": "shard_0",
                "split": "train",
                "task": "classification",
                "dataset_index": 1,
                "train_path": "train.parquet",
                "test_path": "test.parquet",
                "catalog_path": "metadata.ndjson",
                "catalog_offset_bytes": 0,
                "catalog_size_bytes": 16,
                "catalog_sha256": "0" * 64,
                "n_train": 24,
                "n_test": 8,
                "n_features": 6,
                "n_classes": 2,
                "seed": 1,
                "filter_mode": "curated",
                "filter_status": "accepted",
                "filter_accepted": True,
            }
        ]
    )
    pq.write_table(table, path)
    return path


def test_build_training_surface_record_captures_model_data_and_preprocessing_surfaces(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.parquet")
    raw_cfg = {
        "task": "classification",
        "model": {
            "arch": "tabfoundry_staged",
            "stage": "nano_exact",
            "stage_label": "delta_row_cls_pool",
            "module_overrides": {"row_pool": "row_cls"},
            "d_icl": 96,
            "input_normalization": "train_zscore_clip",
            "many_class_base": 2,
            "tficl_n_heads": 4,
            "tficl_n_layers": 3,
            "head_hidden_dim": 192,
            "tfrow_n_heads": 2,
            "tfrow_n_layers": 1,
            "tfrow_cls_tokens": 2,
        },
        "data": {
            "source": "manifest",
            "manifest_path": str(manifest_path),
            "surface_label": "anchor_manifest_default",
            "surface_overrides": {
                "filter_policy": "accepted_only",
                "dagzoo_provenance": {
                    "commands": ["dagzoo filter --in ... --out ... --curated-out ..."],
                    "config_refs": ["configs/dagzoo/binary.yaml"],
                },
            },
        },
        "preprocessing": {
            "surface_label": "runtime_no_impute",
            "overrides": {"impute_missing": False, "all_nan_fill": 1.0},
        },
    }

    record = build_training_surface_record(
        raw_cfg=raw_cfg,
        run_dir=tmp_path / "run",
    )

    assert record["labels"] == {
        "model": "delta_row_cls_pool",
        "data": "anchor_manifest_default",
        "preprocessing": "runtime_no_impute",
    }
    assert record["model"]["module_selection"]["row_pool"] == "row_cls"
    assert record["model"]["module_hyperparameters"]["row_pool"]["n_heads"] == 2
    assert record["data"]["manifest"]["characteristics"]["record_count"] == 2
    assert record["data"]["manifest"]["characteristics"]["split_counts"] == {"train": 1, "val": 1}
    assert record["data"]["manifest"]["characteristics"]["missing_value_policy"] == "forbid_any"
    assert record["data"]["manifest"]["characteristics"]["all_records_no_missing"] is True
    assert record["data"]["allow_missing_values"] is False
    assert record["data"]["filter_policy"] == "accepted_only"
    assert record["data"]["dagzoo_provenance"]["commands"] == [
        "dagzoo filter --in ... --out ... --curated-out ..."
    ]
    assert record["preprocessing"]["impute_missing"] is False
    assert record["preprocessing"]["all_nan_fill"] == 1.0
    assert record["training"]["task_batch_size"] == 1


def test_build_training_surface_record_marks_missing_inputs_when_manifest_is_dirty(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest_dirty.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "dataset_id": "root_a/shard_0/dataset_000001",
                "source_root_id": "root_a",
                "source_shard_relpath": "shard_0",
                "split": "train",
                "task": "classification",
                "dataset_index": 1,
                "train_path": "train.parquet",
                "test_path": "test.parquet",
                "catalog_path": "metadata.ndjson",
                "catalog_offset_bytes": 0,
                "catalog_size_bytes": 16,
                "catalog_sha256": "0" * 64,
                "n_train": 24,
                "n_test": 8,
                "n_features": 6,
                "n_classes": 2,
                "seed": 1,
                "filter_mode": "curated",
                "filter_status": "accepted",
                "filter_accepted": True,
                "missing_value_policy": "allow_any",
                "missing_value_status": "contains_nan_or_inf",
            }
        ]
    )
    pq.write_table(table, manifest_path)

    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry_staged"},
            "data": {
                "source": "manifest",
                "manifest_path": str(manifest_path),
            },
        },
        run_dir=tmp_path / "run_dirty",
    )

    assert record["data"]["manifest"]["characteristics"]["all_records_no_missing"] is False


def test_build_training_surface_record_includes_sandwich_architecture_metadata(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest_sandwich.parquet")

    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {
                "arch": "tabfoundry_sandwich",
                "d_icl": 96,
                "head_hidden_dim": 128,
                "sandwich_latents": 24,
                "sandwich_layers": 2,
                "sandwich_heads": 4,
                "sandwich_ff_expansion": 2,
                "sandwich_summary_tokens_per_axis": 4,
                "sandwich_self_attention_per_cross": 4,
                "sandwich_pre_row_attention_layers": 1,
                "sandwich_pre_column_attention_layers": 1,
            },
            "data": {
                "source": "manifest",
                "manifest_path": str(manifest_path),
            },
            "training": {
                "loss_surface": "cell_bpc",
            },
        },
        run_dir=tmp_path / "run_sandwich",
    )

    assert record["model"]["arch"] == "tabfoundry_sandwich"
    assert record["model"]["architecture"] == {
        "initial_input_tokens": "full_cell_plus_row_col_summary_stream",
        "initial_input_token_count": "R_times_C_plus_K_times_(R_plus_C)",
        "repeated_input_tokens": "row_col_summary_stream",
        "repeated_input_token_count": "K_times_(R_plus_C)",
        "summary_tokens_per_axis": 4,
        "pre_perceiver_cell_mixer": "row_feature_self_attention_then_column_row_isab",
        "pre_row_attention_layers": 1,
        "pre_column_attention_layers": 1,
        "pre_column_inducing_tokens": 16,
        "label_injection": "fused_into_row_summaries_and_feature_cells",
        "summary_builder": "summary_query_attention",
        "position_encoding": "shared_fourier_row_col",
        "feature_type_encoding": "film",
        "floating_likelihood": "single_gaussian",
        "integer_likelihood": "hybrid_mixture",
        "sandwich_activation": "gelu",
        "sandwich_block_norm": "layernorm",
        "latent_core": "stage0_full_cell_plus_summary_then_summary_repeated_cross_self_stages",
        "layer_semantics": "stage0_hybrid_then_summary_repeated_stages",
        "readout": "latent_then_full_cell_cross_attention_then_latent_conditioned_query_pool",
        "latents": 24,
        "layers": 2,
        "heads": 4,
        "ff_expansion": 2,
        "self_attention_per_cross": 4,
    }
    assert record["training"]["loss_surface"] == "cell_bpc"


def test_build_training_surface_record_includes_compile_model_runtime_flag(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest_compile.parquet")

    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {
                "arch": "tabfoundry_sandwich",
                "d_icl": 32,
                "head_hidden_dim": 64,
                "sandwich_latents": 12,
                "sandwich_layers": 1,
                "sandwich_heads": 4,
                "sandwich_ff_expansion": 2,
                "sandwich_summary_tokens_per_axis": 2,
                "sandwich_self_attention_per_cross": 1,
                "sandwich_pre_row_attention_layers": 1,
                "sandwich_pre_column_attention_layers": 1,
            },
            "data": {
                "source": "manifest",
                "manifest_path": str(manifest_path),
            },
            "runtime": {
                "device": "cuda",
                "output_dir": str(tmp_path / "outputs"),
                "compile_model": True,
                "trace_activations": False,
                "activation_checkpointing": True,
            },
        },
        run_dir=tmp_path / "run_compile",
    )

    assert record["runtime"]["compile_model"] is True
    assert record["runtime"]["trace_activations"] is False
    assert record["runtime"]["activation_checkpointing"] is True
    assert "device" not in record["runtime"]
    assert "output_dir" not in record["runtime"]


def test_build_training_surface_record_omits_cross_arch_sandwich_build_spec_fields(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest_sandwich_build_spec.parquet")

    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {
                "arch": "tabfoundry_sandwich",
                "d_icl": 60,
                "head_hidden_dim": 96,
                "sandwich_latents": 24,
                "sandwich_layers": 2,
                "sandwich_heads": 4,
                "sandwich_ff_expansion": 2,
                "sandwich_summary_tokens_per_axis": 3,
                "sandwich_self_attention_per_cross": 4,
                "sandwich_pre_row_attention_layers": 1,
                "sandwich_pre_column_attention_layers": 1,
                "sandwich_pre_column_inducing_tokens": 16,
                "sandwich_activation": "rational",
                "sandwich_block_norm": "none",
                "tficl_n_heads": 4,
                "tficl_n_layers": 3,
            },
            "data": {
                "source": "manifest",
                "manifest_path": str(manifest_path),
            },
        },
        run_dir=tmp_path / "run_sandwich_build_spec",
    )

    build_spec = record["model"]["build_spec"]

    assert build_spec["arch"] == "tabfoundry_sandwich"
    assert build_spec["sandwich_summary_tokens_per_axis"] == 3
    assert build_spec["sandwich_pre_column_inducing_tokens"] == 16
    assert build_spec["sandwich_activation"] == "rational"
    assert build_spec["sandwich_block_norm"] == "none"
    assert build_spec["feature_type_conditioning"] == "film"
    for unsupported_key in (
        "stage",
        "stage_label",
        "module_overrides",
        "tfcol_n_heads",
        "tfcol_n_layers",
        "tfcol_n_inducing",
        "tfrow_n_heads",
        "tfrow_n_layers",
        "tfrow_cls_tokens",
        "tfrow_norm",
        "tficl_n_heads",
        "tficl_n_layers",
        "tficl_ff_expansion",
        "use_digit_position_embed",
        "staged_dropout",
    ):
        assert unsupported_key not in build_spec


def test_build_training_surface_record_keeps_manifest_path_when_file_is_missing(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "missing_manifest.parquet"

    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry_staged"},
            "data": {
                "source": "manifest",
                "manifest_path": str(manifest_path),
            },
        },
        run_dir=tmp_path / "run_missing_manifest",
    )

    assert record["data"]["manifest"] == {
        "manifest_path": str(manifest_path.resolve()),
    }


def test_build_training_surface_record_persists_corpus_identity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    _write_recipe_registry(repo_root)
    dagzoo_root = tmp_path / "dagzoo"
    dagzoo_python = dagzoo_root / ".venv" / "bin" / "python"
    dagzoo_python.parent.mkdir(parents=True, exist_ok=True)
    dagzoo_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    dagzoo_python.chmod(0o755)
    (dagzoo_root / "configs").mkdir(parents=True, exist_ok=True)
    (dagzoo_root / "configs" / "default.yaml").write_text("seed: 1\n", encoding="utf-8")
    monkeypatch.setattr(
        corpus_materialization_invocation_module,
        "run_dagzoo_generate",
        _fake_run_dagzoo_generate,
    )
    record = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=dagzoo_root,
        force=True,
        repo_root=repo_root,
    )
    monkeypatch.setattr(corpus_loading_module, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(corpus_lookup_module, "_repo_root", lambda: repo_root)

    surface_record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry_staged"},
            "data": {
                "corpus_ref": "current_recipe",
            },
        },
        run_dir=tmp_path / "run_with_corpus",
    )

    assert surface_record["data"]["corpus_ref"] == record["corpus_ref"]
    assert surface_record["data"]["recipe_id"] == "current_recipe"
    assert surface_record["data"]["corpus_id"] == record["corpus_id"]
    assert surface_record["data"]["corpus_record_path"] == record["corpus_record_path"]
    assert surface_record["data"]["dagzoo_provenance"]["config_refs"] == ["configs/default.yaml"]
    assert "invocations" not in surface_record["data"]["dagzoo_provenance"]


def test_build_training_surface_record_compacts_legacy_corpus_record_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    _write_recipe_registry(repo_root)
    legacy_record = _write_legacy_unscoped_corpus_record(
        repo_root=repo_root,
        sweep_id=None,
        recipe_id="current_recipe",
        seed=16,
    )
    monkeypatch.setattr(corpus_loading_module, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(corpus_lookup_module, "_repo_root", lambda: repo_root)

    surface_record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry_staged"},
            "data": {
                "corpus_ref": "current_recipe",
            },
        },
        run_dir=tmp_path / "run_with_legacy_corpus",
    )

    assert surface_record["data"]["corpus_ref"] == legacy_record["corpus_ref"]
    assert surface_record["data"]["dagzoo_provenance"]["corpus_variant"] == "current_corpus_default"
    assert surface_record["data"]["dagzoo_provenance"]["invocation_count"] == 1
    assert "invocations" not in surface_record["data"]["dagzoo_provenance"]
    assert "commands" not in surface_record["data"]["dagzoo_provenance"]


def test_build_training_surface_record_captures_post_encoder_norm_component(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest_post_encoder_norm.parquet")

    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {
                "arch": "tabfoundry_staged",
                "stage": "nano_exact",
                "stage_label": "delta_shared_norm_post_ln",
                "module_overrides": {
                    "feature_encoder": "shared",
                    "post_encoder_norm": "layernorm",
                },
                "d_icl": 96,
                "input_normalization": "train_zscore_clip",
                "many_class_base": 2,
                "tficl_n_heads": 4,
                "tficl_n_layers": 3,
                "head_hidden_dim": 192,
            },
            "data": {
                "source": "manifest",
                "manifest_path": str(manifest_path),
                "surface_label": "anchor_manifest_default",
            },
        },
        run_dir=tmp_path / "run_post_encoder_norm",
    )

    assert record["model"]["module_selection"]["post_encoder_norm"] == "layernorm"
    assert record["model"]["module_hyperparameters"]["post_encoder_norm"] == {
        "name": "layernorm",
        "norm_type": "layernorm",
    }


def test_build_training_surface_record_captures_post_stack_norm_and_residual_scale(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest_post_stack_norm.parquet")

    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {
                "arch": "tabfoundry_staged",
                "stage": "nano_exact",
                "stage_label": "delta_stack_scale_followup",
                "module_overrides": {
                    "table_block_style": "prenorm",
                    "table_block_residual_scale": "depth_scaled",
                    "post_stack_norm": "rmsnorm",
                },
                "d_icl": 96,
                "input_normalization": "train_zscore_clip",
                "many_class_base": 2,
                "tficl_n_heads": 4,
                "tficl_n_layers": 4,
                "head_hidden_dim": 192,
            },
            "data": {
                "source": "manifest",
                "manifest_path": str(manifest_path),
                "surface_label": "anchor_manifest_default",
            },
        },
        run_dir=tmp_path / "run_post_stack_norm",
    )

    assert record["model"]["module_selection"]["post_stack_norm"] == "rmsnorm"
    assert record["model"]["module_selection"]["table_block_residual_scale"] == "depth_scaled"
    assert record["model"]["module_hyperparameters"]["post_stack_norm"] == {
        "name": "rmsnorm",
        "norm_type": "rmsnorm",
    }
    assert record["model"]["module_hyperparameters"]["table_block"]["residual_scale"] == "depth_scaled"
    assert record["model"]["module_hyperparameters"]["table_block"]["residual_branch_gain"] > 0.0


def test_build_training_surface_record_marks_legacy_manifest_missingness_as_unknown(
    tmp_path: Path,
) -> None:
    manifest_path = _write_legacy_manifest(tmp_path / "manifest_legacy.parquet")

    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry_staged"},
            "data": {
                "source": "manifest",
                "manifest_path": str(manifest_path),
            },
        },
        run_dir=tmp_path / "run_legacy",
    )

    assert record["data"]["manifest"]["characteristics"]["missing_value_status_counts"] == {
        "missing": 1
    }
    assert record["data"]["manifest"]["characteristics"]["all_records_no_missing"] is None


def test_build_training_surface_record_rejects_removed_row_cap_subsampling(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.parquet")
    with pytest.raises(ValueError, match="Row subsampling is no longer supported"):
        _ = build_training_surface_record(
            raw_cfg={
                "task": "classification",
                "model": {"arch": "tabfoundry_staged"},
                "data": {
                    "source": "manifest",
                    "manifest_path": str(manifest_path),
                    "train_row_cap": 10,
                },
            },
            run_dir=tmp_path / "run_override",
        )


def test_build_training_surface_record_includes_optional_training_surface(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.parquet")
    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {
                "arch": "tabfoundry_staged",
                "stage": "nano_exact",
                "d_icl": 96,
                "input_normalization": "train_zscore_clip",
                "many_class_base": 2,
                "tficl_n_heads": 4,
                "tficl_n_layers": 3,
                "head_hidden_dim": 192,
            },
            "data": {
                "source": "manifest",
                "manifest_path": str(manifest_path),
                "surface_label": "anchor_manifest_default",
            },
            "training": {
                "surface_label": "prior_linear_warmup_decay",
                "apply_schedule": True,
                "overrides": {
                    "optimizer": {"min_lr": 4.0e-4},
                },
            },
            "optimizer": {
                "name": "schedulefree_adamw",
                "min_lr": 4.0e-4,
            },
            "schedule": {
                "stages": [
                    {
                        "name": "stage1",
                        "steps": 2500,
                        "lr_max": 4.0e-3,
                        "lr_schedule": "linear",
                        "warmup_ratio": 0.05,
                    }
                ]
            },
        },
        run_dir=tmp_path / "run_training",
    )

    assert record["labels"]["training"] == "prior_linear_warmup_decay"
    assert record["training"]["apply_schedule"] is True
    assert record["training"]["optimizer_name"] == "schedulefree_adamw"
    assert record["training"]["optimizer_min_lr"] == 4.0e-4
    assert record["training"]["schedule_stages"][0]["warmup_ratio"] == 0.05
    assert "legacy_prior" not in record["training"]


def test_build_training_surface_record_infers_manifest_backend_from_data_surface(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest_backend.parquet")

    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry_staged"},
            "data": {
                "source": "manifest",
                "manifest_path": str(manifest_path),
            },
        },
        run_dir=tmp_path / "run_manifest_backend",
    )

    assert record["training"]["backend"] == "manifest"


def test_build_training_surface_record_omits_legacy_prior_block_for_manifest_experiment(
    tmp_path: Path,
) -> None:
    cfg = compose_config(["experiment=cls_benchmark_staged_corpus"])

    record = build_training_surface_record(
        raw_cfg=OmegaConf.to_container(cfg, resolve=True),
        run_dir=tmp_path / "run_manifest_experiment",
    )

    assert record["training"]["backend"] == "manifest"
    assert "legacy_prior" not in record["training"]


def test_build_training_surface_record_allows_unresolved_corpus_refs_for_manifest_backend(
    tmp_path: Path,
) -> None:
    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry_staged"},
            "data": {
                "surface_label": "fresh_current_corpus",
                "corpus_ref": "tf_rd_013_current_corpus_default_v1",
            },
        },
        run_dir=tmp_path / "run_unresolved_corpus_ref",
        allow_unresolved_corpus_ref=True,
    )

    assert record["training"]["backend"] == "manifest"
    assert record["data"]["source"] == "manifest"
    assert record["data"]["corpus_ref"] == "tf_rd_013_current_corpus_default_v1"
    assert record["data"]["requested_corpus_ref"] == "tf_rd_013_current_corpus_default_v1"
    assert record["data"]["materialization_state"] is None
    assert record["data"]["recipe_id"] == "tf_rd_013_current_corpus_default_v1"
    assert record["data"]["corpus_id"] is None


def test_build_training_surface_record_records_requested_corpus_ref_and_materialization_state(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest_direct.parquet")

    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry_sandwich"},
            "data": {
                "source": "manifest",
                "manifest_path": str(manifest_path),
                "requested_corpus_ref": "tf_rd_010_dagzoo_medium_control_curated_v5",
                "materialization_state": "staged",
            },
        },
        run_dir=tmp_path / "run_direct_manifest",
    )

    assert record["data"]["requested_corpus_ref"] == "tf_rd_010_dagzoo_medium_control_curated_v5"
    assert record["data"]["materialization_state"] == "staged"
    assert record["data"]["corpus_ref"] is None


def test_build_training_surface_record_infers_legacy_prior_backend_without_data_cfg(
    tmp_path: Path,
) -> None:
    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry_staged"},
        },
        run_dir=tmp_path / "run_prior_dump_backend",
    )

    assert record["training"]["backend"] == "legacy_prior"


def test_build_training_surface_record_captures_legacy_prior_batch_scaling_metadata(
    tmp_path: Path,
) -> None:
    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry_staged"},
            "training": {
                "surface_label": "prior_linear_warmup_decay",
                "apply_schedule": True,
            },
            "legacy_prior": {
                "non_finite_policy": "skip",
                "batch_size": 64,
                "lr_scale_rule": "sqrt",
                "batch_reference_size": 32,
                "effective_lr_scale_factor": 2 ** 0.5,
            },
            "optimizer": {
                "name": "schedulefree_adamw",
                "min_lr": 5.656854249492381e-4,
            },
            "schedule": {
                "stages": [
                    {
                        "name": "prior_dump",
                        "steps": 2500,
                        "lr_max": 5.656854249492381e-3,
                        "lr_schedule": "linear",
                        "warmup_ratio": 0.05,
                    }
                ]
            },
        },
        run_dir=tmp_path / "run_prior_scaling",
    )

    assert record["training"]["backend"] == "legacy_prior"
    assert record["training"]["legacy_prior"] == {
        "non_finite_policy": "skip",
        "batch_size": 64,
        "lr_scale_rule": "sqrt",
        "batch_reference_size": 32,
        "effective_lr_scale_factor": 2 ** 0.5,
    }
    assert record["training"]["optimizer_min_lr"] == 5.656854249492381e-4
    assert record["training"]["schedule_stages"][0]["lr_max"] == 5.656854249492381e-3


def test_build_training_surface_record_preserves_flat_legacy_prior_overrides(
    tmp_path: Path,
) -> None:
    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry_staged"},
            "training": {
                "surface_label": "prior_linear_warmup_decay",
                "apply_schedule": True,
                "prior_dump_non_finite_policy": "skip",
                "prior_dump_batch_size": 64,
                "prior_dump_lr_scale_rule": "sqrt",
                "prior_dump_batch_reference_size": 32,
                "effective_lr_scale_factor": 2 ** 0.5,
            },
            "legacy_prior": {
                "non_finite_policy": "error",
                "batch_size": 32,
                "lr_scale_rule": "none",
                "batch_reference_size": 32,
            },
        },
        run_dir=tmp_path / "run_flat_legacy_prior_overrides",
    )

    assert record["training"]["backend"] == "legacy_prior"
    assert record["training"]["legacy_prior"] == {
        "non_finite_policy": "skip",
        "batch_size": 64,
        "lr_scale_rule": "sqrt",
        "batch_reference_size": 32,
        "effective_lr_scale_factor": 2 ** 0.5,
    }
