from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

import click
from click.testing import CliRunner
import pytest

import tab_foundry.bench.bounce_diagnosis as bounce_diagnosis_library_module
import tab_foundry.bench.comparison_contract as comparison_contract_library_module
import tab_foundry.bench.control_baseline_freeze as control_baseline_freeze_library_module
import tab_foundry.bench.dagzoo_smoke as dagzoo_smoke_library_module
import tab_foundry.bench.envs as env_library_module
import tab_foundry.bench.iris_smoke as iris_smoke_library_module
import tab_foundry.bench.openml_bundle as bundle_library_module
import tab_foundry.bench.run_registration as run_registration_library_module
import tab_foundry.bench.tune as tune_library_module
import tab_foundry.cli as cli_module
import tab_foundry.cli.bench_bounce_diagnosis as bounce_diagnosis_cli_module
import tab_foundry.cli.bench_bundle_openml as bundle_cli_module
import tab_foundry.cli.bench_compare as compare_cli_module
import tab_foundry.cli.bench_control_baseline_freeze as control_baseline_freeze_cli_module
import tab_foundry.cli.bench_env_bootstrap as env_bootstrap_cli_module
import tab_foundry.cli.bench_run_registration as run_registration_cli_module
import tab_foundry.cli.bench_smoke_dagzoo as dagzoo_smoke_cli_module
import tab_foundry.cli.bench_smoke_iris as iris_smoke_cli_module
import tab_foundry.cli.bench_tune as tune_cli_module
import tab_foundry.cli.data_inspect as data_inspect_module
import tab_foundry.cli.dev as dev_module
import tab_foundry.cli.groups.bench as bench_group
import tab_foundry.cli.groups.data as data_group
import tab_foundry.cli.groups.research as research_group
import tab_foundry.cli.groups.train as train_group
import tab_foundry.cli.research_adequacy as research_adequacy_cli_module
import tab_foundry.cli.research_diff as research_diff_cli_module
import tab_foundry.cli.research_execute as research_execute_cli_module
import tab_foundry.cli.research_graph as research_graph_cli_module
import tab_foundry.cli.research_inspect as research_inspect_cli_module
import tab_foundry.cli.research_promote as research_promote_cli_module
import tab_foundry.cli.research_summarize as research_summarize_cli_module
import tab_foundry.cli.research_sweep_core as research_sweep_core_cli_module
import tab_foundry.cli.train_prior as train_prior_cli_module
import tab_foundry.research.adequacy.pilot as adequacy_pilot_module
import tab_foundry.research.sweep.catalog as sweep_catalog_module
import tab_foundry.research.sweep.diff as diff_module
import tab_foundry.research.sweep.execute as sweep_execute_library_module
import tab_foundry.research.sweep.graph as graph_module
import tab_foundry.research.sweep.inspect as inspect_module
import tab_foundry.research.sweep.manage as sweep_manage_module
import tab_foundry.research.sweep.materialize as sweep_materialize_module
import tab_foundry.research.sweep.matrix as sweep_matrix_module
import tab_foundry.research.sweep.promote as sweep_promote_library_module
import tab_foundry.research.sweep.summarize as summarize_module
import tab_foundry.training.prior_train as prior_train_library_module
from tests.support.cli import capture_handler


Reader = Callable[[Any], object]


@dataclass(frozen=True)
class DispatchCase:
    argv: tuple[str, ...]
    module: object
    attribute: str
    fields: dict[str, Reader]
    expected: dict[str, object]


def _bool_attr(name: str) -> Reader:
    return lambda args: bool(getattr(args, name))


def _int_attr(name: str) -> Reader:
    return lambda args: int(getattr(args, name))


def _list_attr(name: str, item_reader: Callable[[Any], object] | None = None) -> Reader:
    def _reader(args: Any) -> object:
        values = list(getattr(args, name))
        if item_reader is None:
            return values
        return [item_reader(value) for value in values]

    return _reader


def _optional_str_attr(name: str) -> Reader:
    return lambda args: None if getattr(args, name) is None else str(getattr(args, name))


def _path_attr(name: str) -> Reader:
    return lambda args: str(getattr(args, name))


def _optional_path_attr(name: str) -> Reader:
    return lambda args: None if getattr(args, name) is None else str(getattr(args, name))


def _str_attr(name: str) -> Reader:
    return lambda args: str(getattr(args, name))


def _command_names(command: click.Group) -> list[str]:
    return list(command.list_commands(click.Context(command)))


DISPATCH_CASES = (
    pytest.param(
        DispatchCase(
            argv=(
                "bench",
                "compare",
                "--tab-foundry-run-dir",
                "/tmp/run",
                "--tabicl-root",
                "/tmp/tabicl",
                "--tab-realdata-hub-root",
                "/tmp/tab-realdata-hub",
            ),
            module=compare_cli_module,
            attribute="_compare_command",
            fields={
                "tab_foundry_run_dir": _path_attr("tab_foundry_run_dir"),
                "tabicl_root": _path_attr("tabicl_root"),
                "tab_realdata_hub_root": _path_attr("tab_realdata_hub_root"),
            },
            expected={
                "tab_foundry_run_dir": "/tmp/run",
                "tabicl_root": "/tmp/tabicl",
                "tab_realdata_hub_root": "/tmp/tab-realdata-hub",
            },
        ),
        id="bench-compare",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "bench",
                "tune",
                "--manifest-path",
                "/tmp/manifest.parquet",
                "--seed",
                "7",
            ),
            module=tune_cli_module,
            attribute="_tune_command",
            fields={
                "manifest_path": _path_attr("manifest_path"),
                "seed": _int_attr("seed"),
            },
            expected={"manifest_path": "/tmp/manifest.parquet", "seed": 7},
        ),
        id="bench-tune",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "bench",
                "env",
                "bootstrap",
                "--nanotabpfn-root",
                "/tmp/nano",
                "--tabpfn-root",
                "/tmp/tabpfn",
                "--tabicl-root",
                "/tmp/tabicl",
                "--tab-realdata-hub-root",
                "/tmp/tab-realdata-hub",
            ),
            module=env_bootstrap_cli_module,
            attribute="_bootstrap_command",
            fields={
                "nanotabpfn_root": _path_attr("nanotabpfn_root"),
                "tabicl_root": _path_attr("tabicl_root"),
                "tab_realdata_hub_root": _path_attr("tab_realdata_hub_root"),
            },
            expected={
                "nanotabpfn_root": "/tmp/nano",
                "tabicl_root": "/tmp/tabicl",
                "tab_realdata_hub_root": "/tmp/tab-realdata-hub",
            },
        ),
        id="bench-env-bootstrap",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "bench",
                "bundle",
                "build-openml",
                "--out-path",
                "/tmp/bundle.json",
                "--bundle-name",
                "binary_medium",
                "--version",
                "1",
                "--task-source",
                "binary_expanded_v1",
            ),
            module=bundle_cli_module,
            attribute="_build_openml_bundle_command",
            fields={
                "bundle_name": _str_attr("bundle_name"),
                "task_source": _str_attr("task_source"),
            },
            expected={"bundle_name": "binary_medium", "task_source": "binary_expanded_v1"},
        ),
        id="bench-bundle-build-openml",
    ),
    pytest.param(
        DispatchCase(
            argv=("bench", "smoke", "iris", "--device", "cpu", "--checkpoint-every", "5"),
            module=iris_smoke_cli_module,
            attribute="_iris_smoke_command",
            fields={
                "device": _str_attr("device"),
                "checkpoint_every": _int_attr("checkpoint_every"),
            },
            expected={"device": "cpu", "checkpoint_every": 5},
        ),
        id="bench-smoke-iris",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "bench",
                "smoke",
                "dagzoo",
                "--dagzoo-root",
                "/tmp/dagzoo",
                "--num-datasets",
                "16",
            ),
            module=dagzoo_smoke_cli_module,
            attribute="_dagzoo_smoke_command",
            fields={
                "dagzoo_root": _path_attr("dagzoo_root"),
                "num_datasets": _int_attr("num_datasets"),
            },
            expected={"dagzoo_root": "/tmp/dagzoo", "num_datasets": 16},
        ),
        id="bench-smoke-dagzoo",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "bench",
                "diagnose",
                "bounce",
                "--run-dir",
                "/tmp/run",
                "--bootstrap-samples",
                "64",
            ),
            module=bounce_diagnosis_cli_module,
            attribute="_bounce_diagnosis_command",
            fields={
                "run_dir": _path_attr("run_dir"),
                "bootstrap_samples": _int_attr("bootstrap_samples"),
            },
            expected={"run_dir": "/tmp/run", "bootstrap_samples": 64},
        ),
        id="bench-diagnose-bounce",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "bench",
                "registry",
                "register-run",
                "--run-id",
                "run_001",
                "--track",
                "binary_md_v1",
                "--run-dir",
                "/tmp/run",
                "--comparison-summary",
                "/tmp/comparison_summary.json",
                "--experiment",
                "cls_benchmark_staged_prior",
                "--decision",
                "keep",
                "--conclusion",
                "ok",
                "--registry-path",
                "/tmp/registry.json",
            ),
            module=run_registration_cli_module,
            attribute="_register_run_command",
            fields={
                "run_id": _str_attr("run_id"),
                "registry_path": _path_attr("registry_path"),
            },
            expected={"run_id": "run_001", "registry_path": "/tmp/registry.json"},
        ),
        id="bench-registry-register-run",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "bench",
                "registry",
                "freeze-baseline",
                "--run-dir",
                "/tmp/run",
                "--comparison-summary",
                "/tmp/comparison_summary.json",
                "--baseline-id",
                "baseline_v1",
                "--registry-path",
                "/tmp/control_baselines.json",
            ),
            module=control_baseline_freeze_cli_module,
            attribute="_freeze_baseline_command",
            fields={
                "baseline_id": _str_attr("baseline_id"),
                "registry_path": _path_attr("registry_path"),
            },
            expected={
                "baseline_id": "baseline_v1",
                "registry_path": "/tmp/control_baselines.json",
            },
        ),
        id="bench-registry-freeze-baseline",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "train",
                "legacy-prior",
                "simple",
                "--prior-dump",
                "/tmp/prior.h5",
                "runtime.max_steps=1",
            ),
            module=train_prior_cli_module,
            attribute="_run_simple_command",
            fields={
                "prior_dump": _path_attr("prior_dump"),
                "overrides": _list_attr("overrides"),
            },
            expected={
                "prior_dump": "/tmp/prior.h5",
                "overrides": ["runtime.max_steps=1"],
            },
        ),
        id="train-legacy-prior-simple",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "train",
                "legacy-prior",
                "staged",
                "--prior-dump",
                "/tmp/prior.h5",
                "runtime.max_steps=1",
            ),
            module=train_prior_cli_module,
            attribute="_run_staged_command",
            fields={
                "prior_dump": _path_attr("prior_dump"),
                "overrides": _list_attr("overrides"),
            },
            expected={
                "prior_dump": "/tmp/prior.h5",
                "overrides": ["runtime.max_steps=1"],
            },
        ),
        id="train-legacy-prior-staged",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "research",
                "sweep",
                "create-sweep",
                "--sweep-id",
                "binary_md_v1",
                "--anchor-run-id",
                "run_001",
                "--complexity-level",
                "medium",
                "--benchmark-manifest-path",
                "/tmp/bundle.json",
                "--control-baseline-id",
                "baseline_v1",
            ),
            module=research_sweep_core_cli_module,
            attribute="_run_sweep_create",
            fields={
                "sweep_id": _str_attr("sweep_id"),
                "anchor_run_id": _str_attr("anchor_run_id"),
            },
            expected={"sweep_id": "binary_md_v1", "anchor_run_id": "run_001"},
        ),
        id="research-sweep-create",
    ),
    pytest.param(
        DispatchCase(
            argv=("research", "sweep", "list-sweeps", "--index-path", "/tmp/index.yaml"),
            module=research_sweep_core_cli_module,
            attribute="_run_list_sweeps",
            fields={"index_path": _path_attr("index_path")},
            expected={"index_path": "/tmp/index.yaml"},
        ),
        id="research-sweep-list",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "research",
                "sweep",
                "next",
                "--sweep-id",
                "binary_md_v1",
                "--index-path",
                "/tmp/index.yaml",
            ),
            module=research_sweep_core_cli_module,
            attribute="_run_sweep_next",
            fields={
                "sweep_id": _str_attr("sweep_id"),
                "index_path": _path_attr("index_path"),
            },
            expected={"sweep_id": "binary_md_v1", "index_path": "/tmp/index.yaml"},
        ),
        id="research-sweep-next",
    ),
    pytest.param(
        DispatchCase(
            argv=("research", "sweep", "render", "--sweep-id", "binary_md_v1"),
            module=research_sweep_core_cli_module,
            attribute="_run_sweep_render",
            fields={"sweep_id": _optional_str_attr("sweep_id")},
            expected={"sweep_id": "binary_md_v1"},
        ),
        id="research-sweep-render",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "research",
                "sweep",
                "materialize-corpora",
                "--sweep-id",
                "binary_md_v1",
                "--dagzoo-root",
                "/tmp/dagzoo",
                "--materialize-processes",
                "3",
                "--materialize-worker-threads",
                "2",
                "--force",
                "--json",
            ),
            module=research_sweep_core_cli_module,
            attribute="_run_sweep_materialize_corpora",
            fields={
                "sweep_id": _optional_str_attr("sweep_id"),
                "dagzoo_root": _path_attr("dagzoo_root"),
                "materialize_processes": _int_attr("materialize_processes"),
                "materialize_worker_threads": _int_attr("materialize_worker_threads"),
                "force": _bool_attr("force"),
                "json_mode": _bool_attr("json_mode"),
            },
            expected={
                "sweep_id": "binary_md_v1",
                "dagzoo_root": "/tmp/dagzoo",
                "materialize_processes": 3,
                "materialize_worker_threads": 2,
                "force": True,
                "json_mode": True,
            },
        ),
        id="research-sweep-materialize-corpora",
    ),
    pytest.param(
        DispatchCase(
            argv=("research", "sweep", "graph", "--sweep-id", "binary_md_v1", "--anchor", "--order", "7"),
            module=research_graph_cli_module,
            attribute="_graph_command",
            fields={
                "sweep_id": _str_attr("sweep_id"),
                "anchor": _bool_attr("anchor"),
                "order": _list_attr("order", int),
            },
            expected={"sweep_id": "binary_md_v1", "anchor": True, "order": [7]},
        ),
        id="research-sweep-graph",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "research",
                "sweep",
                "execute",
                "--sweep-id",
                "binary_md_v1",
                "--nanotabpfn-root",
                "/tmp/nanoTabPFN",
                "--include-completed",
            ),
            module=research_execute_cli_module,
            attribute="_execute_command",
            fields={
                "sweep_id": _optional_str_attr("sweep_id"),
                "nanotabpfn_root": _path_attr("nanotabpfn_root"),
                "include_completed": _bool_attr("include_completed"),
            },
            expected={
                "sweep_id": "binary_md_v1",
                "nanotabpfn_root": "/tmp/nanoTabPFN",
                "include_completed": True,
            },
        ),
        id="research-sweep-execute",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "research",
                "sweep",
                "execute",
                "--sweep-id",
                "binary_md_v1",
                "--include-completed",
            ),
            module=research_execute_cli_module,
            attribute="_execute_command",
            fields={
                "sweep_id": _optional_str_attr("sweep_id"),
                "nanotabpfn_root": _optional_path_attr("nanotabpfn_root"),
                "include_completed": _bool_attr("include_completed"),
            },
            expected={
                "sweep_id": "binary_md_v1",
                "nanotabpfn_root": None,
                "include_completed": True,
            },
        ),
        id="research-sweep-execute-no-nanotab-root",
    ),
    pytest.param(
        DispatchCase(
            argv=("research", "sweep", "promote", "--sweep-id", "binary_md_v1", "--run-id", "run_001"),
            module=research_promote_cli_module,
            attribute="_promote_command",
            fields={
                "sweep_id": _str_attr("sweep_id"),
                "run_id": _str_attr("run_id"),
            },
            expected={"sweep_id": "binary_md_v1", "run_id": "run_001"},
        ),
        id="research-sweep-promote",
    ),
    pytest.param(
        DispatchCase(
            argv=("research", "sweep", "summarize", "--sweep-id", "cuda_stack_scale_followup", "--json"),
            module=research_summarize_cli_module,
            attribute="_summarize_command",
            fields={
                "sweep_id": _optional_str_attr("sweep_id"),
                "json_mode": _bool_attr("json_mode"),
            },
            expected={"sweep_id": "cuda_stack_scale_followup", "json_mode": True},
        ),
        id="research-sweep-summarize",
    ),
    pytest.param(
        DispatchCase(
            argv=("research", "sweep", "inspect", "--sweep-id", "binary_md_v1", "--order", "6", "--json"),
            module=research_inspect_cli_module,
            attribute="_inspect_command",
            fields={
                "sweep_id": _str_attr("sweep_id"),
                "order": _int_attr("order"),
                "json_mode": _bool_attr("json_mode"),
            },
            expected={"sweep_id": "binary_md_v1", "order": 6, "json_mode": True},
        ),
        id="research-sweep-inspect",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "research",
                "sweep",
                "diff",
                "--sweep-id",
                "binary_md_v1",
                "--order",
                "7",
                "--against-order",
                "6",
            ),
            module=research_diff_cli_module,
            attribute="_diff_command",
            fields={
                "sweep_id": _str_attr("sweep_id"),
                "order": _int_attr("order"),
                "against_order": _int_attr("against_order"),
            },
            expected={"sweep_id": "binary_md_v1", "order": 7, "against_order": 6},
        ),
        id="research-sweep-diff",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "research",
                "adequacy",
                "pilot",
                "--adequacy-id",
                "tf_rd_010_synthetic_adequacy_v3",
                "--dagzoo-root",
                "/tmp/dagzoo",
                "--device",
                "cpu",
                "--materialize-processes",
                "3",
                "--materialize-worker-threads",
                "2",
                "--contract-check",
                "fast",
                "--force",
                "--out-root",
                "/tmp/adequacy",
            ),
            module=research_adequacy_cli_module,
            attribute="_pilot_command",
            fields={
                "adequacy_id": _str_attr("adequacy_id"),
                "dagzoo_root": _path_attr("dagzoo_root"),
                "device": _str_attr("device"),
                "materialize_processes": _int_attr("materialize_processes"),
                "materialize_worker_threads": _int_attr("materialize_worker_threads"),
                "contract_check": _str_attr("contract_check"),
                "force": _bool_attr("force"),
                "out_root": _path_attr("out_root"),
            },
            expected={
                "adequacy_id": "tf_rd_010_synthetic_adequacy_v3",
                "dagzoo_root": "/tmp/dagzoo",
                "device": "cpu",
                "materialize_processes": 3,
                "materialize_worker_threads": 2,
                "contract_check": "fast",
                "force": True,
                "out_root": "/tmp/adequacy",
            },
        ),
        id="research-adequacy-pilot",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "research",
                "adequacy",
                "finalize",
                "--adequacy-id",
                "tf_rd_010_synthetic_adequacy_v3",
                "--dagzoo-root",
                "/tmp/dagzoo",
                "--contract-check",
                "fast",
                "--out-root",
                "/tmp/adequacy",
            ),
            module=research_adequacy_cli_module,
            attribute="_finalize_command",
            fields={
                "adequacy_id": _str_attr("adequacy_id"),
                "dagzoo_root": _path_attr("dagzoo_root"),
                "contract_check": _str_attr("contract_check"),
                "out_root": _path_attr("out_root"),
            },
            expected={
                "adequacy_id": "tf_rd_010_synthetic_adequacy_v3",
                "dagzoo_root": "/tmp/dagzoo",
                "contract_check": "fast",
                "out_root": "/tmp/adequacy",
            },
        ),
        id="research-adequacy-finalize",
    ),
    pytest.param(
        DispatchCase(
            argv=("dev", "resolve-config", "--json", "experiment=cls_smoke"),
            module=dev_module,
            attribute="_run_resolve_config",
            fields={
                "json_mode": _bool_attr("json_mode"),
                "overrides": _list_attr("overrides"),
            },
            expected={"json_mode": True, "overrides": ["experiment=cls_smoke"]},
        ),
        id="dev-resolve-config",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "dev",
                "diff-config",
                "--left",
                "experiment=cls_smoke",
                "--right",
                "experiment=cls_workstation",
            ),
            module=dev_module,
            attribute="_run_diff_config",
            fields={
                "left": _list_attr("left"),
                "right": _list_attr("right"),
            },
            expected={
                "left": ["experiment=cls_smoke"],
                "right": ["experiment=cls_workstation"],
            },
        ),
        id="dev-diff-config",
    ),
    pytest.param(
        DispatchCase(
            argv=("dev", "export-check", "--checkpoint", "/tmp/checkpoint.pt", "--json"),
            module=dev_module,
            attribute="_run_export_check",
            fields={
                "checkpoint": _path_attr("checkpoint"),
                "json_mode": _bool_attr("json_mode"),
            },
            expected={"checkpoint": "/tmp/checkpoint.pt", "json_mode": True},
        ),
        id="dev-export-check",
    ),
    pytest.param(
        DispatchCase(
            argv=("dev", "run-inspect", "--run-dir", "/tmp/run"),
            module=dev_module,
            attribute="_run_run_inspect",
            fields={"run_dir": _path_attr("run_dir")},
            expected={"run_dir": "/tmp/run"},
        ),
        id="dev-run-inspect",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "dev",
                "data",
                "generate-manifest",
                "--dagzoo-root",
                "/tmp/dagzoo",
                "--dagzoo-config",
                "configs/default.yaml",
                "--handoff-root",
                "/tmp/handoff",
                "--out-manifest",
                "/tmp/manifest.parquet",
                "--num-datasets",
                "32",
            ),
            module=data_group,
            attribute="_run_dagzoo_generate_manifest",
            fields={
                "dagzoo_root": _path_attr("dagzoo_root"),
                "dagzoo_config": _str_attr("dagzoo_config"),
                "handoff_root": _path_attr("handoff_root"),
                "out_manifest": _path_attr("out_manifest"),
                "num_datasets": _int_attr("num_datasets"),
            },
            expected={
                "dagzoo_root": "/tmp/dagzoo",
                "dagzoo_config": "configs/default.yaml",
                "handoff_root": "/tmp/handoff",
                "out_manifest": "/tmp/manifest.parquet",
                "num_datasets": 32,
            },
        ),
        id="dev-data-generate-manifest",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "data",
                "corpus",
                "materialize",
                "--recipe",
                "tf_rd_013_current_corpus_default_v1",
                "--sweep-id",
                "tf_rd_020_harder_dagzoo_ladder_v1",
                "--dagzoo-root",
                "/tmp/dagzoo",
                "--materialize-processes",
                "3",
                "--materialize-worker-threads",
                "2",
                "--force",
            ),
            module=data_group,
            attribute="_run_corpus_materialize",
            fields={
                "recipe": _str_attr("recipe"),
                "sweep_id": _optional_str_attr("sweep_id"),
                "dagzoo_root": _path_attr("dagzoo_root"),
                "materialize_processes": _int_attr("materialize_processes"),
                "materialize_worker_threads": _int_attr("materialize_worker_threads"),
                "force": _bool_attr("force"),
            },
            expected={
                "recipe": "tf_rd_013_current_corpus_default_v1",
                "sweep_id": "tf_rd_020_harder_dagzoo_ladder_v1",
                "dagzoo_root": "/tmp/dagzoo",
                "materialize_processes": 3,
                "materialize_worker_threads": 2,
                "force": True,
            },
        ),
        id="data-corpus-materialize",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "data",
                "corpus",
                "finalize-staged",
                "--recipe",
                "tf_rd_013_current_corpus_default_v1",
                "--sweep-id",
                "tf_rd_020_harder_dagzoo_ladder_v1",
                "--dagzoo-root",
                "/tmp/dagzoo",
                "--stage-root",
                "/tmp/stage",
                "--verify",
                "fast",
                "--experiment",
                "cls_smoke",
                "--override",
                "data.manifest_path=/tmp/manifest.parquet",
                "--force",
            ),
            module=data_group,
            attribute="_run_corpus_finalize_staged",
            fields={
                "recipe": _str_attr("recipe"),
                "sweep_id": _optional_str_attr("sweep_id"),
                "dagzoo_root": _path_attr("dagzoo_root"),
                "stage_root": _path_attr("stage_root"),
                "verify": _str_attr("verify"),
                "experiment": _str_attr("experiment"),
                "override": _list_attr("override"),
                "force": _bool_attr("force"),
            },
            expected={
                "recipe": "tf_rd_013_current_corpus_default_v1",
                "sweep_id": "tf_rd_020_harder_dagzoo_ladder_v1",
                "dagzoo_root": "/tmp/dagzoo",
                "stage_root": "/tmp/stage",
                "verify": "fast",
                "experiment": "cls_smoke",
                "override": ["data.manifest_path=/tmp/manifest.parquet"],
                "force": True,
            },
        ),
        id="data-corpus-finalize-staged",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "data",
                "corpus",
                "inspect",
                "--corpus-ref",
                "tf_rd_013_current_corpus_default_v1/current_recipe__123456789abc",
            ),
            module=data_group,
            attribute="_run_corpus_inspect",
            fields={"corpus_ref": _str_attr("corpus_ref")},
            expected={
                "corpus_ref": "tf_rd_013_current_corpus_default_v1/current_recipe__123456789abc"
            },
        ),
        id="data-corpus-inspect",
    ),
    pytest.param(
        DispatchCase(
            argv=(
                "data",
                "manifest-inspect",
                "--manifest",
                "/tmp/manifest.parquet",
                "--experiment",
                "cls_smoke",
                "--override",
                "data.manifest_path=/tmp/manifest.parquet",
                "--json",
            ),
            module=data_inspect_module,
            attribute="_manifest_inspect_command",
            fields={
                "manifest": _path_attr("manifest"),
                "experiment": _str_attr("experiment"),
                "overrides": _list_attr("overrides"),
                "json_mode": _bool_attr("json_mode"),
            },
            expected={
                "manifest": "/tmp/manifest.parquet",
                "experiment": "cls_smoke",
                "overrides": ["data.manifest_path=/tmp/manifest.parquet"],
                "json_mode": True,
            },
        ),
        id="data-manifest-inspect",
    ),
)


@pytest.mark.parametrize("case", DISPATCH_CASES)
def test_nested_cli_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
    case: DispatchCase,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        case.module,
        case.attribute,
        capture_handler(captured, case.fields),
    )

    result = CliRunner().invoke(cli_module.cli, list(case.argv))

    assert result.exit_code == 0, result.output
    assert captured == case.expected


def test_bench_compare_run_from_args_forwards_tab_realdata_hub_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    hub_root = tmp_path / "tab-realdata-hub"

    monkeypatch.setattr(
        compare_cli_module,
        "run_nanotabpfn_benchmark",
        lambda config: captured.update({"config": config})
        or {"dataset_count": 0, "tab_foundry": {}, "artifacts": {}},
    )

    result = CliRunner().invoke(
        compare_cli_module.COMMAND,
        [
            "--tab-foundry-run-dir",
            str(tmp_path / "run"),
            "--tabicl-root",
            str(tmp_path / "tabicl"),
            "--tab-realdata-hub-root",
            str(hub_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["config"].tab_realdata_hub_root == hub_root


def test_bench_env_bootstrap_run_from_args_forwards_tab_realdata_hub_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    hub_root = tmp_path / "tab-realdata-hub"

    monkeypatch.setattr(
        env_bootstrap_cli_module,
        "bootstrap_benchmark_envs",
        lambda config: captured.update({"config": config}) or {"tabicl_python": "/tmp/python"},
    )

    result = CliRunner().invoke(
        env_bootstrap_cli_module.COMMAND,
        [
            "--nanotabpfn-root",
            str(tmp_path / "nanoTabPFN"),
            "--tabpfn-root",
            str(tmp_path / "TabPFN"),
            "--tabicl-root",
            str(tmp_path / "tabicl"),
            "--tab-realdata-hub-root",
            str(hub_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["config"].tab_realdata_hub_root == hub_root


def test_cli_groups_register_expected_commands() -> None:
    assert _command_names(bench_group.GROUP) == [
        "bundle",
        "compare",
        "diagnose",
        "env",
        "materialize-openml-bundle",
        "registry",
        "smoke",
        "tune",
    ]
    bench_ctx = click.Context(bench_group.GROUP)
    smoke_group = bench_group.GROUP.get_command(bench_ctx, "smoke")
    assert isinstance(smoke_group, click.Group)
    assert _command_names(smoke_group) == ["dagzoo", "iris"]
    registry_group = bench_group.GROUP.get_command(bench_ctx, "registry")
    assert isinstance(registry_group, click.Group)
    assert _command_names(registry_group) == ["freeze-baseline", "register-run"]
    diagnose_group = bench_group.GROUP.get_command(bench_ctx, "diagnose")
    assert isinstance(diagnose_group, click.Group)
    assert _command_names(diagnose_group) == ["bounce"]

    assert _command_names(research_group.GROUP) == ["adequacy", "sweep"]
    research_ctx = click.Context(research_group.GROUP)
    adequacy_group = research_group.GROUP.get_command(research_ctx, "adequacy")
    assert isinstance(adequacy_group, click.Group)
    assert _command_names(adequacy_group) == ["finalize", "pilot"]
    sweep_group = research_group.GROUP.get_command(research_ctx, "sweep")
    assert isinstance(sweep_group, click.Group)
    assert _command_names(sweep_group) == [
        "create-sweep",
        "diff",
        "execute",
        "graph",
        "inspect",
        "list",
        "list-sweeps",
        "materialize-corpora",
        "next",
        "promote",
        "render",
        "summarize",
        "validate",
    ]

    assert _command_names(train_group.GROUP) == ["legacy-prior", "run"]

    data_ctx = click.Context(data_group.GROUP)
    corpus_group = data_group.GROUP.get_command(data_ctx, "corpus")
    assert isinstance(corpus_group, click.Group)
    assert _command_names(corpus_group) == [
        "compare",
        "finalize-staged",
        "inspect",
        "list-recipes",
        "materialize",
        "results",
    ]

    for library_module in (
        comparison_contract_library_module,
        tune_library_module,
        env_library_module,
        bundle_library_module,
        iris_smoke_library_module,
        dagzoo_smoke_library_module,
        bounce_diagnosis_library_module,
        run_registration_library_module,
        control_baseline_freeze_library_module,
        prior_train_library_module,
        sweep_catalog_module,
        adequacy_pilot_module,
        sweep_manage_module,
        sweep_materialize_module,
        sweep_matrix_module,
        sweep_execute_library_module,
        graph_module,
        inspect_module,
        diff_module,
        sweep_promote_library_module,
        summarize_module,
    ):
        for attribute in (
            "configure_parser",
            "configure_core_path_arguments",
            "register_core_subparsers",
            "build_parser",
            "run_from_args",
            "main",
        ):
            assert not hasattr(library_module, attribute)


def test_nested_cli_research_sweep_create_alias_is_rejected(
) -> None:
    result = CliRunner().invoke(cli_module.cli, ["research", "sweep", "create"])

    assert result.exit_code == 2
    assert "No such command 'create'" in result.output


@pytest.mark.parametrize(
    ("argv", "attribute"),
    [
        pytest.param(
            [
                "dev",
                "data",
                "build-manifest",
                "--data-root",
                "/tmp/run",
                "--out-manifest",
                "/tmp/manifest.parquet",
                "--train-ratio",
                "1.0",
            ],
            "build_manifest",
            id="build-manifest",
        ),
        pytest.param(
            [
                "dev",
                "data",
                "generate-manifest",
                "--dagzoo-root",
                "/tmp/dagzoo",
                "--dagzoo-config",
                "configs/default.yaml",
                "--handoff-root",
                "/tmp/handoff",
                "--out-manifest",
                "/tmp/manifest.parquet",
                "--train-ratio",
                "0.95",
                "--val-ratio",
                "0.05",
            ],
            "run_dagzoo_generate_manifest",
            id="generate-manifest",
        ),
    ],
)
def test_nested_cli_data_commands_reject_invalid_split_ratios(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    attribute: str,
) -> None:
    called = False

    def _fake_handler(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(data_group, attribute, _fake_handler)

    result = CliRunner().invoke(cli_module.cli, argv)

    assert result.exit_code != 0
    assert "invalid split ratios" in result.output
    assert called is False


@pytest.mark.parametrize(
    "argv",
    [
        [
            "dev",
            "data",
            "build-manifest",
            "--data-root",
            "/tmp/run",
            "--out-manifest",
            "/tmp/manifest.parquet",
            "--train-ratio",
            "nan",
        ],
        [
            "dev",
            "data",
            "generate-manifest",
            "--dagzoo-root",
            "/tmp/dagzoo",
            "--dagzoo-config",
            "configs/default.yaml",
            "--handoff-root",
            "/tmp/handoff",
            "--out-manifest",
            "/tmp/manifest.parquet",
            "--val-ratio",
            "inf",
        ],
    ],
)
def test_nested_cli_dev_data_commands_reject_non_finite_split_ratios(argv: list[str]) -> None:
    result = CliRunner().invoke(cli_module.cli, argv)

    assert result.exit_code == 2


@pytest.mark.parametrize(
    "argv",
    [
        [
            "data",
            "corpus",
            "materialize",
            "--recipe",
            "recipe_a",
            "--dagzoo-root",
            "/tmp/dagzoo",
            "--materialize-processes",
            "0",
        ],
        [
            "research",
            "adequacy",
            "pilot",
            "--adequacy-id",
            "tf_rd_010_synthetic_adequacy_v3",
            "--dagzoo-root",
            "/tmp/dagzoo",
            "--materialize-processes",
            "-1",
        ],
        [
            "research",
            "sweep",
            "materialize-corpora",
            "--sweep-id",
            "binary_md_v1",
            "--dagzoo-root",
            "/tmp/dagzoo",
            "--materialize-processes",
            "0",
        ],
    ],
)
def test_materialize_processes_rejects_non_positive_values(argv: list[str]) -> None:
    result = CliRunner().invoke(cli_module.cli, argv)

    assert result.exit_code == 2


@pytest.mark.parametrize(
    "argv",
    [
        [
            "data",
            "corpus",
            "materialize",
            "--recipe",
            "recipe_a",
            "--dagzoo-root",
            "/tmp/dagzoo",
            "--materialize-worker-threads",
            "0",
        ],
        [
            "research",
            "adequacy",
            "pilot",
            "--adequacy-id",
            "tf_rd_010_synthetic_adequacy_v3",
            "--dagzoo-root",
            "/tmp/dagzoo",
            "--materialize-worker-threads",
            "-1",
        ],
        [
            "research",
            "sweep",
            "materialize-corpora",
            "--sweep-id",
            "binary_md_v1",
            "--dagzoo-root",
            "/tmp/dagzoo",
            "--materialize-worker-threads",
            "0",
        ],
    ],
)
def test_materialize_worker_threads_rejects_non_positive_values(argv: list[str]) -> None:
    result = CliRunner().invoke(cli_module.cli, argv)

    assert result.exit_code == 2


def test_nested_cli_rejects_unexpected_extra_arguments() -> None:
    result = CliRunner().invoke(
        cli_module.cli,
        [
            "data",
            "manifest-inspect",
            "--manifest",
            "/tmp/manifest.parquet",
            "--unexpected",
        ],
    )

    assert result.exit_code == 2
    assert "No such option: --unexpected" in result.output


def test_nested_cli_dev_data_generate_manifest_returns_subprocess_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_workflow(_config: object) -> None:
        raise subprocess.CalledProcessError(2, ["uv", "run", "dagzoo", "generate"])

    monkeypatch.setattr(data_group, "run_dagzoo_generate_manifest", _fake_workflow)

    exit_code = cli_module.main(
        [
            "dev",
            "data",
            "generate-manifest",
            "--dagzoo-root",
            "/tmp/dagzoo",
            "--dagzoo-config",
            "configs/default.yaml",
            "--handoff-root",
            "/tmp/handoff",
            "--out-manifest",
            "/tmp/manifest.parquet",
        ]
    )

    assert exit_code == 2
