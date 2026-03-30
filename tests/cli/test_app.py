from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

import tab_foundry.bench.bounce_diagnosis as bounce_diagnosis_library_module
import tab_foundry.bench.comparison_contract as comparison_contract_library_module
import tab_foundry.bench.control_baseline_freeze as control_baseline_freeze_library_module
import tab_foundry.bench.dagzoo_smoke as dagzoo_smoke_library_module
import tab_foundry.bench.envs as env_library_module
import tab_foundry.bench.iris_smoke as iris_smoke_library_module
import tab_foundry.bench.openml_benchmark_bundle as bundle_library_module
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
import tab_foundry.cli.research_diff as research_diff_cli_module
import tab_foundry.cli.research_graph as research_graph_cli_module
import tab_foundry.cli.research_inspect as research_inspect_cli_module
import tab_foundry.cli.groups.bench as bench_group
import tab_foundry.cli.groups.data as data_group
import tab_foundry.cli.groups.research as research_group
import tab_foundry.cli.groups.train as train_group
import tab_foundry.cli.research_execute as research_execute_cli_module
import tab_foundry.cli.research_promote as research_promote_cli_module
import tab_foundry.cli.research_summarize as research_summarize_cli_module
import tab_foundry.cli.research_sweep_core as research_sweep_core_cli_module
import tab_foundry.cli.train_prior as train_prior_cli_module
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


def test_nested_cli_bench_compare_delegates_to_compare_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_compare(args):
        captured["tab_foundry_run_dir"] = str(args.tab_foundry_run_dir)
        captured["tab_realdata_hub_root"] = str(args.tab_realdata_hub_root)
        return 0

    monkeypatch.setattr(compare_cli_module, "run_from_args", _fake_compare)

    exit_code = cli_module.main(
        [
            "bench",
            "compare",
            "--tab-foundry-run-dir",
            "/tmp/run",
            "--tab-realdata-hub-root",
            "/tmp/tab-realdata-hub",
        ]
    )

    assert exit_code == 0
    assert captured["tab_foundry_run_dir"] == "/tmp/run"
    assert captured["tab_realdata_hub_root"] == "/tmp/tab-realdata-hub"


def test_nested_cli_bench_tune_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_tune(args):
        captured["manifest_path"] = str(args.manifest_path)
        captured["seed"] = int(args.seed)
        return 0

    monkeypatch.setattr(tune_cli_module, "run_from_args", _fake_tune)

    exit_code = cli_module.main(
        [
            "bench",
            "tune",
            "--manifest-path",
            "/tmp/manifest.parquet",
            "--seed",
            "7",
        ]
    )

    assert exit_code == 0
    assert captured == {"manifest_path": "/tmp/manifest.parquet", "seed": 7}


def test_nested_cli_bench_env_bootstrap_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_env_bootstrap(args):
        captured["nanotabpfn_root"] = str(args.nanotabpfn_root)
        captured["tabicl_root"] = str(args.tabicl_root)
        captured["tab_realdata_hub_root"] = str(args.tab_realdata_hub_root)
        return 0

    monkeypatch.setattr(env_bootstrap_cli_module, "run_from_args", _fake_env_bootstrap)

    exit_code = cli_module.main(
        [
            "bench",
            "env",
            "bootstrap",
            "--nanotabpfn-root",
            "/tmp/nano",
            "--tabicl-root",
            "/tmp/tabicl",
            "--tab-realdata-hub-root",
            "/tmp/tab-realdata-hub",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "nanotabpfn_root": "/tmp/nano",
        "tabicl_root": "/tmp/tabicl",
        "tab_realdata_hub_root": "/tmp/tab-realdata-hub",
    }


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

    exit_code = compare_cli_module.run_from_args(
        compare_cli_module.build_parser().parse_args(
            [
                "--tab-foundry-run-dir",
                str(tmp_path / "run"),
                "--tab-realdata-hub-root",
                str(hub_root),
            ]
        )
    )

    assert exit_code == 0
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

    exit_code = env_bootstrap_cli_module.run_from_args(
        env_bootstrap_cli_module.build_parser().parse_args(
            [
                "--tab-realdata-hub-root",
                str(hub_root),
            ]
        )
    )

    assert exit_code == 0
    assert captured["config"].tab_realdata_hub_root == hub_root


def test_nested_cli_bench_bundle_build_openml_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_bundle(args):
        captured["bundle_name"] = str(args.bundle_name)
        captured["task_source"] = str(args.task_source)
        return 0

    monkeypatch.setattr(bundle_cli_module, "run_from_args", _fake_bundle)

    exit_code = cli_module.main(
        [
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
        ]
    )

    assert exit_code == 0
    assert captured == {"bundle_name": "binary_medium", "task_source": "binary_expanded_v1"}


def test_nested_cli_bench_smoke_iris_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_smoke(args):
        captured["device"] = str(args.device)
        captured["checkpoint_every"] = int(args.checkpoint_every)
        return 0

    monkeypatch.setattr(iris_smoke_cli_module, "run_from_args", _fake_smoke)

    exit_code = cli_module.main(
        ["bench", "smoke", "iris", "--device", "cpu", "--checkpoint-every", "5"]
    )

    assert exit_code == 0
    assert captured == {"device": "cpu", "checkpoint_every": 5}


def test_nested_cli_bench_smoke_dagzoo_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_smoke(args):
        captured["dagzoo_root"] = str(args.dagzoo_root)
        captured["num_datasets"] = int(args.num_datasets)
        return 0

    monkeypatch.setattr(dagzoo_smoke_cli_module, "run_from_args", _fake_smoke)

    exit_code = cli_module.main(
        [
            "bench",
            "smoke",
            "dagzoo",
            "--dagzoo-root",
            "/tmp/dagzoo",
            "--num-datasets",
            "16",
        ]
    )

    assert exit_code == 0
    assert captured == {"dagzoo_root": "/tmp/dagzoo", "num_datasets": 16}


def test_nested_cli_bench_diagnose_bounce_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_diagnose(args):
        captured["run_dir"] = str(args.run_dir)
        captured["bootstrap_samples"] = int(args.bootstrap_samples)
        return 0

    monkeypatch.setattr(bounce_diagnosis_cli_module, "run_from_args", _fake_diagnose)

    exit_code = cli_module.main(
        [
            "bench",
            "diagnose",
            "bounce",
            "--run-dir",
            "/tmp/run",
            "--bootstrap-samples",
            "64",
        ]
    )

    assert exit_code == 0
    assert captured == {"run_dir": "/tmp/run", "bootstrap_samples": 64}


def test_nested_cli_bench_registry_register_run_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_register_handler(args):
        captured["run_id"] = str(args.run_id)
        captured["registry_path"] = str(args.registry_path)
        return 0

    monkeypatch.setattr(run_registration_cli_module, "run_from_args", _fake_register_handler)

    exit_code = cli_module.main(
        [
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
        ]
    )

    assert exit_code == 0
    assert captured == {
        "run_id": "run_001",
        "registry_path": "/tmp/registry.json",
    }


def test_nested_cli_bench_registry_freeze_baseline_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_freeze_handler(args):
        captured["baseline_id"] = str(args.baseline_id)
        captured["registry_path"] = str(args.registry_path)
        return 0

    monkeypatch.setattr(control_baseline_freeze_cli_module, "run_from_args", _fake_freeze_handler)

    exit_code = cli_module.main(
        [
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
        ]
    )

    assert exit_code == 0
    assert captured == {
        "baseline_id": "baseline_v1",
        "registry_path": "/tmp/control_baselines.json",
    }


def test_nested_cli_train_prior_simple_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_prior_handler(args):
        captured["prior_dump"] = str(args.prior_dump)
        captured["overrides"] = list(args.overrides)
        return 0

    monkeypatch.setattr(train_prior_cli_module, "run_from_args", _fake_prior_handler)

    exit_code = cli_module.main(
        [
            "train",
            "legacy-prior",
            "simple",
            "--prior-dump",
            "/tmp/prior.h5",
            "runtime.max_steps=1",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "prior_dump": "/tmp/prior.h5",
        "overrides": ["runtime.max_steps=1"],
    }


def test_nested_cli_train_prior_staged_injects_default_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_prior_handler(args):
        captured["prior_dump"] = str(args.prior_dump)
        captured["overrides"] = list(args.overrides)
        return 0

    monkeypatch.setattr(train_prior_cli_module, "run_from_args", _fake_prior_handler)

    exit_code = cli_module.main(
        [
            "train",
            "legacy-prior",
            "staged",
            "--prior-dump",
            "/tmp/prior.h5",
            "runtime.max_steps=1",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "prior_dump": "/tmp/prior.h5",
        "overrides": [
            "runtime.max_steps=1",
            "experiment=cls_benchmark_staged_prior",
        ],
    }


def test_nested_cli_research_sweep_create_sweep_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_sweep_create(args):
        captured["sweep_id"] = str(args.sweep_id)
        captured["anchor_run_id"] = str(args.anchor_run_id)
        return 0

    monkeypatch.setattr(research_sweep_core_cli_module, "_run_sweep_create", _fake_run_sweep_create)

    exit_code = cli_module.main(
        [
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
        ]
    )

    assert exit_code == 0
    assert captured == {
        "sweep_id": "binary_md_v1",
        "anchor_run_id": "run_001",
    }


def test_nested_cli_research_sweep_list_sweeps_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_list_sweeps(args):
        captured["index_path"] = str(args.index_path)
        return 0

    monkeypatch.setattr(research_sweep_core_cli_module, "_run_list_sweeps", _fake_run_list_sweeps)

    exit_code = cli_module.main(
        ["research", "sweep", "list-sweeps", "--index-path", "/tmp/index.yaml"]
    )

    assert exit_code == 0
    assert captured == {"index_path": "/tmp/index.yaml"}


def test_nested_cli_research_sweep_next_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_sweep_next(args):
        captured["sweep_id"] = str(args.sweep_id)
        captured["index_path"] = str(args.index_path)
        return 0

    monkeypatch.setattr(research_sweep_core_cli_module, "_run_sweep_next", _fake_run_sweep_next)

    exit_code = cli_module.main(
        ["research", "sweep", "next", "--sweep-id", "binary_md_v1", "--index-path", "/tmp/index.yaml"]
    )

    assert exit_code == 0
    assert captured == {"sweep_id": "binary_md_v1", "index_path": "/tmp/index.yaml"}


def test_nested_cli_research_sweep_render_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_sweep_render(args):
        captured["sweep_id"] = None if args.sweep_id is None else str(args.sweep_id)
        return 0

    monkeypatch.setattr(research_sweep_core_cli_module, "_run_sweep_render", _fake_run_sweep_render)

    exit_code = cli_module.main(["research", "sweep", "render", "--sweep-id", "binary_md_v1"])

    assert exit_code == 0
    assert captured["sweep_id"] == "binary_md_v1"


def test_nested_cli_research_sweep_materialize_corpora_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_sweep_materialize_corpora(args):
        captured["sweep_id"] = None if args.sweep_id is None else str(args.sweep_id)
        captured["dagzoo_root"] = str(args.dagzoo_root)
        captured["force"] = bool(args.force)
        captured["json"] = bool(args.json)
        return 0

    monkeypatch.setattr(
        research_sweep_core_cli_module,
        "_run_sweep_materialize_corpora",
        _fake_run_sweep_materialize_corpora,
    )

    exit_code = cli_module.main(
        [
            "research",
            "sweep",
            "materialize-corpora",
            "--sweep-id",
            "binary_md_v1",
            "--dagzoo-root",
            "/tmp/dagzoo",
            "--force",
            "--json",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "sweep_id": "binary_md_v1",
        "dagzoo_root": "/tmp/dagzoo",
        "force": True,
        "json": True,
    }


def test_nested_cli_research_sweep_graph_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_graph_handler(args):
        captured["sweep_id"] = str(args.sweep_id)
        captured["anchor"] = bool(args.anchor)
        captured["order"] = list(args.order)
        return 0

    monkeypatch.setattr(research_graph_cli_module, "run_from_args", _fake_graph_handler)

    exit_code = cli_module.main(
        ["research", "sweep", "graph", "--sweep-id", "binary_md_v1", "--anchor", "--order", "7"]
    )

    assert exit_code == 0
    assert captured == {"sweep_id": "binary_md_v1", "anchor": True, "order": [7]}


def test_nested_cli_research_sweep_execute_dispatches_to_sweep_native_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_execute_handler(args):
        captured["sweep_id"] = None if args.sweep_id is None else str(args.sweep_id)
        captured["include_completed"] = bool(args.include_completed)
        return 0

    monkeypatch.setattr(research_execute_cli_module, "run_from_args", _fake_execute_handler)

    exit_code = cli_module.main(
        ["research", "sweep", "execute", "--sweep-id", "binary_md_v1", "--include-completed"]
    )

    assert exit_code == 0
    assert captured == {"sweep_id": "binary_md_v1", "include_completed": True}


def test_nested_cli_research_sweep_promote_dispatches_to_sweep_native_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_promote_handler(args):
        captured["sweep_id"] = str(args.sweep_id)
        captured["run_id"] = str(args.run_id)
        return 0

    monkeypatch.setattr(research_promote_cli_module, "run_from_args", _fake_promote_handler)

    exit_code = cli_module.main(
        ["research", "sweep", "promote", "--sweep-id", "binary_md_v1", "--run-id", "run_001"]
    )

    assert exit_code == 0
    assert captured == {"sweep_id": "binary_md_v1", "run_id": "run_001"}


def test_cli_groups_use_cli_only_execute_promote_and_bench_modules() -> None:
    assert bench_group.compare_cli.__name__ == "tab_foundry.cli.bench_compare"
    assert bench_group.tune_cli.__name__ == "tab_foundry.cli.bench_tune"
    assert bench_group.env_bootstrap_cli.__name__ == "tab_foundry.cli.bench_env_bootstrap"
    assert bench_group.bundle_openml_cli.__name__ == "tab_foundry.cli.bench_bundle_openml"
    assert bench_group.iris_smoke_cli.__name__ == "tab_foundry.cli.bench_smoke_iris"
    assert bench_group.dagzoo_smoke_cli.__name__ == "tab_foundry.cli.bench_smoke_dagzoo"
    assert bench_group.bounce_diagnosis_cli.__name__ == "tab_foundry.cli.bench_bounce_diagnosis"
    assert bench_group.run_registration_cli.__name__ == "tab_foundry.cli.bench_run_registration"
    assert bench_group.control_baseline_freeze_cli.__name__ == "tab_foundry.cli.bench_control_baseline_freeze"
    assert train_group.train_prior_cli.__name__ == "tab_foundry.cli.train_prior"
    assert research_group.research_sweep_core_cli.__name__ == "tab_foundry.cli.research_sweep_core"
    assert research_group.research_graph_cli.__name__ == "tab_foundry.cli.research_graph"
    assert research_group.research_execute_cli.__name__ == "tab_foundry.cli.research_execute"
    assert research_group.research_inspect_cli.__name__ == "tab_foundry.cli.research_inspect"
    assert research_group.research_diff_cli.__name__ == "tab_foundry.cli.research_diff"
    assert research_group.research_promote_cli.__name__ == "tab_foundry.cli.research_promote"
    assert research_group.research_summarize_cli.__name__ == "tab_foundry.cli.research_summarize"
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


def test_nested_cli_research_sweep_summarize_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_summarize_handler(args):
        captured["sweep_id"] = None if args.sweep_id is None else str(args.sweep_id)
        captured["json"] = bool(args.json)
        return 0

    monkeypatch.setattr(research_summarize_cli_module, "run_from_args", _fake_summarize_handler)

    exit_code = cli_module.main(
        ["research", "sweep", "summarize", "--sweep-id", "cuda_stack_scale_followup", "--json"]
    )

    assert exit_code == 0
    assert captured == {"sweep_id": "cuda_stack_scale_followup", "json": True}


def test_nested_cli_research_sweep_inspect_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_inspect_handler(args):
        captured["sweep_id"] = str(args.sweep_id)
        captured["order"] = int(args.order)
        captured["json"] = bool(args.json)
        return 0

    monkeypatch.setattr(research_inspect_cli_module, "run_from_args", _fake_inspect_handler)

    exit_code = cli_module.main(
        ["research", "sweep", "inspect", "--sweep-id", "binary_md_v1", "--order", "6", "--json"]
    )

    assert exit_code == 0
    assert captured == {"sweep_id": "binary_md_v1", "order": 6, "json": True}


def test_nested_cli_research_sweep_diff_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_diff_handler(args):
        captured["sweep_id"] = str(args.sweep_id)
        captured["order"] = int(args.order)
        captured["against_order"] = int(args.against_order)
        return 0

    monkeypatch.setattr(research_diff_cli_module, "run_from_args", _fake_diff_handler)

    exit_code = cli_module.main(
        [
            "research",
            "sweep",
            "diff",
            "--sweep-id",
            "binary_md_v1",
            "--order",
            "7",
            "--against-order",
            "6",
        ]
    )

    assert exit_code == 0
    assert captured == {"sweep_id": "binary_md_v1", "order": 7, "against_order": 6}


def test_nested_cli_research_sweep_create_alias_is_rejected(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _ = cli_module.main(["research", "sweep", "create"])

    assert exc_info.value.code == 2
    assert "invalid choice: 'create'" in capsys.readouterr().err


def test_nested_cli_dev_resolve_config_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_resolve_config(args):
        captured["json"] = bool(args.json)
        captured["overrides"] = list(args.overrides)
        return 0

    monkeypatch.setattr(dev_module, "_run_resolve_config", _fake_run_resolve_config)

    exit_code = cli_module.main(["dev", "resolve-config", "--json", "experiment=cls_smoke"])

    assert exit_code == 0
    assert captured == {"json": True, "overrides": ["experiment=cls_smoke"]}


def test_nested_cli_dev_diff_config_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_diff_config(args):
        captured["left"] = list(args.left)
        captured["right"] = list(args.right)
        return 0

    monkeypatch.setattr(dev_module, "_run_diff_config", _fake_run_diff_config)

    exit_code = cli_module.main(
        ["dev", "diff-config", "--left", "experiment=cls_smoke", "--right", "experiment=cls_workstation"]
    )

    assert exit_code == 0
    assert captured == {
        "left": ["experiment=cls_smoke"],
        "right": ["experiment=cls_workstation"],
    }


def test_nested_cli_dev_export_check_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_export_check(args):
        captured["checkpoint"] = str(args.checkpoint)
        captured["json"] = bool(args.json)
        return 0

    monkeypatch.setattr(dev_module, "_run_export_check", _fake_run_export_check)

    exit_code = cli_module.main(["dev", "export-check", "--checkpoint", "/tmp/checkpoint.pt", "--json"])

    assert exit_code == 0
    assert captured == {"checkpoint": "/tmp/checkpoint.pt", "json": True}


def test_nested_cli_dev_run_inspect_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_inspect(args):
        captured["run_dir"] = str(args.run_dir)
        return 0

    monkeypatch.setattr(dev_module, "_run_run_inspect", _fake_run_inspect)

    exit_code = cli_module.main(["dev", "run-inspect", "--run-dir", "/tmp/run"])

    assert exit_code == 0
    assert captured["run_dir"] == "/tmp/run"


def test_nested_cli_dev_data_generate_manifest_dispatches_to_data_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_handler(args):
        captured["dagzoo_root"] = str(args.dagzoo_root)
        captured["dagzoo_config"] = str(args.dagzoo_config)
        captured["handoff_root"] = str(args.handoff_root)
        captured["out_manifest"] = str(args.out_manifest)
        captured["num_datasets"] = int(args.num_datasets)
        return 0

    monkeypatch.setattr(data_group, "_run_dagzoo_generate_manifest", _fake_handler)

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
            "--num-datasets",
            "32",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "dagzoo_root": "/tmp/dagzoo",
        "dagzoo_config": "configs/default.yaml",
        "handoff_root": "/tmp/handoff",
        "out_manifest": "/tmp/manifest.parquet",
        "num_datasets": 32,
    }


def test_nested_cli_data_corpus_materialize_dispatches_to_data_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_handler(args):
        captured["recipe"] = str(args.recipe)
        captured["sweep_id"] = None if args.sweep_id is None else str(args.sweep_id)
        captured["dagzoo_root"] = str(args.dagzoo_root)
        captured["force"] = bool(args.force)
        return 0

    monkeypatch.setattr(data_group, "_run_corpus_materialize", _fake_handler)

    exit_code = cli_module.main(
        [
            "data",
            "corpus",
            "materialize",
            "--recipe",
            "tf_rd_013_current_corpus_default_v1",
            "--sweep-id",
            "tf_rd_020_harder_dagzoo_ladder_v1",
            "--dagzoo-root",
            "/tmp/dagzoo",
            "--force",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "recipe": "tf_rd_013_current_corpus_default_v1",
        "sweep_id": "tf_rd_020_harder_dagzoo_ladder_v1",
        "dagzoo_root": "/tmp/dagzoo",
        "force": True,
    }


def test_nested_cli_data_corpus_inspect_dispatches_to_data_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_handler(args):
        captured["corpus_ref"] = str(args.corpus_ref)
        return 0

    monkeypatch.setattr(data_group, "_run_corpus_inspect", _fake_handler)

    exit_code = cli_module.main(
        [
            "data",
            "corpus",
            "inspect",
            "--corpus-ref",
            "tf_rd_013_current_corpus_default_v1/current_recipe__123456789abc",
        ]
    )

    assert exit_code == 0
    assert captured["corpus_ref"] == "tf_rd_013_current_corpus_default_v1/current_recipe__123456789abc"


def test_nested_cli_dev_data_build_manifest_rejects_invalid_split_ratios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def _fake_build_manifest(**_kwargs):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(data_group, "build_manifest", _fake_build_manifest)

    with pytest.raises(SystemExit, match="invalid split ratios"):
        _ = cli_module.main(
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
            ]
        )

    assert called is False


def test_nested_cli_data_manifest_inspect_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_manifest_inspect(args):
        captured["manifest"] = str(args.manifest)
        captured["experiment"] = str(args.experiment)
        captured["overrides"] = list(args.override)
        captured["json"] = bool(args.json)
        return 0

    monkeypatch.setattr(data_inspect_module, "run_from_args", _fake_run_manifest_inspect)

    exit_code = cli_module.main(
        [
            "data",
            "manifest-inspect",
            "--manifest",
            "/tmp/manifest.parquet",
            "--experiment",
            "cls_smoke",
            "--override",
            "data.manifest_path=/tmp/manifest.parquet",
            "--json",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "manifest": "/tmp/manifest.parquet",
        "experiment": "cls_smoke",
        "overrides": ["data.manifest_path=/tmp/manifest.parquet"],
        "json": True,
    }


def test_nested_cli_dev_data_generate_manifest_rejects_invalid_split_ratios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def _fake_workflow(_config):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(data_group, "run_dagzoo_generate_manifest", _fake_workflow)

    with pytest.raises(SystemExit, match="invalid split ratios"):
        _ = cli_module.main(
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
            ]
        )

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
    with pytest.raises(SystemExit):
        _ = cli_module.build_parser().parse_args(argv)


def test_nested_cli_rejects_unexpected_extra_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _ = cli_module.main(
            [
                "data",
                "manifest-inspect",
                "--manifest",
                "/tmp/manifest.parquet",
                "--unexpected",
            ]
        )
    assert exc_info.value.code == 2
    assert "unrecognized arguments: --unexpected" in capsys.readouterr().err


def test_nested_cli_dev_data_generate_manifest_returns_subprocess_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_workflow(_config):
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
