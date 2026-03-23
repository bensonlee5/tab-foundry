from __future__ import annotations

import subprocess

import pytest

import tab_foundry.bench.compare as compare_module
import tab_foundry.cli as cli_module
import tab_foundry.cli.data_inspect as data_inspect_module
import tab_foundry.cli.dev as dev_module
import tab_foundry.cli.groups.data as data_group
import tab_foundry.cli.groups.research as research_group
import tab_foundry.research.sweep.core as sweep_core_module
import tab_foundry.research.sweep.diff as diff_module
import tab_foundry.research.sweep.execute as sweep_execute_module
import tab_foundry.research.sweep.graph as graph_module
import tab_foundry.research.sweep.inspect as inspect_module
import tab_foundry.research.sweep.promote as sweep_promote_module
import tab_foundry.research.sweep.summarize as summarize_module
import tab_foundry.training.prior_train as prior_train_module


def test_nested_cli_bench_compare_delegates_to_compare_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_compare(args):
        captured["tab_foundry_run_dir"] = str(args.tab_foundry_run_dir)
        return 0

    monkeypatch.setattr(compare_module, "run_from_args", _fake_compare)

    exit_code = cli_module.main(["bench", "compare", "--tab-foundry-run-dir", "/tmp/run"])

    assert exit_code == 0
    assert captured["tab_foundry_run_dir"] == "/tmp/run"


def test_nested_cli_train_prior_simple_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_prior_handler(args):
        captured["prior_dump"] = str(args.prior_dump)
        captured["overrides"] = list(args.overrides)
        return 0

    monkeypatch.setattr(prior_train_module, "run_from_args", _fake_prior_handler)

    exit_code = cli_module.main(
        [
            "train",
            "prior",
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

    monkeypatch.setattr(prior_train_module, "run_from_args", _fake_prior_handler)

    exit_code = cli_module.main(
        [
            "train",
            "prior",
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

    monkeypatch.setattr(sweep_core_module, "_run_sweep_create", _fake_run_sweep_create)

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
            "--benchmark-bundle-path",
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

    monkeypatch.setattr(sweep_core_module, "_run_list_sweeps", _fake_run_list_sweeps)

    exit_code = cli_module.main(
        ["research", "sweep", "list-sweeps", "--index-path", "/tmp/index.yaml"]
    )

    assert exit_code == 0
    assert captured == {"index_path": "/tmp/index.yaml"}


def test_nested_cli_research_sweep_show_active_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_show_active(args):
        captured["index_path"] = str(args.index_path)
        return 0

    monkeypatch.setattr(sweep_core_module, "_run_show_active", _fake_run_show_active)

    exit_code = cli_module.main(
        ["research", "sweep", "show-active", "--index-path", "/tmp/index.yaml"]
    )

    assert exit_code == 0
    assert captured == {"index_path": "/tmp/index.yaml"}


def test_nested_cli_research_sweep_render_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_sweep_render(args):
        captured["sweep_id"] = None if args.sweep_id is None else str(args.sweep_id)
        return 0

    monkeypatch.setattr(sweep_core_module, "_run_sweep_render", _fake_run_sweep_render)

    exit_code = cli_module.main(["research", "sweep", "render", "--sweep-id", "binary_md_v1"])

    assert exit_code == 0
    assert captured["sweep_id"] == "binary_md_v1"


def test_nested_cli_research_sweep_graph_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_graph_handler(args):
        captured["anchor"] = bool(args.anchor)
        captured["order"] = list(args.order)
        return 0

    monkeypatch.setattr(graph_module, "run_from_args", _fake_graph_handler)

    exit_code = cli_module.main(["research", "sweep", "graph", "--anchor", "--order", "7"])

    assert exit_code == 0
    assert captured == {"anchor": True, "order": [7]}


def test_nested_cli_research_sweep_execute_dispatches_to_sweep_native_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_execute_handler(args):
        captured["sweep_id"] = None if args.sweep_id is None else str(args.sweep_id)
        captured["include_completed"] = bool(args.include_completed)
        return 0

    monkeypatch.setattr(sweep_execute_module, "run_from_args", _fake_execute_handler)

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

    monkeypatch.setattr(sweep_promote_module, "run_from_args", _fake_promote_handler)

    exit_code = cli_module.main(
        ["research", "sweep", "promote", "--sweep-id", "binary_md_v1", "--run-id", "run_001"]
    )

    assert exit_code == 0
    assert captured == {"sweep_id": "binary_md_v1", "run_id": "run_001"}


def test_research_cli_group_imports_sweep_native_execute_and_promote_modules() -> None:
    assert research_group.sweep_execute.__name__ == "tab_foundry.research.sweep.execute"
    assert research_group.sweep_promote.__name__ == "tab_foundry.research.sweep.promote"


def test_nested_cli_research_sweep_summarize_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_summarize_handler(args):
        captured["sweep_id"] = None if args.sweep_id is None else str(args.sweep_id)
        captured["json"] = bool(args.json)
        return 0

    monkeypatch.setattr(summarize_module, "run_from_args", _fake_summarize_handler)

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
        captured["order"] = int(args.order)
        captured["json"] = bool(args.json)
        return 0

    monkeypatch.setattr(inspect_module, "run_from_args", _fake_inspect_handler)

    exit_code = cli_module.main(["research", "sweep", "inspect", "--order", "6", "--json"])

    assert exit_code == 0
    assert captured == {"order": 6, "json": True}


def test_nested_cli_research_sweep_diff_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_diff_handler(args):
        captured["order"] = int(args.order)
        captured["against_order"] = int(args.against_order)
        return 0

    monkeypatch.setattr(diff_module, "run_from_args", _fake_diff_handler)

    exit_code = cli_module.main(
        ["research", "sweep", "diff", "--order", "7", "--against-order", "6"]
    )

    assert exit_code == 0
    assert captured == {"order": 7, "against_order": 6}


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


def test_nested_cli_data_dagzoo_generate_manifest_dispatches_to_data_handler(
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
            "data",
            "dagzoo",
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
            "--dagzoo-root",
            "/tmp/dagzoo",
            "--force",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "recipe": "tf_rd_013_current_corpus_default_v1",
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


def test_nested_cli_data_build_manifest_rejects_invalid_split_ratios(
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


def test_nested_cli_data_dagzoo_generate_manifest_rejects_invalid_split_ratios(
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
                "data",
                "dagzoo",
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
            "data",
            "dagzoo",
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
def test_nested_cli_data_commands_reject_non_finite_split_ratios(argv: list[str]) -> None:
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


def test_nested_cli_data_dagzoo_generate_manifest_returns_subprocess_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_workflow(_config):
        raise subprocess.CalledProcessError(2, ["uv", "run", "dagzoo", "generate"])

    monkeypatch.setattr(data_group, "run_dagzoo_generate_manifest", _fake_workflow)

    exit_code = cli_module.main(
        [
            "data",
            "dagzoo",
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
