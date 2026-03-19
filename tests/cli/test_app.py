from __future__ import annotations

import subprocess

import pytest

import tab_foundry.bench.prior_train as prior_train_module
import tab_foundry.cli as cli_module
import tab_foundry.cli.groups.dev as dev_group
import tab_foundry.cli.groups.data as data_group
import tab_foundry.research.system_delta as system_delta_module
import tab_foundry.research.sweep.diff as diff_module
import tab_foundry.research.sweep.graph as graph_module
import tab_foundry.research.sweep.inspect as inspect_module
import tab_foundry.research.sweep.summarize as summarize_module


def test_nested_cli_bench_compare_delegates_to_compare_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tab_foundry.cli.groups.bench as bench_group

    captured: dict[str, object] = {}

    def _fake_compare(argv=None):
        captured["argv"] = list(argv) if argv is not None else None
        return 0

    monkeypatch.setattr(bench_group, "_run_compare", _fake_compare)

    exit_code = cli_module.main(["bench", "compare", "--tab-foundry-run-dir", "/tmp/run"])

    assert exit_code == 0
    assert captured["argv"] == ["--tab-foundry-run-dir", "/tmp/run"]


def test_nested_cli_train_prior_staged_injects_default_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_prior_main(argv=None):
        captured["argv"] = list(argv) if argv is not None else None
        return 0

    monkeypatch.setattr(prior_train_module, "main", _fake_prior_main)

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
    assert captured["argv"] == [
        "--prior-dump",
        "/tmp/prior.h5",
        "runtime.max_steps=1",
        "experiment=cls_benchmark_staged_prior",
    ]


def test_nested_cli_research_sweep_render_delegates_to_system_delta_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_system_delta_main(argv=None):
        captured["argv"] = list(argv) if argv is not None else None
        return 0

    monkeypatch.setattr(system_delta_module, "main", _fake_system_delta_main)

    exit_code = cli_module.main(["research", "sweep", "render", "--sweep-id", "binary_md_v1"])

    assert exit_code == 0
    assert captured["argv"] == ["render", "--sweep-id", "binary_md_v1"]


def test_nested_cli_research_sweep_graph_delegates_to_graph_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_graph_main(argv=None):
        captured["argv"] = list(argv) if argv is not None else None
        return 0

    monkeypatch.setattr(graph_module, "main", _fake_graph_main)

    exit_code = cli_module.main(["research", "sweep", "graph", "--anchor", "--order", "7"])

    assert exit_code == 0
    assert captured["argv"] == ["--anchor", "--order", "7"]


def test_nested_cli_research_sweep_summarize_delegates_to_summarize_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_summarize_main(argv=None):
        captured["argv"] = list(argv) if argv is not None else None
        return 0

    monkeypatch.setattr(summarize_module, "main", _fake_summarize_main)

    exit_code = cli_module.main(
        ["research", "sweep", "summarize", "--sweep-id", "cuda_stack_scale_followup", "--json"]
    )

    assert exit_code == 0
    assert captured["argv"] == ["--sweep-id", "cuda_stack_scale_followup", "--json"]


def test_nested_cli_research_sweep_inspect_delegates_to_inspect_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_inspect_main(argv=None):
        captured["argv"] = list(argv) if argv is not None else None
        return 0

    monkeypatch.setattr(inspect_module, "main", _fake_inspect_main)

    exit_code = cli_module.main(["research", "sweep", "inspect", "--order", "6", "--json"])

    assert exit_code == 0
    assert captured["argv"] == ["--order", "6", "--json"]


def test_nested_cli_research_sweep_diff_delegates_to_diff_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_diff_main(argv=None):
        captured["argv"] = list(argv) if argv is not None else None
        return 0

    monkeypatch.setattr(diff_module, "main", _fake_diff_main)

    exit_code = cli_module.main(
        ["research", "sweep", "diff", "--order", "7", "--against-order", "6"]
    )

    assert exit_code == 0
    assert captured["argv"] == ["--order", "7", "--against-order", "6"]


def test_nested_cli_dev_resolve_config_delegates_to_dev_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_resolve_config(argv=None):
        captured["argv"] = list(argv) if argv is not None else None
        return 0

    monkeypatch.setattr(dev_group, "_run_resolve_config", _fake_run_resolve_config)

    exit_code = cli_module.main(["dev", "resolve-config", "--json", "experiment=cls_smoke"])

    assert exit_code == 0
    assert captured["argv"] == ["--json", "experiment=cls_smoke"]


def test_nested_cli_dev_run_inspect_delegates_to_dev_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_inspect(argv=None):
        captured["argv"] = list(argv) if argv is not None else None
        return 0

    monkeypatch.setattr(dev_group, "_run_run_inspect", _fake_run_inspect)

    exit_code = cli_module.main(["dev", "run-inspect", "--run-dir", "/tmp/run"])

    assert exit_code == 0
    assert captured["argv"] == ["--run-dir", "/tmp/run"]


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
