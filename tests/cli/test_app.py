from __future__ import annotations

import pytest

import tab_foundry.bench.prior_train as prior_train_module
import tab_foundry.cli as cli_module
import tab_foundry.research.system_delta as system_delta_module
import tab_foundry.research.sweep.graph as graph_module


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
