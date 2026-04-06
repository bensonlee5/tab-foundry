from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import tab_foundry.research.tf_rd_022_compile_debug as compile_debug_module


def test_tf_rd_022_compile_debug_builds_the_expected_variant_matrix(tmp_path: Path) -> None:
    requests = compile_debug_module._build_variant_requests(tmp_path / "compile_debug", max_steps=24)

    assert [request["variant"]["name"] for request in requests] == [
        "baseline_uncompiled",
        "compile_eager",
        "compile_aot_eager",
        "compile_inductor_default",
        "compile_inductor_max_autotune",
    ]
    assert requests[0]["variant"]["compile_model"] is False
    assert requests[1]["variant"]["compile_backend"] == "eager"
    assert requests[2]["variant"]["compile_backend"] == "aot_eager"
    assert requests[3]["variant"]["compile_mode"] == "default"
    assert requests[4]["variant"]["compile_mode"] == "max-autotune-no-cudagraphs"
    assert str(requests[0]["run_output_dir"]).endswith("/baseline_uncompiled/run")


def test_tf_rd_022_compile_debug_supports_variant_filtering(tmp_path: Path) -> None:
    requests = compile_debug_module._build_variant_requests(
        tmp_path / "compile_debug",
        max_steps=24,
        variant_names=("baseline_uncompiled", "compile_inductor_default"),
    )

    assert [request["variant"]["name"] for request in requests] == [
        "baseline_uncompiled",
        "compile_inductor_default",
    ]


def test_tf_rd_022_compile_debug_builds_env_and_output_layout(tmp_path: Path) -> None:
    requests = compile_debug_module._build_variant_requests(tmp_path / "compile_debug", max_steps=8)
    variant_dir = Path(str(requests[0]["variant_dir"]))
    env = compile_debug_module._build_variant_env(variant_dir)

    assert env["TORCH_LOGS"] == "graph_breaks,recompiles,dynamic,guards"
    assert env["TORCH_TRACE"] == str(variant_dir / "torch_trace")


def test_tf_rd_022_compile_debug_log_parser_summarizes_compile_diagnostics() -> None:
    diagnostics = compile_debug_module._parse_compile_debug_log(
        "\n".join(
            [
                "Graph break in user code at /tmp/model.py:681",
                "AUTOTUNE gemm",
                "Recompiling forward path",
                "Graph break in user code at /tmp/model.py:681",
                "Graph break in user code at /tmp/other.py:10",
                "AUTOTUNE reduction",
            ]
        )
    )

    assert diagnostics == {
        "graph_break_count": 3,
        "graph_break_locations": ["/tmp/model.py:681", "/tmp/other.py:10"],
        "recompile_count": 1,
        "autotune_count": 2,
    }


def test_run_tf_rd_022_compile_debug_suite_writes_machine_readable_summary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def _fake_run(command: list[str], **kwargs):
        request_path = Path(str(command[-1]))
        request = json.loads(request_path.read_text(encoding="utf-8"))
        variant_dir = Path(str(request["variant_dir"]))
        run_output_dir = Path(str(request["run_output_dir"]))
        compile_debug_module._write_json(
            variant_dir / "resolved_config.json",
            {"runtime": {"compile_model": request["variant"]["compile_model"]}},
        )
        compile_debug_module._write_json(
            variant_dir / "subprocess_result.json",
            {"output_dir": str(run_output_dir), "global_step": 8, "best_checkpoint": None},
        )
        Path(str(kwargs["env"]["TORCH_TRACE"])).mkdir(parents=True, exist_ok=True)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        (run_output_dir / "training_surface_record.json").write_text("{}", encoding="utf-8")
        (run_output_dir / "train_history.jsonl").write_text("", encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout="Graph break in user code at /tmp/model.py:681\nAUTOTUNE gemm\nRecompiling forward",
            stderr="",
        )

    monkeypatch.setattr(compile_debug_module.subprocess, "run", _fake_run)

    summary = compile_debug_module.run_tf_rd_022_compile_debug_suite(
        tmp_path / "compile_debug_suite",
        max_steps=8,
        python_executable="/tmp/fake-python",
        variant_names=("baseline_uncompiled", "compile_inductor_default"),
    )

    summary_path = tmp_path / "compile_debug_suite" / "compile_debug_summary.json"
    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary == persisted_summary
    assert summary["python_executable"] == "/tmp/fake-python"
    assert summary["max_steps"] == 8
    assert len(summary["variants"]) == 2
    assert [variant["name"] for variant in summary["variants"]] == [
        "baseline_uncompiled",
        "compile_inductor_default",
    ]
    assert summary["variants"][0]["graph_break_count"] == 1
    assert summary["variants"][0]["autotune_count"] == 1
    assert summary["variants"][0]["recompile_count"] == 1
    assert summary["variants"][0]["artifacts"]["training_surface_record"] is True
    assert summary["variants"][0]["artifacts"]["torch_trace"] is True
