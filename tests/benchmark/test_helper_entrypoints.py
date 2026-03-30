from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NANOTABPFN_HELPER_SCRIPT_PATH = REPO_ROOT / "scripts" / "bench" / "nanotabpfn_helper.py"
TABICLV2_HELPER_SCRIPT_PATH = REPO_ROOT / "scripts" / "bench" / "tabiclv2_helper.py"


def _load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nanotabpfn_helper_entrypoint_parses_tab_realdata_hub_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    script_module = _load_script(
        NANOTABPFN_HELPER_SCRIPT_PATH,
        "bench_nanotabpfn_helper_script",
    )
    captured: dict[str, object] = {}
    hub_root = tmp_path / "tab-realdata-hub"

    monkeypatch.setattr(
        script_module.helper_module,
        "run_nanotabpfn_helper",
        lambda **kwargs: captured.update(kwargs) or 0,
    )

    exit_code = script_module.main(
        [
            "--tab-foundry-src",
            str((REPO_ROOT / "src").resolve()),
            "--benchmark-manifest",
            str(tmp_path / "manifest.parquet"),
            "--prior-dump",
            str(tmp_path / "prior.h5"),
            "--out-path",
            str(tmp_path / "curve.jsonl"),
            "--tab-realdata-hub-root",
            str(hub_root),
        ]
    )

    assert exit_code == 0
    assert captured["tab_realdata_hub_root"] == hub_root


def test_tabiclv2_helper_entrypoint_parses_tab_realdata_hub_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    script_module = _load_script(
        TABICLV2_HELPER_SCRIPT_PATH,
        "bench_tabiclv2_helper_script_argparse",
    )
    captured: dict[str, object] = {}
    hub_root = tmp_path / "tab-realdata-hub"

    monkeypatch.setattr(
        script_module.helper_module,
        "run_tabiclv2_helper",
        lambda **kwargs: captured.update(kwargs) or 0,
    )

    exit_code = script_module.main(
        [
            "--tab-foundry-src",
            str((REPO_ROOT / "src").resolve()),
            "--benchmark-manifest",
            str(tmp_path / "manifest.parquet"),
            "--out-path",
            str(tmp_path / "curve.jsonl"),
            "--task-type",
            "supervised_classification",
            "--checkpoint-version",
            "classifier.ckpt",
            "--tab-realdata-hub-root",
            str(hub_root),
        ]
    )

    assert exit_code == 0
    assert captured["tab_realdata_hub_root"] == hub_root
