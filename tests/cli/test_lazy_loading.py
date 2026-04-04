from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

_ROOT_HEAVY_SENTINELS = {
    "tab_foundry.cli.groups.bench",
    "tab_foundry.cli.groups.research",
    "tab_foundry.research.sweep.models",
    "tab_foundry.bench.openml_bundle.discovery",
}


def _probe_cli(argv: list[str] | None) -> dict[str, Any]:
    script = f"""
import json
import sys

before_import = set(sys.modules)
import tab_foundry.cli as cli_module
after_import = set(sys.modules)

payload = {{
    "import_loaded": sorted(
        name for name in after_import - before_import if name.startswith("tab_foundry")
    ),
}}

argv = {argv!r}
if argv is not None:
    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(cli_module.cli, argv)
    after_invoke = set(sys.modules)
    payload.update(
        exit_code=result.exit_code,
        output=result.output,
        invoke_loaded=sorted(
            name for name in after_invoke - before_import if name.startswith("tab_foundry")
        ),
    )

print(json.dumps(payload))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


def test_import_tab_foundry_cli_is_lazy() -> None:
    payload = _probe_cli(None)

    assert "tab_foundry.cli" in payload["import_loaded"]
    assert "tab_foundry.cli.app" in payload["import_loaded"]
    assert "tab_foundry.cli.click_utils" in payload["import_loaded"]
    for sentinel in _ROOT_HEAVY_SENTINELS:
        assert sentinel not in payload["import_loaded"]


def test_root_help_avoids_heavy_cli_modules() -> None:
    payload = _probe_cli(["--help"])

    assert payload["exit_code"] == 0
    assert "Usage: tab-foundry" in payload["output"]
    for sentinel in _ROOT_HEAVY_SENTINELS:
        assert sentinel not in payload["invoke_loaded"]


def test_bench_help_loads_group_shell_without_child_commands() -> None:
    payload = _probe_cli(["bench", "--help"])

    assert payload["exit_code"] == 0
    assert "tab_foundry.cli.groups.bench" in payload["invoke_loaded"]
    assert "tab_foundry.cli.bench_compare" not in payload["invoke_loaded"]
    assert "tab_foundry.cli.bench_smoke_iris" not in payload["invoke_loaded"]
    assert "tab_foundry.bench.openml_bundle.discovery" not in payload["invoke_loaded"]
    assert "tab_foundry.research.sweep.models" not in payload["invoke_loaded"]


def test_research_help_loads_group_shell_without_deep_commands() -> None:
    payload = _probe_cli(["research", "--help"])

    assert payload["exit_code"] == 0
    assert "tab_foundry.cli.groups.research" in payload["invoke_loaded"]
    assert "tab_foundry.cli.research_execute" not in payload["invoke_loaded"]
    assert "tab_foundry.cli.research_sweep_core" not in payload["invoke_loaded"]
    assert "tab_foundry.research.sweep.models" not in payload["invoke_loaded"]


def test_research_sweep_help_stays_lazy_until_command_selection() -> None:
    payload = _probe_cli(["research", "sweep", "--help"])

    assert payload["exit_code"] == 0
    assert "tab_foundry.cli.groups.research" in payload["invoke_loaded"]
    assert "tab_foundry.cli.research_execute" not in payload["invoke_loaded"]
    assert "tab_foundry.cli.research_sweep_core" not in payload["invoke_loaded"]
    assert "tab_foundry.research.sweep.models" not in payload["invoke_loaded"]


def test_selected_lazy_commands_import_their_target_modules() -> None:
    bench_payload = _probe_cli(["bench", "compare", "--help"])
    research_payload = _probe_cli(["research", "sweep", "execute", "--help"])

    assert bench_payload["exit_code"] == 0
    assert "tab_foundry.cli.bench_compare" in bench_payload["invoke_loaded"]
    assert research_payload["exit_code"] == 0
    assert "tab_foundry.cli.research_execute" in research_payload["invoke_loaded"]
