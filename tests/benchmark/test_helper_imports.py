from __future__ import annotations

from pathlib import Path

import pytest

from tab_foundry.bench.helper_imports import (
    prepend_explicit_tab_realdata_hub_src,
    resolve_tab_realdata_hub_root,
    resolve_tab_realdata_hub_src_root,
)


def test_resolve_tab_realdata_hub_root_returns_none_without_override() -> None:
    assert resolve_tab_realdata_hub_root(tab_realdata_hub_root=None) is None


def test_prepend_explicit_tab_realdata_hub_src_adds_valid_src_root(tmp_path: Path) -> None:
    hub_root = tmp_path / "tab-realdata-hub"
    (hub_root / "src" / "tab_realdata_hub").mkdir(parents=True)
    (hub_root / "pyproject.toml").write_text("[project]\nname='tab-realdata-hub'\n", encoding="utf-8")
    sys_path = ["existing"]

    prepend_explicit_tab_realdata_hub_src(
        sys_path,
        tab_realdata_hub_root=hub_root,
    )

    assert sys_path[0] == str((hub_root / "src").resolve())
    assert resolve_tab_realdata_hub_src_root(tab_realdata_hub_root=hub_root) == (
        hub_root / "src"
    ).resolve()


def test_prepend_explicit_tab_realdata_hub_src_rejects_invalid_root(tmp_path: Path) -> None:
    hub_root = tmp_path / "tab-realdata-hub"
    hub_root.mkdir()
    sys_path = ["existing"]

    with pytest.raises(RuntimeError, match="pyproject.toml"):
        prepend_explicit_tab_realdata_hub_src(
            sys_path,
            tab_realdata_hub_root=hub_root,
        )

    assert sys_path == ["existing"]
