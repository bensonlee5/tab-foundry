"""Shared import-path helpers for external benchmark helper entrypoints."""

from __future__ import annotations

from pathlib import Path


def resolve_tab_realdata_hub_root(*, tab_realdata_hub_root: Path | None) -> Path | None:
    """Resolve and validate an explicit tab-realdata-hub checkout root."""

    if tab_realdata_hub_root is None:
        return None
    resolved_root = tab_realdata_hub_root.expanduser().resolve()
    pyproject_path = resolved_root / "pyproject.toml"
    package_root = resolved_root / "src" / "tab_realdata_hub"
    if not pyproject_path.exists():
        raise RuntimeError(
            "tab-realdata-hub root must contain pyproject.toml: "
            f"{resolved_root}"
        )
    if not package_root.is_dir():
        raise RuntimeError(
            "tab-realdata-hub root must contain src/tab_realdata_hub: "
            f"{resolved_root}"
        )
    return resolved_root


def resolve_tab_realdata_hub_src_root(*, tab_realdata_hub_root: Path | None) -> Path | None:
    """Return the validated tab-realdata-hub ``src`` root for explicit helper overrides."""

    resolved_root = resolve_tab_realdata_hub_root(tab_realdata_hub_root=tab_realdata_hub_root)
    if resolved_root is None:
        return None
    return resolved_root / "src"


def prepend_explicit_tab_realdata_hub_src(
    sys_path: list[str],
    *,
    tab_realdata_hub_root: Path | None,
) -> None:
    """Prepend an explicit tab-realdata-hub checkout to ``sys.path`` when configured."""

    resolved_src_root = resolve_tab_realdata_hub_src_root(
        tab_realdata_hub_root=tab_realdata_hub_root,
    )
    if resolved_src_root is None:
        return
    resolved_src_root_str = str(resolved_src_root)
    if resolved_src_root_str not in sys_path:
        sys_path.insert(0, resolved_src_root_str)
