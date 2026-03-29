"""Shared import-path helpers for external benchmark helper entrypoints."""

from __future__ import annotations

from pathlib import Path


def optional_tab_realdata_hub_src_roots(*, tab_foundry_src: Path) -> tuple[Path, ...]:
    """Return sibling tab-realdata-hub src roots that should shadow installed packages."""

    resolved_tab_foundry_src = tab_foundry_src.expanduser().resolve()
    candidate = resolved_tab_foundry_src.parent.parent / "tab-realdata-hub" / "src"
    if candidate.exists():
        return (candidate,)
    return ()


def prepend_optional_tab_realdata_hub_src(sys_path: list[str], *, tab_foundry_src: Path) -> None:
    """Prepend sibling tab-realdata-hub src roots to ``sys.path`` when they exist."""

    for candidate in reversed(optional_tab_realdata_hub_src_roots(tab_foundry_src=tab_foundry_src)):
        candidate_str = str(candidate)
        if candidate_str not in sys_path:
            sys_path.insert(0, candidate_str)
