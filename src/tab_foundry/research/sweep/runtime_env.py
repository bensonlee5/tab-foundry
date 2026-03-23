"""Runtime-environment helpers for system-delta sweep execution."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def python_can_import_torch(python_path: Path) -> bool:
    try:
        result = subprocess.run(
            [str(python_path), "-c", "import torch"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0


def absolute_path_without_resolving_symlinks(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return Path.cwd() / expanded


def planned_nanotabpfn_python_path(nanotabpfn_root: Path) -> Path:
    return nanotabpfn_root.expanduser().resolve() / ".venv" / "bin" / "python"


def ensure_nanotabpfn_python(*, nanotabpfn_root: Path, fallback_python: Path) -> Path:
    nanotab_python = planned_nanotabpfn_python_path(nanotabpfn_root)
    fallback_executable = absolute_path_without_resolving_symlinks(fallback_python)
    nanotab_python.parent.mkdir(parents=True, exist_ok=True)
    if nanotab_python.exists() and python_can_import_torch(nanotab_python):
        return nanotab_python
    if nanotab_python.exists() or nanotab_python.is_symlink():
        nanotab_python.unlink()
    if not python_can_import_torch(fallback_executable):
        raise RuntimeError(
            "fallback interpreter cannot import torch: "
            f"{fallback_executable}"
        )
    nanotab_python.write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(str(fallback_executable))} \"$@\"\n",
        encoding="utf-8",
    )
    nanotab_python.chmod(0o755)
    return nanotab_python
