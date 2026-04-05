from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"
NANOTABPFN_MODEL_PATH_ENV = "TAB_FOUNDRY_TEST_NANOTABPFN_MODEL_PATH"


def optional_env_path(env_var: str) -> Path | None:
    raw_value = os.environ.get(env_var)
    if raw_value is None or not raw_value.strip():
        return None
    return Path(raw_value).expanduser().resolve()


def nanotabpfn_model_path() -> Path | None:
    return optional_env_path(NANOTABPFN_MODEL_PATH_ENV)
