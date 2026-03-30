from __future__ import annotations

from collections.abc import Callable
from typing import Any


def capture_handler(
    captured: dict[str, object],
    fields: dict[str, Callable[[Any], object]],
) -> Callable[[Any], int]:
    def _handler(args: Any) -> int:
        for key, reader in fields.items():
            captured[key] = reader(args)
        return 0

    return _handler
