from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any


def capture_handler(
    captured: dict[str, object],
    fields: dict[str, Callable[[Any], object]],
) -> Callable[..., int]:
    def _handler(*args: Any, **kwargs: Any) -> int:
        if kwargs:
            target = SimpleNamespace(**kwargs)
        elif args:
            target = args[0]
        else:
            target = SimpleNamespace()
        for key, reader in fields.items():
            captured[key] = reader(target)
        return 0

    return _handler
