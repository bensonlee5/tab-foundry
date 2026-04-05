from __future__ import annotations


BANNED_LOCAL_PATH_MARKERS = (
    "/" + "Users" + "/",
    "~/" + "dev/",
    "$HOME/" + "dev",
    "/" + "workspace" + "/",
)


def find_banned_local_path_markers(text: str) -> tuple[str, ...]:
    return tuple(marker for marker in BANNED_LOCAL_PATH_MARKERS if marker in text)
