# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def settings_path() -> Path:
    """Locate the shared YOLO settings without importing the ML package."""
    if config_dir := os.environ.get("YOLO_CONFIG_DIR"):
        directory = Path(config_dir).expanduser() / "Ultralytics"
    elif sys.platform == "linux":
        directory = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "Ultralytics"
    elif sys.platform == "win32":
        directory = Path.home() / "AppData" / "Roaming" / "Ultralytics"
    elif sys.platform == "darwin":
        directory = Path.home() / "Library" / "Application Support" / "Ultralytics"
    else:
        raise OSError(f"Unsupported settings platform: {sys.platform}")
    # Select the same directory as the settings writer; never revive a key from another location after logout.
    for candidate in (directory, Path("/tmp") / "Ultralytics", Path.cwd() / "Ultralytics"):
        if candidate.exists() or os.access(candidate.parent, os.W_OK):
            break
    return candidate / "settings.json"


def get_api_key() -> str | None:
    """Read the Platform key saved by yolo login without importing the ML package."""
    try:
        settings = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    api_key = settings.get("api_key") if isinstance(settings, dict) else None
    return api_key if isinstance(api_key, str) else None


def save_api_key(api_key: str) -> None:
    """Update the credential in the shared YOLO settings JSON file."""
    path = settings_path()
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError("YOLO settings must be a JSON object; refusing to overwrite them")  # noqa: TRY004
    settings["api_key"] = api_key
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)
