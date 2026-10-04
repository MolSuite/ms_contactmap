"""MS-ContactMap's own settings: how new diagrams look.

One user-level TOML file.  The standalone window reads and writes it here; a host
app (AMDockVS) shows the same file as the "MS-ContactMap" entry of its settings
dialog, so both contexts share one configuration.  Pure Python: no Qt.
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

SETTINGS_PATH = Path.home() / ".config" / "MS-ContactMap" / "config.toml"


def load_view_settings() -> dict:
    """Saved view settings; empty when there are none or the file is unreadable."""
    try:
        return tomllib.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_view_settings(values: dict) -> None:
    # A flat table of strings, booleans and integers: their JSON spelling is valid TOML.
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = (f"{key} = {json.dumps(value)}\n" for key, value in values.items())
    SETTINGS_PATH.write_text("".join(lines), encoding="utf-8")
