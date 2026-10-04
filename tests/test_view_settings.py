"""The widget's host-owned settings.  Run with ``python tests/test_view_settings.py``."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Qt must be loaded before RDKit in the same process (see render.py).
from PySide6.QtWidgets import QApplication  # noqa: F401

from ms_contactmap.export import ensure_app
from ms_contactmap.style import DEFAULT_LEGEND_POSITION, DEFAULT_LEGEND_ROWS, DiagramStyle
from ms_contactmap.widget import InteractionDiagramWidget


def test_view_settings_round_trip_and_host_owned_action() -> None:
    ensure_app()
    widget = InteractionDiagramWidget()
    assert widget.view_settings() == {
        **DiagramStyle().to_dict(),
        "legend_position": DEFAULT_LEGEND_POSITION,
        "legend_rows": DEFAULT_LEGEND_ROWS,
    }
    assert not widget._configure_action.isVisible()  # no host, no entry

    chosen = {**widget.view_settings(), "palette": "vivid", "legend_position": "top", "legend_rows": 2}
    widget.apply_view_settings({**chosen, "not_a_setting": 1})
    assert widget.view_settings() == chosen

    calls: list[int] = []
    widget.set_configure_action("Defaults…", lambda: calls.append(1))
    widget._configure_action.trigger()
    assert calls == [1] and widget._configure_action.text() == "Defaults…"


if __name__ == "__main__":
    test_view_settings_round_trip_and_host_owned_action()
    print("ok")
