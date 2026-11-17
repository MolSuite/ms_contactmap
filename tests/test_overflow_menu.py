"""The toolbar's "»" opens a menu of exactly what the toolbar clipped."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QToolButton  # noqa: E402

from ms_contactmap.widget import InteractionDiagramWidget  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(["contactmap-overflow-test"])


def _shown(widget, width):
    widget.resize(width, 500)
    QApplication.processEvents()
    QApplication.processEvents()
    return widget._toolbar.findChild(QToolButton, "qt_toolbar_ext_button")


def test_overflow_menu_lists_every_clipped_action_with_its_choices(app):
    widget = InteractionDiagramWidget()
    widget.show()
    ext = _shown(widget, 200)
    assert ext.isVisible() and ext.menu() is widget._overflow_menu

    own_menus = [(m, m.title(), list(m.actions())) for m in
                 (widget._reshuffle_action.menu(), widget._legend_menu)]
    widget._fill_overflow_menu()
    entries = {a.text(): a for a in widget._overflow_menu.actions() if not a.isSeparator()}
    clipped = [a for a in widget._overflow_actions() if a is not None]
    toolbar = widget._toolbar
    # Everything clipped is in the menu; the toolbar holds buttons only.
    assert all(type(toolbar.widgetForAction(a)).__name__ == "QToolButton" for a in clipped)
    for action in clipped:
        assert action.text() in entries, action.text()
    # Drop-downs arrive as submenus with all their choices, toggles keep their state.
    legend = [a.text() for a in entries["Legend"].menu().actions()]
    assert legend[:2] == ["Position", "Horizontal rows"]
    assert "Residues only" in [a.text() for a in entries["Reshuffle"].menu().actions()]
    # The submenus are copies: the toolbar's own drop-downs keep their title and actions.
    assert all(m.title() == t and m.actions() == a for m, t, a in own_menus)
    assert entries["Settle neighbors"].isCheckable() and entries["Settle neighbors"].isChecked()

    # A real press opens the menu (filled with the clipped actions), not Qt's
    # in-place expansion.  A blocked `pressed` signal once left it dead.
    height = toolbar.height()
    seen = {}

    def look():
        menu = widget._overflow_menu
        seen["visible"] = menu.isVisible()
        seen["items"] = [a.text() for a in menu.actions() if a.text()]
        menu.close()

    QTimer.singleShot(200, look)
    QTimer.singleShot(400, app.quit)
    QTest.mousePress(ext, Qt.LeftButton)
    app.exec()
    QTest.mouseRelease(ext, Qt.LeftButton)
    assert seen["visible"] and "Legend" in seen["items"]
    assert toolbar.height() == height


def test_no_overflow_button_when_everything_fits(app):
    widget = InteractionDiagramWidget()
    widget.show()
    assert not _shown(widget, 2400).isVisible()


def test_toolbar_goes_vertical_and_icons_only(app):
    widget = InteractionDiagramWidget()
    widget.show()
    _shown(widget, 900)
    assert all(not a.icon().isNull() for a, _ in widget._icon_names)
    widget.set_toolbar_layout(vertical=True)
    assert widget._toolbar.orientation() == Qt.Vertical
    assert widget._toolbar.toolButtonStyle() == Qt.ToolButtonTextUnderIcon
    widget._toolbar_actions["icons_only=True"].trigger()
    assert widget._toolbar.toolButtonStyle() == Qt.ToolButtonIconOnly
    widget.set_toolbar_layout(vertical=False)
    assert widget._toolbar.orientation() == Qt.Horizontal
    assert widget._toolbar.toolButtonStyle() == Qt.ToolButtonIconOnly
    # Status and rotate hint sit below the view, not in the bar.
    assert widget._status_label.parent() is widget
