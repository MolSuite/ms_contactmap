"""The interactive optimizer must never occupy Qt's GUI thread."""
from __future__ import annotations

import os
import time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGraphicsScene, QToolButton
from rdkit import Chem
from rdkit.Chem import AllChem

from ms_contactmap.export import ensure_app
from ms_contactmap.model import Diagram, Interaction, Residue, ResidueRef
from ms_contactmap.widget import DiagramView, InteractionDiagramWidget, RESIZE_FIT_DELAY_MS


def _diagram() -> Diagram:
    """Small real solve with one actual backbone connector."""
    ensure_app()
    mol = Chem.MolFromSmiles("CCO")
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    coords = [
        (conf.GetAtomPosition(i).x * 40.0, conf.GetAtomPosition(i).y * 40.0)
        for i in range(mol.GetNumAtoms())
    ]
    left = Residue(ResidueRef("A", 10, "SER"), has_interactions=True)
    right = Residue(ResidueRef("A", 11, "ASP"), has_interactions=True)
    return Diagram(
        "async", "LIG", mol, coords,
        residues=[left, right],
        interactions=[
            Interaction("hbond", left.key, (2,), 2.8, False),
            Interaction("hbond", right.key, (2,), 3.0, False),
        ],
        exposure={}, nearest_atom={left.key: 2, right.key: 2},
    )


def test_threaded_layout_keeps_qt_event_loop_responsive():
    diagram = _diagram()
    widget = InteractionDiagramWidget()
    loop = QEventLoop()
    heartbeat = []
    finished = []
    failed = []
    QTimer.singleShot(0, lambda: heartbeat.append(True))
    QTimer.singleShot(10000, loop.quit)
    widget.layoutFinished.connect(lambda result: (finished.append(result), loop.quit()))
    widget.layoutFailed.connect(lambda message: (failed.append(message), loop.quit()))

    widget.set_diagram_async(diagram)
    assert widget.is_solving
    loop.exec()

    assert heartbeat, "Qt timers were blocked while solve_layout was running"
    assert not failed, failed
    assert finished
    assert not widget.is_solving


def test_backbone_can_hide_while_optimizer_is_running():
    diagram = _diagram()
    widget = InteractionDiagramWidget(diagram)
    assert widget._build.backbone.isVisible()

    loop = QEventLoop()
    hidden = []
    failed = []
    tick_times = [time.perf_counter()]
    heartbeat = QTimer(widget)
    heartbeat.setInterval(10)
    heartbeat.timeout.connect(lambda: tick_times.append(time.perf_counter()))
    heartbeat.start()
    widget.layoutFinished.connect(loop.quit)
    widget.layoutFailed.connect(lambda message: (failed.append(message), loop.quit()))
    QTimer.singleShot(
        0,
        lambda: (
            widget._backbone_action.setChecked(False),
            hidden.append(not widget._build.backbone.isVisible()),
        ),
    )
    QTimer.singleShot(10000, loop.quit)
    widget.reshuffle()
    loop.exec()
    heartbeat.stop()

    assert not failed, failed
    assert hidden == [True]
    assert not widget._backbone_action.isChecked()
    assert not widget._build.backbone.isVisible()
    gaps = [right - left for left, right in zip(tick_times, tick_times[1:])]
    assert not gaps or max(gaps) < 0.75, \
        f"GUI event loop stalled for {max(gaps):.2f} s"


def test_repeated_reshuffles_finish_and_report_progress():
    diagram = _diagram()
    widget = InteractionDiagramWidget()

    def wait_for(action) -> None:
        loop = QEventLoop()
        finished = []
        failed = []

        def on_finished(result) -> None:
            finished.append(result)
            loop.quit()

        def on_failed(message: str) -> None:
            failed.append(message)
            loop.quit()

        widget.layoutFinished.connect(on_finished)
        widget.layoutFailed.connect(on_failed)
        QTimer.singleShot(10000, loop.quit)
        action()
        loop.exec()
        widget.layoutFinished.disconnect(on_finished)
        widget.layoutFailed.disconnect(on_failed)
        assert not failed, failed
        assert finished
        assert not widget.is_solving

    # This is the demo's exact lifecycle: initial background solve followed by
    # several thread-backed reshuffles.  Reusing a child process was the source
    # of the intermittent second-click freeze in the former implementation.
    wait_for(lambda: widget.set_diagram_async(diagram))
    for _ in range(3):
        wait_for(widget.reshuffle)
        assert "Reshuffling complete" in widget._status_label.text()


def test_resize_coalesces_refits_and_overflow_stays_inside_toolbar():
    app = ensure_app()
    scene = QGraphicsScene()
    view = DiagramView(scene)
    ticks = []
    view._resize_fit_timer.timeout.connect(lambda: ticks.append(True))
    view.show()
    QTest.qWait(RESIZE_FIT_DELAY_MS + 25)
    ticks.clear()

    for width in range(400, 420):
        view.resize(width, 300)

    assert view._resize_fit_timer.isActive()
    QTest.qWait(RESIZE_FIT_DELAY_MS + 25)
    assert ticks == [True]
    view.close()

    widget = InteractionDiagramWidget()
    widget._toolbar.setFixedWidth(100)
    widget.show()
    app.processEvents()
    overflow = widget._toolbar.findChild(QToolButton, "qt_toolbar_ext_button")
    assert overflow is not None and not overflow.isHidden()
    assert overflow.width() == 28
    assert overflow.text() == "»"
    assert overflow.icon().isNull()
    assert widget._toolbar.contentsMargins().right() == 4
    widget.close()
