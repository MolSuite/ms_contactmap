"""The interactive optimizer must never occupy Qt's GUI thread."""
from __future__ import annotations

import os
import time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, Qt, QTimer
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


def test_dragging_a_line_bends_it_and_double_click_resets():
    from ms_contactmap.layout import solve_layout

    diagram = _diagram()
    widget = InteractionDiagramWidget()
    widget.resize(900, 600)
    widget.show()
    widget.set_layout(diagram, solve_layout(diagram))
    QApplication.processEvents()
    route = widget._build.routes._routes[0]
    start, end = route.path.pointAtPercent(0.0), route.path.pointAtPercent(1.0)
    mid = route.path.pointAtPercent(0.5)
    length = ((end.x() - start.x()) ** 2 + (end.y() - start.y()) ** 2) ** 0.5
    normal = type(mid)(-(end.y() - start.y()) / length, (end.x() - start.x()) / length)
    view, positions = widget._view, dict(widget._positions)
    port = view.viewport()
    grab, drop = view.mapFromScene(mid), view.mapFromScene(mid + normal * 20.0)

    QTest.mousePress(port, Qt.LeftButton, Qt.NoModifier, grab)
    QTest.mouseMove(port, drop)
    QTest.mouseRelease(port, Qt.LeftButton, Qt.NoModifier, drop)
    QApplication.processEvents()
    assert list(widget._bends) == [route.bend_key]
    assert widget._bends[route.bend_key] > 0
    assert widget._positions == positions

    widget._rebuild()
    bent = widget._build.routes._routes[0]
    apex = bent.path.pointAtPercent(0.5)
    target = mid + normal * 20.0
    assert abs(apex.x() - target.x()) < 3 and abs(apex.y() - target.y()) < 3

    QTest.mouseDClick(port, Qt.LeftButton, Qt.NoModifier, view.mapFromScene(apex))
    QApplication.processEvents()
    assert widget._bends == {}


def test_selecting_a_glyph_announces_its_residue_once():
    widget = InteractionDiagramWidget()
    widget.set_diagram(_diagram())
    announced = []
    widget.residueSelected.connect(announced.append)
    glyph = next(item for item in widget._scene.items() if getattr(item, "residue_key", None) == "A:10:SER")
    glyph.setSelected(True)
    glyph.setSelected(True)
    widget._scene.clearSelection()
    assert announced == ["A:10:SER", ""]
    widget.deleteLater()


def test_a_zoomed_diagram_shows_no_scroll_bars_and_still_pans():
    app = ensure_app()
    scene = QGraphicsScene()
    scene.addRect(0, 0, 1000, 1000)
    view = DiagramView(scene)
    view.resize(300, 300)
    view.show()
    view.scale(4, 4)  # far past the fit: the scene overflows both ways
    app.processEvents()
    assert not view.horizontalScrollBar().isVisible() and not view.verticalScrollBar().isVisible()
    bar = view.horizontalScrollBar()
    before = bar.value()
    bar.setValue(before + 50)  # what the middle-button hand drag moves
    assert bar.value() == before + 50
    view.close()



def _wait_until(condition, seconds: float = 30.0) -> bool:
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    return condition()


def test_set_complex_waits_for_build_diagram_then_builds_and_reports(tmp_path):
    ensure_app()
    mol = Chem.AddHs(Chem.MolFromSmiles("OCCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    ligand = tmp_path / "pose.sdf"
    ligand.write_text(Chem.MolToMolBlock(mol) + "$$$$\n")
    x, y, z = mol.GetConformer().GetAtomPosition(0)
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("".join(
        f"ATOM  {i:>5}  {name:<3} SER A  10    {x + dx:8.3f}{y + dy:8.3f}{z:8.3f}  1.00  0.00           {name[0]}\n"
        for i, (name, dx, dy) in enumerate([("N", 5, 0), ("CA", 4, 1), ("C", 4.5, 2.3), ("O", 5.5, 2.5),
                                             ("CB", 3.2, 0.5), ("OG", 2.8, 0)], 1)) + "END\n")

    widget = InteractionDiagramWidget()
    built = []
    widget.diagramBuilt.connect(lambda diagram, layout: built.append((diagram, layout)))
    widget.set_complex(receptor, tmp_path / "missing.sdf", "lig", message="Nothing yet.")
    assert widget._pages.currentIndex() == 1 and widget._prompt_label.text() == "Nothing yet."
    widget._build_button.click()  # a bad complex: the error lands under the message, button back on
    assert _wait_until(lambda: "Build failed" in widget._prompt_label.text())
    assert widget._build_button.isEnabled() and not built

    widget.set_complex(receptor, ligand, "lig")
    assert widget.diagram is None  # held, not read
    widget._build_button.click()
    assert _wait_until(lambda: bool(built))
    assert built[0][0] is widget.diagram and widget._pages.currentWidget() is widget._view
    assert widget._toolbar.isEnabled()
