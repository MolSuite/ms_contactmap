"""Interactive Qt component: an interaction diagram you can zoom, pan and edit.

This is the reusable piece.  Drop :class:`InteractionDiagramWidget` into any
PySide6 layout, hand it a :class:`~ms_contactmap.model.Diagram`, and it solves the
layout and draws it.  Residue droplets stay draggable; releasing one rebuilds
the scene around the new position so the connectors, ribbons and routes follow.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QMenu,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import export as export_mod
from . import render
from .layout import LayoutResult, solve_layout
from .model import Diagram

#: How much one wheel notch scales the view.
ZOOM_STEP = 1.15
ZOOM_LIMITS = (0.15, 12.0)

# A splitter emits one resize per pixel.  Re-fitting each one changes the view
# transform, invalidates every device-coordinate cache and repaints the full
# scene.  One fit after the drag pauses keeps resize work bounded.
RESIZE_FIT_DELAY_MS = 75

#: One click of the rotate buttons.
ROTATION_STEP = 15.0

LEGEND_POSITIONS = ("left", "right", "top", "bottom")
LEGEND_ROW_OPTIONS = (2, 3, 4)


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    return (sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points))


def _turn(point: tuple[float, float], center: tuple[float, float],
          radians: float) -> tuple[float, float]:
    dx, dy = point[0] - center[0], point[1] - center[1]
    cos, sin = math.cos(radians), math.sin(radians)
    return (center[0] + dx * cos - dy * sin, center[1] + dx * sin + dy * cos)


class DiagramView(QGraphicsView):
    """A QGraphicsView with wheel zoom, drag-pan and a drag-release signal."""

    residueMoved = Signal()

    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.TextAntialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.white)
        # Dragging a droplet leaves smears of the ribbon and shadow underneath
        # it with the default minimal-update mode: the soft layers are drawn
        # far outside their nominal geometry. Device caches keep repaints at a
        # stable transform cheap; resize refits are coalesced below because a
        # changed transform invalidates those caches.
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self._zoom = 1.0
        # Keep re-fitting until the user takes control of the zoom.  A view
        # built before its window is shown still has its default size, so the
        # first fit() would otherwise scale the diagram to a placeholder
        # viewport and leave it as a speck in the middle.
        self._auto_fit = True
        self._resize_fit_timer = QTimer(self)
        self._resize_fit_timer.setSingleShot(True)
        self._resize_fit_timer.setInterval(RESIZE_FIT_DELAY_MS)
        self._resize_fit_timer.timeout.connect(self._fit_after_resize)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._auto_fit:
            self._resize_fit_timer.stop()
            self.fit(user_zoom=False)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._auto_fit:
            self._resize_fit_timer.start()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        factor = ZOOM_STEP if event.angleDelta().y() > 0 else 1.0 / ZOOM_STEP
        target = self._zoom * factor
        if not (ZOOM_LIMITS[0] <= target <= ZOOM_LIMITS[1]):
            return
        self._auto_fit = False
        self._resize_fit_timer.stop()
        self._zoom = target
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        # Middle button pans; left button keeps selecting and dragging glyphs.
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.RubberBandDrag)
        elif event.button() == Qt.LeftButton:
            self.residueMoved.emit()

    def fit(self, user_zoom: bool = True) -> None:
        """Scale the diagram to the viewport.

        ``user_zoom=True`` (the default, and what the toolbar/API call does)
        re-arms auto-fitting, so an explicit "fit" undoes any manual zoom.
        """
        rect = self.scene().itemsBoundingRect()
        if rect.isEmpty():
            return
        self.fitInView(rect.adjusted(-12, -12, 12, 12), Qt.KeepAspectRatio)
        self._zoom = self.transform().m11()
        if user_zoom:
            self._auto_fit = True

    def fit_if_auto(self) -> None:
        """Re-fit after scene bounds change unless the user chose a zoom."""
        if self._auto_fit:
            self.fit(user_zoom=False)

    def _fit_after_resize(self) -> None:
        if self._auto_fit:
            self.fit(user_zoom=False)


class _LayoutSignals(QObject):
    """Queued result delivery from the bounded layout thread."""

    finished = Signal(int, object, object)
    failed = Signal(int, str)


class _LayoutTask(QRunnable):
    """One pure-Python/NumPy solve; no child process or serialization."""

    def __init__(self, request_id: int, diagram: Diagram, options: dict) -> None:
        super().__init__()
        self.request_id = request_id
        self.diagram = diagram
        self.options = options
        self.signals = _LayoutSignals()

    @Slot()
    def run(self) -> None:
        try:
            layout = solve_layout(self.diagram, **self.options)
        except Exception as exc:
            self.signals.failed.emit(
                self.request_id, f"{type(exc).__name__}: {exc}"
            )
            return
        self.signals.finished.emit(self.request_id, None, layout)


class InteractionDiagramWidget(QWidget):
    """Reusable 2D protein-ligand interaction diagram."""

    layoutStarted = Signal()
    layoutFinished = Signal(object)
    layoutFailed = Signal(str)

    def __init__(self, diagram: Diagram | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._diagram: Diagram | None = None
        self._build: render.SceneBuild | None = None
        self._layout: LayoutResult | None = None
        self._positions: dict[str, tuple[float, float]] = {}
        self._ligand_coords: list[tuple[float, float]] = []
        self._legend_position = "left"
        self._legend_rows = 3
        self._layout_actions: list[QAction] = []
        self._solving = False
        self._solve_serial = 0
        self._active_solve = 0
        self._pending_solve: tuple[int, Diagram, dict, str] | None = None
        self._solve_task: _LayoutTask | None = None
        self._solve_diagram: Diagram | None = None
        self._solve_started = 0.0
        self._solve_operation = "Optimizing layout"
        self._reshuffle_variant = 0
        self._residue_variant = 0

        self._scene = QGraphicsScene(self)
        self._scene.selectionChanged.connect(self._on_selection_changed)
        self._view = DiagramView(self._scene, self)
        self._view.residueMoved.connect(self._on_residue_moved)

        self._status_label = QLabel("  Ready  ")
        self._status_label.setMinimumWidth(170)
        self._selection_label = QLabel()
        self._toolbar = QToolBar(self)
        self._toolbar.setContentsMargins(0, 0, 4, 0)
        self._toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)

        self._solve_status_timer = QTimer(self)
        self._solve_status_timer.setInterval(100)
        self._solve_status_timer.timeout.connect(self._update_solve_status)
        self._layout_pool = QThreadPool(self)
        self._layout_pool.setMaxThreadCount(1)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        box.addWidget(self._toolbar)
        box.addWidget(self._view)

        # Also exposed through QWidget.actions(), so a host can drop the same
        # actions into its own menu; right-click shows them here.
        self._build_actions()
        self._configure_overflow_button()
        self.setContextMenuPolicy(Qt.ActionsContextMenu)

        if diagram is not None:
            self.set_diagram(diagram)
        self._on_selection_changed()

    def _build_actions(self) -> None:
        # Keep progress at the left edge: the selection hint at the far right
        # can be clipped in a narrow host window.
        self._toolbar.addWidget(self._status_label)
        self._toolbar.addSeparator()
        # The rotate actions act on whatever is selected -- a metal turns its
        # coordination polygon, any other glyph swings around the ligand, and
        # with nothing selected the ligand itself turns.  One pair of buttons
        # for all three, because from the user's side it is one gesture.
        self._auto_action = QAction("Settle neighbors", self)
        self._auto_action.setCheckable(True)
        self._auto_action.setChecked(True)
        self._auto_action.setToolTip(
            "After a rotation, optimize the rest of the diagram around the moved item"
        )
        self._layout_actions.append(self._auto_action)
        # Both view options, not layout options: they toggle what an existing
        # scene draws, so neither one re-solves anything.
        self._backbone_action = QAction("Backbone", self)
        self._backbone_action.setCheckable(True)
        self._backbone_action.setChecked(True)
        self._backbone_action.setShortcut(QKeySequence("C"))
        self._backbone_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self._backbone_action.setToolTip(
            "Lines between sequence-consecutive residues  (C)")
        self._backbone_action.toggled.connect(self._apply_layers)

        self._trail_action = QAction("Exposure trail", self)
        self._trail_action.setCheckable(True)
        self._trail_action.setShortcut(QKeySequence("E"))
        self._trail_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self._trail_action.setToolTip(
            "Show solvent exposure as an arc over the exposed face instead "
            "of a halo around the atom  (E)")
        self._trail_action.toggled.connect(self._apply_layers)

        for text, shortcut, slot in (
            ("↺", "[", lambda: self.rotate_selection(-ROTATION_STEP)),
            ("↻", "]", lambda: self.rotate_selection(ROTATION_STEP)),
            ("Mirror", "M", self.mirror_ligand),
            (None, None, None),
            (self._backbone_action, None, None),
            (self._trail_action, None, None),
            (None, None, None),
            (self._auto_action, None, None),
            (None, None, None),
            ("Reshuffle", "R", self.reshuffle),
            ("Recalculate", "Ctrl+R", self.reset_layout),
            ("Fit", "F", self.fit),
        ):
            if text is None:
                self._toolbar.addSeparator()
                continue
            action = text if isinstance(text, QAction) else QAction(text, self)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
                action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
                action.setToolTip(f"{action.text()}  ({shortcut})")
            if slot is not None:
                action.triggered.connect(slot)
            if action.text() in {"↺", "↻", "Mirror", "Reshuffle", "Recalculate"}:
                self._layout_actions.append(action)
            self._toolbar.addAction(action)
            self.addAction(action)
            if action.text() == "Reshuffle":
                self._reshuffle_action = action

        # One primary gesture, with the cheaper residues-only alternative in
        # the arrow menu.  The latter preserves the ligand depiction exactly.
        reshuffle_menu = QMenu("Reshuffle mode", self)
        reshuffle_menu.addAction("Ligand and residues", self.reshuffle)
        residues_action = reshuffle_menu.addAction(
            "Residues only", self.reshuffle_residues
        )
        residues_action.setShortcut(QKeySequence("Shift+R"))
        residues_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(residues_action)
        self._layout_actions.append(residues_action)
        self._reshuffle_action.setMenu(reshuffle_menu)
        reshuffle_button = self._toolbar.widgetForAction(self._reshuffle_action)
        if isinstance(reshuffle_button, QToolButton):
            reshuffle_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)

        self._build_legend_menu()
        self._toolbar.addSeparator()
        self._toolbar.addWidget(self._selection_label)

    def _build_legend_menu(self) -> None:
        """Add the position and horizontal row controls as one compact menu."""
        self._legend_menu = QMenu("Legend", self)
        position_menu = self._legend_menu.addMenu("Position")
        position_group = QActionGroup(self)
        position_group.setExclusive(True)
        self._legend_position_actions: dict[str, QAction] = {}
        for position, label in (
            ("left", "Left"),
            ("right", "Right"),
            ("top", "Top"),
            ("bottom", "Bottom"),
        ):
            action = position_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(position == self._legend_position)
            action.triggered.connect(
                lambda checked=False, value=position: self.set_legend_position(value)
            )
            position_group.addAction(action)
            self._legend_position_actions[position] = action

        self._legend_rows_menu = self._legend_menu.addMenu("Horizontal rows")
        rows_group = QActionGroup(self)
        rows_group.setExclusive(True)
        self._legend_row_actions: dict[int, QAction] = {}
        for rows in LEGEND_ROW_OPTIONS:
            action = self._legend_rows_menu.addAction(f"{rows} rows")
            action.setCheckable(True)
            action.setChecked(rows == self._legend_rows)
            action.triggered.connect(
                lambda checked=False, value=rows: self.set_legend_rows(value)
            )
            rows_group.addAction(action)
            self._legend_row_actions[rows] = action
        self._legend_rows_menu.setEnabled(False)

        button = QToolButton(self._toolbar)
        button.setText("Legend")
        button.setToolTip("Place and arrange the legend")
        button.setMenu(self._legend_menu)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._toolbar.addWidget(button)
        self.addAction(self._legend_menu.menuAction())

    def _configure_overflow_button(self) -> None:
        """Keep the native hidden-actions control centred and off the edge."""
        button = self._toolbar.findChild(QToolButton, "qt_toolbar_ext_button")
        if button is None:
            return
        button.setArrowType(Qt.NoArrow)
        button.setIcon(QIcon())
        button.setText("»")
        button.setStyleSheet(
            "QToolButton { color: palette(text); border: 0; padding: 0;"
            " font-size: 18px; }"
        )
        button.setFixedWidth(28)
        button.setMinimumHeight(22)
        button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        button.setToolTip("More actions")

    # -- public API --------------------------------------------------------

    @property
    def scene(self) -> QGraphicsScene:
        return self._scene

    @property
    def diagram(self) -> Diagram | None:
        return self._diagram

    @property
    def layout_result(self) -> LayoutResult | None:
        return self._layout

    @property
    def auto_settle(self) -> bool:
        """Whether a manual rotation re-optimises everything else around it."""
        return self._auto_action.isChecked()

    @auto_settle.setter
    def auto_settle(self, value: bool) -> None:
        self._auto_action.setChecked(bool(value))

    @property
    def legend_position(self) -> str:
        return self._legend_position

    @property
    def legend_rows(self) -> int:
        return self._legend_rows

    @property
    def is_solving(self) -> bool:
        """Whether a background layout optimization is currently running."""
        return self._solving

    def set_legend_position(self, position: str) -> None:
        """Place the legend on one of the four sides; left is the default."""
        position = str(position).lower()
        if position not in LEGEND_POSITIONS:
            raise ValueError(f"legend position must be one of {LEGEND_POSITIONS}; got {position!r}")
        self._legend_position = position
        self._legend_position_actions[position].setChecked(True)
        # Side legends always use one column.  The saved row choice becomes
        # active again when the legend returns to the top or bottom.
        self._legend_rows_menu.setEnabled(position in {"top", "bottom"})
        self._rebuild()
        self._view.fit_if_auto()

    def set_legend_rows(self, rows: int) -> None:
        """Set the number of rows used by top and bottom legends."""
        rows = int(rows)
        if rows not in LEGEND_ROW_OPTIONS:
            raise ValueError(f"legend rows must be one of {LEGEND_ROW_OPTIONS}; got {rows}")
        self._legend_rows = rows
        self._legend_row_actions[rows].setChecked(True)
        if self._legend_position in {"top", "bottom"}:
            self._rebuild()
            self._view.fit_if_auto()

    def selected_key(self) -> str | None:
        """Residue key of the selected glyph, or ``None`` when none is."""
        for item in self._scene.selectedItems():
            key = getattr(item, "residue_key", None)
            if key is not None:
                return key
        return None

    def set_diagram(self, diagram: Diagram, **layout_options) -> None:
        """Synchronously solve ``diagram`` and draw it.

        This deterministic path remains useful for command-line image export.
        Interactive callers should use :meth:`set_diagram_async`, which keeps
        painting, panning and zooming responsive while layout runs.
        """
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            layout = solve_layout(diagram, **layout_options)
        finally:
            QApplication.restoreOverrideCursor()
        self.set_layout(diagram, layout)

    def set_diagram_async(self, diagram: Diagram, **layout_options) -> None:
        """Schedule a bounded layout solve and update the scene when ready.

        The bounded solver runs in a Qt thread so painting and controls remain
        responsive.  Unlike the former process pool, this has no interpreter
        startup, serialization or child process that can become stranded.
        If several requests arrive first, only the newest pending one is kept.
        """
        self._solve_serial += 1
        options = dict(layout_options)
        operation = str(options.pop("_status_text", "Optimizing layout"))
        request = (self._solve_serial, diagram, options, operation)
        if self._solving:
            self._pending_solve = request
            return
        self._start_solve(request)

    def set_layout(self, diagram: Diagram, layout: LayoutResult) -> None:
        """Draw a layout that was solved elsewhere.

        This is also the zero-analysis path used by cached JSON: hand in the
        saved positions and drawing itself takes only milliseconds.
        """
        self._diagram = diagram
        self._layout = layout
        self._positions = dict(layout.positions)
        self._ligand_coords = list(layout.ligand_coords)
        self._rebuild()
        self._view.fit()

    @classmethod
    def from_json(
        cls, path: str | Path, parent: QWidget | None = None
    ) -> "InteractionDiagramWidget":
        """Create a viewer from cached analysis/layout JSON.

        When an older or analysis-only document has no layout, the layout is
        solved once.  Documents emitted by :meth:`export_json` draw directly.
        """
        from .json_io import load_json

        diagram, layout, view = load_json(path)
        widget = cls(parent=parent)
        widget.set_legend_position(view.get("legend_position", "left"))
        widget.set_legend_rows(int(view.get("legend_rows", 3)))
        widget._backbone_action.setChecked(bool(view.get("backbone", True)))
        widget._trail_action.setChecked(view.get("exposure_style", "halo") == "trail")
        if layout is None:
            widget.set_diagram(diagram)
        else:
            widget.set_layout(diagram, layout)
        return widget

    def reset_layout(self) -> None:
        """Throw away manual edits and re-solve from scratch."""
        if self._diagram is not None and self._layout is not None:
            self._residue_variant = 0
            self.set_diagram_async(
                self._diagram,
                projection=self._layout.projection,
                variant=0,
                _status_text="Recalculating",
            )

    def reshuffle(self) -> None:
        """Re-solve the whole canvas around the next candidate depiction.

        Different projection, different ligand shape to arrange around.  This is
        the escape hatch when the selected candidate reads badly to a
        human eye, so it walks the candidates rather than re-picking the one it
        already rejected.

        It used to pin a random orientation as well, which is what made it look
        like it threw the layout rules away: pinning the angle skips the
        orientation scan, so the structural waters and coordination spheres got
        exactly one attempt at settling instead of the scan's many trials.  The
        depiction alone gives as many distinct layouts as there are candidates,
        each of them properly solved.
        """
        if self._diagram is None or self._layout is None:
            return
        views = 1 + len(self._diagram.coords_alt)
        self._reshuffle_variant += 1
        self.set_diagram_async(
            self._diagram,
            projection=(self._layout.projection + 1) % views,
            variant=self._reshuffle_variant,
            _status_text="Reshuffling",
        )

    def reshuffle_residues(self) -> None:
        """Try another residue arrangement without changing the ligand."""
        if self._diagram is None or self._layout is None:
            return
        self._residue_variant += 1
        self.set_diagram_async(
            self._diagram,
            projection=self._layout.projection,
            orientation=(self._layout.rotation, self._layout.mirror),
            variant=self._residue_variant,
            _status_text="Reshuffling residues",
        )

    def rotate_selection(self, degrees: float) -> None:
        """Turn the selected element, or the ligand when nothing is selected."""
        if self._diagram is None:
            return
        key = self.selected_key()
        if key is None:
            self.rotate_ligand(degrees)
        elif key in self._diagram.metal_coordination:
            self.rotate_metal(key, degrees)
        else:
            self.orbit_residue(key, degrees)

    def rotate_ligand(self, degrees: float) -> None:
        """Turn the ligand by ``degrees``.

        With "Settle neighbors" on this re-solves the whole layout for the new
        orientation, which is the good answer; with it off only the depiction
        turns and every glyph stays where the user put it.
        """
        if self._diagram is None or self._layout is None:
            return
        if self.auto_settle:
            self.set_diagram_async(
                self._diagram,
                orientation=(self._layout.rotation + math.radians(degrees),
                             self._layout.mirror),
                projection=self._layout.projection,
            )
            return
        cx, cy = _centroid(self._ligand_coords)
        self._ligand_coords = [
            _turn(p, (cx, cy), math.radians(degrees)) for p in self._ligand_coords
        ]
        self._rebuild()

    def rotate_metal(self, key: str, degrees: float) -> None:
        """Swing a metal's whole coordination sphere around the metal.

        The polygon glyph follows on its own: :func:`render.build_scene` aims
        its corners at wherever the partners ended up.
        """
        if self._diagram is None or key not in self._positions:
            return
        center = self._positions[key]
        moved = {key} | {
            leg.partner_key for leg in self._diagram.metal_legs if leg.metal_key == key
        }
        for partner in moved - {key}:
            if partner in self._positions:
                self._positions[partner] = _turn(
                    self._positions[partner], center, math.radians(degrees)
                )
        self._settle(moved)

    def orbit_residue(self, key: str, degrees: float) -> None:
        """Swing one glyph around the ligand, keeping its distance."""
        if self._diagram is None or key not in self._positions:
            return
        self._positions[key] = _turn(
            self._positions[key], _centroid(self._ligand_coords), math.radians(degrees)
        )
        self._settle({key})

    def mirror_ligand(self) -> None:
        if self._diagram is None or self._layout is None:
            return
        self.set_diagram_async(
            self._diagram,
            orientation=(self._layout.rotation, not self._layout.mirror),
            projection=self._layout.projection,
        )

    def fit(self) -> None:
        self._view.fit()

    def export_png(self, path: str | Path, scale: float = 2.0, background: str | None = "#ffffff") -> Path:
        return export_mod.export_png(self._scene, path, scale=scale, background=background)

    def export_svg(self, path: str | Path) -> Path:
        return export_mod.export_svg(self._scene, path)

    def export_json(self, path: str | Path) -> Path:
        """Save analysis plus the current, possibly hand-edited layout."""
        if self._diagram is None or self._layout is None:
            raise ValueError("cannot export JSON before a diagram is loaded")
        from .json_io import save_json

        return save_json(
            path,
            self._diagram,
            self._layout,
            view={
                "legend_position": self._legend_position,
                "legend_rows": self._legend_rows,
                "backbone": self._backbone_action.isChecked(),
                "exposure_style": "trail" if self._trail_action.isChecked() else "halo",
            },
        )

    # -- internals ---------------------------------------------------------

    def _start_solve(self, request: tuple[int, Diagram, dict, str]) -> None:
        request_id, diagram, options, operation = request
        self._active_solve = request_id
        self._solving = True
        self._pending_solve = None
        self._solve_operation = operation
        self._solve_started = time.perf_counter()
        self._update_solve_status()
        self._solve_status_timer.start()
        self._set_layout_controls_enabled(False)
        self.layoutStarted.emit()
        self._solve_diagram = diagram
        task = _LayoutTask(request_id, diagram, options)
        task.signals.finished.connect(self._on_solve_finished)
        task.signals.failed.connect(self._on_solve_failed)
        self._solve_task = task
        self._layout_pool.start(task)

    @Slot(int, object, object)
    def _on_solve_finished(
        self, request_id: int, _diagram: object, layout: LayoutResult
    ) -> None:
        if request_id != self._active_solve:
            return
        self._solve_task = None
        if self._pending_solve is not None:
            pending = self._pending_solve
            self._start_solve(pending)
            return
        elapsed = time.perf_counter() - self._solve_started
        self._stop_solve_timers()
        diagram = self._solve_diagram
        if diagram is None:
            self._on_solve_failed(request_id, "layout result lost its diagram")
            return
        try:
            self.set_layout(diagram, layout)
        except Exception as exc:
            self._solving = False
            self._set_layout_controls_enabled(True)
            message = f"{type(exc).__name__}: {exc}"
            self._status_label.setText(f"  Layout failed: {message}  ")
            self.layoutFailed.emit(message)
            return
        self._solving = False
        self._set_layout_controls_enabled(True)
        self._status_label.setText(
            f"  {self._solve_operation} complete · {elapsed:.1f} s  "
        )
        self.layoutFinished.emit(layout)

    @Slot(int, str)
    def _on_solve_failed(self, request_id: int, message: str) -> None:
        if request_id != self._active_solve:
            return
        self._solve_task = None
        if self._pending_solve is not None:
            pending = self._pending_solve
            self._start_solve(pending)
            return
        self._stop_solve_timers()
        self._solving = False
        self._set_layout_controls_enabled(True)
        self._status_label.setText(f"  Layout failed: {message}  ")
        self.layoutFailed.emit(message)

    def _update_solve_status(self) -> None:
        if not self._solving:
            return
        elapsed = max(0.0, time.perf_counter() - self._solve_started)
        self._status_label.setText(
            f"  {self._solve_operation}… {elapsed:.1f} s  "
        )

    def _stop_solve_timers(self) -> None:
        self._solve_status_timer.stop()

    def _set_layout_controls_enabled(self, enabled: bool) -> None:
        for action in self._layout_actions:
            action.setEnabled(enabled)
        if enabled:
            self._view.viewport().unsetCursor()
            self._on_selection_changed()
        else:
            self._view.viewport().setCursor(Qt.CursorShape.BusyCursor)

    def _settle(self, moved: set[str]) -> None:
        """Redraw after a manual move, re-optimising the rest if asked to.

        The moved glyphs are pinned rather than seeded: no term in the energy
        model has an opinion about which way a coordination polygon faces, so
        an unpinned re-solve would put the partners straight back.
        """
        if self._diagram is None or self._layout is None:
            return
        if self.auto_settle:
            # Show the user's rotation immediately; bounded settling then
            # replaces it without blocking repainting or navigation.
            self._rebuild()
            self.set_diagram_async(
                self._diagram,
                seed_positions=self._positions,
                pinned=moved,
                orientation=(self._layout.rotation, self._layout.mirror),
                projection=self._layout.projection,
            )
            return
        self._rebuild()

    def _rebuild(self) -> None:
        if self._diagram is None or self._layout is None:
            return
        # Keep the serializable layout synchronized with manual drags and
        # rotations; otherwise JSON export would silently restore the old pose.
        self._layout.positions = dict(self._positions)
        self._layout.ligand_coords = list(self._ligand_coords)
        selected = self.selected_key()
        self._build = render.build_scene(
            self._diagram,
            self._positions,
            self._ligand_coords,
            scene=self._scene,
            legend_position=self._legend_position,
            legend_rows=self._legend_rows,
        )
        build = self._build
        self._apply_layers()
        # A droplet costs ~0.6 ms to paint (four stacked shadow strokes over a
        # 58-vertex polygon), and FullViewportUpdate repaints every one of them
        # on every mouse-move: 21 ms a frame on a 23-residue scene, 4 ms with
        # the caches.  Dragging changes an item's position, not the view
        # transform, so the caches survive the whole gesture.  Set here rather
        # than in the items: ``export_svg`` draws the same scene into a vector
        # backend, where a cached item would come out as an embedded bitmap
        # (``export`` switches them off for the duration).  Items that brought
        # their own cache mode -- the ligand, which Qt already caches -- keep
        # it, so the exports stay byte-for-byte what they were.
        for item in self._scene.items():
            if item.cacheMode() == QGraphicsItem.CacheMode.NoCache:
                item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        # scene.clear() destroys the old items, and with them the selection.
        if selected in build.droplets:
            build.droplets[selected].setSelected(True)

    def _apply_layers(self) -> None:
        """Push the view toggles onto the current scene.  No re-solve, no rebuild."""
        build = getattr(self, "_build", None)
        if build is None:
            return
        if build.backbone is not None:
            build.backbone.setVisible(self._backbone_action.isChecked())
        if build.halos is not None:
            build.halos.set_mode("trail" if self._trail_action.isChecked() else "halo")

    def _on_selection_changed(self) -> None:
        key = self.selected_key()
        if key is None:
            what = "ligand"
        else:
            name, code = key.split(":")[-1], ":".join(key.split(":")[:2])
            what = f"{name} {code}"
            if self._diagram is not None and key in self._diagram.metal_coordination:
                what += " (coordination sphere)"
        self._selection_label.setText(f"  Rotate: {what}  ")

    def _on_residue_moved(self) -> None:
        """Adopt any dragged droplet positions and redraw around them.

        # ponytail: full scene rebuild on release.  These scenes are a few
        # hundred items, so it is imperceptible; switch to per-item updates
        # only if dragging ever feels sluggish.
        """
        if self._diagram is None:
            return
        moved = False
        for item in self._scene.items():
            key = getattr(item, "residue_key", None)
            if key is None:
                continue
            pos: QPointF = item.scenePos()
            new = (pos.x(), pos.y())
            if self._positions.get(key) != new:
                self._positions[key] = new
                moved = True
        if moved:
            self._rebuild()
