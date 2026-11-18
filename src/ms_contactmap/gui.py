"""The standalone window: diagram, 3D view and interaction table, linked by selection.

``panels`` picks any of ``diagram``, ``3d`` and ``table``.  The first one is the
centre of the window and the others dock beside it.  The 3D panel embeds PyMOL
through ms_components (the ``3d`` extra); without it the panel is left out with a
note on stderr.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import (QAbstractItemView, QDockWidget, QInputDialog, QMainWindow, QTableWidget,
                               QTableWidgetItem, QWidget)

from .model import INTERACTION_STYLES, Diagram

PANELS = ("diagram", "3d", "table")
RECEPTOR, LIGAND = "receptor", "ligand"


def pick_ligand(sites, parent: QWidget | None = None):
    """Ask which ligand to draw; ``None`` when the user cancels."""
    names = [str(site) for site in sites]
    name, ok = QInputDialog.getItem(parent, "Choose the ligand", "Several ligands were found:",
                                    names, 0, False)
    return sites[names.index(name)] if ok else None


def _ligand_atom(diagram: Diagram, index: int) -> str:
    mol = getattr(diagram, "mol", None)
    if mol is None or index >= mol.GetNumAtoms():
        return str(index + 1)
    return f"{mol.GetAtomWithIdx(index).GetSymbol()}{index + 1}"


def interaction_table(diagram: Diagram, parent: QWidget | None = None) -> QTableWidget:
    """One row per contact; sortable, selection by row."""
    headers = ("Interaction", "Residue", "Protein atom", "Ligand atoms", "Distance (Å)", "Via water")
    table = QTableWidget(len(diagram.interactions), len(headers), parent)
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.verticalHeader().hide()
    for row, contact in enumerate(diagram.interactions):
        style = INTERACTION_STYLES.get(contact.kind)
        cells = (
            style.label if style else contact.kind,
            contact.residue_key,
            contact.protein_atom or "",
            " ".join(_ligand_atom(diagram, i) for i in contact.ligand_atoms),
            contact.distance,
            contact.via_water or "",
        )
        for column, value in enumerate(cells):
            item = QTableWidgetItem()
            item.setData(Qt.ItemDataRole.DisplayRole, round(value, 2) if isinstance(value, float) else value)
            item.setData(Qt.ItemDataRole.UserRole, row)  # survives sorting
            table.setItem(row, column, item)
    table.resizeColumnsToContents()
    table.setSortingEnabled(True)
    return table


class ContactMapWindow(QMainWindow):
    def __init__(self, diagram: Diagram, panels: list[str], *, widget=None, pdb: Path | None = None,
                 ligand_selection: str | None = None, ligand_file: Path | None = None) -> None:
        super().__init__()
        self._diagram = diagram
        self._widget = widget
        self._cmd = None
        self._table: QTableWidget | None = None
        views: dict[str, QWidget] = {}
        if "diagram" in panels and widget is not None:
            views["diagram"] = widget
        if "3d" in panels:
            pymol = self._pymol_panel(pdb, ligand_selection, ligand_file)
            if pymol is not None:
                views["3d"] = pymol
        if "table" in panels:
            self._table = interaction_table(diagram, self)
            self._table.itemSelectionChanged.connect(self._on_table_selection)
            views["table"] = self._table

        order = [p for p in panels if p in views]
        for index, name in enumerate(order):
            view = views[name]
            if index == 0:
                if isinstance(view, QDockWidget):  # PyMOL comes as a dock: its body is the centre
                    view.setTitleBarWidget(QWidget())
                    view.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
                    self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, view)
                else:
                    self.setCentralWidget(view)
                continue
            dock = view if isinstance(view, QDockWidget) else QDockWidget(
                {"diagram": "Diagram", "table": "Interactions"}[name], self)
            if dock is not view:
                dock.setWidget(view)
            area = (Qt.DockWidgetArea.BottomDockWidgetArea if name == "table" and "diagram" in order
                    else Qt.DockWidgetArea.RightDockWidgetArea)
            self.addDockWidget(area, dock)
        if widget is not None and self._cmd is not None:
            widget.residueSelected.connect(self._on_residue_selected)

    def _pymol_panel(self, pdb, ligand_selection, ligand_file):
        if pdb is None:
            print("ms_contactmap: the 3D panel needs the PDB input, not a JSON", file=sys.stderr)
            return None
        try:
            from ms_components.ms_pymol.widget import PymolDockWidget
        except ImportError:
            print("ms_contactmap: the 3D panel needs ms_components "
                  "(pip install 'ms_contactmap[3d]')", file=sys.stderr)
            return None
        dock = PymolDockWidget("3D", None, self)
        cmd = dock.cmd
        if cmd is None:
            return dock
        cmd.load(str(pdb), RECEPTOR)
        if ligand_selection:
            if ligand_file is not None:
                cmd.remove(f"{RECEPTOR} and ({ligand_selection})")
                cmd.load(str(ligand_file), LIGAND)
            else:
                cmd.extract(LIGAND, f"{RECEPTOR} and ({ligand_selection})")
        cmd.hide("everything")
        cmd.show("cartoon", RECEPTOR)
        cmd.show("sticks", f"{LIGAND} and not hydro")
        self._cmd = cmd
        self._show_all()
        return dock

    def _show_all(self) -> None:
        from .pymol3d import show_all_interactions

        show_all_interactions(self._cmd, receptor_obj=RECEPTOR, ligand_obj=LIGAND, diagram=self._diagram)

    def _on_residue_selected(self, key: str) -> None:
        from .pymol3d import show_residue_interactions

        if not key:
            self._show_all()
            return
        show_residue_interactions(self._cmd, receptor_obj=RECEPTOR, ligand_obj=LIGAND, key=key,
                                  diagram=self._diagram)
        if self._table is not None:
            self._table.blockSignals(True)
            self._table.clearSelection()
            flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
            for row in range(self._table.rowCount()):
                if self._table.item(row, 1).text() == key:
                    self._table.selectionModel().select(self._table.model().index(row, 0), flags)
            self._table.blockSignals(False)

    def _on_table_selection(self) -> None:
        if self._cmd is None:
            return
        from .pymol3d import show_interaction

        rows = self._table.selectionModel().selectedRows()
        if not rows:
            self._show_all()
            return
        contact = self._diagram.interactions[self._table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)]
        show_interaction(self._cmd, receptor_obj=RECEPTOR, ligand_obj=LIGAND, contact=contact)
