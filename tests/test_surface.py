"""The pocket surface leaves the solvent-exposed part of the ligand open."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF

from ms_contactmap.export import ensure_app
from ms_contactmap.interactions import build_diagram
from ms_contactmap.layout import solve_layout
from ms_contactmap.render import _exposed_bearings, _ribbon_runs, centroid

DATA = Path(__file__).resolve().parents[1] / "data"


def test_surface_breaks_at_exposed_atoms():
    ensure_app()
    diagram = build_diagram(DATA / "4ps5.pdb", "2TA", chain="A", resnum=201)
    result = solve_layout(diagram)
    positions = {k: QPointF(*xy) for k, xy in result.positions.items()}
    center = QPointF(*centroid(result.ligand_coords))
    assert _exposed_bearings(diagram, result.ligand_coords, center)
    drawn = sum(len(r.points) for r in _ribbon_runs(diagram, positions, center, result.ligand_coords))
    diagram.exposure = {}
    sealed = sum(len(r.points) for r in _ribbon_runs(diagram, positions, center, result.ligand_coords))
    assert drawn < sealed


def test_surface_keeps_off_the_ligand():
    import math
    from ms_contactmap.render import RIBBON_SURFACE_CLEARANCE
    ensure_app()
    diagram = build_diagram(DATA / "4ps5.pdb", "2TA", chain="A", resnum=201)
    result = solve_layout(diagram)
    positions = {k: QPointF(*xy) for k, xy in result.positions.items()}
    center = QPointF(*centroid(result.ligand_coords))
    points = [p for r in _ribbon_runs(diagram, positions, center, result.ligand_coords)
              for p in r.points]
    assert min(math.dist((p.x(), p.y()), a) for p in points
               for a in result.ligand_coords) >= RIBBON_SURFACE_CLEARANCE - 1


def test_cation_partner_is_drawn_charged():
    from rdkit import Chem
    from ms_contactmap.render import _charge_cations
    diagram = build_diagram(DATA / "4uwh.pdb", "JXM", chain="A", resnum=1876)
    bridge = next(i for i in diagram.interactions if i.kind == "salt_bridge" and i.ligand_is_donor)
    mol = Chem.Mol(diagram.mol)
    _charge_cations(mol, diagram)
    assert any(mol.GetAtomWithIdx(a).GetFormalCharge() == 1 for a in bridge.ligand_atoms)
