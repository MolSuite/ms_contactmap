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
    from types import SimpleNamespace
    from rdkit import Chem
    from ms_contactmap.render import _charge_cations
    mol = Chem.MolFromSmiles("CCN(C)C")
    bridge = SimpleNamespace(kind="salt_bridge", ligand_is_donor=True, ligand_atoms=(2,))
    _charge_cations(mol, SimpleNamespace(interactions=[bridge]))
    assert mol.GetAtomWithIdx(2).GetFormalCharge() == 1


def test_no_cation_without_bond_orders():
    # 4uwh's JXM has amide/amidine N only; with every bond read as single
    # they once all looked like aliphatic amines.
    diagram = build_diagram(DATA / "4uwh.pdb", "JXM", chain="A", resnum=1876)
    assert not [i for i in diagram.interactions
                if i.kind in ("salt_bridge", "pi_cation") and i.ligand_is_donor]


def test_cif_supplies_bond_orders(tmp_path):
    from rdkit import Chem
    from ms_contactmap.chem import cif_smiles
    cif = tmp_path / "x.cif"
    cif.write_text("""data_X
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
ACM C1 C 0
ACM C2 C 0
ACM O1 O 0
ACM N1 N 0
ACM H1 H 0
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
ACM C1 C2 SING
ACM C2 O1 DOUB
ACM C2 N1 SING
ACM N1 H1 SING
""")
    assert cif_smiles(cif, "ACM") == Chem.CanonSmiles("CC(N)=O")
    assert cif_smiles(cif, "XXX") is None


def test_residues_sit_beside_solvent_gaps():
    """Exposure wins: no protein glyph is placed in front of an open gap."""
    import math
    ensure_app()
    diagram = build_diagram(DATA / "4ps5.pdb", "2TA", chain="A", resnum=201)
    result = solve_layout(diagram)
    center = QPointF(*centroid(result.ligand_coords))
    gaps = _exposed_bearings(diagram, result.ligand_coords, center)
    assert gaps
    for key, (x, y) in result.positions.items():
        if diagram.residue(key).ref.residue_class in ("water", "metal"):
            continue
        angle = math.atan2(y - center.y(), x - center.x())
        assert all(abs((angle - b + math.pi) % (2 * math.pi) - math.pi) > half
                   for b, half in gaps), key
