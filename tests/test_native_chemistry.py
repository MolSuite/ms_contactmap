"""Regression checks for chemistry that is easy to misread in the 2D view."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ms_contactmap import build_diagram
from ms_contactmap.chem import load_ligand, phosphate_axis_variants


ROOT = Path(__file__).resolve().parent.parent
LIGANDS = json.loads((ROOT / "data" / "ligands.json").read_text())


def test_phosphate_terminal_oxygens_are_not_protonated():
    geom = load_ligand(
        ROOT / "data" / "6wak.pdb", "ANP", LIGANDS["ANP"]["smiles"],
        chain="A", resnum=1101,
    )
    terminal = []
    for phosphorus in (a for a in geom.mol.GetAtoms() if a.GetAtomicNum() == 15):
        for oxygen in phosphorus.GetNeighbors():
            bond = geom.mol.GetBondBetweenAtoms(
                phosphorus.GetIdx(), oxygen.GetIdx()
            )
            if oxygen.GetAtomicNum() == 8 and oxygen.GetDegree() == 1 \
                    and str(bond.GetBondType()) == "SINGLE":
                terminal.append(oxygen)
    assert terminal
    assert all(atom.GetFormalCharge() == -1 for atom in terminal)
    assert all(atom.GetTotalNumHs() == 0 for atom in terminal)


def test_phosphate_depiction_rotates_whole_fragments_without_deformation():
    geom = load_ligand(
        ROOT / "data" / "6wak.pdb", "ANP", LIGANDS["ANP"]["smiles"],
        chain="A", resnum=1101,
    )
    original = [geom.coords_2d, *geom.alt_coords_2d]
    variants = phosphate_axis_variants(geom.mol, original, [[1, 5, 10]])
    assert len(variants) == len(original)
    assert any(
        not np.allclose(np.asarray(before), np.asarray(after))
        for before, after in zip(original, variants)
    )

    # A torsional flip may change distances between phosphate groups, but each
    # phosphorus and all its immediate neighbours must remain one rigid group.
    for before, after in zip(original, variants):
        before, after = np.asarray(before), np.asarray(after)
        for phosphorus in (a for a in geom.mol.GetAtoms() if a.GetAtomicNum() == 15):
            ids = [phosphorus.GetIdx(), *[n.GetIdx() for n in phosphorus.GetNeighbors()]]
            old = np.linalg.norm(
                before[ids, None, :] - before[np.asarray(ids)[None, :], :], axis=2
            )
            new = np.linalg.norm(
                after[ids, None, :] - after[np.asarray(ids)[None, :], :], axis=2
            )
            assert np.allclose(old, new, atol=1e-10)


def test_zinc_uses_only_the_tetrahedral_first_shell():
    diagram = build_diagram(
        ROOT / "data" / "2gfk.pdb", "VII", LIGANDS["VII"]["smiles"],
        chain="A", resnum=1410, compute_exposure=False,
    )
    assert diagram.metal_coordination == {"A:401:ZN": 4, "A:402:ZN": 4}
    assert all(leg.distance <= 2.25 for leg in diagram.metal_legs)
    assert all(not leg.partner_key.endswith(":HOH") for leg in diagram.metal_legs)
    assert all(
        interaction.distance <= 2.25
        for interaction in diagram.interactions
        if interaction.kind == "metal_coordination"
    )


def test_4uwh_strict_hbond_geometry_removes_weak_contacts():
    diagram = build_diagram(
        ROOT / "data" / "4uwh.pdb", "JXM", LIGANDS["JXM"]["smiles"],
        chain="A", resnum=1876, compute_exposure=False,
    )
    direct = {
        row.residue_key: row for row in diagram.interactions
        if row.kind == "hbond"
    }
    assert "A:636:LYS" not in direct
    assert "A:761:ASP" not in direct
    # This contact is not equivalent to the two weak ones: 2.77 A and 168.4
    # degrees satisfy the strict 3.0 A / 150 degree profile.
    assert set(direct) == {"A:685:ILE"}
    assert direct["A:685:ILE"].distance < 3.0
    assert direct["A:685:ILE"].angle > 150.0

    water_2260 = [
        row for row in diagram.interactions
        if row.kind == "water_bridge" and row.via_water == "A:2260:HOH"
    ]
    assert {row.residue_key for row in water_2260} == {
        "A:644:ASP", "A:670:TYR",
    }
    assert all(row.distance <= 3.0 for row in water_2260)
    assert all(row.protein_distance <= 3.0 for row in water_2260)
    directions = {row.residue_key: row.protein_is_donor for row in water_2260}
    assert directions == {"A:644:ASP": False, "A:670:TYR": True}
