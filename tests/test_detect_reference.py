"""What the native detector finds on the four reference complexes.

A change detector, not ground truth: the numbers below are what
:func:`~ms_contactmap.detect.detect_interactions` produces today with the strong
contact profile in :mod:`ms_contactmap.detect.geometry`.  Loosening a threshold
or fixing a perception bug is expected to move them -- what must never happen
silently is a whole interaction group disappearing, which is invisible
downstream: the diagram just comes out missing lines.

(This replaces the old PLIP-adapter test; PLIP is gone, and its histograms were
its own, measured with permissive candidate criteria we deliberately do not
use.)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ms_contactmap.chem import load_ligand, read_pdb_atoms
from ms_contactmap.detect import detect_interactions

ROOT = Path(__file__).resolve().parent.parent
LIGANDS = json.loads((ROOT / "data" / "ligands.json").read_text())

#: pdb -> ligand resname, kind histogram, metal legs, coordination numbers.
EXPECTED = {
    ("2gfk", "VII"): (
        # The four HIS that bind the zincs are not cations: no salt bridge
        # and no pi-cation for them.
        {"hbond": 1, "hydrophobic": 4, "metal_coordination": 2,
         "pi_stacking": 1, "water_bridge": 1},
        6, {"A:401:ZN": 4, "A:402:ZN": 4},
    ),
    ("4ps5", "2TA"): ({"hbond": 2, "hydrophobic": 10}, 0, {}),
    ("4uwh", "JXM"): ({"hbond": 1, "hydrophobic": 5, "water_bridge": 3}, 0, {}),
    ("6wak", "ANP"): (
        {"metal_coordination": 2, "salt_bridge": 3, "water_bridge": 1},
        2, {"D:1102:MG": 4},
    ),
}


def test_reference_complexes():
    for (pdb, resname), (kinds, n_legs, coord) in EXPECTED.items():
        path = ROOT / "data" / f"{pdb}.pdb"
        geom = load_ligand(path, resname, LIGANDS[resname]["smiles"])
        serial_to_atom = {a.serial: a for a in read_pdb_atoms(path)}
        interactions, legs, coordination, skipped = detect_interactions(
            path, geom, serial_to_atom
        )

        assert dict(Counter(i.kind for i in interactions)) == kinds, pdb
        assert len(legs) == n_legs, pdb
        assert coordination == coord, pdb
        assert skipped == 0, f"{pdb}: {skipped} contacts had unmappable ligand atoms"

        n_atoms = geom.mol.GetNumAtoms()
        for i in interactions:
            assert i.ligand_atoms, f"{pdb} {i.kind}: no ligand atom"
            assert all(0 <= a < n_atoms for a in i.ligand_atoms), f"{pdb} {i.kind}"
            assert i.distance > 0.0, f"{pdb} {i.kind}: distance {i.distance}"
            if i.kind == "water_bridge":
                assert i.via_water, f"{pdb}: water bridge with no water"
        # A metal leg must point at something the metal actually coordinates.
        for leg in legs:
            assert leg.metal_key in coordination, f"{pdb}: stray leg {leg}"


def test_group_contacts_name_the_atoms_their_distance_is_measured_from():
    """A ring or charged-group contact is measured centroid to centroid, so it
    carries the protein group's atom names: a 3D view that ends its line on
    ``protein_atom`` (the group's first atom) shows a different number."""
    import numpy as np

    seen = set()
    for pdb, resname in (("2gfk", "VII"), ("6wak", "ANP")):
        path = ROOT / "data" / f"{pdb}.pdb"
        geom = load_ligand(path, resname, LIGANDS[resname]["smiles"])
        atoms = read_pdb_atoms(path)
        serial_to_atom = {a.serial: a for a in atoms}
        interactions, *_ = detect_interactions(path, geom, serial_to_atom)
        for i in interactions:
            if i.kind not in ("pi_stacking", "salt_bridge"):
                assert not i.is_protein_group, f"{pdb} {i.kind}"
                continue
            seen.add((i.kind, i.is_protein_group))
            # A one-atom group (LYS NZ) is measured from that atom.
            names = i.protein_atoms or (i.protein_atom,)
            chain, number, name = i.residue_key.split(":")
            residue = [a for a in atoms if a.chain.strip() == chain.strip()
                       and str(a.resnum) + a.icode.strip() == number
                       and a.name in names]
            assert len(residue) == len(names), f"{pdb} {i.residue_key}"
            protein = np.mean([(a.x, a.y, a.z) for a in residue], axis=0)
            ligand = np.mean([geom.coords_3d[a] for a in i.ligand_atoms], axis=0)
            assert abs(float(np.linalg.norm(protein - ligand)) - i.distance) < 1e-3, \
                f"{pdb} {i.kind} {i.residue_key}"
    assert {("pi_stacking", True), ("salt_bridge", True)} <= seen
