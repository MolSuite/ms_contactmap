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
        {"hbond": 1, "hydrophobic": 4, "metal_coordination": 2,
         "pi_cation": 1, "pi_stacking": 1, "salt_bridge": 5},
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
