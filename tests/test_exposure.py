"""The ligand-only SASA marks the same atoms, by the same amount, as Biopython's.

Biopython's Shrake-Rupley over the whole pocket was the original
implementation; it is kept here as the reference the fast one must reproduce.
It is fed the altlocs :func:`read_pdb_atoms` keeps: left alone, Biopython picks
the higher-occupancy one (4ps5 ASN B:116 is B at 0.51), not the copy the
detector sees.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

from ms_contactmap.chem import KEPT_ALTLOCS, load_ligand, read_pdb_atoms
from ms_contactmap.interactions import SASA_MIN, SASA_POINTS, exposed_atoms
from ms_contactmap.model import WATER_NAMES

ROOT = Path(__file__).resolve().parent.parent
LIGANDS = json.loads((ROOT / "data" / "ligands.json").read_text())
COMPLEXES = {"2gfk": "VII", "4ps5": "2TA", "4uwh": "JXM", "6wak": "ANP"}


def _biopython(pdb_path, geom) -> dict[int, float]:
    model = PDBParser(QUIET=True).get_structure("cx", str(pdb_path))[0]
    ligand = None
    for chain in model:
        for residue in list(chain):
            name = residue.get_resname().strip().upper()
            if name == geom.resname and chain.id.strip() == geom.chain and residue.id[1] == geom.resnum:
                ligand = residue
            elif name in WATER_NAMES:
                chain.detach_child(residue.id)
    shrake = ShrakeRupley(n_points=SASA_POINTS)
    shrake.compute(model, level="A")
    in_complex = {a.serial_number: a.sasa for a in ligand}
    shrake.compute(ligand, level="A")
    alone = {a.serial_number: a.sasa for a in ligand}
    return {
        idx: min(1.0, in_complex[serial] / alone[serial])
        for serial, idx in sorted(geom.serial_to_idx.items())
        if in_complex.get(serial, 0.0) > SASA_MIN and alone.get(serial, 0.0) > 0.0
    }


def test_matches_biopython_on_the_reference_complexes(tmp_path):
    for pdb, resname in COMPLEXES.items():
        path = ROOT / "data" / f"{pdb}.pdb"
        same_altlocs = tmp_path / path.name
        same_altlocs.write_text("".join(
            line for line in path.read_text().splitlines(keepends=True)
            if line[:6] not in ("ATOM  ", "HETATM") or line[16] in KEPT_ALTLOCS
        ))
        geom = load_ligand(path, resname, LIGANDS[resname]["smiles"])
        fast, reference = exposed_atoms(read_pdb_atoms(path), geom), _biopython(same_altlocs, geom)
        assert fast.keys() == reference.keys(), pdb
        assert np.allclose([fast[k] for k in fast], [reference[k] for k in fast], atol=1e-6), pdb
