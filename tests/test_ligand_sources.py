"""Ligand chemistry from a pose file, and the proximity-bond guard of the SMILES path."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from ms_contactmap import build_diagram
from ms_contactmap.chem import load_ligand


ROOT = Path(__file__).resolve().parent.parent
LIGANDS = json.loads((ROOT / "data" / "ligands.json").read_text())
JXM = dict(pdb_path=ROOT / "data" / "4uwh.pdb", resname="JXM", chain="A", resnum=1876)

# CCD ADX (adenosine 5'-phosphosulfate) from rdkit#9581: its S and P sit 2.51 A
# apart through the bridging oxygen, close enough for proximity bonding.
ADX_SMILES = "Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OS(=O)(=O)O)C(O)C1O"
ADX_PDB = """\
HETATM    1  SB  ADX A   1       1.120  -0.280  -5.646  1.00 20.00           S
HETATM    2  O1B ADX A   1       1.559  -1.323  -6.506  1.00 20.00           O
HETATM    3  O2B ADX A   1      -0.013   0.546  -5.868  1.00 20.00           O
HETATM    4  O3B ADX A   1       2.303   0.667  -5.509  1.00 20.00           O
HETATM    5  PA  ADX A   1      -0.128   0.012  -3.483  1.00 20.00           P
HETATM    6  O1A ADX A   1       0.457   1.363  -3.330  1.00 20.00           O
HETATM    7  O2A ADX A   1      -1.512   0.117  -4.299  1.00 20.00           O
HETATM    8  O3A ADX A   1       0.899  -0.930  -4.288  1.00 20.00           O
HETATM    9  O5' ADX A   1      -0.409  -0.618  -2.029  1.00 20.00           O
HETATM   10  C5' ADX A   1      -1.323   0.258  -1.367  1.00 20.00           C
HETATM   11  C4' ADX A   1      -1.634  -0.286   0.027  1.00 20.00           C
HETATM   12  O4' ADX A   1      -0.429  -0.350   0.821  1.00 20.00           O
HETATM   13  C3' ADX A   1      -2.560   0.685   0.793  1.00 20.00           C
HETATM   14  O3' ADX A   1      -3.918   0.250   0.712  1.00 20.00           O
HETATM   15  C2' ADX A   1      -2.053   0.615   2.252  1.00 20.00           C
HETATM   16  O2' ADX A   1      -3.085   0.138   3.118  1.00 20.00           O
HETATM   17  C1' ADX A   1      -0.880  -0.386   2.193  1.00 20.00           C
HETATM   18  N9  ADX A   1       0.195   0.033   3.094  1.00 20.00           N
HETATM   19  C8  ADX A   1       1.227   0.869   2.784  1.00 20.00           C
HETATM   20  N7  ADX A   1       1.998   1.026   3.820  1.00 20.00           N
HETATM   21  C5  ADX A   1       1.509   0.308   4.860  1.00 20.00           C
HETATM   22  C6  ADX A   1       1.913   0.092   6.188  1.00 20.00           C
HETATM   23  N6  ADX A   1       3.049   0.702   6.691  1.00 20.00           N
HETATM   24  N1  ADX A   1       1.175  -0.705   6.953  1.00 20.00           N
HETATM   25  C2  ADX A   1       0.089  -1.290   6.482  1.00 20.00           C
HETATM   26  N3  ADX A   1      -0.323  -1.122   5.243  1.00 20.00           N
HETATM   27  C4  ADX A   1       0.350  -0.341   4.405  1.00 20.00           C
HETATM   28 HOB3 ADX A   1       2.436   1.075  -6.376  1.00 20.00           H
HETATM   29 HOA2 ADX A   1      -1.860  -0.781  -4.377  1.00 20.00           H
HETATM   30  H5' ADX A   1      -2.245   0.325  -1.945  1.00 20.00           H
HETATM   31 H5'' ADX A   1      -0.877   1.249  -1.280  1.00 20.00           H
HETATM   32  H4' ADX A   1      -2.094  -1.271  -0.044  1.00 20.00           H
HETATM   33  H3' ADX A   1      -2.460   1.697   0.400  1.00 20.00           H
HETATM   34 HO3' ADX A   1      -4.447   0.891   1.205  1.00 20.00           H
HETATM   35  H2' ADX A   1      -1.702   1.593   2.582  1.00 20.00           H
HETATM   36 HO2' ADX A   1      -3.811   0.774   3.065  1.00 20.00           H
HETATM   37  H1' ADX A   1      -1.222  -1.388   2.454  1.00 20.00           H
HETATM   38  H8  ADX A   1       1.381   1.332   1.821  1.00 20.00           H
HETATM   39 HN61 ADX A   1       3.315   0.548   7.611  1.00 20.00           H
HETATM   40 HN62 ADX A   1       3.582   1.279   6.122  1.00 20.00           H
HETATM   41  H2  ADX A   1      -0.481  -1.931   7.137  1.00 20.00           H
END
"""


def _pose_sdf(tmp_path: Path, *, shift: float = 0.0) -> Path:
    """The JXM pose as an SDF, with the bond orders the SMILES path settled on."""
    geom = load_ligand(smiles=LIGANDS["JXM"]["smiles"], **JXM)
    mol = Chem.Mol(geom.mol)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, (x, y, z) in enumerate(geom.coords_3d):
        conf.SetAtomPosition(idx, Point3D(x + shift, y, z))
    mol.AddConformer(conf)
    # Shuffle the atom order: matching must go by position, not by index.
    mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    path = tmp_path / "pose.sdf"
    with Chem.SDWriter(str(path)) as writer:
        writer.write(Chem.AddHs(mol, addCoords=True))
    return path


def test_ligand_file_gives_the_smiles_result(tmp_path):
    sdf = _pose_sdf(tmp_path)
    by_smiles = load_ligand(smiles=LIGANDS["JXM"]["smiles"], **JXM)
    by_file = load_ligand(ligand=sdf, **JXM)
    assert Chem.MolToSmiles(by_file.mol) == Chem.MolToSmiles(by_smiles.mol)
    assert set(by_file.serial_to_idx) == set(by_smiles.serial_to_idx)
    for serial, idx in by_file.serial_to_idx.items():
        assert by_file.coords_3d[idx] == pytest.approx(
            by_smiles.coords_3d[by_smiles.serial_to_idx[serial]]
        )
    # The SDF carries hydrogens; they become the donor hydrogens.
    assert by_file.donor_hydrogens

    def contacts(diagram):
        return sorted((i.kind, i.residue_key) for i in diagram.interactions)

    common = dict(chain="A", resnum=1876, compute_exposure=False)
    assert contacts(build_diagram(JXM["pdb_path"], "JXM", ligand=sdf, **common)) == contacts(
        build_diagram(JXM["pdb_path"], "JXM", LIGANDS["JXM"]["smiles"], **common)
    )


def test_ligand_from_another_pose_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="do not match the HETATM block"):
        load_ligand(ligand=_pose_sdf(tmp_path, shift=1.0), **JXM)


def test_exactly_one_chemistry_source():
    with pytest.raises(ValueError, match="exactly one"):
        load_ligand(JXM["pdb_path"], "JXM")


def test_smiles_path_drops_proximity_bonds_the_template_lacks(tmp_path):
    pdb = tmp_path / "adx.pdb"
    pdb.write_text(ADX_PDB)
    geom = load_ligand(pdb, "ADX", ADX_SMILES)
    sulfur = next(a for a in geom.mol.GetAtoms() if a.GetSymbol() == "S")
    assert sorted(n.GetSymbol() for n in sulfur.GetNeighbors()) == ["O", "O", "O", "O"]
    assert geom.mol.GetNumBonds() == Chem.MolFromSmiles(ADX_SMILES).GetNumBonds()
