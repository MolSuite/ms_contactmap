"""Ligand chemistry from a pose file, and the proximity-bond guard of the SMILES path."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from ms_contactmap import build_diagram, build_pose_diagram
from ms_contactmap.chem import load_ligand
from ms_contactmap.interactions import _receptor_records


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

    common = dict(chain="A", resnum=1876, compute_exposure=False)
    assert _contacts(build_diagram(JXM["pdb_path"], "JXM", ligand=sdf, **common)) == _contacts(
        build_diagram(JXM["pdb_path"], "JXM", LIGANDS["JXM"]["smiles"], **common)
    )


def _contacts(diagram):
    return sorted((i.kind, i.residue_key) for i in diagram.interactions)


def test_pose_diagram_matches_the_complex(tmp_path):
    lines = (ROOT / "data" / "4uwh.pdb").read_text().splitlines()
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("\n".join(l for l in lines if l[17:20] != "JXM") + "\n")
    common = dict(compute_exposure=False)
    by_pose = build_pose_diagram(receptor, _pose_sdf(tmp_path), **common)
    by_complex = build_diagram(JXM["pdb_path"], "JXM", LIGANDS["JXM"]["smiles"], chain="A", resnum=1876, **common)
    assert _contacts(by_pose) and _contacts(by_pose) == _contacts(by_complex)


def test_pdbqt_receptor_gets_element_symbols(tmp_path):
    pdbqt = tmp_path / "receptor.pdbqt"
    pdbqt.write_text(
        "MODEL 1\n"
        "ATOM      7  N   ASP A 285      -3.256  -3.631  -1.685  1.00  0.00    -0.273 NA\n"
        "ATOM      8  HD1 ASP A 285      -3.100  -3.100  -1.100  1.00  0.00     0.100 HD\n"
        "ATOM      9 CL1  LIG A 285      -4.000  -4.000  -2.000  1.00  0.00     0.000 Cl\n"
        "ENDMDL\nMODEL 2\n"
        "ATOM      7  N   ASP A 285      -3.256  -3.631  -1.685  1.00  0.00    -0.273 NA\n"
    )
    # NA is an acceptor nitrogen, not sodium; Cl keeps both letters; only the first model.
    assert [line[76:78] for line in _receptor_records(pdbqt)] == [" N", " H", "Cl"]


def test_ligand_from_another_pose_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="do not match the HETATM block"):
        load_ligand(ligand=_pose_sdf(tmp_path, shift=1.0), **JXM)


def test_smiles_and_ligand_are_exclusive(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        load_ligand(ligand=_pose_sdf(tmp_path), smiles=LIGANDS["JXM"]["smiles"], **JXM)


def test_smiles_path_drops_proximity_bonds_the_template_lacks(tmp_path):
    pdb = tmp_path / "adx.pdb"
    pdb.write_text(ADX_PDB)
    geom = load_ligand(pdb, "ADX", ADX_SMILES)
    sulfur = next(a for a in geom.mol.GetAtoms() if a.GetSymbol() == "S")
    assert sorted(n.GetSymbol() for n in sulfur.GetNeighbors()) == ["O", "O", "O", "O"]
    assert geom.mol.GetNumBonds() == Chem.MolFromSmiles(ADX_SMILES).GetNumBonds()


def test_without_chemistry_the_legend_warns():
    from ms_contactmap.export import ensure_app
    from ms_contactmap.render import LEGEND_BOND_ORDER_WARNING, Legend

    ensure_app()
    common = dict(chain="A", resnum=1876, compute_exposure=False)
    guessed = build_diagram(JXM["pdb_path"], "JXM", **common)
    known = build_diagram(JXM["pdb_path"], "JXM", LIGANDS["JXM"]["smiles"], **common)
    assert guessed.metadata["bond_orders_known"] is False
    assert known.metadata["bond_orders_known"] is True
    assert guessed.mol.GetNumAtoms() == known.mol.GetNumAtoms()

    def labels(diagram):
        return [label for column in Legend(diagram)._columns for *_, label in column]

    assert LEGEND_BOND_ORDER_WARNING in labels(guessed)
    assert LEGEND_BOND_ORDER_WARNING not in labels(known)


def test_pdb_pose_takes_bond_orders_from_smiles(tmp_path):
    lines = (ROOT / "data" / "4uwh.pdb").read_text().splitlines()
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("\n".join(l for l in lines if l[17:20] != "JXM") + "\n")
    pose = tmp_path / "pose.pdb"
    pose.write_text("\n".join(l for l in lines if l[17:20] == "JXM" and l[21] == "A") + "\nEND\n")
    common = dict(compute_exposure=False)
    with_smiles = build_pose_diagram(receptor, pose, smiles=LIGANDS["JXM"]["smiles"], **common)
    from_sdf = build_pose_diagram(receptor, _pose_sdf(tmp_path), **common)
    assert Chem.MolToSmiles(with_smiles.mol) == Chem.MolToSmiles(from_sdf.mol)
    assert _contacts(with_smiles) == _contacts(from_sdf)
    assert build_pose_diagram(receptor, pose, **common).metadata["bond_orders_known"] is False
