from rdkit import Chem
from rdkit.Chem import AllChem

from ms_contactmap.json_io import load_json, save_json
from ms_contactmap.layout import LayoutResult
from ms_contactmap.model import Diagram, Interaction, Residue, ResidueRef


def _document():
    mol = Chem.MolFromSmiles("CCO")
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    coords = [
        (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y)
        for i in range(mol.GetNumAtoms())
    ]
    ref = ResidueRef("A", 12, "SER")
    diagram = Diagram(
        name="roundtrip",
        ligand_name="LIG",
        mol=mol,
        coords_2d=coords,
        residues=[Residue(ref, True, (1.0, 2.0, 3.0))],
        interactions=[Interaction(
            "hbond", ref.key, (2,), 2.8, False,
            protein_atom="OG", angle=167.5,
        )],
        exposure={2: 0.75},
        nearest_atom={ref.key: 2},
        metadata={"detector": "native"},
    )
    layout = LayoutResult(
        positions={ref.key: (120.0, 40.0)},
        ligand_coords=[(10.0 * x, 10.0 * y) for x, y in coords],
        rotation=0.25,
        mirror=True,
        energy=12.5,
        energy_terms={"anchor": 12.5},
        crossings=0,
        projection=0,
    )
    return diagram, layout


def test_json_roundtrip_is_self_contained(tmp_path):
    diagram, layout = _document()
    path = save_json(
        tmp_path / "diagram.json",
        diagram,
        layout,
        view={"legend_position": "left", "legend_rows": 3},
    )

    restored, restored_layout, view = load_json(path)
    assert restored.mol.GetNumAtoms() == 3
    assert restored.interactions[0].ligand_atoms == (2,)
    assert restored.interactions[0].ligand_is_donor is False
    assert restored.interactions[0].protein_atom == "OG"
    assert restored.interactions[0].angle == 167.5
    assert restored.exposure == {2: 0.75}
    assert restored.metadata["detector"] == "native"
    assert restored_layout is not None
    assert restored_layout.positions == layout.positions
    assert restored_layout.ligand_coords == layout.ligand_coords
    assert restored_layout.mirror is True
    assert view == {"legend_position": "left", "legend_rows": 3}

    text = path.read_text()
    assert '"hbond_min_donor_angle_degree": 150.0' in text
    assert '"hbond_profile": "strong_geometric"' in text
    assert '"mol_block"' in text


def test_json_keeps_a_group_contacts_protein_atoms(tmp_path):
    diagram, layout = _document()
    ring = ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")
    diagram.interactions = [Interaction(
        "pi_stacking", diagram.interactions[0].residue_key, (2,), 4.86,
        protein_atom="CG", protein_atoms=ring,
    )]
    restored, _, _ = load_json(save_json(tmp_path / "diagram.json", diagram, layout))
    assert restored.interactions[0].protein_atoms == ring
    assert restored.interactions[0].is_protein_group

    # A file written before the field existed reads as a one-atom contact.
    text = (tmp_path / "diagram.json").read_text().replace('"protein_atoms"', '"_unused"')
    (tmp_path / "old.json").write_text(text)
    old, _, _ = load_json(tmp_path / "old.json")
    assert old.interactions[0].protein_atoms == ()


def test_tooltip_names_the_centroid_for_a_group_contact():
    from ms_contactmap.render import _route_label

    ring = Interaction("pi_stacking", "A:227:PHE", (2,), 4.86, protein_atom="CG",
                       protein_atoms=("CG", "CD1", "CD2", "CE1", "CE2", "CZ"))
    assert "ring centroid" in _route_label(ring, ring.style)
    assert "atom CG" not in _route_label(ring, ring.style)
    bridge = Interaction("salt_bridge", "A:52:ASP", (2,), 3.9, protein_atom="OD1",
                         protein_atoms=("OD1", "OD2"))
    assert "charged group centroid" in _route_label(bridge, bridge.style)
    hbond = Interaction("hbond", "A:10:SER", (2,), 2.8, protein_atom="OG")
    assert "atom OG" in _route_label(hbond, hbond.style)
