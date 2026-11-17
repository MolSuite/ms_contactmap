"""A residue picked in the 2D diagram, or one contact picked in the interaction table, shown in
PyMOL with its contacts drawn in 3D.

Each contact is a PyMOL distance between two pseudoatoms, coloured like its line in the 2D
diagram, so both views read the same. A ring or charged group is drawn from its centroid (a
small sphere) on either side, because that is where ms_contactmap measures the distance from;
water bridges go through the water. No atom-index mapping crosses into PyMOL:
the diagram's ligand atoms are the pose's heavy atoms in file order, which is also the
order PyMOL keeps, so the coordinates come straight from the loaded pose.

Needs a PyMOL ``cmd`` (the ``3d`` extra brings ms_components' embedded PyMOL).
"""
from __future__ import annotations

GROUP = "contacts"


def interaction_group(receptor_obj: str, ligand_obj: str) -> str:
    return f"{GROUP}_{receptor_obj}_{ligand_obj}"


def residue_selection(receptor_obj: str, key: str) -> str:
    """``A:248B:LEU`` -> the PyMOL selection of that residue in ``receptor_obj``."""
    chain, number, _name = key.split(":")
    selection = f"{receptor_obj} and resi {number}"
    return f"{selection} and chain {chain}" if chain.strip() else selection


def _centroid(points) -> tuple[float, float, float]:
    n = len(points)
    return tuple(sum(p[i] for p in points) / n for i in range(3))


#: Contacts ms_contactmap measures centroid to centroid (its group pairs).
_GROUP_KINDS = frozenset({"pi_stacking", "pi_cation", "salt_bridge"})


def _protein_group(interaction) -> tuple[str, ...]:
    """Atom names of the protein group a ring or charged contact is measured from; ``()`` for a
    contact to one atom.

    ``protein_atoms`` when the contact carries it. A row saved before it did (or an ms_contactmap
    without it) still names ``protein_atom``, the group's first atom, which picks the residue's
    group out of ms_contactmap's own tables: CG is PHE's ring, ND1 the HIS cation, NZ all of LYS.
    """
    names = tuple(getattr(interaction, "protein_atoms", None) or ())
    if names or getattr(interaction, "kind", None) not in _GROUP_KINDS:
        return names
    first = getattr(interaction, "protein_atom", None)
    try:
        from .detect.protein import AROMATIC_RINGS, CHARGED_GROUPS
    except ImportError:
        return ()
    resname = interaction.residue_key.split(":")[-1].strip().upper()
    for group in (AROMATIC_RINGS.get(resname), (CHARGED_GROUPS.get(resname) or (None, None))[1]):
        if group and group[0] == first:
            return tuple(group)
    return ()


def _nearest(cmd, selection: str, to) -> tuple[float, float, float] | None:
    coords = cmd.get_coords(selection)
    if coords is None or not len(coords):
        return None
    return tuple(min(coords, key=lambda p: sum((p[i] - to[i]) ** 2 for i in range(3))))


def _nearest_pair(cmd, ligand, selection: str):
    """The ligand point closest to any atom of ``selection``, or ``None`` when it selects nothing."""
    import numpy as np

    target = cmd.get_coords(selection)
    if target is None or not len(target):
        return None
    gaps = np.linalg.norm(np.asarray(ligand)[:, None, :] - np.asarray(target)[None, :, :], axis=2)
    return tuple(ligand[int(gaps.min(axis=1).argmin())])


def show_residue_interactions(cmd, *, receptor_obj: str, ligand_obj: str, key: str, diagram) -> int:
    """Sticks + zoom on residue ``key``, one dashed distance per contact; the number drawn."""
    ligand = cmd.get_coords(f"{ligand_obj} and not elem H")
    if ligand is None or len(ligand) != diagram.mol.GetNumAtoms():
        ligand = None  # not the pose this diagram was built from: show the residue, draw nothing
    contacts = [c for c in diagram.interactions if c.residue_key == key or c.via_water == key]
    return _draw(cmd, receptor_obj, ligand_obj, residue_selection(receptor_obj, key), contacts, ligand)


def show_all_interactions(cmd, *, receptor_obj: str, ligand_obj: str, diagram) -> int:
    """Every contact of the diagram at once, its residues as sticks."""
    ligand = cmd.get_coords(f"{ligand_obj} and not elem H")
    if ligand is None or len(ligand) != diagram.mol.GetNumAtoms():
        ligand = None
    keys = {c.residue_key for c in diagram.interactions}
    residues = " or ".join(f"({residue_selection(receptor_obj, k)})" for k in sorted(keys)) or "none"
    return _draw(cmd, receptor_obj, ligand_obj, residues, diagram.interactions, ligand)


def show_interaction(cmd, *, receptor_obj: str, ligand_obj: str, contact) -> int:
    """One contact picked from the interaction table (``kind``, ``residue_key``, ``ligand_atoms``,
    ``protein_atom``, ``via_water`` and optionally ``protein_atoms`` attributes), drawn like the
    diagram draws it."""
    ligand = cmd.get_coords(f"{ligand_obj} and not elem H")
    return _draw(cmd, receptor_obj, ligand_obj, residue_selection(receptor_obj, contact.residue_key),
                 [contact], ligand)


def _draw(cmd, receptor_obj: str, ligand_obj: str, residue: str, contacts, ligand) -> int:
    from .model import INTERACTION_STYLES

    group = interaction_group(receptor_obj, ligand_obj)
    cmd.delete(group)
    cmd.show("sticks", f"({residue}) and not hydro")
    cmd.label(f"({residue}) and name CA", '"%s %s%s" % (resn, chain, resi)')
    if ligand is None:
        cmd.zoom(residue, 5)
        return 0

    names: list[str] = []

    def point(xyz, *, sphere: bool = False) -> str:
        name = f"{group}_p{len(names)}"
        cmd.pseudoatom(name, pos=[float(v) for v in xyz])
        cmd.hide("everything", name)
        if sphere:
            cmd.show("spheres", name)
            cmd.set("sphere_scale", 0.25, name)
        names.append(name)
        return name

    def dash(a: str, b: str, color: str) -> None:
        name = f"{group}_d{len(names)}"
        cmd.distance(name, a, b)
        cmd.set("dash_color", color.replace("#", "0x"), name)
        cmd.set("dash_gap", 0.35, name)
        cmd.set("label_distance_digits", 2, name)  # as the diagram reports it: 4.86, not 4.9
        names.append(name)

    drawn = 0
    for interaction in contacts:
        protein = residue_selection(receptor_obj, interaction.residue_key)
        if interaction.protein_atom:
            protein += f" and name {interaction.protein_atom}"
        atoms = [ligand[i] for i in interaction.ligand_atoms if i < len(ligand)]
        if not atoms:  # a table row saved without its ligand atoms: the ligand atom nearest the residue
            near = _nearest_pair(cmd, ligand, protein) or _nearest_pair(cmd, ligand, residue_selection(receptor_obj, interaction.residue_key))
            if near is None:
                continue
            atoms = [near]
        start = _centroid(atoms)
        color = INTERACTION_STYLES[interaction.kind].color if interaction.kind in INTERACTION_STYLES else "#808080"
        a = point(start, sphere=len(atoms) > 1)
        if interaction.via_water:
            water = _nearest(cmd, residue_selection(receptor_obj, interaction.via_water), start)
            if water is not None:
                w = point(water, sphere=True)
                dash(a, w, color)
                a = w
        members = _protein_group(interaction)
        coords = None
        if len(members) > 1:
            coords = cmd.get_coords(f"({residue_selection(receptor_obj, interaction.residue_key)}) "
                                    f"and name {'+'.join(members)}")
        if coords is not None and len(coords) == len(members):
            # The centroid, as measured: the group's first atom is up to a ring's radius off.
            dash(a, point(_centroid(coords), sphere=True), color)
            drawn += 1
            continue
        end = _nearest(cmd, protein, start) or _nearest(cmd, residue_selection(receptor_obj, interaction.residue_key), start)
        if end is None:
            continue
        dash(a, point(end), color)
        drawn += 1
    if names:
        cmd.group(group, " ".join(names))
        try:  # keep the drawing in its structure's grid panel when ms_components runs PyMOL
            from ms_components.ms_pymol import associate_scene_objects
        except ImportError:
            pass
        else:
            associate_scene_objects(cmd, [*names, group], reference=receptor_obj)
    cmd.zoom(f"({residue}) or {ligand_obj}", 3)
    return drawn
