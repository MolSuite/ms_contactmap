"""Turn geometric hits into the diagram's interaction list.

:func:`detect_interactions` is the package's only detector.  It returns the
``(interactions, metal_legs, coordination, skipped)`` tuple that
:func:`ms_contactmap.interactions.build_diagram` consumes; everything above this
line -- the layout optimiser, the renderer, the widget -- only ever sees that
tuple, so detection stays swappable without touching them.

The division of labour: :mod:`.protein` and :mod:`.ligand` say what each atom
*is*, :mod:`.geometry` says which pairs are close and well-oriented enough to
count, and this module is the only one that knows about residues, water names
and the diagram's data classes.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import cKDTree

from ..chem import PdbAtom
from ..model import Interaction, MetalLeg, ResidueRef
from .geometry import (
    METAL_EXPECTED_COORDINATION,
    METAL_FIRST_SHELL_MAX,
    WATER_OMEGA_IDEAL,
    detect_contacts,
)
from .ligand import perceive_ligand
from .protein import perceive_protein
from .roles import Hit

#: Radius around the ligand within which protein atoms are perceived at all.
#: The longest criterion is pi-cation at 6.0 A, and a water bridge reaches
#: 4.1 + 4.1; 10 A clears both with room for a metal sitting between the ligand
#: and its coordination sphere.  Perceiving a whole protein would be correct and
#: slow, and every atom it adds is one no criterion could ever reach.
POCKET_RADIUS = 10.0

# First-shell metal-ligand limits.  The old generic 3.0 A search radius is
# retained inside geometry.py as a candidate generator, then chemistry-specific
# limits and coordination geometry decide which candidates form the sphere.
def _off_tetrahedral(hit: Hit) -> float:
    return abs((hit.angle or 0.0) - WATER_OMEGA_IDEAL)


def _key(atom: PdbAtom) -> str:
    return ResidueRef(atom.chain, atom.resnum, atom.resname.strip().upper(), atom.icode).key


def _nearby(serial_to_atom: dict[int, PdbAtom], ligand_xyz: np.ndarray,
            exclude: set[int]) -> list[PdbAtom]:
    """Heavy atoms of every residue with an atom near the ligand, ligand aside.

    Whole residues, not loose atoms: a charged group or an aromatic ring is only
    perceivable if all of its atoms are present, and clipping one off would
    silently cost a salt bridge.
    """
    atoms = [a for s, a in serial_to_atom.items()
             if s not in exclude and a.element.strip().upper() not in ("H", "D")]
    if not atoms:
        return []
    coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=float)
    near = cKDTree(coords).query_ball_point(ligand_xyz, POCKET_RADIUS)
    hit = {j for hits in near for j in hits}
    keep = {(atoms[i].chain, atoms[i].resnum, atoms[i].icode) for i in hit}
    return [a for a in atoms if (a.chain, a.resnum, a.icode) in keep]


def _refine_geometric_hits(
    hits: list[Hit], lig, serial_to_atom: dict[int, PdbAtom]
) -> list[Hit]:
    """Apply the published redundancy rules while atom identities still exist.

    Hydrophobic pairs are reduced from both perspectives: first to the nearest
    protein atom in a residue for each ligand carbon, then to the nearest of
    neighbouring ligand carbons for each protein atom.  Ring-carbon contacts
    already represented by pi stacking are removed.  A hydrogen-bond donor is
    allowed one bond, selected by the donor angle closest to linearity.
    """
    stacks = [
        (set(hit.ligand_atoms), set(hit.protein_atoms))
        for hit in hits if hit.kind == "pi_stacking"
    ]
    hydrophobic = [hit for hit in hits if hit.kind == "hydrophobic"]
    hydrophobic = [
        hit for hit in hydrophobic
        if not any(
            hit.ligand_atoms[0] in ligand_ring
            and hit.protein_atoms[0] in protein_ring
            for ligand_ring, protein_ring in stacks
        )
    ]

    nearest_by_residue: dict[tuple[int, str], Hit] = {}
    for hit in sorted(hydrophobic, key=lambda row: row.distance):
        atom = serial_to_atom.get(hit.protein_atoms[0])
        if atom is None:
            continue
        nearest_by_residue.setdefault((hit.ligand_atoms[0], _key(atom)), hit)

    nearest_by_protein: dict[int, list[Hit]] = {}
    for hit in sorted(nearest_by_residue.values(), key=lambda row: row.distance):
        protein_atom = hit.protein_atoms[0]
        ligand_atom = hit.ligand_atoms[0]
        selected = nearest_by_protein.setdefault(protein_atom, [])
        if any(
            ligand_atom == other.ligand_atoms[0]
            or ligand_atom in lig.sites[other.ligand_atoms[0]].neighbours
            or other.ligand_atoms[0] in lig.sites[ligand_atom].neighbours
            for other in selected
        ):
            continue
        selected.append(hit)

    best_hbond: dict[tuple[str, int], Hit] = {}
    for hit in (row for row in hits if row.kind == "hbond"):
        donor = ("L", hit.ligand_atoms[0]) if hit.ligand_is_donor else (
            "P", hit.protein_atoms[0]
        )
        rival = best_hbond.get(donor)
        score = (hit.angle if hit.angle is not None else 0.0, -hit.distance)
        rival_score = (
            rival.angle if rival is not None and rival.angle is not None else 0.0,
            -rival.distance if rival is not None else float("-inf"),
        )
        if rival is None or score > rival_score:
            best_hbond[donor] = hit

    other = [hit for hit in hits if hit.kind not in {"hydrophobic", "hbond"}]
    kept_hydrophobic = [hit for group in nearest_by_protein.values() for hit in group]
    return other + list(best_hbond.values()) + kept_hydrophobic


def _refine_metal_hits(hits: list[Hit], lig, prot,
                       serial_to_atom: dict[int, PdbAtom]) -> list[Hit]:
    """Apply element-specific first-shell limits and coordination geometry."""
    ordinary = [hit for hit in hits if hit.kind != "metal_coordination"]
    grouped: dict[int, list[Hit]] = {}
    for hit in (row for row in hits if row.kind == "metal_coordination"):
        metal = serial_to_atom.get(hit.protein_atoms[0])
        if metal is None:
            continue
        element = metal.element.strip().upper() or metal.resname.strip().upper()
        if hit.distance <= METAL_FIRST_SHELL_MAX.get(element, 3.0):
            grouped.setdefault(hit.protein_atoms[0], []).append(hit)

    selected: list[Hit] = []
    for metal_id, sphere in grouped.items():
        metal = serial_to_atom[metal_id]
        element = metal.element.strip().upper() or metal.resname.strip().upper()
        expected = METAL_EXPECTED_COORDINATION.get(element)
        if expected is None or len(sphere) <= expected:
            selected.extend(sphere)
            continue

        centre = prot.coords[metal_id]

        def partner(hit: Hit) -> np.ndarray:
            if hit.ligand_atoms:
                return lig.coords[hit.ligand_atoms[0]]
            return prot.coords[hit.protein_atoms[1]]

        ideal = 109.47 if element == "ZN" else 90.0

        def score(combo: tuple[Hit, ...]) -> float:
            vectors = np.array([partner(hit) - centre for hit in combo])
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1), 1e-9)[:, None]
            angles = []
            for i, j in itertools.combinations(range(len(vectors)), 2):
                angle = float(np.degrees(np.arccos(np.clip(
                    np.dot(vectors[i], vectors[j]), -1.0, 1.0
                ))))
                if element == "MG":
                    angles.append(min(abs(angle - 90.0), abs(angle - 180.0)))
                else:
                    angles.append(abs(angle - ideal))
            return float(np.sum(np.square(angles))) + 4.0 * sum(
                hit.distance * hit.distance for hit in combo
            )

        selected.extend(min(itertools.combinations(sphere, expected), key=score))
    return ordinary + selected


def detect_interactions(pdb_path, geom, serial_to_atom: dict[int, PdbAtom]):
    """Detect the ligand's contacts in-process.

    ``pdb_path`` is accepted and unused, since everything needed is already in
    ``geom`` and ``serial_to_atom``.  ``skipped`` counts hits whose protein atom
    is not in ``serial_to_atom`` -- structurally impossible today, kept because
    the caller reports it and a silent drop is what it exists to catch.
    """
    lig = perceive_ligand(geom)
    ligand_xyz = np.array(geom.coords_3d, dtype=float)
    prot = perceive_protein(_nearby(serial_to_atom, ligand_xyz, set(geom.serial_to_idx)))

    interactions: list[Interaction] = []
    metal_legs: list[MetalLeg] = []
    coordination: dict[str, int] = {}
    skipped = 0

    hits = _refine_metal_hits(detect_contacts(lig, prot), lig, prot, serial_to_atom)
    hits = _refine_geometric_hits(hits, lig, serial_to_atom)

    # A structural water can make a small network, not just one arbitrary
    # pair.  Keep up to two distinct protein partners per ligand-water leg,
    # ranked by tetrahedral geometry and distance.  This preserves 4UWH's
    # ligand--HOH2260--{TYR670, ASP761 backbone N} network while preventing a
    # crowded hydration site from becoming a starburst.
    bridge_groups: dict[tuple[int, tuple[int, ...]], list[Hit]] = {}
    for hit in (row for row in hits if row.kind == "water_bridge"):
        bridge_groups.setdefault((hit.water, hit.ligand_atoms), []).append(hit)
    selected_bridges: set[int] = set()
    for group in bridge_groups.values():
        seen_protein: set[int] = set()
        ranked = sorted(
            group,
            key=lambda row: (_off_tetrahedral(row), row.distance),
        )
        for hit in ranked:
            partner = hit.protein_atoms[0]
            if partner in seen_protein:
                continue
            selected_bridges.add(id(hit))
            seen_protein.add(partner)
            if len(seen_protein) == 2:
                break
    hits = [
        hit for hit in hits
        if hit.kind != "water_bridge" or id(hit) in selected_bridges
    ]

    # Two atoms in the same coordination sphere sit close and well-oriented
    # because the metal holds them there, not because they bond to each other.
    # Reading that as a hydrogen bond draws a line for something the metal
    # already explains, so the spheres are collected first and used to veto.
    spheres: dict[int, set[tuple[str, int]]] = {}
    for hit in hits:
        if hit.kind == "metal_coordination":
            sphere = spheres.setdefault(hit.protein_atoms[0], set())
            sphere.update(("L", a) for a in hit.ligand_atoms)
            if not hit.ligand_atoms:
                sphere.add(("P", hit.protein_atoms[1]))

    # A binuclear centre -- 2gfk's two zincs -- is one site, not two, and the
    # ligand that bridges them puts the partners of both metals within reach of
    # each other.  Spheres sharing an atom are therefore merged before the veto,
    # or a contact across the pair reads as a hydrogen bond of its own.
    merged: list[set[tuple[str, int]]] = []
    for sphere in spheres.values():
        touching = [m for m in merged if m & sphere]
        for m in touching:
            merged.remove(m)
            sphere = sphere | m
        merged.append(sphere)

    for hit in hits:
        if hit.kind in ("hbond", "water_bridge"):
            ends = {("L", a) for a in hit.ligand_atoms}
            ends.add(("P", hit.protein_atoms[0]))
            if any(ends <= sphere for sphere in merged):
                continue
        anchor = serial_to_atom.get(hit.protein_atoms[0]) if hit.protein_atoms else None
        if anchor is None:
            skipped += 1
            continue
        residue_key = _key(anchor)

        if hit.kind == "metal_coordination":
            # The whole coordination sphere is the point: a Zn has to read as a
            # tetrahedron, so the legs that end on the protein or on a water are
            # kept as their own edges rather than dropped for not touching the
            # ligand.  The metal is the residue on both.
            coordination[residue_key] = coordination.get(residue_key, 0) + 1
            if not hit.ligand_atoms:
                partner = serial_to_atom[hit.protein_atoms[1]]
                metal_legs.append(MetalLeg(residue_key, _key(partner), hit.distance))
                continue

        via = None
        if hit.water is not None:
            water = serial_to_atom.get(hit.water)
            if water is None:
                skipped += 1
                continue
            via = _key(water)

        interactions.append(Interaction(
            kind=hit.kind,
            residue_key=residue_key,
            ligand_atoms=hit.ligand_atoms,
            distance=hit.distance,
            ligand_is_donor=hit.ligand_is_donor,
            via_water=via,
            protein_atom=anchor.name,
            angle=hit.angle,
            protein_distance=hit.partner_distance,
            protein_is_donor=hit.protein_is_donor,
        ))
    return _refine(interactions), metal_legs, coordination, skipped


def _refine(interactions: list[Interaction]) -> list[Interaction]:
    """Collapse what the diagram cannot draw twice.

    :mod:`.geometry` reports every qualifying atom pair because it has no notion
    of a residue, and several of those pairs are the same line on the sheet.
    Nothing here is chemistry -- the criteria already ran -- it is about what
    reads once drawn, so each rule keeps the shortest contact of its group.

    Three rules, in order:

    * one contact per (kind, residue, ligand atoms): an ASP accepting from the
      same ligand atom through both carboxylate oxygens is one bond;
    * a hydrogen bond inside a salt bridge is part of that salt bridge -- a
      charged pair holds itself together with hydrogen bonds, and drawing both
      says the same thing twice in two colours;
    * a water bridge whose two ends already share a direct hydrogen bond is
      dropped, because the direct bond says it better.

    Which bridge each water contributes was already settled in
    :func:`detect_interactions`, where the angle at the oxygen is still around.
    """
    best: dict[tuple, Interaction] = {}
    for i in sorted(interactions, key=lambda i: i.distance):
        best.setdefault((i.kind, i.residue_key, i.ligand_atoms), i)

    # No "one salt bridge per residue" rule here on purpose: a guanidinium or an
    # imidazolium is Y-shaped and bridges two anions at once, which 2gfk's HIS
    # 196 does to two separate phosphonates of the ligand.  Collapsing those
    # loses a real contact to save a doubtful one.
    charged: dict[str, set[int]] = {}
    for i in best.values():
        if i.kind == "salt_bridge":
            charged.setdefault(i.residue_key, set()).update(i.ligand_atoms)
    best = {
        k: i for k, i in best.items()
        if i.kind != "hbond"
        or not set(i.ligand_atoms) <= charged.get(i.residue_key, set())
    }

    direct = {(i.residue_key, i.ligand_atoms) for i in best.values() if i.kind == "hbond"}
    kept: list[Interaction] = []
    for i in sorted(best.values(), key=lambda i: i.distance):
        if i.kind == "water_bridge" and (i.residue_key, i.ligand_atoms) in direct:
            continue
        kept.append(i)
    return sorted(kept, key=lambda i: (i.kind, i.residue_key, i.ligand_atoms))


def _self_check() -> None:
    """``python -m ms_contactmap.detect.engine`` -- the four reference systems."""
    import json
    from collections import Counter
    from pathlib import Path

    from ..chem import load_ligand, read_pdb_atoms

    root = Path(__file__).resolve().parent.parent.parent
    ligands = json.loads((root / "data" / "ligands.json").read_text())
    for pdb, resname in (("2gfk", "VII"), ("4ps5", "2TA"),
                         ("4uwh", "JXM"), ("6wak", "ANP")):
        path = root / "data" / f"{pdb}.pdb"
        geom = load_ligand(path, resname, ligands[resname]["smiles"])
        serial_to_atom = {a.serial: a for a in read_pdb_atoms(path)}
        found, legs, coordination, skipped = detect_interactions(path, geom, serial_to_atom)

        assert found, f"{pdb}: no interactions at all"
        assert skipped == 0, f"{pdb}: {skipped} unmappable contacts"
        n_atoms = geom.mol.GetNumAtoms()
        for i in found:
            assert i.ligand_atoms, f"{pdb} {i.kind}: no ligand atom"
            assert all(0 <= a < n_atoms for a in i.ligand_atoms), \
                f"{pdb} {i.kind}: ligand index out of range {i.ligand_atoms}"
            assert i.distance > 0.0, f"{pdb} {i.kind}: distance {i.distance}"
            assert i.kind != "water_bridge" or i.via_water, \
                f"{pdb}: water bridge with no water"
        for leg in legs:
            assert leg.metal_key in coordination, f"{pdb}: stray metal leg {leg}"
        print(f"  {pdb} {resname}: {len(found)} interactions, {len(legs)} metal legs  "
              f"{dict(Counter(i.kind for i in found))}")
    print("\ndetect engine self-check passed")


if __name__ == "__main__":
    _self_check()
