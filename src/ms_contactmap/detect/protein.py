"""Protein/water/metal perception: PDB atoms -> :class:`~ms_contactmap.detect.roles.Perception`.

This is the left-hand branch of the split described in :mod:`.roles`.  Bond
orders never appear anywhere below -- crystal structures do not carry them and
a standard-residue lookup table does not need them -- so instead of RDKit
sanitisation, connectivity comes back to distance: two heavy atoms close
enough to be bonded are bonded, worked out once with a k-d tree rather than
hand-maintained per residue.  Roles, in contrast, are looked up: the table
below is the textbook donor/acceptor/hydrophobic/metal-binder assignment for
the 20 standard amino acids, spelled out atom by atom rather than derived,
because a table is easier to check against a textbook than a rule ever is.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from ms_contactmap.chem import KEPT_ALTLOCS, PdbAtom
from ms_contactmap.detect.roles import Group, Perception, Site
from ms_contactmap.model import AA_CLASS, METAL_NAMES, WATER_NAMES

# ---------------------------------------------------------------------------
# Sidechain role table
# ---------------------------------------------------------------------------

# ponytail: a handful of these carbons are, strictly, bonded to a heteroatom
# (CYS/SER/THR's CB to S/O, MET's CE to S, TRP's CD1/CE2 to the indole N) and
# so fail the textbook "bonded only to carbon/hydrogen" hydrophobic test. They
# are kept hydrophobic anyway because that is how every hydrophobicity scale
# and every pocket-drawing tool treats them -- the ceiling this trades away is
# a purist reading of "bonded only to C/H", not chemical accuracy.

#: ``(resname, atom_name) -> Site kwargs`` for sidechain atoms of the 20
#: standard amino acids.  Backbone ``N``/``CA``/``C``/``O``/``OXT`` are handled
#: by :func:`_backbone_roles` and never appear here.  An atom that is neither
#: here nor a backbone atom (ASN/ASP ``CG``, GLU/GLN ``CD``, ARG ``CZ``, HIS/
#: TRP ring carbons next to a ring nitrogen, GLY has no sidechain at all, ...)
#: gets an all-``False`` Site: present for connectivity, nothing else.
SIDECHAIN_ROLES: dict[tuple[str, str], dict[str, bool]] = {
    # ALA -- CB only.
    ("ALA", "CB"): dict(hydrophobic=True),
    # ARG -- CB/CG/CD are a plain alkyl tether to the guanidinium; NE/NH1/NH2
    # donate, CZ is the sp2 guanidinium carbon and stays out of hydrophobic.
    ("ARG", "CB"): dict(hydrophobic=True),
    ("ARG", "CG"): dict(hydrophobic=True),
    ("ARG", "CD"): dict(hydrophobic=True),
    ("ARG", "NE"): dict(donor=True),
    ("ARG", "NH1"): dict(donor=True),
    ("ARG", "NH2"): dict(donor=True),
    # ASN -- carboxamide: OD1 accepts, ND2 donates. CG is the amide carbon.
    ("ASN", "CB"): dict(hydrophobic=True),
    ("ASN", "OD1"): dict(acceptor=True),
    ("ASN", "ND2"): dict(donor=True),
    # ASP -- carboxylate, both oxygens equivalent by resonance.
    ("ASP", "CB"): dict(hydrophobic=True),
    ("ASP", "OD1"): dict(acceptor=True, metal_binder=True),
    ("ASP", "OD2"): dict(acceptor=True, metal_binder=True),
    # CYS -- thiol: weak donor and acceptor both, plus a soft metal ligand.
    ("CYS", "CB"): dict(hydrophobic=True),
    ("CYS", "SG"): dict(donor=True, acceptor=True, metal_binder=True),
    # GLN -- one CH2 further out than ASN, so CG is a plain alkyl carbon and
    # CD carries the amide.
    ("GLN", "CB"): dict(hydrophobic=True),
    ("GLN", "CG"): dict(hydrophobic=True),
    ("GLN", "OE1"): dict(acceptor=True),
    ("GLN", "NE2"): dict(donor=True),
    # GLU -- carboxylate, one CH2 further out than ASP.
    ("GLU", "CB"): dict(hydrophobic=True),
    ("GLU", "CG"): dict(hydrophobic=True),
    ("GLU", "OE1"): dict(acceptor=True, metal_binder=True),
    ("GLU", "OE2"): dict(acceptor=True, metal_binder=True),
    # GLY -- no sidechain atoms at all.
    # HIS -- imidazole. Both ring nitrogens can be protonated (donor), can
    # accept when not, and either can chelate a metal; the ring carbons each
    # sit next to a ring nitrogen so none of them count as hydrophobic.
    ("HIS", "CB"): dict(hydrophobic=True),
    ("HIS", "ND1"): dict(donor=True, acceptor=True, metal_binder=True),
    ("HIS", "NE2"): dict(donor=True, acceptor=True, metal_binder=True),
    # ILE
    ("ILE", "CB"): dict(hydrophobic=True),
    ("ILE", "CG1"): dict(hydrophobic=True),
    ("ILE", "CG2"): dict(hydrophobic=True),
    ("ILE", "CD1"): dict(hydrophobic=True),
    # LEU
    ("LEU", "CB"): dict(hydrophobic=True),
    ("LEU", "CG"): dict(hydrophobic=True),
    ("LEU", "CD1"): dict(hydrophobic=True),
    ("LEU", "CD2"): dict(hydrophobic=True),
    # LYS -- CB/CG/CD are a plain alkyl tether, NZ is the ammonium that donates
    # and anchors the cation group.
    ("LYS", "CB"): dict(hydrophobic=True),
    ("LYS", "CG"): dict(hydrophobic=True),
    ("LYS", "CD"): dict(hydrophobic=True),
    ("LYS", "NZ"): dict(donor=True),
    # MET -- SD is both a soft metal ligand and, chemically, still nonpolar
    # enough that both flags make sense on the same atom.
    ("MET", "CB"): dict(hydrophobic=True),
    ("MET", "CG"): dict(hydrophobic=True),
    ("MET", "SD"): dict(hydrophobic=True, metal_binder=True),
    ("MET", "CE"): dict(hydrophobic=True),
    # PHE -- pure carbon ring, entirely hydrophobic and entirely aromatic.
    ("PHE", "CB"): dict(hydrophobic=True),
    ("PHE", "CG"): dict(hydrophobic=True),
    ("PHE", "CD1"): dict(hydrophobic=True),
    ("PHE", "CD2"): dict(hydrophobic=True),
    ("PHE", "CE1"): dict(hydrophobic=True),
    ("PHE", "CE2"): dict(hydrophobic=True),
    ("PHE", "CZ"): dict(hydrophobic=True),
    # PRO -- ring closes back to N (handled in _backbone_roles: no N donor).
    ("PRO", "CB"): dict(hydrophobic=True),
    ("PRO", "CG"): dict(hydrophobic=True),
    ("PRO", "CD"): dict(hydrophobic=True),
    # SER
    ("SER", "CB"): dict(hydrophobic=True),
    ("SER", "OG"): dict(donor=True, acceptor=True),
    # THR
    ("THR", "CB"): dict(hydrophobic=True),
    ("THR", "OG1"): dict(donor=True, acceptor=True),
    ("THR", "CG2"): dict(hydrophobic=True),
    # TRP -- indole. NE1 donates; the fused 6-ring is what stacks and is
    # reported as the aromatic group (see _groups); its carbons are all kept
    # hydrophobic even though two of them (CD1, CE2) touch NE1.
    ("TRP", "CB"): dict(hydrophobic=True),
    ("TRP", "CG"): dict(hydrophobic=True),
    ("TRP", "CD1"): dict(hydrophobic=True),
    ("TRP", "CD2"): dict(hydrophobic=True),
    ("TRP", "NE1"): dict(donor=True),
    ("TRP", "CE2"): dict(hydrophobic=True),
    ("TRP", "CE3"): dict(hydrophobic=True),
    ("TRP", "CZ2"): dict(hydrophobic=True),
    ("TRP", "CZ3"): dict(hydrophobic=True),
    ("TRP", "CH2"): dict(hydrophobic=True),
    # TYR -- CZ carries the phenolic OH and is excluded from hydrophobic; that
    # one carbon is the whole chemical difference from PHE's ring.
    ("TYR", "CB"): dict(hydrophobic=True),
    ("TYR", "CG"): dict(hydrophobic=True),
    ("TYR", "CD1"): dict(hydrophobic=True),
    ("TYR", "CD2"): dict(hydrophobic=True),
    ("TYR", "CE1"): dict(hydrophobic=True),
    ("TYR", "CE2"): dict(hydrophobic=True),
    ("TYR", "OH"): dict(donor=True, acceptor=True),
    # VAL
    ("VAL", "CB"): dict(hydrophobic=True),
    ("VAL", "CG1"): dict(hydrophobic=True),
    ("VAL", "CG2"): dict(hydrophobic=True),
}

#: Ring atom names for the aromatic Group of each residue that has one, in the
#: order the brief specifies.  HIS's is the imidazole, PHE/TYR the phenyl,
#: TRP the indole's fused 6-ring only (the 5-ring is not reported: it is the
#: 6-ring that stacks).
AROMATIC_RINGS: dict[str, tuple[str, ...]] = {
    "HIS": ("CG", "ND1", "CE1", "NE2", "CD2"),
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}

#: Atom names forming each residue's charged Group, keyed like
#: :data:`AROMATIC_RINGS`.  HIS is included even though the ring above is also
#: aromatic: protonated histidine (HIP) is drawn as charged_positive in
#: Maestro's own convention, and a cation Group that a docked pose never
#: brings near a partner simply never fires -- see :mod:`.geometry`.
CHARGED_GROUPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ARG": ("cation", ("CZ", "NH1", "NH2")),
    "LYS": ("cation", ("NZ",)),
    "HIS": ("cation", ("ND1", "CE1", "NE2")),
    "ASP": ("anion", ("OD1", "OD2")),
    "GLU": ("anion", ("OE1", "OE2")),
}

#: Ring planarity gate for :func:`_ring_group`.  A genuine aromatic ring is
#: flat to a few hundredths of an angstrom; anything past this is either a
#: badly modelled sidechain or a bug feeding the wrong atoms in, and drawing a
#: ring normal for it would just be wrong.
_RING_FLATNESS = 0.25  # angstrom, max deviation from the least-squares plane

#: Covalent-bond distance cutoffs for :func:`_neighbours`: generous enough for
#: any single bond between two heavy atoms, tight enough to exclude even a
#: short hydrogen bond or salt bridge.
_BOND_MAX = 1.9  # angstrom
_BOND_MAX_SULFUR = 2.2  # angstrom, either atom is S (covers C-S and S-S)


def _backbone_roles(atom: PdbAtom) -> dict[str, bool]:
    """Roles that follow from the atom name alone, true in every residue.

    Proline's backbone nitrogen is the one exception in the whole table: it is
    part of the pyrrolidine ring and has no hydrogen to donate.
    """
    name = atom.name.strip().upper()
    if name == "N":
        return {} if atom.resname.strip().upper() == "PRO" else {"donor": True}
    if name in ("O", "OXT"):
        return {"acceptor": True}
    return {}


def _is_metal(atom: PdbAtom) -> bool:
    """Resname or element says metal.  METAL_NAMES already covers both:

    it lists real element symbols (ZN, FE, CU, ...) alongside a few resname
    variants that are not elements (FE2, CU1), so checking the element column
    against the same set is free and correctly ignores those variants.
    """
    return (
        atom.resname.strip().upper() in METAL_NAMES
        or atom.element.strip().upper() in METAL_NAMES
    )


def _atom_roles(atom: PdbAtom) -> dict[str, bool]:
    """Roles for one heavy atom, resolved in the order the brief specifies."""
    resname = atom.resname.strip().upper()
    if _is_metal(atom):
        return dict(is_metal=True)
    element = atom.element.strip().upper()
    if resname in WATER_NAMES:
        roles = dict(donor=True, acceptor=True, is_water=True)
    elif resname in AA_CLASS:  # the 20 standard amino acids, keys of AA_CLASS
        roles = _backbone_roles(atom)
        roles.update(SIDECHAIN_ROLES.get((resname, atom.name.strip().upper()), {}))
    # Fallback for everything else: modified residues, cofactors, and any
    # ligand other than the one of interest (which the caller normally leaves
    # out, but nothing here depends on that).  A silently empty Site would
    # drop real contacts, so every heavy element still gets a generic role.
    elif element in ("N", "O"):
        roles = dict(donor=True, acceptor=True)
    elif element == "S":
        roles = dict(acceptor=True)
    elif element == "C":
        roles = dict(hydrophobic=True)
    else:
        roles = {}  # halogens and anything rarer: no role on the protein side

    # Coordinating a metal is donating a lone pair, so every N/O/S that accepts
    # a hydrogen bond can also bind a metal -- a backbone carbonyl, an ASN OD1
    # and a water all do it routinely.  The table above marks metal_binder only
    # where the sidechain is a *good* ligand; this widens it to the physically
    # capable set, which is what the 3.0 A distance criterion then filters.
    if roles.get("acceptor") and element in ("N", "O", "S"):
        roles["metal_binder"] = True
    return roles


def _neighbours(
    atoms: list[PdbAtom], coords: dict[int, np.ndarray]
) -> dict[int, tuple[int, ...]]:
    """Bonded heavy atoms by distance, derived rather than tabulated.

    Two heavy atoms are bonded if they are within a covalent-bond distance of
    each other and either share a residue or are the backbone C/N of
    consecutive residues in the same chain (the peptide bond).  cKDTree finds
    every pair within the looser of the two cutoffs in one pass; the loop
    below tightens the cutoff per pair and applies the residue/peptide filter.
    Metals get no neighbours: their contacts to the rest of the structure are
    coordination bonds, not covalent ones, and belong to :mod:`.geometry`.
    """
    bonded_atoms = [a for a in atoms if not _is_metal(a)]
    if not bonded_atoms:
        return {}
    ids = [a.serial for a in bonded_atoms]
    xyz = np.array([coords[s] for s in ids])
    element = {a.serial: a.element.strip().upper() for a in bonded_atoms}
    res_id = {a.serial: a.res_id for a in bonded_atoms}
    chain = {a.serial: a.chain for a in bonded_atoms}
    resnum = {a.serial: a.resnum for a in bonded_atoms}
    name = {a.serial: a.name.strip().upper() for a in bonded_atoms}

    out: dict[int, list[int]] = {s: [] for s in ids}
    tree = cKDTree(xyz)
    for i, j in tree.query_pairs(r=_BOND_MAX_SULFUR):
        si, sj = ids[i], ids[j]
        cutoff = _BOND_MAX_SULFUR if "S" in (element[si], element[sj]) else _BOND_MAX
        if np.linalg.norm(xyz[i] - xyz[j]) >= cutoff:
            continue
        same_residue = res_id[si] == res_id[sj]
        peptide = False
        if chain[si] == chain[sj] and {name[si], name[sj]} == {"C", "N"}:
            c, n = (si, sj) if name[si] == "C" else (sj, si)
            peptide = resnum[n] == resnum[c] + 1
        if not (same_residue or peptide):
            continue
        out[si].append(sj)
        out[sj].append(si)
    return {s: tuple(sorted(v)) for s, v in out.items()}


def _atoms_by_name(names: tuple[str, ...], by_name: dict[str, int]) -> tuple[int, ...] | None:
    """Serials for every name in ``names``, or ``None`` if one is missing."""
    if not all(n in by_name for n in names):
        return None
    return tuple(by_name[n] for n in names)


def _ring_group(atoms: tuple[int, ...], coords: dict[int, np.ndarray]) -> Group | None:
    """An aromatic Group with a unit normal, or ``None`` if the ring isn't flat."""
    points = np.array([coords[s] for s in atoms])
    centroid = points.mean(axis=0)
    normal = np.linalg.svd(points - centroid, full_matrices=False)[2][2]
    normal = normal / np.linalg.norm(normal)
    if np.abs((points - centroid) @ normal).max() > _RING_FLATNESS:
        return None
    return Group(kind="aromatic", atoms=atoms, normal=normal)


def _groups(atoms: list[PdbAtom], coords: dict[int, np.ndarray]) -> list[Group]:
    """One cation/anion/aromatic Group per residue occurrence that has one."""
    by_residue: dict[tuple, dict[str, int]] = {}
    for a in atoms:
        by_residue.setdefault(a.res_id, {})[a.name.strip().upper()] = a.serial

    groups: list[Group] = []
    for res_id, by_name in by_residue.items():
        # res_id alone doesn't carry the resname; recover it from any atom.
        resname = next(a.resname.strip().upper() for a in atoms if a.res_id == res_id)

        kind_atoms = CHARGED_GROUPS.get(resname)
        if kind_atoms is not None:
            kind, names = kind_atoms
            picked = _atoms_by_name(names, by_name)
            if picked is not None:
                groups.append(Group(kind=kind, atoms=picked))

        ring_names = AROMATIC_RINGS.get(resname)
        if ring_names is not None:
            picked = _atoms_by_name(ring_names, by_name)
            if picked is not None:
                ring = _ring_group(picked, coords)
                if ring is not None:
                    groups.append(ring)

        if "OXT" in by_name:  # C-terminal carboxylate: a one-atom anion Group
            groups.append(Group(kind="anion", atoms=(by_name["OXT"],)))

    return groups


def perceive_protein(atoms: list[PdbAtom]) -> Perception:
    """Chemical roles of every protein/water/metal heavy atom, keyed by PDB serial."""
    kept = [a for a in atoms if a.element not in ("H", "D") and a.altloc in KEPT_ALTLOCS]
    coords = {a.serial: np.array((a.x, a.y, a.z), dtype=float) for a in kept}
    neighbours = _neighbours(kept, coords)
    elements = {a.serial: a.element.strip().upper() for a in kept}
    by_residue: dict[tuple[str, int, str], list[PdbAtom]] = {}
    for atom in kept:
        by_residue.setdefault(atom.res_id, []).append(atom)
    hydrogen_coords: dict[int, list[tuple[float, float, float]]] = {}
    for hydrogen in (
        atom for atom in atoms
        if atom.element in ("H", "D") and atom.altloc in KEPT_ALTLOCS
    ):
        candidates = by_residue.get(hydrogen.res_id, ())
        if not candidates:
            continue
        nearest = min(
            candidates,
            key=lambda atom: (atom.x - hydrogen.x) ** 2
            + (atom.y - hydrogen.y) ** 2
            + (atom.z - hydrogen.z) ** 2,
        )
        distance = np.linalg.norm(coords[nearest.serial] - (hydrogen.x, hydrogen.y, hydrogen.z))
        if distance <= 1.7:
            hydrogen_coords.setdefault(nearest.serial, []).append(
                (hydrogen.x, hydrogen.y, hydrogen.z)
            )
    sites = {
        a.serial: Site(
            **_atom_roles(a),
            neighbours=neighbours.get(a.serial, ()),
            hydrogens=tuple(hydrogen_coords.get(a.serial, ())),
        )
        for a in kept
    }
    # Atom-level hydrophobicity is derived from the actual connectivity: a
    # carbon qualifies only when every bonded heavy neighbour is also carbon.
    # This follows the published criterion and avoids residue-table exceptions.
    for serial, site in tuple(sites.items()):
        hydrophobic = elements[serial] == "C" and all(
            elements.get(other) == "C" for other in site.neighbours
        )
        if site.hydrophobic != hydrophobic:
            sites[serial] = replace(site, hydrophobic=hydrophobic)
    return Perception(sites=sites, coords=coords, groups=_groups(kept, coords))


def _self_check() -> None:
    """Run against the reference structures. ``python -m ms_contactmap.detect.protein``"""
    from pathlib import Path

    from ms_contactmap.chem import read_pdb_atoms

    root = Path(__file__).resolve().parent.parent.parent
    for stem in ("4ps5", "2gfk"):
        atoms = read_pdb_atoms(root / "data" / f"{stem}.pdb")
        heavy = [a for a in atoms if a.element not in ("H", "D") and a.altloc in KEPT_ALTLOCS]
        by_serial = {a.serial: a for a in heavy}
        perception = perceive_protein(atoms)

        assert set(perception.coords) == set(perception.sites)

        def residue_ids(resname: str) -> set:
            return {a.res_id for a in heavy if a.resname.strip().upper() == resname}

        def group_residue(group: Group) -> tuple:
            return by_serial[group.atoms[0]].res_id

        def groups_of(kind: str, size: int, resname: str) -> set:
            return {
                group_residue(g)
                for g in perception.groups
                if g.kind == kind
                and len(g.atoms) == size
                and by_serial[g.atoms[0]].resname.strip().upper() == resname
            }

        # Every ARG -> one 3-atom cation group; every ASP/GLU -> one 2-atom
        # anion group.
        assert groups_of("cation", 3, "ARG") == residue_ids("ARG"), stem
        assert groups_of("anion", 2, "ASP") == residue_ids("ASP"), stem
        assert groups_of("anion", 2, "GLU") == residue_ids("GLU"), stem

        # Every PHE/TYR -> one 6-atom aromatic group with a unit normal.
        for resname in ("PHE", "TYR"):
            found = groups_of("aromatic", 6, resname)
            assert found == residue_ids(resname), (stem, resname)
        for g in perception.groups:
            if g.kind == "aromatic":
                assert abs(np.linalg.norm(g.normal) - 1.0) < 1e-6, (stem, g.atoms)

        # Backbone N: donor except PRO.
        pro_n = next(
            a for a in heavy
            if a.resname.strip().upper() == "PRO" and a.name.strip().upper() == "N"
        )
        assert perception.sites[pro_n.serial].donor is False, stem
        std_n = next(
            a for a in heavy
            if not a.hetatm
            and a.name.strip().upper() == "N"
            and a.resname.strip().upper() != "PRO"
        )
        assert perception.sites[std_n.serial].donor is True, stem

        # Metals come through as is_metal.
        zn_atoms = [a for a in heavy if a.resname.strip().upper() == "ZN"]
        if stem == "2gfk":
            assert zn_atoms, "2gfk should contain ZN"
        assert all(perception.sites[a.serial].is_metal for a in zn_atoms), stem

        # A known hydrophobic carbon vs. a known non-hydrophobic carbonyl.
        leu_cd1 = next(
            a for a in heavy
            if a.resname.strip().upper() == "LEU" and a.name.strip().upper() == "CD1"
        )
        assert perception.sites[leu_cd1.serial].hydrophobic is True, stem
        leu_c = next(
            a for a in heavy
            if a.resname.strip().upper() == "LEU" and a.name.strip().upper() == "C"
        )
        assert perception.sites[leu_c.serial].hydrophobic is False, stem

        # Dual donor/acceptor sidechains and the weak-donor thiol, spot-checked.
        ser_og = next(
            (a for a in heavy if a.resname.strip().upper() == "SER" and a.name.strip().upper() == "OG"),
            None,
        )
        if ser_og is not None:
            s = perception.sites[ser_og.serial]
            assert s.donor and s.acceptor, stem
        cys_sg = next(
            (a for a in heavy if a.resname.strip().upper() == "CYS" and a.name.strip().upper() == "SG"),
            None,
        )
        if cys_sg is not None:
            s = perception.sites[cys_sg.serial]
            assert s.donor and s.acceptor and s.metal_binder, stem

        # Waters: O is a donor, an acceptor, and flagged as water.
        water = next((a for a in heavy if a.resname.strip().upper() in WATER_NAMES), None)
        if water is not None:
            s = perception.sites[water.serial]
            assert s.donor and s.acceptor and s.is_water, stem

        # Neighbour counts are sane: every non-glycine backbone CA sees
        # exactly N, C and CB (3), occasionally 4 with a coincidental close
        # contact; glycine has no CB and so sees only N and C (2).
        for a in heavy:
            if a.hetatm or a.name.strip().upper() != "CA":
                continue
            n = len(perception.sites[a.serial].neighbours)
            if a.resname.strip().upper() == "GLY":
                assert n == 2, (stem, "GLY CA", a.resnum, n)
            else:
                assert n in (3, 4), (stem, a.resname, a.resnum, n)

        kinds = Counter(g.kind for g in perception.groups)
        print(
            f"  {stem}: {len(perception.sites)} sites, {len(perception.groups)} groups "
            f"{dict(kinds)}"
        )

    print("\nprotein perception self-check passed")


if __name__ == "__main__":
    _self_check()
