"""Ligand perception: what every heavy atom of the ligand is chemically able to do.

The ligand reaches this module from :func:`ms_contactmap.chem.load_ligand` already
correctly typed -- bond orders, aromaticity and formal charges come from the
user's SMILES rather than from a distance-based guess -- so the job here is to
ask RDKit what it already knows, not to re-derive chemistry from coordinates.
The atom typing is RDKit's own feature factory (BSD-3-Clause, ships with
RDKit); the charged groups are a short SMARTS table below.  Nothing here is
taken from PLIP, which is the whole point: this package stays permissive.

The coordinates handed on are the **3D** ones.  Detection is geometric and runs
on the crystallographic pose; ``coords_2d`` is a drawing, and measuring
distances in it would invent contacts the structure does not have.

Two things RDKit's features get right and a naive reading of the graph does
not, and one it gets wrong for our purposes:

* a Donor feature marks an atom *type*, so a tertiary amine matches even though
  it has no hydrogen to donate -- we re-check ``GetTotalNumHs()``;
* a LumpedHydrophobe is a ring centroid, so it arrives as a multi-atom feature
  that has to be expanded before it means anything per atom;
* a ZnBinder feature spans the whole chelating group including its carbon,
  and a carbon does not coordinate a metal -- only the lone-pair donors do.
"""
from __future__ import annotations

import os

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

from ..chem import LigandGeometry
from .roles import Group, Perception, Site

#: Built once: parsing the fdef costs more than perceiving a ligand does, and
#: the factory is stateless, so there is no reason to pay for it per call.
_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)

#: Elements that can coordinate a metal through a lone pair.  ``ZnBinder``
#: features cover the whole chelating group, so they are filtered to these.
_LONE_PAIR_DONORS = frozenset({7, 8, 16, 34})  # N, O, S, Se

#: Anything outside this set counts as a metal.  Cofactor ligands do turn up
#: with an iron or a zinc in the HETATM block, and a metal atom is a partner
#: for the other side's lone pairs rather than an atom with roles of its own.
_NONMETALS = frozenset(
    "H He B C N O F Ne Si P S Cl Ar As Se Br Kr Te I Xe At Rn".split()
)

#: Halogen-bond donors: the sigma hole only gets deep enough to matter on the
#: heavy halogens.  Fluorine is deliberately absent -- it is too electronegative
#: and too small to develop a positive cap, and C-F contacts that look like
#: halogen bonds are electrostatics, not sigma-hole donation.
_HALOGENS = frozenset({"Cl", "Br", "I"})

# ---------------------------------------------------------------------------
# Charged groups
#
# A formal charge alone does not locate the charge: a carboxylate drawn as
# C(=O)[O-] carries its -1 on one oxygen while the ion is the whole group, and
# an amidinium spreads +1 over two nitrogens that are equivalent in the crystal.
# Salt bridges are measured centroid to centroid, so the group is what matters.
#
# ponytail: this is a pH-7 guess made from the connectivity, and it has no way
# to see an unusual pKa (an aniline pushed basic by its substituents, a phenol
# pulled acidic by four nitro groups), a tautomer that moves the proton, or a
# residue-perturbed pKa in the pocket itself.  The upgrade path is an explicit
# protonation step -- run the ligand through a pKa predictor and honour the
# formal charges it assigns -- at which point these tables become the fallback
# for ligands nobody protonated rather than the primary answer.
# ---------------------------------------------------------------------------

#: Groups anionic at physiological pH.  The hydroxyl forms (``OX2H1``) are
#: listed alongside the deprotonated ones because a SMILES is usually drawn
#: neutral: a carboxylic acid at pH 7 is a carboxylate whatever the file says.
_ANION_SMARTS = {
    "carboxylate": "[CX3](=O)[OX1H0-,OX2H1]",
    "phosphate": "[PX4](=O)([OX1-,OX2H1])[OX1-,OX2H1]",
    # A phosphate mono- or diester keeps only one ionisable oxygen -- the others
    # are bridging -- and its first pKa is still near 1, so the alpha and beta
    # phosphates of a nucleotide triphosphate are as anionic as the terminal
    # one.  Without this the largest anion in the reference set goes unseen.
    "phosphate_ester": "[PX4](=[OX1])[OX1-,OX2H1]",
    "sulfonate": "[SX4](=O)(=O)[OX1-,OX2H1]",
    "tetrazolate": "c1nnn[nH,n-]1",
}

#: Groups cationic at physiological pH.  ``!$(N-[a])`` and ``!$(N-[CX3])``
#: between them keep the amine pattern to *aliphatic* amines: an aniline
#: (pKa ~4.6), an aminopyrimidine, an amide and an enamine all put their lone
#: pair into a pi system and none of them is protonated at pH 7.  The imine
#: nitrogen is written ``[NX2,NX3+]`` so that an amidine drawn already
#: protonated -- which is how an arginine mimetic usually arrives -- matches as
#: the whole delocalised group rather than as the one atom carrying the charge.
_CATION_SMARTS = {
    "quaternary_amine": "[NX4+]",
    "aliphatic_amine": "[NX3;!$(N-[!#6]);!$(N-[a]);!$(N-[CX3])]",
    "amidine": "[CX3](=[NX2,NX3+])[NX3]",
    "guanidine": "[NX3][CX3](=[NX2,NX3+])[NX3]",
}

_ANION_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in _ANION_SMARTS.items()}
_CATION_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in _CATION_SMARTS.items()}


def _hydrophobic(atom: Chem.Atom) -> bool:
    """The classic hydrophobic-contact atom: carbon with nothing polar on it.

    RDKit's Hydrophobe families are broader than that -- they include ether and
    thioether carbons, which are not what a hydrophobic contact means here -- so
    a feature hit is a candidate and this is the filter.  Hydrogens are implicit
    on ``mol``, so every neighbour returned is a heavy atom.
    """
    return atom.GetAtomicNum() == 6 and all(
        n.GetAtomicNum() == 6 for n in atom.GetNeighbors()
    )


def _feature_atoms(mol: Chem.Mol) -> dict[str, set[int]]:
    """Feature family -> the atoms it covers, multi-atom features expanded."""
    found: dict[str, set[int]] = {}
    for feature in _FACTORY.GetFeaturesForMol(mol):
        found.setdefault(feature.GetFamily(), set()).update(feature.GetAtomIds())
    return found


def _aromatic_groups(mol: Chem.Mol, coords: dict[int, np.ndarray]) -> list[Group]:
    """One group per aromatic ring, with the unit normal of its bound pose.

    Rings are reported individually rather than merged per fused system because
    stacking is ring-to-ring: an indole stacks through one of its two rings and
    the centroid of the pair sits between them, where nothing is.
    """
    groups = []
    for ring in mol.GetRingInfo().AtomRings():
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        xyz = np.array([coords[i] for i in ring])
        # The plane of best fit through the ring; its normal is the singular
        # vector for the smallest singular value, which is unit by construction.
        normal = np.linalg.svd(xyz - xyz.mean(axis=0))[2][2]
        groups.append(Group(kind="aromatic", atoms=tuple(ring), normal=normal))
    return groups


def _anion_charge_atoms(mol: Chem.Mol, match: tuple[int, ...]) -> set[int]:
    """The atoms of an anionic match that actually share the charge.

    The centroid of the group is where the salt bridge is measured from, so
    membership decides geometry and not just bookkeeping.  A carboxylate carbon
    is sp2 and formally neutral, sitting between the two oxygens that carry the
    charge, so it is dropped.  A phosphorus or a sulfur is the centre of a
    delocalised oxo-anion and stays, together with its whole oxygen shell --
    which is what pulls in the bridging esters of a nucleotide phosphate that
    the SMARTS itself does not name.
    """
    atoms = set(match)
    for idx in tuple(atoms):
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() in (15, 16):
            atoms.update(n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 8)
        elif atom.GetAtomicNum() == 6:
            atoms.discard(idx)
    return atoms


def _charged_groups(mol: Chem.Mol) -> list[Group]:
    """Cationic and anionic groups, SMARTS first and formal charge as backstop."""
    found: dict[frozenset[int], str] = {}
    for kind, patterns in (("anion", _ANION_PATTERNS), ("cation", _CATION_PATTERNS)):
        for pattern in patterns.values():
            for match in mol.GetSubstructMatches(pattern):
                atoms = _anion_charge_atoms(mol, match) if kind == "anion" else set(match)
                if atoms:
                    found.setdefault(frozenset(atoms), kind)

    # A guanidine also matches the amidine pattern, and a fully protonated
    # phosphate matches both phosphate patterns twice over.  Keeping only the
    # maximal sets leaves one group per chemical unit.
    groups = [
        Group(kind=kind, atoms=tuple(sorted(atoms)))
        for atoms, kind in found.items()
        if not any(atoms < other for other in found)
    ]

    # Whatever the tables missed but the SMILES was explicit about: a charged
    # atom nobody claimed is its own group.  This is what catches the oddities
    # -- a persulfide, a boronate, an N-oxide.  Metals are left out on purpose:
    # a bound zinc is always cationic and would otherwise be reported as a salt
    # bridge on top of the coordination that already describes it.
    covered = {atom for group in groups for atom in group.atoms}
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge and atom.GetIdx() not in covered and atom.GetSymbol() in _NONMETALS:
            groups.append(
                Group(
                    kind="cation" if charge > 0 else "anion",
                    atoms=(atom.GetIdx(),),
                )
            )
    return groups


def perceive_ligand(geom: LigandGeometry) -> Perception:
    """Chemical roles of the ligand's heavy atoms, keyed by RDKit atom index.

    The returned :class:`~ms_contactmap.detect.roles.Perception` is in the same
    index space as ``geom.mol``, so
    :attr:`~ms_contactmap.chem.LigandGeometry.idx_to_serial` translates it back to
    the PDB when a caller needs to name an atom.
    """
    mol = geom.mol
    coords = {i: np.array(xyz, dtype=float) for i, xyz in enumerate(geom.coords_3d)}
    features = _feature_atoms(mol)

    donors = features.get("Donor", set())
    acceptors = features.get("Acceptor", set())
    binders = features.get("ZnBinder", set())

    sites: dict[int, Site] = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        charge = atom.GetFormalCharge()
        neighbours = tuple(sorted(n.GetIdx() for n in atom.GetNeighbors()))
        if atom.GetSymbol() not in _NONMETALS:
            # A metal has no roles of its own: it is the partner the other
            # side's lone pairs point at, and every geometric test treats it
            # that way.  Flagging it as an acceptor too would double-count.
            sites[idx] = Site(is_metal=True, neighbours=neighbours)
            continue
        sites[idx] = Site(
            # The Donor family types the atom, not its protonation: a tertiary
            # amine and an ether oxygen both match and neither has an H to give.
            donor=idx in donors and atom.GetTotalNumHs() > 0,
            # RDKit's feature table intentionally does not classify every
            # isolated oxyanion (notably terminal phosphate O-) as Acceptor.
            # Chemically it still accepts hydrogen bonds and coordinates Mg.
            acceptor=(
                idx in acceptors
                or (charge < 0 and atom.GetAtomicNum() in _LONE_PAIR_DONORS)
            ),
            # The published atom criterion is complete on its own; using an
            # RDKit Hydrophobe feature as an additional gate misses valid
            # aliphatic carbons in some ring and fused-system environments.
            hydrophobic=_hydrophobic(atom),
            # ZnBinder marks the sidechain-quality zinc ligands; metals in
            # general are coordinated by any lone pair, which is why an
            # acceptor qualifies whether or not the feature factory says so --
            # ANP's phosphate oxygens hold a Mg and match no ZnBinder pattern.
            metal_binder=(
                (idx in binders or idx in acceptors or charge < 0)
                and atom.GetAtomicNum() in _LONE_PAIR_DONORS
                and charge <= 0
            ),
            halogen=(
                atom.GetSymbol() in _HALOGENS
                and len(neighbours) == 1
                and mol.GetAtomWithIdx(neighbours[0]).GetAtomicNum() == 6
            ),
            neighbours=neighbours,
            hydrogens=tuple(geom.donor_hydrogens.get(idx, ())),
        )

    return Perception(
        sites=sites,
        coords=coords,
        groups=_aromatic_groups(mol, coords) + _charged_groups(mol),
    )


def _self_check() -> None:
    """Perceive the four reference ligands.  ``python -m ms_contactmap.detect.ligand``"""
    import json
    from pathlib import Path

    from ms_contactmap.chem import load_ligand

    root = Path(__file__).resolve().parent.parent.parent
    ligands = json.loads((root / "data" / "ligands.json").read_text())
    hydroxyl = Chem.MolFromSmarts("[OX2H1][CX4]")

    perceptions = {}
    for stem, resname in (("2gfk", "VII"), ("4ps5", "2TA"), ("4uwh", "JXM"),
                          ("6wak", "ANP")):
        geom = load_ligand(root / "data" / f"{stem}.pdb", resname,
                           ligands[resname]["smiles"])
        p = perceive_ligand(geom)
        perceptions[resname] = (geom, p)

        n = geom.mol.GetNumAtoms()
        assert len(p.coords) == n, f"{resname}: {len(p.coords)} coords for {n} atoms"
        assert len(p.sites) == n, f"{resname}: {len(p.sites)} sites for {n} atoms"

        for idx, site in p.sites.items():
            if site.is_metal:
                assert not (site.donor or site.acceptor or site.hydrophobic
                            or site.metal_binder or site.halogen), \
                    f"{resname}: metal atom {idx} also has ligand roles"
            assert not (site.hydrophobic and geom.mol.GetAtomWithIdx(idx).GetSymbol()
                        != "C"), f"{resname}: non-carbon {idx} is hydrophobic"

        for group in p.groups:
            assert group.kind in ("aromatic", "cation", "anion"), group.kind
            assert group.atoms, f"{resname}: empty {group.kind} group"
            assert all(0 <= a < n for a in group.atoms), \
                f"{resname}: {group.kind} group out of range {group.atoms}"
            assert not any(p.sites[a].is_metal for a in group.atoms), \
                f"{resname}: a metal was put in a {group.kind} group"
            if group.kind == "aromatic":
                assert abs(np.linalg.norm(group.normal) - 1.0) < 1e-9, \
                    f"{resname}: ring normal is not a unit vector"
                assert all(geom.mol.GetAtomWithIdx(a).GetIsAromatic()
                           for a in group.atoms), \
                    f"{resname}: non-aromatic atom in an aromatic group"
            else:
                assert group.normal is None, f"{resname}: charged group has a normal"

        kinds = [g.kind for g in p.groups]
        print(f"  {stem}/{resname:4s} {n:3d} atoms   "
              f"donors {sum(s.donor for s in p.sites.values()):2d}   "
              f"acceptors {sum(s.acceptor for s in p.sites.values()):2d}   "
              f"hydrophobes {sum(s.hydrophobic for s in p.sites.values()):2d}   "
              f"rings {kinds.count('aromatic')}   "
              f"charged {kinds.count('cation')}+ {kinds.count('anion')}-")

    # ANP is an ATP analogue: three phosphates and an adenine.  The phosphates
    # are the reason a nucleotide sits where it does, so missing them is not a
    # rounding error in the diagram, it is the wrong diagram.
    geom, p = perceptions["ANP"]
    anions = [g for g in p.groups if g.kind == "anion"]
    assert len(anions) >= 2, f"ANP: {len(anions)} anion groups, expected the phosphates"
    assert any(any(geom.mol.GetAtomWithIdx(a).GetSymbol() == "P" for a in g.atoms)
               for g in anions), "ANP: no anion group is a phosphate"
    assert sum(g.kind == "aromatic" for g in p.groups) >= 1, "ANP: adenine lost"
    hydroxyls = [m[0] for m in geom.mol.GetSubstructMatches(hydroxyl)]
    assert hydroxyls, "ANP: the ribose hydroxyls are gone"
    for o in hydroxyls:
        assert p.sites[o].donor and p.sites[o].acceptor, \
            f"ANP: hydroxyl {o} is not both donor and acceptor"

    # 2TA's tert-butyl is the textbook hydrophobe; its sulfonamide and ether
    # oxygens are the textbook thing that must never be called one.
    geom, p = perceptions["2TA"]
    tbu = geom.mol.GetSubstructMatches(Chem.MolFromSmarts("[CH3][CX4]([CH3])[CH3]"))
    assert tbu, "2TA: no tert-butyl found"
    assert p.sites[tbu[0][0]].hydrophobic, "2TA: a tert-butyl methyl is not hydrophobic"
    assert not any(p.sites[a.GetIdx()].hydrophobic
                   for a in geom.mol.GetAtoms() if a.GetSymbol() == "O"), \
        "2TA: an oxygen is hydrophobic"

    print("\nligand self-check passed")


if __name__ == "__main__":
    _self_check()
