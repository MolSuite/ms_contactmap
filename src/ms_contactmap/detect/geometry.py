"""The geometric half of the detector: coordinates in, :class:`Hit` list out.

This module knows no chemistry.  Everything it needs to decide has already been
decided by perception and handed over as booleans on a :class:`~.roles.Site` or
as a :class:`~.roles.Group` with a centroid and a normal, which is why the same
code runs unchanged over a PDB protein and an RDKit ligand whose atom ids mean
entirely different things.

Every threshold below is a published number with its source in a trailing
comment.  That provenance is the point: a measured cutoff is a fact, facts are
not copyrightable, and citing them is how this package states that it reimplements
the literature rather than a GPL-2.0 program.  If a number here ever changes,
change the citation with it or the argument stops holding.

Two approximations run through the whole file and are worth knowing before
reading any single predicate:

* Crystal structures carry no hydrogens, so every directional test that wants a
  D-H vector uses the donor's *free direction* instead -- away from the mean of
  its bonded neighbours.  See :func:`_donates_towards`.
* Searches are KD-tree neighbour queries, never full pairwise matrices.  A
  binding site is a handful of atoms inside a protein of tens of thousands, and
  the cost should scale with the former.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .roles import Group, Hit, Perception

#: Below this, two "atoms" are a clash or the same atom counted twice by an
#: altloc or a symmetry mate.  Nothing real is a contact at this range.
MIN_SEPARATION = 0.5

#: Twice Bondi's 1.7 A carbon van der Waals radius (J. Phys. Chem. 68:441, 1964)
#: plus crystallographic slack.  The weakest-sourced number in the file: the
#: hydrophobic-contact literature reports no single agreed cutoff.
HYDROPHOBIC_MAX_DIST = 4.0

#: Strong-contact profile.  These are the defaults of MDAnalysis'
#: HydrogenBondAnalysis: D...A <= 3.0 A and D-H...A >= 150 degrees.  PLIP's
#: 4.1 A / 100 degree limits are intentionally permissive candidate criteria;
#: using them as the final display rule was what admitted the weak 4UWH
#: LYS636 and ASP761 contacts.
HBOND_MAX_DIST = 3.0
HBOND_MIN_DONOR_ANGLE = 150.0
#: Complement used internally by the hydrogen-free direction estimate.
HBOND_MAX_H_ANGLE = 180.0 - HBOND_MIN_DONOR_ANGLE
#: Where a hydrogen sits relative to its donor's one bond when that bond is the
#: only thing holding it -- a hydroxyl or a thiol.  It rotates on this cone, so
#: the free direction says nothing about it; see :func:`_donates_towards`.
TETRAHEDRAL_ANGLE = 109.5

#: Barlow & Thornton, J. Mol. Biol. 168:867 (1983), relaxed by 1.5 A.
SALT_BRIDGE_MAX_DIST = 5.5

#: McGaughey, Gagne & Rappe, J. Biol. Chem. 273:15458 (1998).
PI_STACK_MAX_DIST = 5.5
#: Same source: parallel (P) stacking near 0 degrees, perpendicular (T) near 90.
PI_STACK_ANGLE_TOL = 30.0
#: Rings that are close but slid past each other do not stack; the in-plane
#: component of the centroid-to-centroid vector is what tells them apart.
PI_MAX_OFFSET = 2.0

#: Gallivan & Dougherty, PNAS 96:9459 (1999).
PI_CATION_MAX_DIST = 6.0

#: Auffinger, Hays, Westhof & Ho, PNAS 101:16789 (2004), relaxed by 0.5 A.
HALOGEN_MAX_DIST = 4.0
#: Same source: the sigma hole lies opposite the C-X bond, and the acceptor
#: presents its lone pair at roughly a tetrahedral angle.
HALOGEN_DONOR_ANGLE = 165.0
HALOGEN_ACCEPTOR_ANGLE = 120.0
HALOGEN_ANGLE_TOL = 30.0

#: Harding, Acta Crystallogr. D57:401 (2001).
METAL_MAX_DIST = 3.0
#: Element-specific first-shell limits applied after the broad neighbour query.
#: Harding's survey places protein Zn--donor bonds around 2.0--2.3 A; Mg--O
#: first-shell distances occupy the same narrow band.  The generic value above
#: remains only a cheap candidate-query radius, never a reported bond cutoff.
METAL_FIRST_SHELL_MAX = {"ZN": 2.25, "MG": 2.30}
METAL_EXPECTED_COORDINATION = {"ZN": 4, "MG": 6}
#: Not chemistry: the diagram sets this one.  A metal that coordinates nothing
#: near the ligand still has a coordination sphere, and reporting it would put
#: a cluster of residues in the picture with no path to the ligand.  Tune it if
#: the drawing wants more or less of the second shell.
METAL_LIGAND_CUTOFF = 6.0

#: Jiang et al. (2005), the survey this whole predicate is shaped by.
WATER_BRIDGE_MIN_DIST = 2.5
WATER_BRIDGE_MAX_DIST = 3.0
#: Angle subtended at the bridging oxygen by its two partners.
WATER_OMEGA_MIN = 75.0
WATER_OMEGA_MAX = 140.0
#: Where a water's two lone pairs actually point.  The band above says which
#: bridges are possible; this says which one a water with several candidates is
#: really holding, and the driver uses it to pick.
WATER_OMEGA_IDEAL = 104.5

#: A hydrophobic contact between two atoms that also make one of these is not
#: a second interaction, it is the same one described worse.
SUBSUMES_HYDROPHOBIC = frozenset({"hbond", "salt_bridge", "pi_stacking", "pi_cation"})


def reference_parameters() -> dict[str, float | int | str]:
    """Public detector settings, suitable for reports and JSON provenance."""
    return {
        "hydrophobic_max_distance_angstrom": HYDROPHOBIC_MAX_DIST,
        "hbond_max_distance_angstrom": HBOND_MAX_DIST,
        "hbond_min_donor_angle_degree": HBOND_MIN_DONOR_ANGLE,
        "hbond_profile": "strong_geometric",
        "salt_bridge_max_distance_angstrom": SALT_BRIDGE_MAX_DIST,
        "pi_stacking_max_distance_angstrom": PI_STACK_MAX_DIST,
        "pi_stacking_max_angle_deviation_degree": PI_STACK_ANGLE_TOL,
        "pi_max_offset_angstrom": PI_MAX_OFFSET,
        "pi_cation_max_distance_angstrom": PI_CATION_MAX_DIST,
        "halogen_max_distance_angstrom": HALOGEN_MAX_DIST,
        "halogen_donor_angle_degree": HALOGEN_DONOR_ANGLE,
        "halogen_acceptor_angle_degree": HALOGEN_ACCEPTOR_ANGLE,
        "halogen_max_angle_deviation_degree": HALOGEN_ANGLE_TOL,
        "metal_max_distance_angstrom": METAL_MAX_DIST,
        "zinc_first_shell_max_distance_angstrom": METAL_FIRST_SHELL_MAX["ZN"],
        "magnesium_first_shell_max_distance_angstrom": METAL_FIRST_SHELL_MAX["MG"],
        "zinc_expected_coordination": METAL_EXPECTED_COORDINATION["ZN"],
        "magnesium_expected_coordination": METAL_EXPECTED_COORDINATION["MG"],
        "water_bridge_min_distance_angstrom": WATER_BRIDGE_MIN_DIST,
        "water_bridge_max_distance_angstrom": WATER_BRIDGE_MAX_DIST,
        "water_bridge_min_omega_degree": WATER_OMEGA_MIN,
        "water_bridge_max_omega_degree": WATER_OMEGA_MAX,
    }


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray | None:
    """``v`` scaled to length one, or ``None`` if it has no direction."""
    v = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-9 else None


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two vectors in degrees, in ``[0, 180]``.

    Returns NaN when either vector is degenerate.  NaN compares false against
    everything, so a degenerate geometry silently fails every test it reaches
    instead of raising in the middle of a scan.
    """
    a, b = _unit(u), _unit(v)
    if a is None or b is None:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))


def _acute(angle: float) -> float:
    """``angle`` folded into ``[0, 90]``, for the undirected normals of rings."""
    return min(angle, 180.0 - angle)


def _xyz(p: Perception, ids: list[int]) -> np.ndarray:
    """``(n, 3)`` for ``ids``, shaped even when empty so cKDTree accepts it."""
    return np.array([p.coords[i] for i in ids], dtype=float).reshape(-1, 3)


def _atoms(p: Perception, pred, *, solvent: bool = False) -> list[int]:
    """Ids whose :class:`~.roles.Site` satisfies ``pred``.

    Waters and metal ions are excluded unless ``solvent`` says otherwise: they
    sit in the protein's perception so that the water-bridge and metal
    predicates can reach them, and letting them fall into the ordinary tests
    would report every ligand-water hydrogen bond as a protein contact.
    """
    return [
        i for i, s in p.sites.items()
        if pred(s) and (solvent or not (s.is_water or s.is_metal))
    ]


def _pairs_within(a: Perception, a_ids: list[int], b: Perception,
                  b_ids: list[int], cutoff: float):
    """Yield ``(a_id, b_id, distance)`` for every pair closer than ``cutoff``."""
    if not a_ids or not b_ids:
        return
    a_xyz, b_xyz = _xyz(a, a_ids), _xyz(b, b_ids)
    neighbours = cKDTree(a_xyz).query_ball_tree(cKDTree(b_xyz), cutoff)
    for i, js in enumerate(neighbours):
        for j in js:
            dist = float(np.linalg.norm(a_xyz[i] - b_xyz[j]))
            if dist >= MIN_SEPARATION:
                yield a_ids[i], b_ids[j], dist


def _group_pairs(a: Perception, a_kind: str, b: Perception, b_kind: str,
                 cutoff: float):
    """Yield ``(group_a, group_b, distance, centroid_a, centroid_b)``.

    Groups are addressed by their centroid, so this is the same neighbour
    search as :func:`_pairs_within` over a handful of derived points.
    """
    a_groups = [g for g in a.groups if g.kind == a_kind]
    b_groups = [g for g in b.groups if g.kind == b_kind]
    if not a_groups or not b_groups:
        return
    a_cent = np.array([a.centroid(g) for g in a_groups], dtype=float).reshape(-1, 3)
    b_cent = np.array([b.centroid(g) for g in b_groups], dtype=float).reshape(-1, 3)
    neighbours = cKDTree(a_cent).query_ball_tree(cKDTree(b_cent), cutoff)
    for i, js in enumerate(neighbours):
        for j in js:
            dist = float(np.linalg.norm(a_cent[i] - b_cent[j]))
            if dist >= MIN_SEPARATION:
                yield a_groups[i], b_groups[j], dist, a_cent[i], b_cent[j]


def _free_direction(p: Perception, atom: int) -> np.ndarray | None:
    """Unit vector away from ``atom``'s substituents, or ``None`` if it has none."""
    others = [p.coords[n] for n in p.sites[atom].neighbours if n in p.coords]
    if not others:
        return None
    return _unit(p.coords[atom] - np.mean(others, axis=0))


def _donor_angle(p: Perception, donor: int, target: np.ndarray) -> float | None:
    """Best estimated D-H...A angle, or ``None`` when it cannot qualify.

    A donor with no bonded neighbours -- a water oxygen -- has no free
    direction and so no angle to test; it passes.

    A donor with exactly one -- a hydroxyl or a thiol -- has a hydrogen that
    rotates freely about that single bond, so its H is not near the free
    direction at all: it sweeps a cone one tetrahedral angle away from the
    bond.  Testing such a donor against the free direction as if the H sat
    there rejects real bonds, which is what cost 6wak's HOH 9920 the bridge
    PLIP reports.  For those the test is on the bond instead, and it only asks
    that the target is far enough off it for some rotamer to reach.

    # ponytail: this is the ceiling of a hydrogen-free model.  The real test is
    # the D-H...A angle, and with explicit hydrogens it would be exact rather
    # than approximate; the free direction is only the average of where the
    # hydrogens could be, so a donor with two H (an -NH2) or a rotatable one
    # (a hydroxyl) is served worst.  The upgrade path is to protonate with
    # RDKit or reduce/openbabel in perception, store the H coordinates on the
    # Site, and replace this function with the exact angle -- the callers and
    # the threshold constant stay as they are.
    """
    explicit = p.sites[donor].hydrogens
    if explicit:
        angles = [
            _angle(p.coords[donor] - np.asarray(hydrogen, dtype=float),
                   np.asarray(target, dtype=float) - np.asarray(hydrogen, dtype=float))
            for hydrogen in explicit
        ]
        best = max(angles)
        return best if best >= 180.0 - HBOND_MAX_H_ANGLE else None

    h = _free_direction(p, donor)
    if h is None:
        return 180.0
    towards = np.asarray(target, dtype=float) - p.coords[donor]
    neighbours = p.sites[donor].neighbours
    if len(neighbours) == 1:
        bond = p.coords[neighbours[0]] - p.coords[donor]
        deviation = abs(_angle(bond, towards) - TETRAHEDRAL_ANGLE)
    else:
        deviation = _angle(h, towards)
    return 180.0 - deviation if deviation <= HBOND_MAX_H_ANGLE else None


def _donates_towards(p: Perception, donor: int, target: np.ndarray) -> bool:
    """Whether ``donor`` can satisfy the published donor-angle threshold."""
    return _donor_angle(p, donor, target) is not None


def _offset(between: np.ndarray, normal: np.ndarray) -> float:
    """Length of ``between`` projected into the plane of ``normal``.

    Zero when one ring sits squarely over the other, and it grows as they slide
    apart, which is exactly the quantity the stacking criterion bounds.
    """
    n = _unit(normal)
    if n is None:
        return float("inf")
    return float(np.linalg.norm(between - float(np.dot(between, n)) * n))


# --------------------------------------------------------------------------
# the eight predicates
# --------------------------------------------------------------------------

def hydrophobic_contacts(lig: Perception, prot: Perception) -> list[Hit]:
    """Two hydrophobic atoms within :data:`HYDROPHOBIC_MAX_DIST`.

    Every qualifying pair is emitted; see :func:`detect_contacts` on why the
    collapsing is somebody else's job.
    """
    return [
        Hit("hydrophobic", (p,), (l,), dist)
        for l, p, dist in _pairs_within(
            lig, _atoms(lig, lambda s: s.hydrophobic),
            prot, _atoms(prot, lambda s: s.hydrophobic),
            HYDROPHOBIC_MAX_DIST,
        )
    ]


def hydrogen_bonds(lig: Perception, prot: Perception) -> list[Hit]:
    """Donor-acceptor pairs in range and roughly aimed at each other.

    Both directions are searched, and ``ligand_is_donor`` records which side
    brought the hydrogen -- the diagram draws the arrow from it.
    """
    hits: list[Hit] = []
    for src, dst, lig_donates in ((lig, prot, True), (prot, lig, False)):
        donors = _atoms(src, lambda s: s.donor)
        acceptors = _atoms(dst, lambda s: s.acceptor)
        for d, a, dist in _pairs_within(src, donors, dst, acceptors, HBOND_MAX_DIST):
            donor_angle = _donor_angle(src, d, dst.coords[a])
            if donor_angle is None:
                continue
            hits.append(Hit(
                "hbond",
                protein_atoms=(a,) if lig_donates else (d,),
                ligand_atoms=(d,) if lig_donates else (a,),
                distance=dist,
                ligand_is_donor=lig_donates,
                angle=donor_angle,
            ))
    return hits


def salt_bridges(lig: Perception, prot: Perception) -> list[Hit]:
    """Oppositely charged groups whose centroids are within range.

    Charge is a property of the group, not of one atom, so both sides of the
    :class:`Hit` carry the whole group and the distance is centroid to centroid.
    """
    hits: list[Hit] = []
    for lig_kind, prot_kind, lig_positive in (("cation", "anion", True),
                                              ("anion", "cation", False)):
        for lg, pg, dist, _, _ in _group_pairs(lig, lig_kind, prot, prot_kind,
                                               SALT_BRIDGE_MAX_DIST):
            hits.append(Hit("salt_bridge", pg.atoms, lg.atoms, dist, lig_positive))
    return hits


def pi_stacking(lig: Perception, prot: Perception) -> list[Hit]:
    """Aromatic rings stacked face to face (P) or edge to face (T).

    Distance alone accepts rings that merely pass near one another, so the
    in-plane offset has to be small as well: measured in both ring planes, the
    smaller of the two is what counts, since a T-stacked pair is offset in one
    plane by construction.
    """
    hits: list[Hit] = []
    for lg, pg, dist, lc, pc in _group_pairs(lig, "aromatic", prot, "aromatic",
                                             PI_STACK_MAX_DIST):
        theta = _acute(_angle(lg.normal, pg.normal))
        parallel = theta <= PI_STACK_ANGLE_TOL
        perpendicular = theta >= 90.0 - PI_STACK_ANGLE_TOL
        if not (parallel or perpendicular):
            continue
        between = pc - lc
        if min(_offset(between, lg.normal), _offset(between, pg.normal)) > PI_MAX_OFFSET:
            continue
        hits.append(Hit("pi_stacking", pg.atoms, lg.atoms, dist))
    return hits


def pi_cation(lig: Perception, prot: Perception) -> list[Hit]:
    """A cationic group over the face of an aromatic ring.

    The offset test is against the ring plane only -- a cation beside the ring
    rather than above it meets no pi system.
    """
    hits: list[Hit] = []
    for cat_side, ring_side, cation_kind, ring_kind, lig_positive in (
        (lig, prot, "cation", "aromatic", True),
        (prot, lig, "cation", "aromatic", False),
    ):
        for cg, rg, dist, cc, rc in _group_pairs(cat_side, cation_kind,
                                                 ring_side, ring_kind,
                                                 PI_CATION_MAX_DIST):
            if _offset(cc - rc, rg.normal) > PI_MAX_OFFSET:
                continue
            hits.append(Hit(
                "pi_cation",
                protein_atoms=rg.atoms if lig_positive else cg.atoms,
                ligand_atoms=cg.atoms if lig_positive else rg.atoms,
                distance=dist,
                ligand_is_donor=lig_positive,
            ))
    return hits


def halogen_bonds(lig: Perception, prot: Perception) -> list[Hit]:
    """A ligand halogen presenting its sigma hole to a protein acceptor.

    Both angles matter and both are approximate: the donor angle is exact,
    since the C-X bond is real connectivity, while the acceptor angle uses the
    mean of the acceptor's neighbours and is skipped when it has none.

    Only the ligand-donates direction is searched.  Protein halogens exist but
    are not in the twenty standard residues, and a modified one would need
    perception to flag it before there were anything here to find.
    """
    hits: list[Hit] = []
    halogens = _atoms(lig, lambda s: s.halogen)
    acceptors = _atoms(prot, lambda s: s.acceptor)
    for x, a, dist in _pairs_within(lig, halogens, prot, acceptors, HALOGEN_MAX_DIST):
        towards_acceptor = prot.coords[a] - lig.coords[x]
        carbon = _free_direction(lig, x)
        if carbon is not None:
            # The free direction points away from the carbon, so the published
            # C-X...A angle is its complement.
            if abs((180.0 - _angle(carbon, towards_acceptor)) - HALOGEN_DONOR_ANGLE) > HALOGEN_ANGLE_TOL:
                continue
        lone_pair = _free_direction(prot, a)
        if lone_pair is not None:
            if abs((180.0 - _angle(lone_pair, -towards_acceptor)) - HALOGEN_ACCEPTOR_ANGLE) > HALOGEN_ANGLE_TOL:
                continue
        hits.append(Hit("halogen_bond", (a,), (x,), dist))
    return hits


def metal_coordination(lig: Perception, prot: Perception) -> list[Hit]:
    """Every lone-pair donor inside a metal ion's coordination sphere.

    ``protein_atoms[0]`` is always the metal's serial; the driver reads exactly
    that.  A leg to a ligand atom is ``protein_atoms=(metal,)`` with the partner
    in ``ligand_atoms``; a leg to a protein or water atom is
    ``protein_atoms=(metal, partner)`` with ``ligand_atoms`` empty.  The second
    shape exists because a coordination sphere only reads correctly when it is
    drawn whole -- how much of it to draw is the driver's call, not ours.

    Ions live in the protein's perception (they are HETATMs of the receptor
    file), so there is no ligand-side-metal branch here.  A ligand that carries
    its own metal would need perception to say so first, and the convention
    above would have to grow a third shape before this could handle it.

    Only ions within :data:`METAL_LIGAND_CUTOFF` of a ligand atom are reported:
    a distant one coordinates a piece of protein the diagram never shows.

    ``ligand_is_donor`` is always ``False``.  The ligand does donate the lone
    pair, but the diagram's arrow points at the metal, and the flag exists to
    orient the arrow.
    """
    binds = lambda s: s.metal_binder and not s.is_metal
    metals = [i for i, s in prot.sites.items() if s.is_metal]
    ligand_ids = lig.ids()
    if not metals or not ligand_ids:
        return []
    near = {m for m, _, _ in _pairs_within(prot, metals, lig, ligand_ids,
                                           METAL_LIGAND_CUTOFF)}
    metals = [m for m in metals if m in near]

    hits = [
        Hit("metal_coordination", (m,), (b,), dist, ligand_is_donor=False)
        for m, b, dist in _pairs_within(prot, metals, lig,
                                        _atoms(lig, binds, solvent=True),
                                        METAL_MAX_DIST)
    ]
    hits += [
        Hit("metal_coordination", (m, b), (), dist, ligand_is_donor=False)
        for m, b, dist in _pairs_within(prot, metals, prot,
                                        _atoms(prot, binds, solvent=True),
                                        METAL_MAX_DIST)
    ]
    return hits


def water_bridges(lig: Perception, prot: Perception) -> list[Hit]:
    """A ligand atom and a protein atom hydrogen-bonded to one water oxygen.

    One :class:`Hit` per (ligand atom, protein atom, water) triple; a water
    with two ligand partners and two protein ones is four bridges, and which
    of them survives into the picture is the driver's decision.  ``distance``
    is the ligand leg, the one the diagram measures.

    Each leg is an ordinary hydrogen bond with a floor as well as a ceiling --
    a water closer than :data:`WATER_BRIDGE_MIN_DIST` is modelled into density
    it does not own -- and the angle at the water keeps the two partners from
    sitting on the same side of it.

    Each leg is classified independently.  This matters for a structural water
    network: water has two donor hydrogens and two accepting lone-pair
    directions, so it can donate to two acceptors at once.  4UWH's HOH2260 is
    the concrete case: TYR670 donates to the water, while the water donates to
    both the ligand carbonyl and ASP644 OD1.
    """
    waters = [i for i, s in prot.sites.items()
              if s.is_water and (s.donor or s.acceptor)]
    if not waters:
        return []
    legs = {}
    for side in (lig, prot):
        found: dict[int, list[tuple[int, float, bool, bool]]] = defaultdict(list)
        polar = _atoms(side, lambda s: s.donor or s.acceptor)
        for w, atom, dist in _pairs_within(prot, waters, side, polar,
                                           WATER_BRIDGE_MAX_DIST):
            if dist < WATER_BRIDGE_MIN_DIST:
                continue
            site = side.sites[atom]
            # The water donates to an acceptor -- and has no free direction to
            # check -- or the partner donates to the water, and then it does.
            donates = site.donor and _donates_towards(side, atom, prot.coords[w])
            if site.acceptor or donates:
                found[w].append((atom, dist, donates, site.acceptor))
        legs[id(side)] = found

    hits: list[Hit] = []
    for w in waters:
        origin = prot.coords[w]
        for l, l_dist, l_don, l_acc in legs[id(lig)].get(w, ()):
            for p, p_dist, p_don, p_acc in legs[id(prot)].get(w, ()):
                omega = _angle(lig.coords[l] - origin, prot.coords[p] - origin)
                if not WATER_OMEGA_MIN <= omega <= WATER_OMEGA_MAX:
                    continue
                hits.append(Hit(
                    "water_bridge", (p,), (l,), l_dist,
                    ligand_is_donor=bool(l_don),
                    water=w, partner_distance=p_dist,
                    protein_is_donor=bool(p_don), angle=omega,
                ))
    return hits


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def _drop_subsumed_hydrophobic(hits: list[Hit]) -> list[Hit]:
    """Remove hydrophobic hits between atoms already joined by something stronger."""
    stronger = {
        (l, p)
        for h in hits if h.kind in SUBSUMES_HYDROPHOBIC
        for l in h.ligand_atoms for p in h.protein_atoms
    }
    return [h for h in hits if h.kind != "hydrophobic"
            or (h.ligand_atoms[0], h.protein_atoms[0]) not in stronger]


def _drop_stacked_pi_cation(hits: list[Hit]) -> list[Hit]:
    """Let stacking win where a cationic ring also stacks with its partner.

    Histidinium and the like are aromatic *and* charged, so the same pair of
    rings can satisfy both criteria; reporting both would double-count one
    contact.
    """
    stacked = [(set(h.ligand_atoms), set(h.protein_atoms))
               for h in hits if h.kind == "pi_stacking"]
    if not stacked:
        return hits
    return [
        h for h in hits
        if h.kind != "pi_cation"
        or not any(sl & set(h.ligand_atoms) and sp & set(h.protein_atoms)
                   for sl, sp in stacked)
    ]


def detect_contacts(lig: Perception, prot: Perception) -> list[Hit]:
    """Every geometric contact between the two sides, at the published criteria.

    The result is deliberately unfiltered in one respect: hydrophobic contacts
    come out one per atom pair, not one per ligand-atom/residue, because this
    module has no idea what a residue is -- it sees serial numbers.  Collapsing
    them belongs to the driver that owns the residue map, so please do not add
    residue logic here; it would need a second, worse copy of that map.

    What is filtered is genuine double-counting: a hydrophobic contact between
    two atoms that also make a hydrogen bond, a salt bridge or a pi interaction
    is dropped, and a ring pair that both stacks and satisfies pi-cation is
    reported as stacking only.
    """
    hits = [
        *hydrogen_bonds(lig, prot),
        *water_bridges(lig, prot),
        *salt_bridges(lig, prot),
        *hydrophobic_contacts(lig, prot),
        *pi_stacking(lig, prot),
        *pi_cation(lig, prot),
        *halogen_bonds(lig, prot),
        *metal_coordination(lig, prot),
    ]
    return _drop_subsumed_hydrophobic(_drop_stacked_pi_cation(hits))


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------

def _self_check() -> None:
    """Hand-placed geometries whose answers are known.  ``python -m ...geometry``

    Synthetic on purpose: perception is being written alongside this module, so
    a check that needed a real PDB would be a check on somebody else's progress.
    Every case here is a number one can verify with a ruler.
    """
    from math import cos, radians, sin

    def build(atoms: dict[int, tuple], groups: list[Group] | None = None) -> Perception:
        """``{id: (xyz, {flag: True})}`` into a Perception."""
        from .roles import Site
        p = Perception(groups=list(groups or []))
        for i, (xyz, flags) in atoms.items():
            p.coords[i] = np.array(xyz, dtype=float)
            p.sites[i] = Site(**flags)
        return p

    def ring(centre, radius=1.4, ids=range(6), z_axis=(0.0, 0.0, 1.0)):
        """Benzene-sized hexagon in the plane normal to ``z_axis``."""
        n = _unit(np.array(z_axis, dtype=float))
        # Any seed not parallel to n will do; picking by n's largest component
        # is what keeps the cross product away from zero for an edge-on ring.
        seed = np.array([0.0, 1.0, 0.0]) if abs(n[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        u = _unit(np.cross(n, seed))
        v = np.cross(n, u)
        centre = np.array(centre, dtype=float)
        atoms = {}
        for k, i in enumerate(ids):
            a = 2.0 * np.pi * k / 6.0
            atoms[i] = (centre + radius * (cos(a) * u + sin(a) * v), {"hydrophobic": True})
        return atoms, Group("aromatic", tuple(ids), normal=n)

    def kinds(hits):
        return sorted(h.kind for h in hits)

    # -- hydrophobic: a distance test and nothing else -----------------------
    for gap, expected in ((3.5, 1), (4.5, 0)):
        l = build({1: ((0.0, 0.0, 0.0), {"hydrophobic": True})})
        p = build({101: ((gap, 0.0, 0.0), {"hydrophobic": True})})
        got = detect_contacts(l, p)
        assert len(got) == expected, f"hydrophobic at {gap} A: {got}"
        if expected:
            assert got[0].kind == "hydrophobic" and abs(got[0].distance - gap) < 1e-9

    # -- H-bond: the free direction decides ----------------------------------
    acceptor = build({101: ((2.9, 0.0, 0.0), {"acceptor": True})})
    aimed = build({1: ((0.0, 0.0, 0.0), {"donor": True, "neighbours": (2, 3)}),
                   2: ((-1.5, 0.5, 0.0), {}), 3: ((-1.5, -0.5, 0.0), {})})
    turned = build({1: ((0.0, 0.0, 0.0), {"donor": True, "neighbours": (2, 3)}),
                    2: ((1.5, 0.5, 0.0), {}), 3: ((1.5, -0.5, 0.0), {})})
    assert kinds(detect_contacts(aimed, acceptor)) == ["hbond"], "donor aimed at acceptor"
    got = detect_contacts(aimed, acceptor)[0]
    assert got.ligand_is_donor and got.protein_atoms == (101,) and got.ligand_atoms == (1,)
    assert detect_contacts(turned, acceptor) == [], "acceptor behind the donor is not a bond"

    # The protein-donates direction is the same test with the sides swapped.
    assert kinds(detect_contacts(build({1: ((0.0, 0.0, 0.0), {"acceptor": True})}),
                                 build({101: ((2.9, 0.0, 0.0),
                                              {"donor": True, "neighbours": (102, 103)}),
                                        102: ((4.5, 0.5, 0.0), {}),
                                        103: ((4.5, -0.5, 0.0), {})}))) == ["hbond"]

    # -- pi-stacking: parallel, then slid sideways ---------------------------
    lig_atoms, lig_ring = ring((0.0, 0.0, 0.0))
    # 4 A sideways puts the centroids past the distance cutoff; 3 A at 3.4 A of
    # height stays in range and is rejected by the offset alone, which is the
    # case the offset test exists for.
    for shift, height, stacks in (((0.0, 0.0), 3.8, True),
                                  ((4.0, 0.0), 3.8, False),
                                  ((3.0, 0.0), 3.4, False)):
        prot_atoms, prot_ring = ring((shift[0], shift[1], height), ids=range(101, 107))
        got = kinds(detect_contacts(build(lig_atoms, [lig_ring]),
                                    build(prot_atoms, [prot_ring])))
        assert ("pi_stacking" in got) is stacks, f"stacking, ring slid {shift}: {got}"
        if stacks:
            assert got == ["pi_stacking"], f"stacking absorbs its own carbons: {got}"

    # T-stacking: the same rings, one turned on edge, centroids 5.0 A apart.
    edge_atoms, edge_ring = ring((0.0, 0.0, 5.0), ids=range(101, 107), z_axis=(1.0, 0.0, 0.0))
    assert "pi_stacking" in kinds(detect_contacts(build(lig_atoms, [lig_ring]),
                                                  build(edge_atoms, [edge_ring]))), "T-stacking"

    # -- pi-cation: a charge over the ring face ------------------------------
    cation = build({1: ((0.0, 0.0, 4.0), {})}, [Group("cation", (1,))])
    prot_atoms, prot_ring = ring((0.0, 0.0, 0.0), ids=range(101, 107))
    got = detect_contacts(cation, build(prot_atoms, [prot_ring]))
    assert kinds(got) == ["pi_cation"], f"cation over a ring: {kinds(got)}"
    assert got[0].ligand_is_donor and got[0].ligand_atoms == (1,)

    # -- salt bridge ---------------------------------------------------------
    got = detect_contacts(build({1: ((0.0, 0.0, 0.0), {})}, [Group("cation", (1,))]),
                          build({101: ((4.0, 0.0, 0.0), {})}, [Group("anion", (101,))]))
    assert kinds(got) == ["salt_bridge"] and got[0].ligand_is_donor

    # -- halogen bond: 165 degrees at the halogen, 120 at the acceptor -------
    a = radians(180.0 - 165.0)
    lig_halogen = build({1: ((0.0, 0.0, 0.0), {"halogen": True, "neighbours": (2,)}),
                         2: ((-1.8, 0.0, 0.0), {})})
    acc_xyz = np.array((3.0 * cos(a), 3.0 * sin(a), 0.0))
    prot_acc = build({101: (acc_xyz, {"acceptor": True, "neighbours": (102,)}),
                      102: (acc_xyz + np.array((1.0, 1.0, 0.0)), {})})
    assert "halogen_bond" in kinds(detect_contacts(lig_halogen, prot_acc)), "halogen bond"
    # Pull the halogen round behind its own carbon and the sigma hole misses.
    bent = build({1: ((0.0, 0.0, 0.0), {"halogen": True, "neighbours": (2,)}),
                  2: ((1.8, 0.2, 0.0), {})})
    assert "halogen_bond" not in kinds(detect_contacts(bent, prot_acc)), "sigma hole points away"

    # -- metal coordination --------------------------------------------------
    metal = build({101: ((0.0, 0.0, 0.0), {"is_metal": True}),
                   102: ((0.0, 2.4, 0.0), {"metal_binder": True, "acceptor": True}),
                   103: ((0.0, 0.0, 2.3), {"metal_binder": True, "acceptor": True,
                                           "donor": True, "is_water": True})})
    got = detect_contacts(build({1: ((2.2, 0.0, 0.0), {"metal_binder": True})}), metal)
    assert kinds(got) == ["metal_coordination"] * 3, f"coordination sphere: {kinds(got)}"
    assert all(h.protein_atoms[0] == 101 for h in got), "protein_atoms[0] is the metal"
    ligand_leg = [h for h in got if h.ligand_atoms]
    assert len(ligand_leg) == 1 and abs(ligand_leg[0].distance - 2.2) < 1e-9
    assert ligand_leg[0].protein_atoms == (101,) and ligand_leg[0].ligand_atoms == (1,)
    assert not ligand_leg[0].ligand_is_donor, "the arrow points at the metal"
    other_legs = sorted(h.protein_atoms for h in got if not h.ligand_atoms)
    assert other_legs == [(101, 102), (101, 103)], f"protein and water legs: {other_legs}"
    # An ion out of the ligand's reach keeps its whole sphere out of the picture.
    assert detect_contacts(build({1: ((9.0, 0.0, 0.0), {"metal_binder": True})}),
                           metal) == [], "distant metals are not the diagram's business"

    # -- water bridge: two legs at 2.9 A, 100 degrees apart ------------------
    omega = radians(100.0)
    water = build({
        101: ((0.0, 0.0, 0.0), {"is_water": True, "donor": True, "acceptor": True}),
        102: ((2.9 * cos(omega), 2.9 * sin(omega), 0.0), {"acceptor": True}),
    })
    donor = build({1: ((2.9, 0.0, 0.0), {"donor": True, "neighbours": (2, 3)}),
                   2: ((4.5, 0.5, 0.0), {}), 3: ((4.5, -0.5, 0.0), {})})
    got = detect_contacts(donor, water)
    assert kinds(got) == ["water_bridge"], f"water bridge: {kinds(got)}"
    assert got[0].water == 101 and abs(got[0].distance - 2.9) < 1e-9
    assert got[0].ligand_atoms == (1,) and got[0].protein_atoms == (102,)
    assert got[0].ligand_is_donor, "the ligand donates its leg"
    # Fold the partners onto the same side of the water and omega rules it out.
    narrow = build({
        101: ((0.0, 0.0, 0.0), {"is_water": True, "donor": True, "acceptor": True}),
        102: ((2.9 * cos(radians(20.0)), 2.9 * sin(radians(20.0)), 0.0), {"acceptor": True}),
    })
    assert "water_bridge" not in kinds(detect_contacts(donor, narrow)), "omega too narrow"

    # -- subsumption: the stronger contact absorbs the hydrophobic one -------
    both = build({1: ((0.0, 0.0, 0.0), {"donor": True, "hydrophobic": True,
                                        "neighbours": (2, 3)}),
                  2: ((-1.5, 0.5, 0.0), {}), 3: ((-1.5, -0.5, 0.0), {})})
    partner = build({101: ((2.9, 0.0, 0.0), {"acceptor": True, "hydrophobic": True})})
    assert kinds(detect_contacts(both, partner)) == ["hbond"], "hydrophobic must be absorbed"
    assert kinds(hydrophobic_contacts(both, partner)) == ["hydrophobic"], \
        "the predicate on its own still reports it"

    # -- the clash floor and the empty case ----------------------------------
    assert detect_contacts(build({1: ((0.0, 0.0, 0.0), {"hydrophobic": True})}),
                           build({101: ((0.2, 0.0, 0.0), {"hydrophobic": True})})) == [], \
        "a duplicate atom is not a contact"
    assert detect_contacts(Perception(), Perception()) == []

    print("geometry self-check passed")


if __name__ == "__main__":
    _self_check()
