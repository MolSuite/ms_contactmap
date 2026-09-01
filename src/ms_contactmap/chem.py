"""Ligand perception: PDB coordinates + a user-supplied SMILES -> a 2D RDKit mol.

The PDB tells us *where* the ligand atoms are; the SMILES tells us *what* they
are.  Neither alone is enough: a HETATM block has no bond orders, and a SMILES
has no pose.  :func:`load_ligand` welds the two together and hands the rest of
the package a :class:`LigandGeometry` that keeps the PDB serial of every atom,
so the detector's atom references can be translated into RDKit atom indices
later.

The SMILES is a trust boundary.  If it does not describe the HETATM block we
refuse to build anything rather than draw a confident, wrong diagram.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdCoordGen, rdDepictor

#: Altloc codes we keep.  Anything else is a minor conformer of an atom we
#: already have, and feeding both copies to ``proximityBonding`` invents bonds.
KEPT_ALTLOCS = frozenset({"", " ", "A"})

#: Viewing directions sampled when projecting the 3D pose: this many rotations
#: about x times the same about y, over a half turn each (a half turn is all
#: there is -- the other half is the same plane seen from behind).  4 x 4 is
#: enough to reach every distinct arrangement the flip search produces on the
#: four reference ligands; finer grids only return duplicates.
_VIEW_GRID = 4
#: How many "clashes" worse than the best depiction a candidate may be and
#: still be offered to the layout.  Above this it is folded, not merely
#: unusual, and no arrangement of residues rescues it.
_SCORE_SLACK = 2.0
#: Candidates handed on.  Each one costs the layout a coarse orientation scan.
_MAX_DEPICTIONS = 4


# ---------------------------------------------------------------------------
# Raw PDB parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PdbAtom:
    """One ATOM/HETATM record, in PDB column order.

    Shared with :mod:`ms_contactmap.interactions`, which needs the protein atoms to
    find pocket-lining residues and to name the waters and metals that
    detection reports only by serial number.
    """

    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resnum: int
    icode: str
    x: float
    y: float
    z: float
    element: str
    hetatm: bool
    line: str

    @property
    def res_id(self) -> tuple[str, int, str]:
        return (self.chain, self.resnum, self.icode)


def read_pdb_atoms(pdb_path) -> list[PdbAtom]:
    """Parse every ATOM/HETATM of the first model, keeping only major altlocs.

    Hydrogens are kept here -- callers that want heavy atoms filter on
    ``element``.  Records with unparsable coordinates are skipped silently;
    they are always malformed vendor output, never something we could use.
    """
    atoms: list[PdbAtom] = []
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tag = line[:6]
            if tag == "ENDMDL":
                break
            if tag not in ("ATOM  ", "HETATM"):
                continue
            altloc = line[16].strip()
            if altloc not in KEPT_ALTLOCS:
                continue
            try:
                atoms.append(
                    PdbAtom(
                        serial=int(line[6:11]),
                        name=line[12:16].strip(),
                        altloc=altloc,
                        resname=line[17:20].strip(),
                        chain=line[21].strip() or "_",
                        resnum=int(line[22:26]),
                        icode=line[26].strip(),
                        x=float(line[30:38]),
                        y=float(line[38:46]),
                        z=float(line[46:54]),
                        element=(line[76:78].strip() or line[12:16].strip()[:1]).upper(),
                        hetatm=tag == "HETATM",
                        line=line.rstrip("\n"),
                    )
                )
            except ValueError:
                continue
    return atoms


# ---------------------------------------------------------------------------
# Ligand geometry
# ---------------------------------------------------------------------------

@dataclass
class LigandGeometry:
    """The ligand as both a chemical graph and two sets of coordinates."""

    #: RDKit ``Mol`` with correct bond orders and a single 2D conformer.
    mol: Chem.Mol
    #: 2D depiction coordinates, one per RDKit atom index.
    coords_2d: list[tuple[float, float]]
    #: Original PDB coordinates, same indexing as ``mol`` atoms.
    coords_3d: list[tuple[float, float, float]]
    #: PDB serial number -> RDKit atom index.  The bridge to the detector's output.
    serial_to_idx: dict[int, int]
    resname: str
    chain: str
    resnum: int
    #: Runner-up depictions of the same pose, same indexing as ``coords_2d``.
    #: The layout optimiser tries them all and keeps whichever leaves the
    #: residues the most room; see :func:`_depictions`.
    alt_coords_2d: list[list[tuple[float, float]]] = field(default_factory=list)
    #: Explicit donor-hydrogen coordinates from the input pose, keyed by the
    #: heavy-atom RDKit index.  Empty when the source omits hydrogens.
    donor_hydrogens: dict[int, list[tuple[float, float, float]]] = field(default_factory=dict)

    @property
    def idx_to_serial(self) -> dict[int, int]:
        return {idx: serial for serial, idx in self.serial_to_idx.items()}


def _formula(symbols) -> str:
    """Hill-order formula from a sequence of element symbols."""
    counts = Counter(s.capitalize() for s in symbols)
    order = [s for s in ("C", "H") if s in counts]
    order += sorted(s for s in counts if s not in ("C", "H"))
    return "".join(s if counts[s] == 1 else f"{s}{counts[s]}" for s in order)


def _pick_copy(atoms: list[PdbAtom], resname: str, chain, resnum) -> list[PdbAtom]:
    """Collect the HETATM records of one ligand copy.

    With several copies in the asymmetric unit and no chain/resnum given we
    take the first in file order and say so on stderr -- silently picking one
    of two binding sites is exactly the kind of thing that makes a diagram
    quietly disagree with the structure it claims to show.
    """
    name = resname.strip().upper()
    copies: dict[tuple[str, int, str], list[PdbAtom]] = {}
    for atom in atoms:
        if not atom.hetatm or atom.resname.upper() != name:
            continue
        if chain is not None and atom.chain != chain:
            continue
        if resnum is not None and atom.resnum != int(resnum):
            continue
        copies.setdefault(atom.res_id, []).append(atom)

    if not copies:
        raise ValueError(
            f"no HETATM records for ligand {name}"
            + (f" chain {chain}" if chain else "")
            + (f" resnum {resnum}" if resnum is not None else "")
        )
    chosen_id = next(iter(copies))
    if len(copies) > 1:
        others = ", ".join(f"{c}:{n}{i}" for c, n, i in list(copies)[1:])
        print(
            f"[ms_contactmap.chem] {name} has {len(copies)} copies; using "
            f"{chosen_id[0]}:{chosen_id[1]}{chosen_id[2]} (also present: {others})",
            file=sys.stderr,
        )
    return copies[chosen_id]


def _depiction_score(mol: Chem.Mol, xyz: np.ndarray) -> float:
    """How bad a 2D depiction is.  Lower is better; the units are "one clash".

    Three things decide whether a depiction helps or hurts the diagram:
    bonds that cross each other and atoms that land on top of one another make
    the drawing unreadable on their own, and how faithfully the picture recalls
    the bound pose decides whether the residues can be placed sensibly around
    it.  Fidelity is the term that matters for the layout, so it is weighted
    heavily; the weights below are set against the four reference systems so
    that a projection which folds part of the ligand onto itself (4ps5) loses
    however faithful it is, while a merely tighter drawing (6wak) does not.
    """
    xy = np.array(mol.GetConformer().GetPositions())[:, :2]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    if not bonds:
        return 0.0
    unit = float(np.median([np.hypot(*(xy[i] - xy[j])) for i, j in bonds])) or 1.0

    def side(a, b, c):
        """Which side of the line a-b the point c falls on."""
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    crossings = 0
    for m, (i, j) in enumerate(bonds):
        for k, l in bonds[m + 1:]:
            if {i, j} & {k, l}:
                continue
            d1, d2 = side(xy[i], xy[j], xy[k]), side(xy[i], xy[j], xy[l])
            d3, d4 = side(xy[k], xy[l], xy[i]), side(xy[k], xy[l], xy[j])
            crossings += (d1 * d2 < 0) and (d3 * d4 < 0)

    d2d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    near = np.ones_like(d2d, dtype=bool)
    np.fill_diagonal(near, False)
    for i, j in bonds:
        near[i, j] = near[j, i] = False
    gap = np.clip(1.0 - d2d[near] / (0.85 * unit), 0.0, None)
    clash = float(np.sum(gap * gap))

    d3d = np.hypot(*(xyz[:, None, :] - xyz[None, :, :]).transpose(2, 0, 1))
    iu = np.triu_indices(len(xy), 1)
    fidelity = float(np.corrcoef(d2d[iu] / unit, d3d[iu])[0, 1]) if len(xy) > 2 else 0.0

    return 3.0 * crossings + clash - 2.0 * fidelity


def _spin(coords: np.ndarray, a: float, b: float) -> np.ndarray:
    """``coords`` turned by ``a`` radians about x, then ``b`` about y."""
    ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    return coords @ rx.T @ ry.T


def phosphate_axis_variants(
    mol: Chem.Mol,
    views: list[list[tuple[float, float]]],
    metal_groups: list[list[int]],
) -> list[list[tuple[float, float]]]:
    """Choose rigid phosphate-chain flips that clear metal interaction lines.

    Rotation about a bond has no literal depth in a 2D drawing.  Its standard
    depiction equivalent is to reflect the complete fragment on one side of
    that bond across the bond axis.  Every phosphate and every substituent
    therefore remains rigid; only the torsional relationship between adjacent
    phosphate groups changes.  This is intentionally different from moving a
    terminal oxygen around phosphorus, which deforms the group and is wrong.

    For each existing RDKit depiction we enumerate the few P-X bridge flips
    (X is the O or N joining two phosphorus atoms) and retain the variant whose
    virtual metal-coordination rays cross the ligand least.  The untouched
    depiction is one of the candidates, so a flip is never mandatory.
    """
    if not metal_groups:
        return views
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    adjacency = {i: set() for i in range(mol.GetNumAtoms())}
    for i, j in bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)

    def component(start: int, cut: tuple[int, int]) -> set[int]:
        seen, stack = set(), [start]
        blocked = {cut, cut[::-1]}
        while stack:
            atom = stack.pop()
            if atom in seen:
                continue
            seen.add(atom)
            stack.extend(
                other for other in adjacency[atom]
                if (atom, other) not in blocked and other not in seen
            )
        return seen

    # Every P-X bond where X itself joins two phosphorus atoms is a true axis
    # between phosphate groups.  Reflect the smaller graph component; the two
    # axis atoms remain fixed because they lie on the reflection line.
    axes: list[tuple[int, int, tuple[int, ...]]] = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        phosphorus, bridge = (a, b) if a.GetAtomicNum() == 15 else (b, a)
        if phosphorus.GetAtomicNum() != 15 or bridge.GetAtomicNum() not in (7, 8):
            continue
        if sum(n.GetAtomicNum() == 15 for n in bridge.GetNeighbors()) != 2:
            continue
        i, j = phosphorus.GetIdx(), bridge.GetIdx()
        left, right = component(i, (i, j)), component(j, (i, j))
        moving = left if len(left) <= len(right) else right
        axes.append((i, j, tuple(sorted(moving))))

    if not axes:
        return views

    def reflect(xy: np.ndarray, axis: tuple[int, int, tuple[int, ...]]) -> np.ndarray:
        i, j, moving = axis
        origin = xy[i]
        direction = xy[j] - origin
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return xy
        unit = direction / norm
        out = xy.copy()
        ids = np.asarray(moving, dtype=int)
        delta = xy[ids] - origin
        projection = (delta @ unit)[:, None] * unit
        out[ids] = origin + 2.0 * projection - delta
        return out

    def side(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1])
                     - (b[1] - a[1]) * (c[0] - a[0]))

    def intersects(a, b, c, d) -> bool:
        return side(a, b, c) * side(a, b, d) < -1e-8 \
            and side(c, d, a) * side(c, d, b) < -1e-8

    def score(xy: np.ndarray, flips: int) -> tuple[float, int]:
        lengths = [float(np.linalg.norm(xy[i] - xy[j])) for i, j in bonds]
        unit = float(np.median(lengths)) if lengths else 1.5

        # Reject a torsional view that folds the ligand onto itself before its
        # interaction lines are even considered.
        self_cross = 0
        for n, (i, j) in enumerate(bonds):
            for k, l in bonds[n + 1:]:
                if {i, j} & {k, l}:
                    continue
                self_cross += int(intersects(xy[i], xy[j], xy[k], xy[l]))
        distance = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        np.fill_diagonal(distance, np.inf)
        for i, j in bonds:
            distance[i, j] = distance[j, i] = np.inf
        atom_clash = float(np.square(np.maximum(0.0, 0.62 * unit - distance)).sum())

        route_cross = 0
        route_near = 0.0
        center = xy.mean(axis=0)
        bond_start = xy[[i for i, _ in bonds]]
        bond_end = xy[[j for _, j in bonds]]
        bond_vector = bond_end - bond_start
        fan_offset = np.linspace(-0.65, 0.65, 27)
        for group in metal_groups:
            anchors = [int(i) for i in dict.fromkeys(group) if 0 <= int(i) < len(xy)]
            if not anchors:
                continue
            anchor_center = xy[anchors].mean(axis=0)
            base = math.atan2(*(anchor_center - center)[::-1])
            # Try the same small local fan used by the fast layout.  The score
            # asks whether *some* close metal position makes every leg readable.
            angles = base + fan_offset
            metals = anchor_center + 2.8 * unit * np.stack(
                [np.cos(angles), np.sin(angles)], axis=1
            )
            anchor_xy = xy[anchors]
            rays = metals[:, None, :] - anchor_xy[None, :, :]       # f,a,2

            def cross2(left, right):
                return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]

            # Proper ray/bond intersections, all fan angles and all anchors in
            # one small NumPy operation instead of thousands of Python calls.
            c = bond_start[None, None, :, :] - anchor_xy[None, :, None, :]
            d = bond_end[None, None, :, :] - anchor_xy[None, :, None, :]
            s1 = cross2(rays[:, :, None, :], c)
            s2 = cross2(rays[:, :, None, :], d)
            a_from_bond = anchor_xy[None, :, None, :] - bond_start[None, None, :, :]
            m_from_bond = metals[:, None, None, :] - bond_start[None, None, :, :]
            s3 = cross2(bond_vector[None, None, :, :], a_from_bond)
            s4 = cross2(bond_vector[None, None, :, :], m_from_bond)
            hit = (s1 * s2 < -1e-8) & (s3 * s4 < -1e-8)
            incident = np.array([
                [atom in bond for bond in bonds] for atom in anchors
            ], dtype=bool)
            hit &= ~incident[None, :, :]
            crossing_count = hit.sum(axis=(1, 2))

            # Near misses against ligand atoms are secondary to real crossings.
            atom_delta = xy[None, None, :, :] - anchor_xy[None, :, None, :]
            denominator = np.maximum(np.sum(rays * rays, axis=2), 1e-9)
            along = np.sum(atom_delta * rays[:, :, None, :], axis=3) \
                / denominator[:, :, None]
            along = np.clip(along, 0.18, 0.92)
            closest = anchor_xy[None, :, None, :] \
                + along[:, :, :, None] * rays[:, :, None, :]
            gap = np.linalg.norm(xy[None, None, :, :] - closest, axis=3)
            for a_i, atom in enumerate(anchors):
                gap[:, a_i, atom] = np.inf
            near = np.square(np.maximum(0.0, 0.25 * unit - gap)).sum(axis=(1, 2))
            minimum_crossings = int(crossing_count.min())
            eligible = np.flatnonzero(crossing_count == minimum_crossings)
            best_i = int(eligible[np.argmin(near[eligible])])
            route_cross += minimum_crossings
            route_near += float(near[best_i])
        return (
            1_000_000.0 * self_cross
            + 100_000.0 * route_cross
            + 5_000.0 * atom_clash
            + 500.0 * route_near
            + 0.01 * flips,
            route_cross,
        )

    finished: list[list[tuple[float, float]]] = []
    for view in views:
        base = np.asarray(view, dtype=float)
        candidates = []
        # Four bridge axes in ANP mean only sixteen combinations.  Bound the
        # general case so a polymeric ligand cannot turn this into an unbounded
        # depiction search.
        usable = axes[:6]
        for mask in range(1 << len(usable)):
            xy = base.copy()
            flips = 0
            for bit, axis in enumerate(usable):
                if mask & (1 << bit):
                    xy = reflect(xy, axis)
                    flips += 1
            candidates.append((score(xy, flips), xy))
        candidates.sort(key=lambda row: row[0])
        xy = candidates[0][1]
        finished.append([(float(x), float(y)) for x, y in xy])
    return finished


def _depictions(mol: Chem.Mol, resname: str, limit: int = _MAX_DEPICTIONS) -> list[np.ndarray]:
    """Candidate 2D depictions of ``mol``'s bound pose, best-scoring first.

    ``Compute2DCoords`` draws the *graph*: it is free to swap which side of a
    ring a substituent leaves from, and on a flexible ligand like a nucleotide
    triphosphate it folds the chain back over the sugar.  The binding-site
    residues are then laid out around a shape that has nothing to do with the
    pose, which is what makes 6wak read as a knot.  Maestro depicts the bound
    conformer, and so does ``GenerateDepictionMatching3DStructure``.

    But there is no single "the" projection of a pose.  Seen along one axis a
    ligand is an open sprawl; seen along another it folds onto itself, and the
    residues around it then have nowhere to go.  RDKit's 3D-matched depiction
    always takes the flattest view -- it canonicalises to the principal axes
    first, so rotating the input changes nothing -- which is the right default
    and the wrong answer often enough to matter.

    So the pose is also projected from a grid of viewing directions, and each
    projection goes to ``Compute2DCoordsMimicDistmat``, whose search flips the
    rotatable groups to mimic the distances it is given.  That is the "turn the
    ligand by groups and try again" knob: the rigid fragments keep their shape
    and the joints between them are what moves.

    :func:`_depiction_score` only *prunes* -- it can see a fold or a clash, but
    not whether the residues fit around the result.  That question belongs to
    the layout optimiser, so everything still plausible is returned and
    :func:`ms_contactmap.layout.solve_layout` makes the final choice.
    """
    reference = Chem.Mol(mol)  # carries the 3D conformer
    xyz = np.array(reference.GetConformer().GetPositions())
    centred = xyz - xyz.mean(axis=0)
    frame = centred @ np.linalg.svd(centred, full_matrices=False)[2].T
    tril = np.tril_indices(len(xyz), -1)

    def mimic(view: np.ndarray):
        flat = view[:, :2]
        dist = np.hypot(*(flat[:, None, :] - flat[None, :, :]).transpose(2, 0, 1))
        return lambda trial: rdDepictor.Compute2DCoordsMimicDistmat(trial, dist[tril])

    trials = [
        ("3d-matched", lambda trial: rdDepictor.GenerateDepictionMatching3DStructure(
            trial, reference, acceptFailure=False)),
        ("coordgen", rdCoordGen.AddCoords),
        ("graph", AllChem.Compute2DCoords),
    ]
    grid = np.linspace(0.0, math.pi, _VIEW_GRID, endpoint=False)
    for a in grid:
        for b in grid:
            trials.append(
                (f"view{math.degrees(a):3.0f}/{math.degrees(b):3.0f}",
                 mimic(_spin(frame, a, b)))
            )

    pool: list[tuple[float, str, np.ndarray]] = []
    for name, depict in trials:
        trial = Chem.Mol(reference)
        try:
            depict(trial)
            score = _depiction_score(trial, xyz)
        except Exception as exc:  # noqa: BLE001 -- RDKit raises bare RuntimeError
            print(f"[ms_contactmap.chem] {resname}: {name} depiction failed ({exc})",
                  file=sys.stderr)
            continue
        xy = np.array(trial.GetConformer().GetPositions())[:, :2]
        if any(np.abs(xy - seen).max() < 1e-6 for _, _, seen in pool):
            continue
        pool.append((score, name, xy))

    if not pool:
        raise ValueError(f"{resname}: no 2D depiction could be generated")
    pool.sort(key=lambda row: row[0])
    keep = [row for row in pool if row[0] <= pool[0][0] + _SCORE_SLACK][:limit]
    print(f"[ms_contactmap.chem] {resname}: {len(keep)} of {len(pool)} depictions kept ("
          + ", ".join(f"{name} {score:.2f}" for score, name, _ in keep) + ")",
          file=sys.stderr)
    return [xy for _, _, xy in keep]


def _normalize_phosphate_oxyanions(mol: Chem.Mol) -> None:
    """Use the physiological oxyanion form for terminal phosphate oxygens.

    CCD SMILES commonly encode the neutral acid so their formula is portable.
    In a protein interaction diagram that silently creates implicit O-H bonds,
    turns phosphate oxygens into H-bond donors and even prints ``OH`` beside a
    Mg-bound nucleotide.  A terminal, single-bonded oxygen on phosphorus is the
    ionisable site; bridging P-O-P/P-O-C atoms and phosphoryl P=O atoms are
    untouched.
    """
    changed = False
    for phosphorus in (atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 15):
        for bond in phosphorus.GetBonds():
            oxygen = bond.GetOtherAtom(phosphorus)
            if (
                oxygen.GetAtomicNum() != 8
                or oxygen.GetDegree() != 1
                or bond.GetBondType() != Chem.BondType.SINGLE
            ):
                continue
            oxygen.SetFormalCharge(-1)
            oxygen.SetNumExplicitHs(0)
            oxygen.SetNoImplicit(True)
            changed = True
    if changed:
        mol.UpdatePropertyCache(strict=False)


def load_ligand(pdb_path, resname, smiles, chain=None, resnum=None) -> LigandGeometry:
    """Build the ligand mol for ``resname`` in ``pdb_path`` using ``smiles``.

    ``chain``/``resnum`` disambiguate between copies; omit them to take the
    first copy in the file.  Raises :class:`ValueError` if the ligand is absent
    or if the SMILES does not match the HETATM block.
    """
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"SMILES for {resname} does not parse: {smiles!r}")

    records = _pick_copy(read_pdb_atoms(pdb_path), resname, chain, resnum)
    heavy = [a for a in records if a.element not in ("H", "D")]
    if not heavy:
        raise ValueError(f"ligand {resname} has no heavy atoms")

    # Docking poses commonly include polar hydrogens even though the drawing
    # molecule below is intentionally heavy-atom only.  Assign each H to its
    # nearest heavy atom in this ligand; a 1.7 A ceiling covers S-H while
    # remaining far below any intermolecular hydrogen bond.
    hydrogen_by_heavy_serial: dict[int, list[tuple[float, float, float]]] = {}
    for hydrogen in (a for a in records if a.element in ("H", "D")):
        nearest = min(
            heavy,
            key=lambda atom: (atom.x - hydrogen.x) ** 2
            + (atom.y - hydrogen.y) ** 2
            + (atom.z - hydrogen.z) ** 2,
        )
        distance = math.sqrt(
            (nearest.x - hydrogen.x) ** 2
            + (nearest.y - hydrogen.y) ** 2
            + (nearest.z - hydrogen.z) ** 2
        )
        if distance <= 1.7:
            hydrogen_by_heavy_serial.setdefault(nearest.serial, []).append(
                (hydrogen.x, hydrogen.y, hydrogen.z)
            )

    block = "\n".join(a.line for a in heavy) + "\nEND\n"
    pdb_mol = Chem.MolFromPDBBlock(
        block, sanitize=False, removeHs=False, proximityBonding=True
    )
    if pdb_mol is None:
        raise ValueError(f"RDKit could not read the HETATM block of {resname}")

    # Heavy-atom composition, compared before the match is attempted:
    # AssignBondOrdersFromTemplate matches the *template into* the PDB mol, so a
    # template that is merely a fragment of the ligand succeeds quietly and
    # leaves the rest of the molecule with guessed single bonds.
    pdb_formula = _formula(a.element for a in heavy)
    smiles_formula = _formula(a.GetSymbol() for a in template.GetAtoms())
    mismatch = (
        f"SMILES does not match the {resname} HETATM block in {pdb_path}: "
        f"SMILES is {smiles_formula} ({template.GetNumAtoms()} heavy atoms), "
        f"PDB block is {pdb_formula} ({len(heavy)} heavy atoms)"
    )
    if smiles_formula != pdb_formula:
        raise ValueError(mismatch)
    try:
        mol = AllChem.AssignBondOrdersFromTemplate(template, pdb_mol)
    except Exception as exc:  # noqa: BLE001 -- RDKit raises bare ValueError/RuntimeError
        raise ValueError(f"{mismatch}. RDKit said: {exc}") from exc

    # The template match is a substructure match, so it can succeed on a
    # fragment.  Atom-count equality is what makes serial -> index safe.
    if mol.GetNumAtoms() != pdb_mol.GetNumAtoms():
        raise ValueError(
            f"AssignBondOrdersFromTemplate changed the atom count for {resname} "
            f"({pdb_mol.GetNumAtoms()} -> {mol.GetNumAtoms()}); serial mapping "
            f"would be wrong"
        )

    _normalize_phosphate_oxyanions(mol)
    Chem.SanitizeMol(mol)
    # Preserve any explicit pose hydrogens for exact D-H...A angles before the
    # drawing molecule is reduced to heavy atoms.  PDB serials survive
    # RemoveHs on their heavy neighbours and provide a stable remapping.
    full_conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetDegree() != 1:
            continue
        heavy = atom.GetNeighbors()[0]
        info = heavy.GetPDBResidueInfo()
        if info is None:
            continue
        point = full_conf.GetAtomPosition(atom.GetIdx())
        hydrogen_by_heavy_serial.setdefault(info.GetSerialNumber(), []).append(
            (float(point.x), float(point.y), float(point.z))
        )
    mol = Chem.RemoveHs(mol)
    if mol.GetNumAtoms() != len(heavy):
        raise ValueError(
            f"{resname}: {len(heavy)} heavy HETATM records became "
            f"{mol.GetNumAtoms()} atoms; the SMILES describes a different molecule"
        )

    conf = mol.GetConformer()
    coords_3d = [
        (p.x, p.y, p.z) for p in (conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms()))
    ]
    serial_to_idx: dict[int, int] = {}
    for idx, atom in enumerate(mol.GetAtoms()):
        info = atom.GetPDBResidueInfo()
        if info is None:
            raise ValueError(f"{resname}: atom {idx} lost its PDB residue info")
        serial_to_idx[info.GetSerialNumber()] = idx
    donor_hydrogens = {
        serial_to_idx[serial]: points
        for serial, points in hydrogen_by_heavy_serial.items()
        if serial in serial_to_idx
    }

    # The 3D conformer has been read out above, so it can go now: everything
    # downstream draws from the 2D one.
    views = _depictions(mol, resname)
    mol.RemoveAllConformers()
    flat = Chem.Conformer(mol.GetNumAtoms())
    flat.Set3D(False)
    for i, (x, y) in enumerate(views[0]):
        flat.SetAtomPosition(i, (float(x), float(y), 0.0))
    mol.AddConformer(flat, assignId=True)

    first = heavy[0]
    return LigandGeometry(
        mol=mol,
        coords_2d=[(float(x), float(y)) for x, y in views[0]],
        alt_coords_2d=[[(float(x), float(y)) for x, y in v] for v in views[1:]],
        coords_3d=coords_3d,
        serial_to_idx=serial_to_idx,
        resname=first.resname.upper(),
        chain=first.chain,
        resnum=first.resnum,
        donor_hydrogens=donor_hydrogens,
    )


if __name__ == "__main__":
    import json
    from pathlib import Path

    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    root = Path(__file__).resolve().parent.parent
    smiles = json.loads((root / "data" / "ligands.json").read_text())["2TA"]["smiles"]
    geom = load_ligand(root / "data" / "4ps5.pdb", "2TA", smiles)

    assert CalcMolFormula(geom.mol) == CalcMolFormula(Chem.MolFromSmiles(smiles)), (
        CalcMolFormula(geom.mol),
        CalcMolFormula(Chem.MolFromSmiles(smiles)),
    )
    hetatm = [
        a
        for a in read_pdb_atoms(root / "data" / "4ps5.pdb")
        if a.hetatm
        and a.resname == "2TA"
        and (a.chain, a.resnum) == (geom.chain, geom.resnum)
        and a.element not in ("H", "D")
    ]
    missing = [a.serial for a in hetatm if a.serial not in geom.serial_to_idx]
    assert not missing, f"unmapped HETATM serials: {missing}"
    assert len(geom.serial_to_idx) == len(hetatm) == geom.mol.GetNumAtoms()
    assert len(geom.coords_2d) == geom.mol.GetNumAtoms()
    assert len(geom.coords_3d) == geom.mol.GetNumAtoms()

    print(
        f"ok  4ps5/2TA  {CalcMolFormula(geom.mol)}  "
        f"{geom.mol.GetNumAtoms()} atoms  {geom.chain}:{geom.resnum}"
    )
