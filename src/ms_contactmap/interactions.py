"""Turn a PDB file plus a ligand SMILES into a ready-to-lay-out :class:`Diagram`.

Three sources of truth are merged here:

* **The native clean-room detector** identifies contacts in-process from
  published geometric criteria and RDKit/PDB chemistry.
* **The PDB coordinates** supply the pocket-lining residues that carry no
  interaction line but that Maestro still draws, and the 3D anchor (nearest
  ligand atom) that ties every residue to a place on the 2D depiction.
* **Shrake-Rupley SASA** marks the ligand atoms that stay solvent-accessible
  inside the complex; those get the grey halo of the reference images.

Everything leaves this module already expressed in :mod:`ms_contactmap.model`
terms, so the layout and rendering stages never see Biopython.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from Bio.PDB.SASA import ATOMIC_RADII

from .chem import PdbAtom, load_ligand, read_pdb_atoms
from .model import (
    COMMON_ADDITIVES,
    Diagram,
    Residue,
    ResidueRef,
    WATER_NAMES,
)

#: A protein residue is drawn as pocket context when one of its heavy atoms is
#: this close to a ligand heavy atom.  Measured against the four references:
#: every residue Maestro draws sits at <= 4.10 A (2gfk SER A:221 is the
#: farthest), and the nearest one it leaves out is at 3.90 A (4ps5 MET A:149),
#: so the boundary is right here.  4.0 reproduces 4uwh and 6wak exactly and
#: comes within one glyph on the other two; 4.5 added six spurious residues to
#: 4ps5 alone.
CONTEXT_CUTOFF = 4.0

#: Ligand atoms keep their grey solvent halo above this SASA (A^2) *inside the
#: complex*.  A water probe laid flat on an exposed sp3 carbon covers roughly
#: 8-10 A^2, so 2 A^2 reads as "at least a quarter of a water can still touch
#: this atom".  Tuned against data/4ps5.png: it marks the whole pyrrolidine
#: ring, the ethoxy linker, the tert-butyl and the free sulfonyl oxygen -- the
#: groups that carry halos there -- while dropping the buried pyrimidine core.
#: ponytail: one absolute cutoff for every element.  If nitrogen and oxygen
#: halos turn out over-eager next to carbon, switch to SASA relative to the
#: same atom in the isolated ligand (both numbers are already computed).
SASA_MIN = 2.0
#: Shrake-Rupley sphere sampling.  The Biopython default (100) quantises a
#: carbon's area in ~1.2 A^2 steps, which is coarser than SASA_MIN.
SASA_POINTS = 960
#: Water probe radius (A), Biopython's default.
SASA_PROBE = 1.40


# ---------------------------------------------------------------------------
# Pocket context from raw coordinates
# ---------------------------------------------------------------------------

def _residue_atoms(atoms: list[PdbAtom], geom) -> dict[ResidueRef, list[PdbAtom]]:
    """Heavy atoms of every residue except the ligand copy being drawn."""
    groups: dict[ResidueRef, list[PdbAtom]] = {}
    for atom in atoms:
        if atom.element in ("H", "D"):
            continue
        if (
            atom.hetatm
            and atom.resname.upper() == geom.resname
            and atom.chain == geom.chain
            and atom.resnum == geom.resnum
        ):
            continue
        ref = ResidueRef(atom.chain, atom.resnum, atom.resname.upper(), atom.icode)
        groups.setdefault(ref, []).append(atom)
    return groups


def _pocket(groups, ligand_xyz: np.ndarray, interacting: set[str]):
    """Residues to draw, their centroids, and the nearest ligand atom of each.

    A residue qualifies when the detector reported a contact with it or when it
    lines the pocket within :data:`CONTEXT_CUTOFF`.  Buffers, cryoprotectants and
    bulk water only get in through the first route.
    """
    residues: list[Residue] = []
    nearest: dict[str, int] = {}
    for ref, atoms in groups.items():
        has_interactions = ref.key in interacting
        if not has_interactions and ref.name in COMMON_ADDITIVES:
            continue
        xyz = np.array([(a.x, a.y, a.z) for a in atoms], dtype=float)
        # (n_residue_atoms, n_ligand_atoms) -- residues are small, so the dense
        # block is cheaper than building a tree per structure.
        dist = np.linalg.norm(xyz[:, None, :] - ligand_xyz[None, :, :], axis=2)
        if not has_interactions and dist.min() > CONTEXT_CUTOFF:
            continue
        residues.append(
            Residue(
                ref=ref,
                has_interactions=has_interactions,
                center_3d=tuple(float(v) for v in xyz.mean(axis=0)),
            )
        )
        nearest[ref.key] = int(np.unravel_index(dist.argmin(), dist.shape)[1])
    residues.sort(key=lambda r: (r.ref.chain, r.ref.number, r.ref.name))
    return residues, nearest


# ---------------------------------------------------------------------------
# Solvent exposure
# ---------------------------------------------------------------------------

def _sphere(n: int) -> np.ndarray:
    """Golden-spiral points on the unit sphere, the same set Biopython samples."""
    k = np.arange(n)
    z = 1.0 - (2 * k + 1) / n
    longitude = k * np.pi * (3.0 - 5.0 ** 0.5)
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([np.cos(longitude) * r, np.sin(longitude) * r, z])


def exposed_atoms(atoms: list[PdbAtom], geom) -> dict[int, float]:
    """How exposed each solvent-reachable ligand atom is, by RDKit index.

    SASA is computed twice, once for the isolated ligand and once for the
    ligand inside the complex.  The second value decides whether an atom is
    marked at all; their ratio is *how* exposed it is -- the fraction of the
    atom's own surface the protein leaves free -- which is what sizes the mark
    the renderer draws.  A ratio rather than a raw area so a sulphur and a
    fully buried-then-freed nitrogen are compared on the same footing.

    Shrake-Rupley as Biopython does it (same radii, probe and sphere), but only
    for the ligand atoms: the protein only occludes, so its own surface is never
    sampled.  ``atoms`` is the :func:`read_pdb_atoms` parse the caller already has.
    """
    kept = [a for a in atoms if a.resname.upper() not in WATER_NAMES]  # bulk water is not the pocket
    ligand = np.array([
        a.resname.upper() == geom.resname and a.chain == geom.chain and a.resnum == geom.resnum
        for a in kept
    ])
    if not ligand.any():
        raise ValueError(f"{geom.resname} {geom.chain}:{geom.resnum} not found for SASA")

    xyz = np.array([(a.x, a.y, a.z) for a in kept], dtype=float)
    # ponytail: an element Biopython has no radius for gets carbon's instead of a KeyError.
    radius = np.array([ATOMIC_RADII.get(a.element, ATOMIC_RADII["C"]) for a in kept]) + SASA_PROBE
    sphere = _sphere(SASA_POINTS)

    def area(i: int, occluders: np.ndarray) -> float:
        near = occluders[
            (np.linalg.norm(xyz[occluders] - xyz[i], axis=1) < radius[i] + radius[occluders])
            & (occluders != i)
        ]
        points = sphere * radius[i] + xyz[i]
        buried = (
            np.linalg.norm(points[:, None, :] - xyz[near][None, :, :], axis=2) <= radius[near]
        ).any(axis=1)
        return float((~buried).sum()) * 4.0 * np.pi * radius[i] ** 2 / len(sphere)

    everyone = np.arange(len(kept))
    alone_only = np.flatnonzero(ligand)
    by_serial = {kept[i].serial: i for i in alone_only}
    exposure: dict[int, float] = {}
    for serial, idx in sorted(geom.serial_to_idx.items()):
        i = by_serial.get(serial)
        if i is None:
            continue
        in_complex = area(i, everyone)
        if in_complex <= SASA_MIN:
            continue
        alone = area(i, alone_only)
        if alone > 0.0:
            exposure[idx] = min(1.0, in_complex / alone)
    return exposure


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_diagram(
    pdb_path,
    resname,
    smiles=None,
    name=None,
    chain=None,
    resnum=None,
    compute_exposure: bool = True,
    *,
    ligand=None,
) -> Diagram:
    """Assemble one diagram with the native detector.

    The ligand chemistry comes from ``smiles`` or from ``ligand`` (an RDKit
    ``Mol`` or an ``.sdf``/``.mol``/``.mol2`` file of the same pose); with
    neither, the bond orders are unknown and the legend says so.  See
    :func:`~ms_contactmap.chem.load_ligand`.  ``compute_exposure`` may be
    disabled for a fast analysis when solvent halos are not needed.
    """
    pdb_path = Path(pdb_path)
    geom = load_ligand(pdb_path, resname, smiles, chain=chain, resnum=resnum, ligand=ligand)
    atoms = read_pdb_atoms(pdb_path)
    serial_to_atom = {a.serial: a for a in atoms}

    from .detect import detect_interactions

    interactions, metal_legs, coordination, skipped = detect_interactions(
        pdb_path, geom, serial_to_atom
    )
    # ponytail: a stderr note instead of a logger.  Swap for
    # logging.getLogger(__name__) once the widget ships and stderr belongs to
    # the host application.
    print(
        f"[ms_contactmap.interactions] {geom.resname} {geom.chain}:{geom.resnum}: "
        f"{len(interactions)} interactions, {skipped} skipped (unmappable ligand atoms)",
        file=sys.stderr,
    )

    interacting = {i.residue_key for i in interactions}
    interacting |= {i.via_water for i in interactions if i.via_water}
    # A coordinating histidine can sit well outside CONTEXT_CUTOFF of the
    # ligand and still be half of what makes the metal's geometry legible.
    interacting |= {leg.metal_key for leg in metal_legs}
    # A coordinating *water*, though, no longer earns its glyph that way: the
    # legs stopped being drawn, so it would sit there attached to nothing.  It
    # still counts towards the metal's coordination number, which is what
    # decides the polygon; it is only drawn if the pocket scan wants it anyway.
    interacting |= {
        leg.partner_key for leg in metal_legs
        if leg.partner_key.rsplit(":", 1)[-1] not in WATER_NAMES
    }
    ligand_xyz = np.array(geom.coords_3d, dtype=float)
    residues, nearest = _pocket(_residue_atoms(atoms, geom), ligand_xyz, interacting)

    # A residue the detector reported but that our own scan never grouped (an
    # atom the PDB parser dropped, say) would leave a dangling residue_key.
    drawn = {r.key for r in residues}
    orphans = sorted(interacting - drawn)
    if orphans:
        print(
            f"[ms_contactmap.interactions] contacts reference undrawn residues: {orphans}",
            file=sys.stderr,
        )

    # Explore rigid torsional flips between phosphate groups.  Interaction
    # detection has already used the untouched 3D pose; this changes only the
    # legibility of the resulting 2D coordination graph.
    from .chem import phosphate_axis_variants

    metal_groups = [
        [
            atom
            for inter in interactions
            if inter.kind == "metal_coordination" and inter.residue_key == metal_key
            for atom in inter.ligand_atoms
        ]
        for metal_key in sorted(coordination)
    ]
    views = phosphate_axis_variants(
        geom.mol, [geom.coords_2d, *geom.alt_coords_2d], metal_groups
    )

    return Diagram(
        name=name or f"{pdb_path.stem} {geom.resname} {geom.chain}:{geom.resnum}",
        ligand_name=geom.resname,
        mol=geom.mol,
        coords_2d=views[0],
        coords_alt=views[1:],
        residues=residues,
        interactions=[i for i in interactions if i.residue_key in drawn],
        exposure=exposed_atoms(atoms, geom) if compute_exposure else {},
        nearest_atom=nearest,
        metal_legs=[
            leg for leg in metal_legs
            if leg.metal_key in drawn and leg.partner_key in drawn
        ],
        metal_coordination={k: v for k, v in coordination.items() if k in drawn},
        metadata={
            "detector": "native",
            "source_pdb": str(pdb_path.resolve()),
            "ligand": {
                "resname": geom.resname,
                "chain": geom.chain,
                "resnum": geom.resnum,
            },
            "exposure_computed": bool(compute_exposure),
            "bond_orders_known": geom.bond_orders_known,
        },
    )


# ---------------------------------------------------------------------------
# Docked poses: receptor file + ligand chemistry, no complex PDB needed
# ---------------------------------------------------------------------------

#: AutoDock atom types that are not element symbols (PDBQT columns 78-79).
_AUTODOCK_ELEMENTS = {"A": "C", "NA": "N", "OA": "O", "SA": "S", "HD": "H", "HS": "H", "NS": "N"}
#: Residue name given to the pose inside the assembled complex.
POSE_RESNAME = "UNL"


def _receptor_records(receptor, suffix: str = "") -> list[str]:
    """First-model ATOM/HETATM lines of a PDB, PDBQT or mmCIF receptor: a file, or its
    bytes with `suffix` naming the format (".pdb", ".pdbqt", ".cif").

    PDBQT puts the AutoDock type where PDB keeps the element, so that tail is
    replaced by the element symbol.  mmCIF goes through Biopython's PDB writer.
    ponytail: PDB columns cap the receptor at 99,999 atoms and one-letter chains.
    """
    in_memory = isinstance(receptor, bytes)
    data = receptor if in_memory else Path(receptor).read_bytes()
    suffix = (suffix or (".pdb" if in_memory else Path(receptor).suffix)).lower()
    if suffix in (".cif", ".mmcif"):
        import io

        from Bio.PDB import PDBIO, MMCIFParser

        writer = PDBIO()
        structure = MMCIFParser(QUIET=True).get_structure("receptor", io.StringIO(data.decode("utf-8", "replace")))
        writer.set_structure(structure[0])
        buffer = io.StringIO()
        writer.save(buffer)
        text = buffer.getvalue()
    else:
        text = data.decode("utf-8", errors="replace")
    records = []
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if suffix == ".pdbqt":
            tail = line[66:].split()
            kind = tail[-1] if tail else ""
            element = _AUTODOCK_ELEMENTS.get(kind.upper(), kind[:2].capitalize())
            line = f"{line[:66]:<76}{element:>2}"
        records.append(line)
    return records


def build_pose_diagram(
    receptor,
    ligand,
    name=None,
    compute_exposure: bool = True,
    smiles=None,
    receptor_format: str = "pdb",
) -> Diagram:
    """Diagram of a docked pose from the receptor file and the pose itself.

    ``receptor`` is a ``.pdb``, ``.pdbqt`` or ``.cif`` file, or its bytes with
    ``receptor_format`` naming which; ``ligand`` is an RDKit ``Mol`` with the pose
    conformer, the bytes of an SDF/MOL record, or an ``.sdf``/``.mol``/``.mol2``
    file, whose own bonds and hydrogens are kept.  A ``.pdb`` ligand carries no
    bond orders: they come from ``smiles``, or stay unknown without it.  The complex that
    :func:`build_diagram` reads is assembled here, in a temporary file, with
    the pose as residue ``UNL`` 1 on a chain the receptor does not use.
    """
    import tempfile

    from rdkit import Chem

    from .chem import _read_ligand

    in_memory = isinstance(receptor, bytes)
    receptor = receptor if in_memory else Path(receptor)
    from_pdb = not isinstance(ligand, (Chem.Mol, bytes)) and Path(ligand).suffix.lower() == ".pdb"
    if from_pdb:
        mol = Chem.MolFromPDBFile(str(ligand), sanitize=False, removeHs=False, proximityBonding=False)
        if mol is None:
            raise ValueError(f"RDKit could not read {ligand}")
    elif smiles is not None:
        raise ValueError("smiles is only used with a .pdb ligand")
    else:
        mol = _read_ligand(ligand)
    if not mol.GetNumConformers():
        raise ValueError("the ligand has no coordinates")
    records = _receptor_records(receptor, f".{receptor_format.lstrip('.')}" if in_memory else "")
    used_chains = {line[21] for line in records}
    chain = next((c for c in "ZYXWVUTSRQPONMLKJIHGFEDCBA" if c not in used_chains), None)
    if chain is None:
        raise ValueError(f"{'receptor' if in_memory else receptor.name}: no free chain id left for the ligand")
    serial = max((int(line[6:11]) for line in records if line[6:11].strip().isdigit()), default=0)
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        serial += 1
        p = conf.GetAtomPosition(atom.GetIdx())
        symbol = atom.GetSymbol()
        atom_name = f"{symbol}{atom.GetIdx() + 1}"[:4]
        records.append(
            f"HETATM{serial:>5} {atom_name:<4} {POSE_RESNAME:>3} {chain}{1:>4}    "
            f"{p.x:8.3f}{p.y:8.3f}{p.z:8.3f}  1.00  0.00          {symbol.upper():>2}"
        )
    with tempfile.TemporaryDirectory(prefix="ms_contactmap_") as scratch:
        complex_pdb = Path(scratch) / "complex.pdb"
        complex_pdb.write_text("\n".join(records) + "\nEND\n", encoding="utf-8")
        diagram = build_diagram(
            complex_pdb, POSE_RESNAME, smiles, name=name or ("receptor" if in_memory else receptor.stem), chain=chain, resnum=1,
            compute_exposure=compute_exposure, ligand=None if from_pdb else mol,
        )
    if in_memory:  # the complex it names was a temporary file
        diagram.metadata.pop("source_pdb", None)
    else:
        diagram.metadata["source_pdb"] = str(receptor.resolve())
    return diagram
