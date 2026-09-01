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
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

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
#: Atoms further than this from the ligand cannot occlude it (probe 1.4 A plus
#: two heavy-atom radii is < 6 A), so trimming here is exact, not approximate.
SASA_SHELL = 12.0


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

def exposed_atoms(pdb_path, geom) -> dict[int, float]:
    """How exposed each solvent-reachable ligand atom is, by RDKit index.

    SASA is computed twice, once for the isolated ligand and once for the
    ligand inside the complex.  The second value decides whether an atom is
    marked at all; their ratio is *how* exposed it is -- the fraction of the
    atom's own surface the protein leaves free -- which is what sizes the mark
    the renderer draws.  A ratio rather than a raw area so a sulphur and a
    fully buried-then-freed nitrogen are compared on the same footing.
    """
    model = PDBParser(QUIET=True).get_structure("cx", str(pdb_path))[0]
    ligand = None
    for chain in model:
        for residue in list(chain):
            name = residue.get_resname().strip().upper()
            if (
                name == geom.resname
                and chain.id.strip() == geom.chain
                and residue.id[1] == geom.resnum
            ):
                ligand = residue
            elif name in WATER_NAMES:
                chain.detach_child(residue.id)  # bulk water is not the pocket
    if ligand is None:
        raise ValueError(f"{geom.resname} {geom.chain}:{geom.resnum} not found for SASA")

    ligand_xyz = np.array([a.coord for a in ligand], dtype=float)
    for chain in list(model):
        for residue in list(chain):
            if residue is ligand:
                continue
            xyz = np.array([a.coord for a in residue], dtype=float)
            if xyz.size == 0 or np.linalg.norm(
                xyz[:, None, :] - ligand_xyz[None, :, :], axis=2
            ).min() > SASA_SHELL:
                chain.detach_child(residue.id)

    shrake = ShrakeRupley(n_points=SASA_POINTS)
    shrake.compute(model, level="A")
    in_complex = {a.serial_number: a.sasa for a in ligand}
    shrake.compute(ligand, level="A")  # overwrites .sasa, hence the copy above
    alone = {a.serial_number: a.sasa for a in ligand}

    return {
        idx: min(1.0, in_complex[serial] / alone[serial])
        for serial, idx in sorted(geom.serial_to_idx.items())
        if in_complex.get(serial, 0.0) > SASA_MIN and alone.get(serial, 0.0) > 0.0
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_diagram(
    pdb_path,
    resname,
    smiles,
    name=None,
    chain=None,
    resnum=None,
    compute_exposure: bool = True,
) -> Diagram:
    """Assemble one diagram with the native detector.

    ``compute_exposure`` may be disabled for a fast analysis when solvent halos
    are not needed.
    """
    pdb_path = Path(pdb_path)
    geom = load_ligand(pdb_path, resname, smiles, chain=chain, resnum=resnum)
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
        exposure=exposed_atoms(pdb_path, geom) if compute_exposure else {},
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
        },
    )
