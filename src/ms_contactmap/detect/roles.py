"""The data contract shared by the detection engine's four halves.

:mod:`ms_contactmap.detect` is the in-process detector, and the only one: no
external interaction program is called anywhere in this package.  It is a
**clean-room** implementation: the published criteria it
applies are facts from the literature (each threshold in :mod:`.geometry` cites
its source) and the chemistry comes from RDKit, which is BSD-3-Clause.  No PLIP
code, structure, naming or comment is reproduced -- that is the whole point of
the exercise, since PLIP is GPL-2.0-only.

The pipeline is deliberately three independent pieces meeting here:

    protein.py  --\\
                    >--  detect_contacts()  -->  [Hit]  -->  engine.py
    ligand.py   --/         (geometry.py)                    (Interaction[])

:func:`~ms_contactmap.detect.protein.perceive_protein` and
:func:`~ms_contactmap.detect.ligand.perceive_ligand` both return a
:class:`Perception`; :func:`~ms_contactmap.detect.geometry.detect_contacts` consumes
two of them and knows no chemistry at all, only coordinates and flags.  Atom
ids are PDB serial numbers on the protein side and RDKit atom indices on the
ligand side; nothing in this package mixes them, and the driver translates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Site:
    """What one heavy atom can do, independent of who is nearby.

    ``neighbours`` are the bonded heavy atoms.  Crystal structures carry no
    hydrogens, so every directional test approximates the missing H by the
    atom's free direction -- away from the mean of its neighbours -- which is
    why perception has to report connectivity and not just element types.
    """

    donor: bool = False
    acceptor: bool = False
    hydrophobic: bool = False
    #: Lone-pair donor to a metal: N, O, S (and Se) that are not already
    #: cationic.  Kept apart from ``acceptor`` because the geometry differs.
    metal_binder: bool = False
    #: Cl, Br or I bonded to carbon -- the sigma-hole donor of a halogen bond.
    halogen: bool = False
    is_metal: bool = False
    is_water: bool = False
    neighbours: tuple[int, ...] = ()
    #: Coordinates of explicit bonded hydrogens when the input provides them.
    hydrogens: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class Group:
    """Atoms acting as one chemical unit, addressed by their centroid.

    ``kind`` is ``"cation"``, ``"anion"`` or ``"aromatic"``.  Aromatic groups
    are rings and carry a ``normal``; charged groups do not.
    """

    kind: str
    atoms: tuple[int, ...]
    #: Unit ring normal for ``aromatic``; ``None`` otherwise.
    normal: np.ndarray | None = None


@dataclass
class Perception:
    """One side of the complex: coordinates plus what each atom can do.

    ``sites`` and ``coords`` are keyed alike, and only heavy atoms appear.
    """

    sites: dict[int, Site] = field(default_factory=dict)
    coords: dict[int, np.ndarray] = field(default_factory=dict)
    groups: list[Group] = field(default_factory=list)

    def ids(self) -> list[int]:
        return list(self.coords)

    def array(self, ids: list[int]) -> np.ndarray:
        """``(n, 3)`` of the given ids, in that order -- for cKDTree."""
        return np.array([self.coords[i] for i in ids], dtype=float)

    def centroid(self, group: Group) -> np.ndarray:
        return np.mean([self.coords[a] for a in group.atoms], axis=0)


@dataclass(frozen=True)
class Hit:
    """One detected contact, still in atom ids -- no residues, no styling.

    ``kind`` is already in :func:`ms_contactmap.model.normalize_kind`'s vocabulary
    so the driver never has to translate: ``hbond``, ``water_bridge``,
    ``salt_bridge``, ``hydrophobic``, ``pi_stacking``, ``pi_cation``,
    ``halogen_bond``, ``metal_coordination``.
    """

    kind: str
    #: PDB serials.  Several for a ring or a charged group.
    protein_atoms: tuple[int, ...]
    #: RDKit indices.  Several for a ring or a charged group.
    ligand_atoms: tuple[int, ...]
    distance: float
    #: ``True`` when the ligand donates the hydrogen, the cation or the
    #: halogen; ``False`` when the protein does.  Meaningless for symmetric
    #: kinds (hydrophobic, pi-stacking), where it stays ``True``.
    ligand_is_donor: bool = True
    #: PDB serial of the bridging water's oxygen, for ``water_bridge`` only.
    water: int | None = None
    #: Other leg of a water bridge (water to protein), in angstrom.
    partner_distance: float | None = None
    #: Donor direction of the protein-water leg for a water bridge.
    protein_is_donor: bool | None = None
    #: The angle that characterises the contact, where one does: for a water
    #: bridge, the angle subtended at the water oxygen.  The driver needs it to
    #: choose between several bridges through the same water, and recomputing it
    #: there would mean duplicating geometry this module already did.
    angle: float | None = None
