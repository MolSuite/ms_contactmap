"""Shared data contract and visual vocabulary for the 2D interaction diagram.

Every other module in :mod:`ms_contactmap` imports from here and nothing else, so
this file is the single place where the diagram's meaning and its Maestro-like
appearance are defined.

Colours are sampled directly from the reference PNGs in ``data/`` -- see
``tools/sample_reference_colors.py`` for how they were obtained.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

# ---------------------------------------------------------------------------
# Residue chemistry
# ---------------------------------------------------------------------------

#: Maestro's residue colour classes.  ``unspecified`` covers anything that is
#: not a standard amino acid and is not water or a metal (modified residues,
#: nucleotides, cofactors).
RESIDUE_CLASSES = (
    "hydrophobic",
    "polar",
    "charged_negative",
    "charged_positive",
    "glycine",
    "water",
    "metal",
    "unspecified",
)

AA_CLASS: dict[str, str] = {
    "ALA": "hydrophobic",
    "VAL": "hydrophobic",
    "LEU": "hydrophobic",
    "ILE": "hydrophobic",
    "PRO": "hydrophobic",
    "PHE": "hydrophobic",
    "MET": "hydrophobic",
    "TRP": "hydrophobic",
    "CYS": "hydrophobic",
    "SER": "polar",
    "THR": "polar",
    "ASN": "polar",
    "GLN": "polar",
    "TYR": "polar",
    "HIS": "polar",
    "ASP": "charged_negative",
    "GLU": "charged_negative",
    "ARG": "charged_positive",
    "LYS": "charged_positive",
    "GLY": "glycine",
}

WATER_NAMES = frozenset({"HOH", "WAT", "DOD", "H2O"})

METAL_NAMES = frozenset(
    {
        "ZN", "MG", "MN", "FE", "FE2", "CA", "NA", "K", "CU", "CU1", "NI",
        "CO", "CD", "HG", "PT", "AU", "AG", "LI", "SR", "BA", "CS", "RB",
        "PB", "MO", "W", "V", "CR",
    }
)

#: Solvents, buffers and cryoprotectants that are never the ligand of interest
#: and are not worth drawing as binding-site residues.
COMMON_ADDITIVES = frozenset(
    {
        "HOH", "DOD", "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "1PE", "P6G",
        "ACT", "ACY", "FMT", "CIT", "TRS", "MES", "EPE", "IMD", "DMS", "NO3",
        "CL", "BR", "IOD", "F", "MPD", "BME", "TLA", "SCN", "AZI", "NH4",
    }
)


def classify_residue(resname: str) -> str:
    """Map a PDB residue name to one of :data:`RESIDUE_CLASSES`."""
    name = resname.strip().upper()
    if name in WATER_NAMES:
        return "water"
    if name in METAL_NAMES:
        return "metal"
    return AA_CLASS.get(name, "unspecified")


# ---------------------------------------------------------------------------
# Visual vocabulary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GlyphStyle:
    """Fill gradient and outline of a residue droplet."""

    base: str      # saturated colour, used at the glyph rim and in the legend
    light: str     # near-white colour at the highlight
    outline: str
    text: str = "#303030"


#: Sampled from data/4ps5.png (hydrophobic #aad71b), data/2gfk.png (polar
#: #78d7f5) and data/6wak.png (charged positive #8080bf, glycine #f3f3db).
RESIDUE_STYLES: dict[str, GlyphStyle] = {
    "hydrophobic": GlyphStyle("#aad71b", "#f1f8d9", "#8ab512"),
    "polar": GlyphStyle("#78d7f5", "#e4f7fd", "#4fb6da"),
    "charged_negative": GlyphStyle("#f0793c", "#fde3d2", "#d55f26"),
    "charged_positive": GlyphStyle("#8f8fe0", "#e6e6fb", "#6f6fc8"),
    "glycine": GlyphStyle("#f3f3db", "#fdfdf2", "#cfcfae"),
    # Maestro draws water white, which on a white canvas is a ball only the
    # drop shadow makes visible.  A light red keeps it clearly a hydration site
    # (nothing else in the palette is pink) without competing with the orange
    # of the negatively charged residues.
    "water": GlyphStyle("#f4a9a9", "#fdeceb", "#d07f7f"),
    "metal": GlyphStyle("#9a9a9a", "#ececec", "#767676"),
    "unspecified": GlyphStyle("#9a9a9a", "#ececec", "#767676"),
}

#: Legend rows, in Maestro's order and wording.  Column 0 and 1 are residue
#: spheres, columns 2 and 3 are line samples.
LEGEND_RESIDUE_ROWS: tuple[tuple[str, str], ...] = (
    ("charged_negative", "Charged (negative)"),
    ("charged_positive", "Charged (positive)"),
    ("glycine", "Glycine"),
    ("hydrophobic", "Hydrophobic"),
    ("metal", "Metal"),
    ("polar", "Polar"),
    ("unspecified", "Unspecified residue"),
    ("water", "Water"),
)

LEGEND_EXTRA_ROWS: tuple[str, ...] = ("Hydration site", "Hydration site (displaced)")


@dataclass(frozen=True)
class InteractionStyle:
    """How one interaction kind is drawn."""

    label: str
    color: str
    width: float
    #: ``None`` for a solid line, otherwise a Qt dash pattern in line widths.
    dash: tuple[float, ...] | None
    #: ``arrow`` points at the acceptor, ``dot`` marks both ends, ``none`` is a
    #: bare line.
    marker: str = "none"
    #: Lower sorts first when several interactions share a residue.
    priority: int = 5


#: Colours sampled from the legend strip of data/4ps5.png.
INTERACTION_STYLES: dict[str, InteractionStyle] = {
    "hbond": InteractionStyle("H-bond", "#cc33ff", 1.6, (4.0, 2.4), "arrow", 1),
    "water_bridge": InteractionStyle("H-bond", "#cc33ff", 1.4, (3.0, 2.4), "arrow", 2),
    "halogen_bond": InteractionStyle("Halogen bond", "#cc9a00", 1.6, (4.0, 2.4), "arrow", 2),
    "salt_bridge": InteractionStyle("Salt bridge", "#0000ff", 1.6, None, "none", 0),
    "metal_coordination": InteractionStyle("Metal coordination", "#786781", 1.6, None, "none", 1),
    "pi_stacking": InteractionStyle("Pi-Pi stacking", "#149614", 1.8, None, "dot", 3),
    "pi_cation": InteractionStyle("Pi-cation", "#fa0014", 1.8, None, "dot", 3),
    "distance": InteractionStyle("Distance", "#149614", 1.4, (1.6, 2.4), "none", 6),
    "hydrophobic": InteractionStyle("Hydrophobic", "#9a9a9a", 1.2, (2.0, 3.0), "none", 7),
}

#: Legend line rows, in Maestro's order.  ``hydrophobic`` is deliberately not
#: listed: Maestro does not draw a line for it (the residue colour carries the
#: information), and neither do we.
LEGEND_LINE_ROWS: tuple[str, ...] = (
    "distance",
    "hbond",
    "halogen_bond",
    "metal_coordination",
    "pi_stacking",
    "pi_cation",
    "salt_bridge",
)

#: Every spelling of an interaction kind we accept from a detector.
INTERACTION_ALIASES: dict[str, str] = {
    "hbond": "hbond",
    "h_bond": "hbond",
    "hydrogen_bond": "hbond",
    "hbond_donor": "hbond",
    "hbond_acceptor": "hbond",
    "hbonds_ldon": "hbond",
    "hbonds_pdon": "hbond",
    "water_bridge": "water_bridge",
    "water_bridges": "water_bridge",
    "waterbridge": "water_bridge",
    "salt": "salt_bridge",
    "saltbridge": "salt_bridge",
    "salt_bridge": "salt_bridge",
    "saltbridge_lneg": "salt_bridge",
    "saltbridge_pneg": "salt_bridge",
    "hydrophobic": "hydrophobic",
    "hydrophobic_contact": "hydrophobic",
    "hydrophobic_contacts": "hydrophobic",
    "hydrophobic_interaction": "hydrophobic",
    "pi": "pi_stacking",
    "pi_stack": "pi_stacking",
    "pi_stacking": "pi_stacking",
    "pistacking": "pi_stacking",
    "pi_cation": "pi_cation",
    "pication": "pi_cation",
    "pication_laro": "pi_cation",
    "pication_paro": "pi_cation",
    "halogen": "halogen_bond",
    "halogen_bond": "halogen_bond",
    "halogen_bonds": "halogen_bond",
    "metal": "metal_coordination",
    "metal_complexes": "metal_coordination",
    "metal_coordination": "metal_coordination",
    "distance": "distance",
}


def normalize_kind(kind: str) -> str:
    """Map a detector-specific interaction name onto our vocabulary."""
    key = str(kind).strip().lower().replace("-", "_").replace(" ", "_")
    return INTERACTION_ALIASES.get(key, "distance")


#: RDKit's default heteroatom palette already matches the references
#: (N #0000ff, O #ff0000), so the ligand drawing does not override it.  These
#: are only used for the few labels we draw ourselves.
ATOM_COLORS: dict[str, str] = {
    "N": "#0000ff",
    "O": "#ff0000",
    "S": "#cccc00",
    "P": "#ff8000",
    "F": "#00cc00",
    "Cl": "#00cc00",
    "Br": "#992200",
    "I": "#6600cc",
}


# ---------------------------------------------------------------------------
# Diagram data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResidueRef:
    """Identity of a binding-site residue, water or metal ion."""

    chain: str
    number: int
    name: str
    insertion: str = ""

    @property
    def key(self) -> str:
        return f"{self.chain}:{self.number}{self.insertion}:{self.name}"

    @property
    def residue_class(self) -> str:
        return classify_residue(self.name)

    @property
    def label_lines(self) -> tuple[str, str]:
        """Two-line glyph caption, as in the references: ``PHE`` / ``A:93``."""
        return self.name.upper(), f"{self.chain}:{self.number}{self.insertion}"


@dataclass
class Residue:
    ref: ResidueRef
    #: ``False`` for residues drawn only because they line the pocket.
    has_interactions: bool = False
    #: Centroid of the residue's heavy atoms, from the PDB.  Used to anchor
    #: context residues onto the nearest ligand atom.
    center_3d: tuple[float, float, float] | None = None

    @property
    def key(self) -> str:
        return self.ref.key


@dataclass
class Interaction:
    """One protein-ligand contact, already mapped onto RDKit atom indices."""

    kind: str
    residue_key: str
    #: RDKit atom indices on the ligand side.  Several for ring interactions.
    ligand_atoms: tuple[int, ...]
    distance: float = 0.0
    #: ``True`` when the ligand donates (arrow points away from the ligand).
    ligand_is_donor: bool = True
    #: Set for water-mediated contacts; the bridging water's residue key.
    via_water: str | None = None
    #: Protein-side atom name when the detector can identify it (for example
    #: backbone ``N`` versus ASP side-chain ``OD1``).  This distinction is
    #: chemically important even though both contacts share one residue glyph.
    protein_atom: str | None = None
    #: D-H...A angle for a direct H-bond, or partner-water-partner angle for a
    #: water bridge.  Stored so JSON and tooltips expose why a line qualified.
    angle: float | None = None
    #: Water-to-protein distance for a water bridge.  ``distance`` is always
    #: the ligand-to-water leg; direct contacts leave this unset.
    protein_distance: float | None = None
    #: Donor direction of the protein-water leg.  Only meaningful for a water
    #: bridge; ``False`` means the water donates to the protein atom.
    protein_is_donor: bool | None = None

    @property
    def style(self) -> InteractionStyle:
        return INTERACTION_STYLES[self.kind]


@dataclass
class MetalLeg:
    """A leg of a metal ion's coordination sphere whose far end is not the ligand.

    The ligand-side legs are ordinary :class:`Interaction` rows; these are the
    protein residues and waters that complete the sphere.  They are what turns
    the metal glyph from a lone ball into a coordination polyhedron.
    """

    metal_key: str
    partner_key: str
    distance: float = 0.0


@dataclass
class Diagram:
    """Everything needed to lay out and draw one diagram."""

    name: str
    ligand_name: str
    #: RDKit ``Mol`` with 2D conformer; typed loosely so this module does not
    #: import RDKit.
    mol: object
    #: 2D coordinates per RDKit atom index, in RDKit units (~1.5 per bond).
    coords_2d: list[tuple[float, float]]
    residues: list[Residue] = field(default_factory=list)
    interactions: list[Interaction] = field(default_factory=list)
    #: RDKit atom index -> fraction of the atom's own surface the protein
    #: leaves free, for the atoms solvent still reaches in the complex.
    exposure: dict[int, float] = field(default_factory=dict)
    #: Nearest ligand atom (RDKit index) per residue key, from 3D distances.
    nearest_atom: dict[str, int] = field(default_factory=dict)
    #: Protein/water legs of every drawn metal's coordination sphere.
    metal_legs: list[MetalLeg] = field(default_factory=list)
    #: Coordination number per metal residue key, as detection assigned it.  Drives
    #: the metal glyph's shape: 4 -> square, 5 -> pentagon, 6 -> hexagon.
    metal_coordination: dict[str, int] = field(default_factory=dict)
    #: Alternative depictions of the same pose, same atom indexing as
    #: ``coords_2d``.  The layout optimiser picks whichever of them leaves the
    #: residues the most room, so this is where "try several projections of the
    #: 3D structure" is decided.
    coords_alt: list[list[tuple[float, float]]] = field(default_factory=list)
    #: Provenance that should survive JSON export (detector, source and ligand
    #: identity).  Rendering does not interpret these values.
    metadata: dict[str, object] = field(default_factory=dict)

    def residue(self, key: str) -> Residue:
        for residue in self.residues:
            if residue.key == key:
                return residue
        raise KeyError(key)

    def interactions_of(self, key: str) -> list[Interaction]:
        return [i for i in self.interactions if i.residue_key == key]

    def backbone_edges(self) -> list[tuple[str, str]]:
        """Sequence-consecutive residue pairs, both of which are drawn.

        These become the thin black connectors that chain the droplets into
        necklaces in the reference images.  Only a gap of exactly one residue
        number counts, so we never invent a link across a missing loop.
        """
        by_chain: dict[str, list[Residue]] = {}
        for residue in self.residues:
            if residue.ref.residue_class in ("water", "metal"):
                continue
            by_chain.setdefault(residue.ref.chain, []).append(residue)
        edges: list[tuple[str, str]] = []
        for group in by_chain.values():
            group.sort(key=lambda r: r.ref.number)
            for left, right in zip(group, group[1:]):
                if right.ref.number - left.ref.number == 1:
                    edges.append((left.key, right.key))
        return edges


# ---------------------------------------------------------------------------
# Small geometry helpers shared by layout and rendering
# ---------------------------------------------------------------------------

def centroid(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def segments_cross(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    """Proper intersection test for two open segments (shared ends don't count)."""

    def orient(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1, d2 = orient(c, d, a), orient(c, d, b)
    d3, d4 = orient(a, b, c), orient(a, b, d)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def point_segment_distance(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length_sq))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))
