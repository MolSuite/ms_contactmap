"""The diagram's own visual language: glyph shapes, palettes and line colours.

Shape follows the residue's side chain (ring size, charge sign, backbone
special cases); colour follows its chemical nature.  Every colour is computed
here from a hue and a palette level, so the three saturation levels stay
consistent with one another.  Pure Python: no Qt, no RDKit.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, replace

from .model import classify_residue

GLYPH_MODES = ("shapes", "circles")
COLOR_MODES = ("nature", "residue", "blend")
PALETTES = ("soft", "medium", "vivid")
SURFACE_MODES = ("line", "dots", "hidden")
EXPOSURE_MODES = ("arc", "dots", "trail", "hidden")
METAL_MODES = ("star", "polygon")
LEGEND_POSITIONS = ("left", "right", "top", "bottom")
LEGEND_ROW_OPTIONS = (2, 3, 4)
DEFAULT_LEGEND_POSITION = "left"
DEFAULT_LEGEND_ROWS = 3


@dataclass(frozen=True)
class DiagramStyle:
    """User-selectable appearance.  Changing it never re-solves the layout."""

    glyphs: str = "shapes"
    coloring: str = "nature"
    palette: str = "medium"
    surface: str = "line"
    exposure: str = "arc"
    metals: str = "star"
    hydrophobic_lines: bool = True

    def __post_init__(self) -> None:
        for value, allowed in (
            (self.glyphs, GLYPH_MODES),
            (self.coloring, COLOR_MODES),
            (self.palette, PALETTES),
            (self.surface, SURFACE_MODES),
            (self.exposure, EXPOSURE_MODES),
            (self.metals, METAL_MODES),
        ):
            if value not in allowed:
                raise ValueError(f"{value!r} is not one of {allowed}")

    def with_(self, **changes) -> "DiagramStyle":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict | None) -> "DiagramStyle":
        known = {k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

#: Shape family per residue.  Rings keep their ring size, charged side chains
#: point along the sign of their charge, residues without a free side chain
#: (Gly, Pro) are round.
SHAPE_OF: dict[str, str] = {
    **dict.fromkeys(("PHE", "TYR", "TRP"), "hexagon"),
    "HIS": "pentagon",
    **dict.fromkeys(("GLY", "PRO"), "circle"),
    **dict.fromkeys(("LYS", "ARG"), "house_up"),
    **dict.fromkeys(("ASP", "GLU"), "house_down"),
    **dict.fromkeys(("SER", "THR", "ASN", "GLN", "CYS"), "drop"),
    **dict.fromkeys(("ALA", "VAL", "LEU", "ILE", "MET"), "square"),
}

#: Legend rows for the shape families, in reading order.
SHAPE_ROWS: tuple[tuple[str, str], ...] = (
    ("square", "Aliphatic (ALA, VAL, LEU, ILE, MET)"),
    ("hexagon", "Aromatic 6-ring (PHE, TYR, TRP)"),
    ("pentagon", "Imidazole (HIS)"),
    ("drop", "Polar side chain (SER, THR, ASN, GLN, CYS)"),
    ("house_up", "Basic (LYS, ARG)"),
    ("house_down", "Acidic (ASP, GLU)"),
    ("circle", "No free side chain (GLY, PRO)"),
    ("star", "Metal (one point per coordination bond)"),
)


def shape_of(resname: str, style: DiagramStyle) -> str:
    if style.glyphs == "circles":
        return "circle"
    return SHAPE_OF.get(resname.strip().upper(), "circle")


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

#: (saturation, lightness) of the fill per palette level, in HLS.
_LEVELS: dict[str, tuple[float, float]] = {
    "soft": (0.60, 0.78),
    "medium": (0.72, 0.70),
    "vivid": (0.78, 0.60),
}

#: Hue in degrees, a saturation multiplier and an optional lightness shift per
#: chemical nature.  Water is lighter and pinker than the negative red.
_NATURE_HUE: dict[str, tuple[float, ...]] = {
    "hydrophobic": (92.0, 1.0),
    "polar": (192.0, 1.0),
    "charged_negative": (6.0, 1.0),
    "charged_positive": (224.0, 1.0),
    "glycine": (40.0, 0.22),
    "water": (338.0, 0.85, 0.1),
    "metal": (210.0, 0.08),
    "unspecified": (30.0, 0.06),
}

NATURE_ROWS: tuple[tuple[str, str], ...] = (
    ("charged_negative", "Negative"),
    ("charged_positive", "Positive"),
    ("glycine", "Glycine"),
    ("hydrophobic", "Hydrophobic"),
    ("metal", "Metal"),
    ("polar", "Polar"),
    ("water", "Water"),
    ("unspecified", "Other"),
)

#: Per-residue hues for "residue" colouring: the RasMol "amino" convention
#: that molecular viewers share, reduced to hue and softened by the palette.
_RESIDUE_HUE: dict[str, tuple[float, float]] = {
    **dict.fromkeys(("ASP", "GLU"), (0.0, 1.0)),
    **dict.fromkeys(("CYS", "MET"), (58.0, 1.0)),
    **dict.fromkeys(("LYS", "ARG"), (220.0, 1.0)),
    **dict.fromkeys(("SER", "THR"), (36.0, 1.0)),
    **dict.fromkeys(("PHE", "TYR"), (240.0, 0.75)),
    **dict.fromkeys(("ASN", "GLN"), (180.0, 1.0)),
    "GLY": (0.0, 0.0),
    **dict.fromkeys(("LEU", "VAL", "ILE"), (120.0, 0.9)),
    "ALA": (0.0, 0.0),
    "TRP": (300.0, 0.45),
    "HIS": (240.0, 0.45),
    "PRO": (14.0, 0.55),
}

#: Residues whose side chain is of mixed character: (main, secondary) nature.
#: Only used by the experimental "blend" colouring.
BLEND_NATURES: dict[str, tuple[str, str]] = {
    "TYR": ("hydrophobic", "polar"),
    "TRP": ("hydrophobic", "polar"),
    "CYS": ("hydrophobic", "polar"),
    "MET": ("hydrophobic", "polar"),
    "THR": ("polar", "hydrophobic"),
    "HIS": ("polar", "charged_positive"),
    "LYS": ("charged_positive", "hydrophobic"),
}

TEXT_COLOR = "#26303a"


def _hls(hue: float, sat: float, light: float) -> str:
    r, g, b = colorsys.hls_to_rgb((hue % 360.0) / 360.0, light, max(0.0, min(1.0, sat)))
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


@dataclass(frozen=True)
class Fill:
    """Glyph paint: one colour, or a two-stop blend from ``color`` to ``second``."""

    color: str
    outline: str
    second: str | None = None
    text: str = TEXT_COLOR


def _tone(hue_sat: tuple[float, ...], palette: str) -> tuple[str, str]:
    hue, sat_mul, shift = (*hue_sat, 0.0)[:3]
    sat, light = _LEVELS[palette]
    light = min(0.92, light + shift)
    return (_hls(hue, sat * sat_mul, light),
            _hls(hue, min(1.0, sat * sat_mul * 1.15), light - 0.26))


def nature_color(nature: str, palette: str) -> str:
    return _tone(_NATURE_HUE[nature], palette)[0]


def fill_for(resname: str, style: DiagramStyle) -> Fill:
    name = resname.strip().upper()
    nature = classify_residue(name)
    if style.coloring == "residue" and name in _RESIDUE_HUE:
        return Fill(*_tone(_RESIDUE_HUE[name], style.palette))
    if style.coloring == "blend" and name in BLEND_NATURES:
        main, second = BLEND_NATURES[name]
        color, outline = _tone(_NATURE_HUE[main], style.palette)
        return Fill(color, outline, nature_color(second, style.palette))
    return Fill(*_tone(_NATURE_HUE[nature], style.palette))


# ---------------------------------------------------------------------------
# Interaction lines
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LineStyle:
    label: str
    color: str
    width: float
    dash: tuple[float, ...] | None
    #: ``arrow`` points at the acceptor, ``dot`` marks both ends.
    marker: str = "none"
    #: Salt bridges run from the negative to the positive colour.
    gradient: tuple[str, str] | None = None


NEGATIVE_END = "#d8545a"
POSITIVE_END = "#4a74d6"

LINE_STYLES: dict[str, LineStyle] = {
    "hbond": LineStyle("Hydrogen bond", "#8f6bd1", 1.7, (3.2, 2.2), "arrow"),
    "water_bridge": LineStyle("Water bridge", "#a58fdc", 1.5, (2.4, 2.2), "arrow"),
    "halogen_bond": LineStyle("Halogen bond", "#2f9c96", 1.7, (3.2, 2.2), "arrow"),
    "salt_bridge": LineStyle("Salt bridge", NEGATIVE_END, 1.9, (5.0, 1.6), "none",
                             (NEGATIVE_END, POSITIVE_END)),
    "metal_coordination": LineStyle("Metal coordination", "#8a7560", 1.8, None, "none"),
    "pi_stacking": LineStyle("π–π stacking", "#3a9a6e", 1.8, (1.0, 2.2), "dot"),
    "pi_cation": LineStyle("π–cation", "#dd8a2f", 1.8, (1.0, 2.2), "dot"),
    "hydrophobic": LineStyle("Hydrophobic contact", "#8fc36a", 1.3, (1.2, 2.6), "none"),
    "distance": LineStyle("Distance", "#8c8c8c", 1.4, (1.6, 2.4), "none"),
}

LINE_ROWS: tuple[str, ...] = (
    "hbond", "water_bridge", "halogen_bond", "salt_bridge", "pi_stacking",
    "pi_cation", "hydrophobic", "metal_coordination", "distance",
)

#: Hydrophobic contact distance range mapped onto line strength, in angstrom.
HYDROPHOBIC_RANGE = (3.4, 4.0)
_HYDROPHOBIC_STRONG = "#4e9a2e"
_HYDROPHOBIC_WEAK = "#c2dcae"


def hydrophobic_color(distance: float) -> str:
    """Closer contacts draw in a deeper green, far ones fade out."""
    near, far = HYDROPHOBIC_RANGE
    t = 0.5 if not distance else min(1.0, max(0.0, (distance - near) / (far - near)))
    a = int(_HYDROPHOBIC_STRONG[1:], 16)
    b = int(_HYDROPHOBIC_WEAK[1:], 16)
    mix = [round(((a >> s) & 255) + ((((b >> s) & 255) - ((a >> s) & 255)) * t))
           for s in (16, 8, 0)]
    return "#{:02x}{:02x}{:02x}".format(*mix)


SURFACE_COLOR_ALPHA = 150
EXPOSURE_DOT_COLOR = "#6f7f90"
