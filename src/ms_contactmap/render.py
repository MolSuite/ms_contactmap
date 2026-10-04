"""Qt scene construction for the 2D protein-ligand interaction diagram.

:func:`build_scene` turns a :class:`~ms_contactmap.model.Diagram` plus the
coordinates chosen by ``ms_contactmap.layout`` into a ``QGraphicsScene`` whose
appearance follows Schrodinger Maestro's 2D diagrams (see ``data/*.png``).

Every dimension below is expressed in *scene units*.  One scene unit is half a
pixel of the reference PNGs: the reference droplets have a 12 px body, ours a
24 unit body, and everything else -- line widths, fonts, the legend grid -- was
measured off the references and doubled.  ``model.INTERACTION_STYLES`` widths
are already at this scale, which is how the factor was pinned down.

The layers, back to front, are one ``QGraphicsObject`` each:
solvent halos, ribbons, backbone connectors, interaction routes, the RDKit
ligand drawing, the residue droplets and the legend.

.. note::
   Import ``PySide6`` before ``rdkit`` in the hosting process.  The reverse
   order segfaults with rdkit 2025.3.5 / PySide6 6.10.2, so this module does
   its Qt imports first.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QByteArray, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtSvgWidgets import QGraphicsSvgItem
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsScene,
)
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D

from .model import (
    INTERACTION_STYLES,
    LEGEND_EXTRA_ROWS,
    LEGEND_LINE_ROWS,
    LEGEND_RESIDUE_ROWS,
    RESIDUE_STYLES,
    Diagram,
    centroid,
    classify_residue,
    point_segment_distance,
    segments_cross,
)
from . import style as st
from .style import DiagramStyle

# ---------------------------------------------------------------------------
# Measurements, all taken from data/*.png and doubled (see module docstring)
# ---------------------------------------------------------------------------

#: Sizes below were solved so the rendered advance widths match the reference
#: captions ("VAL" 21 units, "Charged (negative)" 144 units) in this family.
FONT_FAMILIES = ("DejaVu Sans", "Verdana", "Liberation Sans", "Arial")

#: Body circle of a droplet; the point sticks out to ``TIP_REACH`` radii.
DROPLET_RADIUS = 24.0
TIP_REACH = 1.62
#: Water is not a pocket residue -- Maestro shows it as a plain hydration-site
#: ball, so it gets a sphere at this fraction of a droplet body instead of a
#: teardrop.  Its number goes beside the ball: no sphere small enough to read as
#: water is big enough to hold "A:302" inside it.
WATER_RADIUS_FRAC = 0.52
WATER_LABEL_GAP = 5.0
DROPLET_OUTLINE_WIDTH = 0.9
DROPLET_NAME_PX = 9.0
DROPLET_CODE_PX = 9.0
DROPLET_LETTER_SPACING = 1.0
DROPLET_LINE_GAP = 13.0
#: The shadow measured off 4ps5.png is nearly hard: #9b9b9b, ~2 px offset and
#: ~1.5 px of falloff at reference scale.  Stacked strokes fake that falloff
#: without a QGraphicsEffect, which QGraphicsScene.render() drops.
SHADOW_OFFSET = 4.5
SHADOW_LAYERS = ((6.0, 8), (4.0, 14), (2.0, 22), (0.0, 62))

BOND_COLOR = "#303030"
LIGAND_BOND_WIDTH = 2
#: Heteroatom labels, in the same scene pixels as the residue captions -- the
#: two sit side by side, so anything else reads as a mistake.  It has to be
#: ``fixedFontSize``: ``baseFontSize`` is a fraction of RDKit's *drawing* scale,
#: and the canvas here is fitted to the layout rather than the other way round,
#: so every value of it came out clamped to ``minFontSize`` (6 px).
LIGAND_FONT_PX = 12.0
LIGAND_CANVAS_MARGIN = 1.6
#: RDKit scales the depiction to fill its canvas, so the hand-computed margin
#: above only sets the aspect ratio -- and a ``fixedFontSize`` label drawn wider
#: than the scale RDKit assumed then spills past the viewBox, where Qt clips it
#: (6wak's phosphates ran to -5.6 px on a 429 px canvas).  Padding is the slack
#: RDKit itself honours; the similarity fit absorbs the scale it costs.
LIGAND_CANVAS_PAD = 0.06
BACKBONE_COLOR = QColor("#141414")
BACKBONE_WIDTH = 3.0
BACKBONE_BOW = 0.09
#: Sequence context is a hint, not an interaction.  If consecutive residues
#: land on opposite faces of a large ligand, a heroic loop across the whole
#: canvas adds clutter and is less truthful than omitting that optional edge.
BACKBONE_MAX_SPAN = 260.0

RIBBON_WIDTH = 8.0
RIBBON_LIGAND_CLEARANCE = 24.0
RIBBON_DROPLET_INSET = 46.0
#: Radius of the final rolling probe.  The original 49 px radius erased useful
#: pocket curvature; 18 px suppresses pixel-scale wobble while preserving the
#: broad indentations made by the optimized residue positions.
RIBBON_SURFACE_PROBE_RADIUS = 18.0
#: The styled surface rolls a smaller probe: it follows the pocket more closely.
STYLED_SURFACE_PROBE_RADIUS = 11.0
#: Angular sampling interval for the radial surface reconstruction.
RIBBON_SAMPLE_STEP = math.radians(1.0)
#: Angular padding, in radians, added past the first and last residue of a run.
RIBBON_OVERHANG = 0.22
RIBBON_FADE = 0.28
RIBBON_MAX_GAP = math.radians(58.0)
#: A residue substantially behind a nearer neighbour at the same bearing is a
#: second-row annotation.  It must not pull the apparent pocket surface out.
RIBBON_FRONT_ANGLE = math.radians(20.0)
RIBBON_BACK_ROW_GAP = 42.0

HALO_COLOR = QColor(70, 70, 70)
#: Halo radius as a fraction of the ligand bond length (~2.5 atom radii).
HALO_RADIUS_FRAC = 0.56
#: Multipliers on that radius for a barely-exposed and a fully-exposed atom.
#: Centred on 1.0 so a mid-exposure atom draws the mark this always drew, and
#: kept narrow: the halo has to stay a texture behind the drawing, and at 1.6x
#: the exposed face of a ring merges into one grey blob.
HALO_SIZE_RANGE = (0.70, 1.30)
#: Trail mode: angular half-width of the arc, its stand-off from the atom as a
#: fraction of the halo radius, and its stroke width at full exposure.
TRAIL_SPAN = math.radians(72.0)
TRAIL_OFFSET = 0.72
TRAIL_WIDTH = 7.0

#: Width of the invisible band around a route that counts as hovering it.  A
#: 1.2 px dashed hydrophobic line is not something a mouse can be asked to hit.
HOVER_SLOP = 11.0
#: Multiplier on the pen width of the hovered route, and on the glyph outline
#: of the residues it joins.
HOVER_BOLD = 2.2
#: Styled metal coordination: peak half-height and period of the wave, px.
WAVE_AMPLITUDE = 3.6
WAVE_LENGTH = 11.0

#: Kinds whose ligand side is a ring or a delocalised system, so the route
#: belongs at the centroid.  Everything else starts on a single atom.
RING_ANCHORED = frozenset({"pi_stacking", "pi_cation"})

ROUTE_ATOM_GAP = 11.0
ROUTE_BEND = 0.17
ARROW_LENGTH = 10.0
ARROW_HALF_WIDTH = 4.6
DOT_RADIUS = 3.2

LEGEND_ROW_PITCH = 20.5
LEGEND_COL_PITCH = 252.0
LEGEND_SPHERE_RADIUS = 10.0
LEGEND_SAMPLE_LENGTH = 28.0
LEGEND_TEXT_OFFSET = 42.0
LEGEND_FONT_PX = 15.0
#: Legend row shown when nothing supplied the ligand's bond orders.
LEGEND_BOND_ORDER_WARNING = "Bond orders undetermined: chemistry not reliable"
LEGEND_WARNING_COLOR = QColor("#d97706")
LEGEND_GAP = 42.0
LEGEND_CROSS_COLOR = QColor("#ee0000")
#: The reference legend draws the salt-bridge sample as a red-to-blue blend
#: (negative to positive), unlike the plain blue of ``INTERACTION_STYLES``.
SALT_BRIDGE_LEGEND = ("#fa0014", "#0000ff")

#: Styled look: surface dots, exposure dots, glyph outline and legend box.
SURFACE_LINE_WIDTH = 6.0
SURFACE_DOT_STEP = 10.0
SURFACE_DOT_RADIUS = 2.4
EXPOSURE_DOT_RADIUS = 1.5
EXPOSURE_DOT_COUNT = 26
GLYPH_OUTLINE_WIDTH = 1.1
#: Corner rounding of the styled glyphs, as a fraction of the glyph radius.
GLYPH_CORNER = 0.3
#: Metal star: tip reach and inner radius (fraction of the tip reach), and
#: the tip rounding as a fraction of the droplet radius.
STAR_REACH = 1.75
STAR_INNER = 0.48
STAR_CORNER = 0.14
GLYPH_SHADOW = (2.2, 34)
TABLE_PAD = 12.0
STYLED_BACKBONE_COLOR = QColor("#7d8793")
STYLED_BACKBONE_WIDTH = 2.0
STYLED_WATER_SCALE = 0.6
EXPOSURE_DOT_REACH = 0.8
#: Exposure arc: sampling of the free surface, the radius (fraction of the
#: halo) at which a direction counts as buried, the halo opacity gain, and
#: the number and angular step of the stacked wedges that fade its sides.
ARC_SAMPLES = 36
ARC_RING = 0.56
ARC_ALPHA = 1.0
#: Arc opacity from the atom (0) out to the halo radius (1): dark at the atom.
ARC_PROFILE = ((0.0, 64), (0.3, 50), (0.62, 22), (0.85, 7), (1.0, 0))
ARC_FEATHER = 4
ARC_FEATHER_STEP = math.radians(7.0)
TABLE_ROW_PITCH = 21.0
TABLE_TITLE_PX = 13.0
TABLE_TEXT_PX = 12.0
TABLE_ICON_RADIUS = 8.5
TABLE_TEXT_OFFSET = 40.0
TABLE_SECTION_GAP = 18.0
TABLE_BORDER = QColor("#c9ced6")
TABLE_FILL = QColor("#fbfbfc")
TABLE_TEXT = QColor("#2b323b")
TABLE_ICON_FILL = QColor("#e4e7eb")
TABLE_ICON_OUTLINE = QColor("#6b7480")

Z_HALOS, Z_RIBBONS, Z_BACKBONE, Z_ROUTES = -40, -30, -20, -10
Z_LIGAND, Z_DROPLETS, Z_LEGEND = 0, 10, 20


def _font(pixel_size: float, letter_spacing: float = 0.0) -> QFont:
    font = QFont()
    font.setFamilies(list(FONT_FAMILIES))
    font.setPixelSize(int(round(pixel_size)))
    if letter_spacing:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, letter_spacing)
    return font


def _darker(color: str, factor: float = 0.78) -> QColor:
    c = QColor(color)
    return QColor.fromHsvF(c.hueF(), min(1.0, c.saturationF() * 1.25), c.valueF() * factor)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _teardrop(center: QPointF, radius: float, tip_angle: float) -> QPolygonF:
    """Maestro's droplet: a circle with a point aimed along ``tip_angle``.

    The two flanks are the tangents from the tip to the body circle, which is
    what keeps the join from kinking.
    """
    reach = radius * TIP_REACH
    half = math.acos(radius / reach)
    tip = QPointF(center.x() + reach * math.cos(tip_angle),
                  center.y() + reach * math.sin(tip_angle))
    pts = [tip]
    span = 2 * math.pi - 2 * half
    steps = 56
    for i in range(steps + 1):
        a = tip_angle + half + span * i / steps
        pts.append(QPointF(center.x() + radius * math.cos(a),
                           center.y() + radius * math.sin(a)))
    return QPolygonF(pts)


def _ngon(center: QPointF, radius: float, sides: int, angle: float) -> QPolygonF:
    """Regular polygon with a vertex on ``angle`` and inradius ``radius``.

    Sizing by the inradius keeps the caption inside whatever the coordination
    number turns out to be, instead of letting a triangle swallow its own text.
    """
    reach = radius / math.cos(math.pi / sides)
    return QPolygonF(
        [
            QPointF(center.x() + reach * math.cos(angle + 2 * math.pi * i / sides),
                    center.y() + reach * math.sin(angle + 2 * math.pi * i / sides))
            for i in range(sides)
        ]
    )


def _star(radius: float, sides: int, angle: float) -> QPolygonF:
    """Star with a point on ``angle`` and one per coordination bond.

    The inner radius leaves about half of ``radius`` free for the caption,
    whatever the number of points.
    """
    inner = radius * STAR_INNER
    pts = []
    for i in range(2 * sides):
        a = angle + math.pi * i / sides
        pts.append(_along(QPointF(0, 0), a, radius if i % 2 == 0 else inner))
    return QPolygonF(pts)


def _round_corners(polygon: QPolygonF, corner: float, steps: int = 6) -> QPolygonF:
    """Replace every sharp vertex with a curve that starts ``corner`` before it.

    The curve is a quadratic Bezier with the vertex as control point; nearly
    straight vertices (the sampled arc of a drop) are left alone.
    """
    pts = [polygon[i] for i in range(polygon.size())]
    out = []
    for i, p in enumerate(pts):
        a, b = pts[i - 1], pts[(i + 1) % len(pts)]
        la, lb = math.dist((a.x(), a.y()), (p.x(), p.y())), math.dist((b.x(), b.y()), (p.x(), p.y()))
        if la < 1e-9 or lb < 1e-9:
            continue
        ua = QPointF((a.x() - p.x()) / la, (a.y() - p.y()) / la)
        ub = QPointF((b.x() - p.x()) / lb, (b.y() - p.y()) / lb)
        if ua.x() * ub.x() + ua.y() * ub.y() < -0.985:
            out.append(p)
            continue
        d = min(corner, 0.45 * la, 0.45 * lb)
        t1, t2 = p + ua * d, p + ub * d
        for k in range(steps + 1):
            t = k / steps
            out.append(t1 * (1 - t) ** 2 + p * (2 * t * (1 - t)) + t2 * t * t)
    return QPolygonF(out)


def _house(radius: float, up: bool) -> QPolygonF:
    """Square body with a roof; the roof points along the sign of the charge."""
    s = -1.0 if up else 1.0
    return QPolygonF([
        QPointF(0.0, s * 1.45 * radius),
        QPointF(radius, s * 0.45 * radius),
        QPointF(radius, -s * radius),
        QPointF(-radius, -s * radius),
        QPointF(-radius, s * 0.45 * radius),
    ])


def glyph_polygon(shape: str, radius: float) -> tuple[QPolygonF, float]:
    """Outline of a styled glyph and the vertical offset of its caption."""
    if shape == "hexagon":
        polygon, dy = _ngon(QPointF(0, 0), radius * 1.02, 6, 0.0), 0.0
    elif shape == "pentagon":
        polygon, dy = _ngon(QPointF(0, 0), radius * 1.02, 5, -math.pi / 2), 0.1 * radius
    elif shape in ("house_up", "house_down"):
        polygon = _house(radius, shape == "house_up")
        dy = 0.25 * radius if shape == "house_up" else -0.25 * radius
    elif shape == "drop":
        dy = 0.2 * radius
        polygon = _teardrop(QPointF(0, dy), radius, -math.pi / 2)
    elif shape == "star":
        return _round_corners(_star(radius * 1.35, 4, -math.pi / 2), radius * STAR_CORNER), 0.0
    elif shape == "square":
        half = radius * 0.98
        polygon, dy = QPolygonF([QPointF(half, half), QPointF(-half, half),
                                 QPointF(-half, -half), QPointF(half, -half)]), 0.0
    else:
        return _circle(QPointF(0, 0), radius * 1.04, 48), 0.0
    return _round_corners(polygon, radius * GLYPH_CORNER), dy


def _along(origin: QPointF, angle: float, distance: float) -> QPointF:
    return QPointF(origin.x() + distance * math.cos(angle),
                   origin.y() + distance * math.sin(angle))


def _arc_path(center: QPointF, radius: float, start: float, end: float,
              pie: bool = False) -> QPainterPath:
    steps = max(1, math.ceil(abs(end - start) / math.radians(4)))
    path = QPainterPath(center if pie else _along(center, start, radius))
    if pie:
        path.lineTo(_along(center, start, radius))
    for k in range(1, steps + 1):
        path.lineTo(_along(center, start + (end - start) * k / steps, radius))
    return path


def _wrap(angle: float) -> float:
    """``angle`` folded into (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _vertex_angle(bearings: list[float], sides: int, steps: int = 180) -> float:
    """Rotation that lines a ``sides``-gon's vertices up with ``bearings``.

    The circular mean of ``sides * bearing`` is the closed-form least-squares
    fit, but it lets every partner pull on whichever corner is nearest *to it*,
    independently.  Corners are handed out exclusively, so that is the wrong
    objective: two partners 30 degrees apart both drag the same corner and the
    polygon settles between them, which is 2gfk's ZN 402 sitting edge-on to
    three of its five ligands when a quarter turn would give each one a corner.

    So score what actually happens: sweep the rotation over one symmetry period
    and, at each step, run the same greedy assignment the renderer uses.  The
    winner is then refined to the exact mean of its own assignment, which makes
    the answer independent of ``steps`` whenever the fit is good.  ``sides`` is
    at most 8, so the sweep is a few thousand operations per metal.
    """
    if not bearings:
        return -math.pi / 2
    order = sorted(bearings)
    period = 2.0 * math.pi / sides

    def assign(angle: float) -> tuple[float, list[tuple[float, int]]]:
        free = list(range(sides))
        cost, pairs = 0.0, []
        for b in order:
            if not free:
                break
            j = min(free, key=lambda i: abs(_wrap(angle + i * period - b)))
            free.remove(j)
            cost += _wrap(angle + j * period - b) ** 2
            pairs.append((b, j))
        return cost, pairs

    best = min((assign(k * period / steps)[0], k * period / steps) for k in range(steps))
    angle = best[1]
    pairs = assign(angle)[1]
    # Least-squares rotation for a fixed assignment: shift by the mean residual.
    return angle + sum(_wrap(b - angle - j * period) for b, j in pairs) / len(pairs)


def _circle(center: QPointF, radius: float, steps: int = 40) -> QPolygonF:
    return QPolygonF(
        [
            QPointF(center.x() + radius * math.cos(2 * math.pi * i / steps),
                    center.y() + radius * math.sin(2 * math.pi * i / steps))
            for i in range(steps)
        ]
    )


def _ray_hit(origin: QPointF, target: QPointF, polygon: QPolygonF) -> QPointF:
    """Where the segment ``origin``-``target`` last crosses ``polygon``."""
    best = target
    best_t = 2.0
    dx, dy = target.x() - origin.x(), target.y() - origin.y()
    n = polygon.count()
    for i in range(n):
        a, b = polygon.at(i), polygon.at((i + 1) % n)
        ex, ey = b.x() - a.x(), b.y() - a.y()
        den = dx * ey - dy * ex
        if abs(den) < 1e-9:
            continue
        t = ((a.x() - origin.x()) * ey - (a.y() - origin.y()) * ex) / den
        u = ((a.x() - origin.x()) * dy - (a.y() - origin.y()) * dx) / den
        if 0.0 <= u <= 1.0 and 0.0 <= t < best_t:
            best_t, best = t, QPointF(origin.x() + t * dx, origin.y() + t * dy)
    return best


def _blend(a: QColor, b: QColor, t: float) -> QColor:
    return QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
    )


# ---------------------------------------------------------------------------
# Layer 1 -- solvent halos
# ---------------------------------------------------------------------------

@dataclass
class _Exposure:
    """One solvent-reachable ligand atom, and how much of it is reachable."""

    at: QPointF
    #: 0-1, the share of the atom's own surface the protein leaves free.
    fraction: float
    #: Scene bearing the free surface faces, for the trail representation.
    facing: float


class SolventHalos(QGraphicsObject):
    """The solvent-exposure marks, in either of two representations.

    ``halo`` is Maestro's: a diffuse grey ring centred on the atom, sized by
    how exposed it is.  ``trail`` instead puts an arc on the side the free
    surface actually faces, which says *where* the solvent reaches rather than
    just *that* it does -- useful on a ligand half-buried edge-on, where a
    field of concentric rings tells you nothing about which face is out.

    Both are drawn from the same numbers, computed once in
    :func:`_exposure_spots`; switching is a repaint.
    """

    MODES = ("halo", "trail", "dots", "arc")

    def __init__(self, spots: list[_Exposure], radius: float, mode: str = "halo",
                 atoms: list[QPointF] | None = None):
        super().__init__()
        self._spots = spots
        self._radius = radius
        self._mode = mode
        self._atoms = atoms or []
        self._dots: list[tuple[QPointF, float]] | None = None
        self._arcs: list[tuple[_Exposure, float, float]] | None = None
        self.setZValue(Z_HALOS)

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown exposure mode {mode!r}")
        if mode != self._mode:
            self.prepareGeometryChange()  # the two reach different distances
            self._mode = mode
            self.update()

    def _radius_of(self, spot: _Exposure) -> float:
        lo, hi = HALO_SIZE_RANGE
        return self._radius * (lo + (hi - lo) * spot.fraction)

    def boundingRect(self) -> QRectF:
        if not self._spots:
            return QRectF()
        # Padded per point before uniting: a rect on a single point is null and
        # QRectF.united() drops a null operand (see Ribbons).
        rect = QRectF()
        for spot in self._spots:
            r = self._radius_of(spot) * (1.0 + TRAIL_OFFSET) + TRAIL_WIDTH
            box = QRectF(spot.at, spot.at).adjusted(-r, -r, r, r)
            rect = box if rect.isNull() else rect.united(box)
        return rect

    def surface_dots(self) -> list[tuple[QPointF, float]]:
        """Points of the 2D probe surface that face the solvent.

        Each exposed atom gets a ring of points; points buried by another atom
        of the drawing are dropped, and of the rest only an arc around the free
        direction is kept, wider the more exposed the atom is.  So the dots sit
        where the solvent is, not over the whole atom.
        """
        if self._dots is None:
            self._dots = []
            for spot in self._spots:
                rho = self._radius * EXPOSURE_DOT_REACH
                for k in range(EXPOSURE_DOT_COUNT):
                    a = spot.facing + 2.0 * math.pi * (k / EXPOSURE_DOT_COUNT - 0.5)
                    if self._is_free(spot, a, rho):
                        self._dots.append((_along(spot.at, a, rho), spot.fraction))
        return self._dots

    def _is_free(self, spot: _Exposure, angle: float, rho: float) -> bool:
        """Whether the surface point at ``angle`` faces the solvent.

        The kept arc widens with exposure; a point closer than ``rho`` to
        another atom of the drawing is buried.
        """
        if abs(_wrap(angle - spot.facing)) > math.pi * (0.25 + 0.6 * spot.fraction):
            return False
        p = _along(spot.at, angle, rho)
        return not any(_sq_dist(p, q) < rho * rho for q in self._atoms
                       if _sq_dist(q, spot.at) > 1e-6)

    def surface_arcs(self) -> list[tuple[_Exposure, float, float]]:
        """``(spot, start, end)`` angle runs of the free surface, per atom."""
        if self._arcs is None:
            self._arcs = []
            step = 2.0 * math.pi / ARC_SAMPLES
            for spot in self._spots:
                rho = self._radius_of(spot) * ARC_RING
                start = None
                for k in range(ARC_SAMPLES + 1):
                    a = spot.facing + step * (k - ARC_SAMPLES / 2)
                    free = k < ARC_SAMPLES and self._is_free(spot, a, rho)
                    if free and start is None:
                        start = a
                    elif not free and start is not None:
                        self._arcs.append((spot, start, a - step))
                        start = None
        return self._arcs

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self._mode == "trail":
            self._paint_trails(painter)
            return
        if self._mode == "arc":
            painter.setPen(Qt.PenStyle.NoPen)
            for spot, start, end in self.surface_arcs():
                _paint_halo_wedge(painter, spot.at, self._radius_of(spot), start, end)
            return
        if self._mode == "dots":
            painter.setPen(Qt.PenStyle.NoPen)
            for p, fraction in self.surface_dots():
                color = QColor(st.EXPOSURE_DOT_COLOR)
                color.setAlpha(int(110 + 120 * fraction))
                painter.setBrush(color)
                painter.drawEllipse(p, EXPOSURE_DOT_RADIUS, EXPOSURE_DOT_RADIUS)
            return
        painter.setPen(Qt.PenStyle.NoPen)
        for spot in self._spots:
            r = self._radius_of(spot)
            painter.setBrush(QBrush(halo_gradient(spot.at, r)))
            painter.drawEllipse(spot.at, r, r)

    def _paint_trails(self, painter: QPainter) -> None:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for spot in self._spots:
            r = self._radius_of(spot) * (1.0 + TRAIL_OFFSET)
            color = QColor(HALO_COLOR)
            color.setAlpha(int(38 + 92 * spot.fraction))
            pen = QPen(color, TRAIL_WIDTH * (0.45 + 0.55 * spot.fraction))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            box = QRectF(spot.at.x() - r, spot.at.y() - r, 2 * r, 2 * r)
            # Qt's arc angles are 1/16 degree and run anticlockwise, while the
            # scene's y axis points down, so the bearing has to be negated.
            painter.drawArc(box, int(math.degrees(-spot.facing - TRAIL_SPAN) * 16),
                            int(math.degrees(2 * TRAIL_SPAN) * 16))


def _exposure_spots(diagram: Diagram, atom_coords: dict[int, QPointF],
                    ligand_coords: list[tuple[float, float]]) -> list[_Exposure]:
    """Where each exposed atom is, how exposed, and which way its free face looks.

    The facing is taken as the direction away from the atom's bonded
    neighbours, not away from the ligand centroid: on a concave ligand the
    centroid is on the wrong side, and it is the bonds that actually shadow
    the atom.
    """
    center = QPointF(*centroid(ligand_coords))
    spots: list[_Exposure] = []
    for idx, fraction in sorted(diagram.exposure.items()):
        at = atom_coords.get(idx)
        if at is None:
            continue
        away = [atom_coords[n.GetIdx()]
                for n in diagram.mol.GetAtomWithIdx(idx).GetNeighbors()
                if n.GetIdx() in atom_coords]
        if away:
            dx = at.x() - sum(p.x() for p in away) / len(away)
            dy = at.y() - sum(p.y() for p in away) / len(away)
        else:
            dx, dy = at.x() - center.x(), at.y() - center.y()
        spots.append(_Exposure(at, fraction, math.atan2(dy, dx or 1e-9)))
    return spots


def halo_gradient(center: QPointF, radius: float, gain: float = 1.0) -> QRadialGradient:
    """The ring profile measured off the pyrrolidine halos in 4ps5.png."""
    g = QRadialGradient(center, radius)
    for stop, alpha in ((0.0, 20), (0.30, 15), (0.52, 46), (0.68, 34), (0.86, 12), (1.0, 0)):
        c = QColor(HALO_COLOR)
        c.setAlpha(min(255, round(alpha * gain)))
        g.setColorAt(stop, c)
    return g




def _paint_halo_wedge(painter: QPainter, center: QPointF, radius: float,
                      start: float, end: float, gain: float = ARC_ALPHA) -> None:
    """A halo sector over ``start``..``end``, darkest at the atom.

    Stacked wedges, each wider than the last and each carrying a share of the
    opacity, fade the straight sides out instead of cutting them: the core
    gets every layer, the flanks fewer.
    """
    g = QRadialGradient(center, radius)
    for stop, alpha in ARC_PROFILE:
        c = QColor(HALO_COLOR)
        c.setAlpha(round(alpha * gain / ARC_FEATHER))
        g.setColorAt(stop, c)
    painter.setBrush(QBrush(g))
    for k in range(ARC_FEATHER):
        grow = ARC_FEATHER_STEP * k
        painter.drawPath(_arc_path(center, radius, start - grow, end + grow, pie=True))


# ---------------------------------------------------------------------------
# Layer 2 -- ribbons
# ---------------------------------------------------------------------------

@dataclass
class _Ribbon:
    points: list[QPointF]
    colors: list[QColor]


def _ligand_reach(coords: list[tuple[float, float]], center: QPointF, angle: float) -> float:
    """How far the ligand extends from ``center`` along ``angle``."""
    ux, uy = math.cos(angle), math.sin(angle)
    return max(
        (0.0, *((x - center.x()) * ux + (y - center.y()) * uy for x, y in coords))
    )


class Ribbons(QGraphicsObject):
    """Thick pastel sweeps standing off the ligand along runs of residues.

    Painted as a chain of round-capped segments rather than one stroked path
    with a ``QLinearGradient``: the sweeps are C-shaped, and a linear gradient
    only ramps correctly along a straight chord.  The per-segment colours also
    reproduce the class-to-class blends in the references (green to cyan in
    4ps5.png, green to orange in 4uwh.png).
    """

    def __init__(self, ribbons: list[_Ribbon], mode: str = "classic"):
        super().__init__()
        self._ribbons = ribbons
        self._mode = mode
        self.setZValue(Z_RIBBONS)

    def boundingRect(self) -> QRectF:
        # Not built by uniting per-point QRectFs: a rect on a single point is
        # null, and QRectF.united() drops a null operand, so that came out as a
        # 16x16 box at the origin.  Harmless while nothing clipped to it, fatal
        # once the item is cached into a pixmap of exactly this size.
        xs = [p.x() for ribbon in self._ribbons for p in ribbon.points]
        ys = [p.y() for ribbon in self._ribbons for p in ribbon.points]
        if not xs:
            return QRectF()
        pad = RIBBON_WIDTH
        return QRectF(min(xs) - pad, min(ys) - pad,
                      max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self._mode == "dots":
            self._paint_dots(painter)
            return
        width = SURFACE_LINE_WIDTH if self._mode == "line" else RIBBON_WIDTH
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for ribbon in self._ribbons:
            pts, cols = ribbon.points, ribbon.colors
            last = len(pts) - 2
            for i in range(len(pts) - 1):
                pen = QPen(cols[i], width)
                # Flat caps: consecutive segments abut instead of overlapping,
                # which would bead visibly wherever the alpha is below 255.
                pen.setCapStyle(Qt.PenCapStyle.RoundCap if i in (0, last)
                                else Qt.PenCapStyle.FlatCap)
                painter.setPen(pen)
                painter.drawLine(pts[i], pts[i + 1])

    def _paint_dots(self, painter: QPainter) -> None:
        """The same surface as a row of evenly spaced beads."""
        painter.setPen(Qt.PenStyle.NoPen)
        for ribbon in self._ribbons:
            pts, cols = ribbon.points, ribbon.colors
            travelled = SURFACE_DOT_STEP / 2
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                length = math.hypot(b.x() - a.x(), b.y() - a.y())
                while travelled <= length:
                    t = travelled / length
                    painter.setBrush(cols[i])
                    painter.drawEllipse(QPointF(a.x() + (b.x() - a.x()) * t,
                                                a.y() + (b.y() - a.y()) * t),
                                        SURFACE_DOT_RADIUS, SURFACE_DOT_RADIUS)
                    travelled += SURFACE_DOT_STEP
                travelled -= length


def _ribbon_runs(diagram: Diagram, positions: dict[str, QPointF],
                 center: QPointF, ligand_coords: list[tuple[float, float]],
                 look: DiagramStyle | None = None) -> list[_Ribbon]:
    palette = None if look is None else look.palette
    probe = RIBBON_SURFACE_PROBE_RADIUS if look is None else STYLED_SURFACE_PROBE_RADIUS
    metal_reach = (DROPLET_RADIUS * STAR_REACH if look is not None and look.metals == "star"
                   else DROPLET_RADIUS)
    entries = []
    for residue in diagram.residues:
        p = positions.get(residue.key)
        if p is None or residue.ref.residue_class in ("water", "metal"):
            continue
        dx, dy = p.x() - center.x(), p.y() - center.y()
        entries.append((math.atan2(dy, dx), math.hypot(dx, dy),
                        residue.ref.residue_class))
    if len(entries) < 2:
        return []
    entries.sort()

    # Surface is reconstructed only from the front row.  Context residues may
    # occupy row two to avoid collisions; using their larger radius here was
    # the reason an otherwise good layout sometimes acquired a huge empty
    # cavity between the ligand and its ribbon.
    front = []
    for angle, radius, residue_class in entries:
        nearer = [
            other_radius
            for other_angle, other_radius, _ in entries
            if abs(float((other_angle - angle + math.pi) % (2 * math.pi) - math.pi))
            <= RIBBON_FRONT_ANGLE
            and other_radius < radius
        ]
        if nearer and radius - min(nearer) > RIBBON_BACK_ROW_GAP:
            continue
        front.append((angle, radius, residue_class))
    entries = front
    if len(entries) < 2:
        return []

    runs: list[list[tuple[float, float, str, float]]] = [[entries[0]]]
    for prev, cur in zip(entries, entries[1:]):
        if cur[0] - prev[0] > RIBBON_MAX_GAP:
            runs.append([])
        runs[-1].append(cur)
    # The circle wraps: join the last run onto the first when they meet.
    if len(runs) > 1 and (entries[0][0] + 2 * math.pi - entries[-1][0]) <= RIBBON_MAX_GAP:
        runs[0] = [(a - 2 * math.pi, r, c)
                   for a, r, c in runs.pop()] + runs[0]

    ribbons: list[_Ribbon] = []
    for run in runs:
        if len(run) < 2:
            continue
        angles = ([run[0][0] - RIBBON_OVERHANG]
                  + [a for a, _, _ in run]
                  + [run[-1][0] + RIBBON_OVERHANG])
        radii = [run[0][1]] + [r for _, r, _ in run] + [run[-1][1]]
        residue_classes = [run[0][2]] + [c for _, _, c in run] + [run[-1][2]]
        classes = [RESIDUE_STYLES[value].base if palette is None
                   else st.nature_color(value, "medium" if palette == "soft" else "vivid")
                   for value in residue_classes]

        # Reconstruct a radial contour after optimization, then roll a probe
        # twice the ligand probe over it.  Unlike an interpolating Catmull-Rom
        # spline, this does not reproduce every local radial wobble or overshoot
        # between two droplets at different depths.
        count = max(2, int(math.ceil((angles[-1] - angles[0]) / RIBBON_SAMPLE_STEP)) + 1)
        sample_angles = np.linspace(angles[0], angles[-1], count)
        raw = np.interp(sample_angles, angles, np.asarray(radii) - RIBBON_DROPLET_INSET)
        step = max(float(sample_angles[1] - sample_angles[0]), 1e-6)
        mean_radius = max(float(np.mean(raw)), 1.0)
        sigma = (probe / mean_radius) / step
        kernel = _gaussian_samples(sigma)
        pad = len(kernel) // 2
        smooth = np.convolve(np.pad(raw, (pad, pad), mode="edge"), kernel, mode="valid")
        clear = np.array([
            _ligand_reach(ligand_coords, center, float(angle)) + RIBBON_LIGAND_CLEARANCE
            for angle in sample_angles
        ])
        smooth = np.maximum(smooth, clear)

        # Waters and metals are exposed pocket waypoints, not protein-surface
        # residues.  A metal can nevertheless sit at the same bearing as two
        # protein droplets; interpolation would then draw the ribbon through
        # its coordination polygon.  Raise a small smooth local shoulder so
        # the complete metal remains on the ligand-facing side of the surface.
        for residue in diagram.residues:
            if residue.ref.residue_class != "metal":
                continue
            metal = positions.get(residue.key)
            if metal is None:
                continue
            dx, dy = metal.x() - center.x(), metal.y() - center.y()
            metal_angle = math.atan2(dy, dx)
            while metal_angle < sample_angles[0]:
                metal_angle += 2.0 * math.pi
            while metal_angle > sample_angles[-1]:
                metal_angle -= 2.0 * math.pi
            if metal_angle < sample_angles[0] or metal_angle > sample_angles[-1]:
                continue
            at = int(np.argmin(np.abs(sample_angles - metal_angle)))
            distance = math.hypot(dx, dy)
            required = distance + metal_reach + 10.0
            lift = max(0.0, required - float(smooth[at]))
            if lift > 0.0:
                # Wide enough to clear the whole glyph, not just its centre.
                sigma = max(0.14, 1.2 * metal_reach / max(distance, 1.0))
                smooth += lift * np.exp(-0.5 * ((sample_angles - metal_angle) / sigma) ** 2)
        pts = [
            QPointF(center.x() + radius * math.cos(angle),
                    center.y() + radius * math.sin(angle))
            for angle, radius in zip(sample_angles, smooth)
        ]
        span = len(pts) - 1
        colors = []
        class_axis = np.asarray(angles)
        for i, angle in enumerate(sample_angles):
            t = i / max(1, span)
            k = float(np.interp(angle, class_axis, np.arange(len(classes), dtype=float)))
            lo = min(int(k), len(classes) - 2)
            col = _blend(QColor(classes[lo]), QColor(classes[lo + 1]), k - lo)
            fade = min(1.0, t / RIBBON_FADE, (1.0 - t) / RIBBON_FADE)
            col.setAlpha(int(255 * max(0.0, fade) ** 0.9))
            colors.append(col)
        ribbons.append(_Ribbon(pts, colors))
    return ribbons


def _gaussian_samples(sigma: float) -> np.ndarray:
    """Open-curve Gaussian kernel used by the final rolling-probe pass."""
    if sigma <= 0.35:
        return np.ones(1)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    values = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (values / sigma) ** 2)
    return kernel / kernel.sum()


# ---------------------------------------------------------------------------
# Layer 3 -- backbone connectors
# ---------------------------------------------------------------------------

class BackboneConnectors(QGraphicsObject):
    """Thin black links between sequence-consecutive droplets."""

    def __init__(self, path: QPainterPath, styled: bool = False):
        super().__init__()
        self._path = path
        self._styled = styled
        self.setZValue(Z_BACKBONE)

    def boundingRect(self) -> QRectF:
        return self._path.boundingRect().adjusted(-4, -4, 4, 4)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        pen = (QPen(STYLED_BACKBONE_COLOR, STYLED_BACKBONE_WIDTH) if self._styled
               else QPen(BACKBONE_COLOR, BACKBONE_WIDTH))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self._path)


def _backbone_path(diagram: Diagram, positions: dict[str, QPointF],
                   center: QPointF,
                   ligand_coords: list[tuple[float, float]]) -> QPainterPath:
    radii = {
        residue.key: (DROPLET_RADIUS * WATER_RADIUS_FRAC
                      if residue.ref.residue_class == "water" else DROPLET_RADIUS)
        for residue in diagram.residues
    }

    def control(a: QPointF, b: QPointF, strength: float) -> QPointF:
        mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
        away = QPointF(mid.x() - center.x(), mid.y() - center.y())
        norm = math.hypot(away.x(), away.y()) or 1.0
        bow = math.hypot(b.x() - a.x(), b.y() - a.y()) * strength
        return QPointF(mid.x() + away.x() / norm * bow,
                       mid.y() + away.y() / norm * bow)

    def clears(a: QPointF, ctrl: QPointF, b: QPointF, own: set[str]) -> bool:
        # The renderer uses a quadratic Bezier, so test that same curve rather
        # than the straight chord optimized by the layout.  Denser near the
        # middle where an outward bow differs most from its chord.
        for step in range(1, 12):
            t = step / 12.0
            u = 1.0 - t
            point = QPointF(
                u * u * a.x() + 2.0 * u * t * ctrl.x() + t * t * b.x(),
                u * u * a.y() + 2.0 * u * t * ctrl.y() + t * t * b.y(),
            )
            angle = math.atan2(point.y() - center.y(), point.x() - center.x())
            if math.hypot(point.x() - center.x(), point.y() - center.y()) < (
                _ligand_reach(ligand_coords, center, angle) + RIBBON_LIGAND_CLEARANCE
            ):
                return False
            for key, obstacle in positions.items():
                if key in own:
                    continue
                clearance = radii.get(key, DROPLET_RADIUS) + BACKBONE_WIDTH + 6.0
                if math.hypot(point.x() - obstacle.x(), point.y() - obstacle.y()) < clearance:
                    return False
        return True

    path = QPainterPath()
    for left, right in diagram.backbone_edges():
        a, b = positions.get(left), positions.get(right)
        if a is None or b is None:
            continue
        if math.hypot(b.x() - a.x(), b.y() - a.y()) > BACKBONE_MAX_SPAN:
            continue
        baseline = control(a, b, BACKBONE_BOW)
        ctrl = baseline if clears(a, baseline, b, {left, right}) else None
        if ctrl is None:
            mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
            dx, dy = b.x() - a.x(), b.y() - a.y()
            length = math.hypot(dx, dy) or 1.0
            # Detour locally around the obstacle, trying the side farther from
            # the ligand first.  A radial bow scaled to the whole connector can
            # become a huge loop on a long sequence edge.
            for strength in (0.14, 0.22, 0.32, 0.44):
                candidates = [
                    QPointF(mid.x() - side * dy * strength,
                            mid.y() + side * dx * strength)
                    for side in (1.0, -1.0)
                ]
                candidates.sort(
                    key=lambda point: -math.hypot(
                        point.x() - center.x(), point.y() - center.y()
                    )
                )
                clear = next(
                    (candidate for candidate in candidates
                     if clears(a, candidate, b, {left, right})),
                    None,
                )
                if clear is not None:
                    ctrl = clear
                    break
        if ctrl is None:
            # Long chords sometimes need to follow the outside of the whole
            # pocket rather than detour locally around one water.  Keep the
            # smallest outward bow that clears both ligand and foreign glyphs.
            for strength in (0.18, 0.28):
                candidate = control(a, b, strength)
                if clears(a, candidate, b, {left, right}):
                    ctrl = candidate
                    break
        if ctrl is None:
            # A sequence hint is optional; drawing it through the ligand would
            # falsely look like a chemical contact and is strictly worse.
            continue
        path.moveTo(a)
        path.quadTo(ctrl, b)
    return path


# ---------------------------------------------------------------------------
# Layer 4 -- interaction routes
# ---------------------------------------------------------------------------

@dataclass
class _Route:
    path: QPainterPath
    kind: str
    #: 0 = no arrow, 1 = arrow at the end, -1 = arrow at the start.
    arrow: int = 0
    dots: bool = False
    #: Tooltip text, and the glyphs to light up with it.
    label: str = ""
    keys: tuple[str, ...] = ()
    #: Styled look only: the line style, a per-route colour override and the
    #: (start, end) colours of a gradient line.
    line: st.LineStyle | None = None
    color: str | None = None
    gradient: tuple[str, str] | None = None
    #: Stable id of this line for a hand-set bend, and whether it is drawn wavy.
    bend_key: str = ""
    wavy: bool = False


class InteractionRoutes(QGraphicsObject):
    """Dashed / solid contact lines from ligand atoms to droplet rims.

    All the routes share one item rather than getting one each: they are
    painted in a fixed order and never move independently, and a single item
    is what keeps a busy scene at one pixmap cache instead of forty.  Hover is
    therefore resolved here, against the stroked paths.
    """

    #: Label of the hovered route, or an empty string once the mouse leaves.
    hotChanged = Signal(str)
    #: A line was bent by hand: (bend key, bow as a fraction of the chord),
    #: or (bend key, None) when a double click hands it back to the router.
    bendChanged = Signal(str, object)

    def __init__(self, routes: list[_Route], droplets: dict[str, "ResidueDroplet"] | None = None):
        super().__init__()
        self._routes = routes
        self._droplets = droplets or {}
        self._hot: int | None = None
        self._dragging: int | None = None
        self._drag_bow = 0.0
        # Fattened once: the hit area has to be reachable with a mouse, which
        # a 1.4 px dashed line is not.
        stroker = QPainterPathStroker()
        stroker.setWidth(HOVER_SLOP)
        self._hit = [stroker.createStroke(r.path) for r in routes]
        self.setAcceptHoverEvents(True)
        self.setZValue(Z_ROUTES)

    def boundingRect(self) -> QRectF:
        # Padded before uniting, not after: a horizontal route has a zero-height
        # bounding rect, and QRectF.united() drops a null operand (see Ribbons).
        rect = QRectF()
        for route in self._routes:
            padded = route.path.boundingRect().adjusted(-12, -12, 12, 12)
            rect = padded if rect.isNull() else rect.united(padded)
        return rect

    def shape(self) -> QPainterPath:
        """Only the lines, so hovering the gaps between them reaches the ligand."""
        out = QPainterPath()
        # Winding, or two crossing lines would cancel each other out.
        out.setFillRule(Qt.FillRule.WindingFill)
        for hit in self._hit:
            out.addPath(hit)
        return out

    # -- hover -------------------------------------------------------------

    def _at(self, pos: QPointF) -> int | None:
        """Index of the route under ``pos``, nearest midpoint first on a tie."""
        hits = [i for i, hit in enumerate(self._hit) if hit.contains(pos)]
        if not hits:
            return None
        return min(hits, key=lambda i: _sq_dist(self._routes[i].path.pointAtPercent(0.5), pos))

    def hoverMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._set_hot(self._at(event.pos()))
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._set_hot(None)
        super().hoverLeaveEvent(event)

    def _set_hot(self, index: int | None) -> None:
        if index == self._hot:
            return
        was = self._routes[self._hot].keys if self._hot is not None else ()
        self._hot = index
        now = self._routes[index].keys if index is not None else ()
        for key in set(was) | set(now):
            droplet = self._droplets.get(key)
            if droplet is not None:
                droplet.set_lit(key in now)
        self.hotChanged.emit(self._routes[index].label if index is not None else "")
        self.update()

    # -- manual bend -------------------------------------------------------
    # The ends stay on the ligand atom and the glyph; dragging only moves the
    # middle of the line, so a route can be steered around a crowded spot.

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        index = self._at(event.pos())
        if event.button() != Qt.MouseButton.LeftButton or index is None \
                or not self._routes[index].bend_key:
            event.ignore()
            return
        self._dragging = index
        self._drag_bow = 0.0
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._dragging is None:
            return
        route = self._routes[self._dragging]
        start, end = route.path.pointAtPercent(0.0), route.path.pointAtPercent(1.0)
        dx, dy = end.x() - start.x(), end.y() - start.y()
        length = math.hypot(dx, dy) or 1.0
        pos = event.pos()
        # The apex of a quadratic sits halfway to its control point.
        offset = ((pos.x() - start.x()) * -dy + (pos.y() - start.y()) * dx) / length
        self._drag_bow = 2.0 * offset / length
        path = _bowed(start, end, self._drag_bow * length)
        route.path = _wave(path) if route.wavy else path
        stroker = QPainterPathStroker()
        stroker.setWidth(HOVER_SLOP)
        self.prepareGeometryChange()
        self._hit[self._dragging] = stroker.createStroke(route.path)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._dragging is not None:
            key = self._routes[self._dragging].bend_key
            self._dragging = None
            self.bendChanged.emit(key, self._drag_bow)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt naming
        index = self._at(event.pos())
        if index is None or not self._routes[index].bend_key:
            event.ignore()
            return
        self._dragging = None
        self.bendChanged.emit(self._routes[index].bend_key, None)

    # -- paint -------------------------------------------------------------

    def paint(self, painter: QPainter, option, widget=None) -> None:
        for i, route in enumerate(self._routes):
            style = route.line or INTERACTION_STYLES[route.kind]
            color = QColor(route.color or style.color)
            hot = i == self._hot
            pen = QPen(color, style.width * (HOVER_BOLD if hot else 1.0))
            if route.gradient:
                grad = QLinearGradient(route.path.pointAtPercent(0.0),
                                       route.path.pointAtPercent(1.0))
                grad.setColorAt(0.0, QColor(route.gradient[0]))
                grad.setColorAt(1.0, QColor(route.gradient[1]))
                pen.setBrush(QBrush(grad))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            if style.dash:
                # The dash pattern is in pen widths, so a bolder pen would
                # stretch the dashes too and the line would read as a
                # different kind of contact.
                pen.setDashPattern([d / (HOVER_BOLD if hot else 1.0) for d in style.dash])
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(route.path)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            if hot:
                # Both ends marked while hovering, whatever the kind: that is
                # what points at the ligand atom the contact actually leaves.
                for t in (0.0, 1.0):
                    painter.drawEllipse(route.path.pointAtPercent(t),
                                        DOT_RADIUS * HOVER_BOLD, DOT_RADIUS * HOVER_BOLD)
            elif route.dots:
                painter.drawEllipse(route.path.pointAtPercent(0.0), DOT_RADIUS, DOT_RADIUS)
                painter.drawEllipse(route.path.pointAtPercent(1.0), DOT_RADIUS, DOT_RADIUS)
            if route.arrow:
                at = 1.0 if route.arrow > 0 else 0.0
                tip = route.path.pointAtPercent(at)
                angle = math.radians(-route.path.angleAtPercent(at))
                if route.arrow < 0:
                    angle += math.pi
                painter.drawPolygon(_arrow_head(tip, angle))


def _wave(base: QPainterPath, amplitude: float = WAVE_AMPLITUDE,
          wavelength: float = WAVE_LENGTH) -> QPainterPath:
    """``base`` redrawn as a sine that swells in the middle and dies at both ends."""
    length = base.length()
    if length < 1.0:
        return base
    cycles = max(1, round(length / wavelength))
    steps = cycles * 12
    out = QPainterPath(base.pointAtPercent(0.0))
    for i in range(1, steps + 1):
        t = i / steps
        at = base.pointAtPercent(t)
        angle = math.radians(-base.angleAtPercent(t))
        offset = amplitude * math.sin(math.pi * t) * math.sin(2 * math.pi * cycles * t)
        out.lineTo(at.x() - math.sin(angle) * offset, at.y() + math.cos(angle) * offset)
    return out


def _bowed(start: QPointF, end: QPointF, bow: float) -> QPainterPath:
    """A line from ``start`` to ``end``, bent sideways by ``bow`` at its control point."""
    path = QPainterPath(start)
    if not bow:
        path.lineTo(end)
        return path
    dx, dy = end.x() - start.x(), end.y() - start.y()
    length = math.hypot(dx, dy) or 1.0
    path.quadTo(QPointF((start.x() + end.x()) / 2 - dy / length * bow,
                        (start.y() + end.y()) / 2 + dx / length * bow), end)
    return path


def _sq_dist(a: QPointF, b: QPointF) -> float:
    return (a.x() - b.x()) ** 2 + (a.y() - b.y()) ** 2


def _arrow_head(tip: QPointF, angle: float) -> QPolygonF:
    back = QPointF(tip.x() - ARROW_LENGTH * math.cos(angle),
                   tip.y() - ARROW_LENGTH * math.sin(angle))
    nx, ny = -math.sin(angle) * ARROW_HALF_WIDTH, math.cos(angle) * ARROW_HALF_WIDTH
    return QPolygonF([tip, QPointF(back.x() + nx, back.y() + ny),
                      QPointF(back.x() - nx, back.y() - ny)])


def _ligand_anchor(inter, atom_coords: dict[int, QPointF]) -> tuple[str, QPointF] | None:
    """Where this interaction leaves the ligand, and a stable id for that spot."""
    anchors = [(i, atom_coords[i]) for i in inter.ligand_atoms if i in atom_coords]
    if not anchors:
        return None
    if inter.kind in RING_ANCHORED:
        # Pi systems really do start at the ring centre.
        cx = sum(p.x() for _, p in anchors) / len(anchors)
        cy = sum(p.y() for _, p in anchors) / len(anchors)
        return f"ring:{anchors[0][0]}", QPointF(cx, cy)
    # A charged group interacts as a whole, but its centroid sits in empty
    # space; the line leaves from the group atom nearest the protein in 3D,
    # which the detector recorded.  Fixed by the structure, not the view.
    chosen = dict(anchors).get(inter.anchor_atom)
    if chosen is not None:
        return f"atom:{inter.anchor_atom}", chosen
    idx, point = min(anchors, key=lambda ip: ip[0])
    return f"atom:{idx}", point


def _coordination_legs(diagram: Diagram, key: str, center: QPointF,
                       positions: dict[str, QPointF],
                       atom_coords: dict[int, QPointF]) -> list[tuple[str, QPointF]]:
    """Every partner in metal ``key``'s sphere: (leg id, where its line comes from)."""
    legs: list[tuple[str, QPointF]] = []
    for inter in diagram.interactions_of(key):
        got = _ligand_anchor(inter, atom_coords)
        if got is not None:
            legs.append(got)
    for leg in diagram.metal_legs:
        other = None
        if leg.metal_key == key:
            other = leg.partner_key
        elif leg.partner_key == key:
            other = leg.metal_key
        if other is not None and other in positions:
            legs.append((other, positions[other]))
    return legs


def _metal_vertices(diagram: Diagram, positions: dict[str, QPointF],
                    atom_coords: dict[int, QPointF],
                    droplets: dict[str, "ResidueDroplet"]) -> dict[tuple[str, str], QPointF]:
    """One polygon corner per coordination leg, keyed ``(metal key, leg id)``.

    Assignment is greedy over the legs sorted by bearing.  Nearest-corner alone
    would let two partners land on the same one, which defeats the point of
    drawing the polyhedron in the first place.
    """
    out: dict[tuple[str, str], QPointF] = {}
    for key, item in droplets.items():
        if not item.is_metal:
            continue
        center = positions[key]
        corners = item.corners or [item.polygon.at(i) for i in range(item.polygon.count())]
        verts = [item.mapToScene(c) for c in corners]
        legs = _coordination_legs(diagram, key, center, positions, atom_coords)
        free = list(range(len(verts)))
        for leg_id, point in sorted(
            legs, key=lambda lp: math.atan2(lp[1].y() - center.y(), lp[1].x() - center.x())
        ):
            if not free:
                break
            j = min(free, key=lambda i: (verts[i].x() - point.x()) ** 2
                    + (verts[i].y() - point.y()) ** 2)
            free.remove(j)
            out[(key, leg_id)] = verts[j]
    return out


# The protein side of a coordination sphere is deliberately not drawn: those
# legs are what the metal does with the *protein*, and the diagram is about
# what it does with the ligand.  Five solid spokes to nearby residues buried
# the metal-ligand bond the reader came for.  The legs are still computed --
# they set the polyhedron's shape and reserve its corners, so the ligand lines
# still land where the geometry says they should.


def _route_label(inter, style) -> str:
    """One line for the hover tooltip: what the contact is, how long, to what.

    The distance is the detector's, in angstrom, and it is the 3D one --
    nothing on this canvas is to scale, so a length measured off the drawing
    would be a lie.  Contacts reported without one just drop the middle
    field.
    """
    name = ":".join(inter.residue_key.split(":")[:2])
    what = f"{inter.residue_key.split(':')[-1]} {name}"
    if inter.protein_atom:
        what += f" · atom {inter.protein_atom}"
    geometry = f"{inter.distance:.2f} Å" if inter.distance else ""
    if inter.protein_distance is not None:
        geometry += f" + {inter.protein_distance:.2f} Å"
    if inter.angle is not None:
        geometry += f" / {inter.angle:.1f}°"
    parts = [style.label, geometry, what]
    return "  ·  ".join(p for p in parts if p)


def _routes(diagram: Diagram, positions: dict[str, QPointF],
            atom_coords: dict[int, QPointF], shapes: dict[str, QPolygonF],
            bonds: list[tuple[QPointF, QPointF]],
            vertices: dict[tuple[str, str], QPointF],
            look: DiagramStyle | None = None,
            bends: dict[str, float] | None = None) -> list[_Route]:
    routes: list[_Route] = []
    bends = bends or {}
    seen_water_ligand: set[tuple[str, str]] = set()
    closest_contact: dict[str, int] = {}
    if look is not None and look.hydrophobic_lines:
        # One line per residue: the closest contact stands for the patch.
        for n, inter in enumerate(diagram.interactions):
            if inter.kind != "hydrophobic":
                continue
            best = closest_contact.get(inter.residue_key)
            if best is None or inter.distance < diagram.interactions[best].distance:
                closest_contact[inter.residue_key] = n
    keep = set(closest_contact.values())
    order = sorted(range(len(diagram.interactions)),
                   key=lambda n: diagram.interactions[n].style.priority)
    for n in order:
        inter = diagram.interactions[n]
        if inter.kind == "hydrophobic" and n not in keep:
            continue
        target = positions.get(inter.residue_key)
        shape = shapes.get(inter.residue_key)
        if target is None or shape is None:
            continue
        got = _ligand_anchor(inter, atom_coords)
        if got is None:
            continue
        leg_id, atom = got

        style = INTERACTION_STYLES[inter.kind]
        arrow = 0
        if style.marker == "arrow":
            arrow = 1 if inter.ligand_is_donor else -1

        # A water bridge is two hydrogen bonds, and drawing it as one line from
        # the ligand to the residue hides the water that makes it -- the water
        # ball would sit unconnected in the middle of the canvas.
        hops = [(inter.residue_key, target, shape)]
        origin = atom
        branch_from_water = False
        water = positions.get(inter.via_water) if inter.via_water else None
        if water is not None and inter.via_water in shapes:
            first_key = (inter.via_water, leg_id)
            if first_key in seen_water_ligand:
                # The ligand-water leg is shared by every protein partner in
                # a water network.  Draw it once, then branch at the sphere.
                origin = water
                branch_from_water = True
            else:
                hops = [(inter.via_water, water, shapes[inter.via_water])] + hops
                seen_water_ligand.add(first_key)

        for hop, (key, dest, dest_shape) in enumerate(hops):
            angle = math.atan2(dest.y() - origin.y(), dest.x() - origin.x())
            gap = ROUTE_ATOM_GAP if hop == 0 else 0.0
            start = QPointF(origin.x() + gap * math.cos(angle),
                            origin.y() + gap * math.sin(angle))
            if branch_from_water and hop == 0:
                start = _ray_hit(origin, dest, shapes[inter.via_water])
            elif hop > 0:
                start = _ray_hit(origin, dest, shapes[hops[hop - 1][0]])
            # A metal takes the line on the corner reserved for it, not on
            # whatever edge happens to face the ligand.
            end = vertices.get((key, leg_id)) or _ray_hit(dest, start, dest_shape)

            bend_key = f"{inter.residue_key}|{leg_id}|{inter.kind}|{hop}"
            if bend_key in bends:
                bow = bends[bend_key] * math.hypot(end.x() - start.x(), end.y() - start.y())
            else:
                bow = _needed_bow(start, end, positions, key, bonds)
            path = _bowed(start, end, bow)
            route_arrow = arrow if hop == len(hops) - 1 else 0
            if inter.via_water:
                if branch_from_water or hop == 1:
                    route_arrow = -1 if inter.protein_is_donor else 1
                else:
                    route_arrow = 1 if inter.ligand_is_donor else -1
            line = color = gradient = None
            if look is not None:
                line = st.LINE_STYLES[inter.kind]
                if inter.kind == "hydrophobic":
                    color = st.hydrophobic_color(inter.distance)
                if line.gradient:
                    # The line runs ligand -> residue, so the residue's own
                    # charge colours the far end.
                    negative = diagram.residue(inter.residue_key).ref.residue_class \
                        == "charged_negative"
                    gradient = (line.gradient[::-1] if negative else line.gradient)
            wavy = look is not None and inter.kind == "metal_coordination"
            if wavy:
                path = _wave(path)
            routes.append(
                _Route(path, inter.kind, route_arrow,
                       (line or style).marker == "dot",
                       label=_route_label(inter, line or style),
                       keys=tuple(k for k, _, _ in hops),
                       line=line, color=color, gradient=gradient,
                       bend_key=bend_key, wavy=wavy)
            )
            origin = dest
    return routes


def _needed_bow(start: QPointF, end: QPointF, positions: dict[str, QPointF],
                own_key: str, bonds: list[tuple[QPointF, QPointF]]) -> float:
    """Perpendicular offset that clears the ligand and foreign droplets, or 0.

    The bow is tried in both directions at growing strength and the first
    clear one wins; if nothing clears, the widest bow is used anyway so the
    line at least reads as deliberately routed.
    """
    obstacles = [(pos, DROPLET_RADIUS * 1.15) for key, pos in positions.items() if key != own_key]
    if _clear(start, end, 0.0, bonds, obstacles):
        return 0.0
    length = math.hypot(end.x() - start.x(), end.y() - start.y())
    for strength in (ROUTE_BEND, ROUTE_BEND * 2, ROUTE_BEND * 3.2):
        for side in (1.0, -1.0):
            bow = length * strength * side
            if _clear(start, end, bow, bonds, obstacles):
                return bow
    return length * ROUTE_BEND * 3.2


def _clear(start: QPointF, end: QPointF, bow: float,
           bonds: list[tuple[QPointF, QPointF]], obstacles) -> bool:
    """Does the (possibly bowed) route miss every bond and foreign droplet?"""
    dx, dy = end.x() - start.x(), end.y() - start.y()
    length = math.hypot(dx, dy) or 1.0
    mid = QPointF((start.x() + end.x()) / 2 - dy / length * bow,
                  (start.y() + end.y()) / 2 + dx / length * bow)
    samples = [(start.x(), start.y())]
    for i in range(1, 9):
        t = i / 8
        u = 1 - t
        samples.append((u * u * start.x() + 2 * u * t * mid.x() + t * t * end.x(),
                        u * u * start.y() + 2 * u * t * mid.y() + t * t * end.y()))
    for p, q in zip(samples, samples[1:]):
        for a, b in bonds:
            if segments_cross(p, q, (a.x(), a.y()), (b.x(), b.y())):
                return False
        for pos, radius in obstacles:
            if point_segment_distance((pos.x(), pos.y()), p, q) < radius:
                return False
    return True


# ---------------------------------------------------------------------------
# Layer 5 -- the ligand
# ---------------------------------------------------------------------------

_BOND_ELEMENT = re.compile(r"<path class='bond-[^>]*>")
_HEX = re.compile(r"#[0-9A-Fa-f]{6}")


class LigandItem(QGraphicsSvgItem):
    """The RDKit depiction, placed so its atoms land on ``ligand_coords``.

    ``atom_coords`` maps RDKit atom index to the atom's position in *scene*
    coordinates -- ``drawer.GetDrawCoords`` run back through the item's own
    transform, so it is exact rather than approximate.
    """

    def __init__(self, svg: str, atom_coords: dict[int, QPointF]):
        super().__init__()
        self._renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
        self.setSharedRenderer(self._renderer)
        self.atom_coords = atom_coords
        self.setZValue(Z_LIGAND)


def _draw_ligand(diagram: Diagram, ligand_coords: list[tuple[float, float]]) -> LigandItem:
    """Render the ligand and reconcile RDKit's canvas with the layout's frame.

    ``ligand_coords`` are scene units with y growing downwards; RDKit's drawer
    flips y, so the conformer is built with y negated and comes back upright.

    RDKit always scales the depiction to fill its canvas (``fixedBondLength``
    is only an upper bound), so the canvas size is what sets the scale.  We
    draw once to learn the ratio, then redraw on a canvas divided by it: the
    second pass comes out at scale ~1, which keeps the SVG's own bond widths
    and label fonts at their intended size once the item is placed.  Whatever
    residue is left is absorbed by a uniform scale plus translation, so
    ``GetDrawCoords(i)`` maps onto ``ligand_coords[i]`` exactly.
    """
    mol = Chem.Mol(diagram.mol)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y) in enumerate(ligand_coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(-y), 0.0))
    mol.AddConformer(conf)
    prepared = rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False, kekulize=True)

    xs = [p[0] for p in ligand_coords]
    ys = [p[1] for p in ligand_coords]
    bond_length = _median_bond_length(diagram, ligand_coords)
    margin = LIGAND_CANVAS_MARGIN * bond_length
    width = max(32.0, max(xs) - min(xs) + 2 * margin)
    height = max(32.0, max(ys) - min(ys) + 2 * margin)

    drawer = None
    for _ in range(2):
        drawer = rdMolDraw2D.MolDraw2DSVG(int(width), int(height))
        opts = drawer.drawOptions()
        opts.clearBackground = False
        opts.addStereoAnnotation = False
        opts.explicitMethyl = False
        opts.bondLineWidth = LIGAND_BOND_WIDTH
        opts.scaleBondWidth = False
        opts.padding = LIGAND_CANVAS_PAD
        opts.fixedFontSize = int(round(LIGAND_FONT_PX))
        drawer.DrawMolecule(prepared)
        drawer.FinishDrawing()
        drawn = [drawer.GetDrawCoords(i) for i in range(len(ligand_coords))]
        scale, dx, dy = _fit_similarity(drawn, ligand_coords)
        if abs(scale - 1.0) < 0.02:
            break
        width, height = width * scale, height * scale

    svg = _BOND_ELEMENT.sub(lambda m: _HEX.sub(BOND_COLOR, m.group(0)), drawer.GetDrawingText())
    item = LigandItem(svg, {i: QPointF(*ligand_coords[i]) for i in range(len(ligand_coords))})
    item.setScale(scale)
    item.setPos(dx, dy)
    return item


def _fit_similarity(drawn, target) -> tuple[float, float, float]:
    """Least-squares uniform scale + offset from draw coords onto layout coords."""
    n = len(target)
    dcx = sum(p.x for p in drawn) / n
    dcy = sum(p.y for p in drawn) / n
    tcx = sum(p[0] for p in target) / n
    tcy = sum(p[1] for p in target) / n
    num = sum((p.x - dcx) * (q[0] - tcx) + (p.y - dcy) * (q[1] - tcy)
              for p, q in zip(drawn, target))
    den = sum((p.x - dcx) ** 2 + (p.y - dcy) ** 2 for p in drawn)
    scale = (num / den) if den > 1e-9 else 1.0
    return scale, tcx - scale * dcx, tcy - scale * dcy


# ---------------------------------------------------------------------------
# Layer 6 -- residue droplets
# ---------------------------------------------------------------------------

class ResidueDroplet(QGraphicsObject):
    """One teardrop glyph, pointed at the ligand atom the residue touches."""

    def __init__(self, residue, tip_angle: float, coordination: tuple[int, float] | None = None,
                 look: DiagramStyle | None = None):
        super().__init__()
        self.residue_key: str = residue.key
        self.residue = residue
        self.tip_angle = tip_angle
        self.style = RESIDUE_STYLES[residue.ref.residue_class]
        self.look = look
        self.fill = st.fill_for(residue.ref.name, look) if look is not None else None
        self.is_water = residue.ref.residue_class == "water"
        self.is_metal = coordination is not None
        self.radius = DROPLET_RADIUS
        self.text_dy = 0.0
        self.corners: list[QPointF] = []
        if look is not None and not self.is_metal and not self.is_water:
            self.polygon, self.text_dy = glyph_polygon(
                st.shape_of(residue.ref.name, look), DROPLET_RADIUS)
        elif self.is_metal:
            # The coordination polyhedron drawn flat: a metal with four
            # partners is a square, five a pentagon, and each partner's line
            # lands on its own corner.
            sides, angle = coordination
            if look is not None and look.metals == "star":
                reach = DROPLET_RADIUS * STAR_REACH
                star = _star(reach, sides, angle)
                # Rounding pulls each tip in a little; land the lines on it.
                pull = 1.0 - 0.5 * STAR_CORNER * DROPLET_RADIUS / reach
                self.corners = [star.at(i) * pull for i in range(0, star.count(), 2)]
                self.polygon = _round_corners(star, DROPLET_RADIUS * STAR_CORNER)
            else:
                self.polygon = _ngon(QPointF(0, 0), DROPLET_RADIUS, sides, angle)
        elif self.is_water:
            self.radius = DROPLET_RADIUS * WATER_RADIUS_FRAC
            if look is not None:
                self.radius *= STYLED_WATER_SCALE
            self.polygon = _circle(QPointF(0, 0), self.radius)
        else:
            self.polygon = _teardrop(QPointF(0, 0), DROPLET_RADIUS, tip_angle)
        self._route_lit = False
        self._hovered = False
        self.setAcceptHoverEvents(True)
        self.setZValue(Z_DROPLETS)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self._shadow = QPolygonF([QPointF(p.x() + SHADOW_OFFSET, p.y() + SHADOW_OFFSET)
                                  for p in self.polygon])

    def shape_in_scene(self) -> QPolygonF:
        return QPolygonF([self.mapToScene(p) for p in self.polygon])

    @property
    def lit(self) -> bool:
        return self._route_lit or self._hovered

    def set_lit(self, lit: bool) -> None:
        """Thicken the outline while a route touching this residue is hovered."""
        if lit != self._route_lit:
            self._route_lit = lit
            self.update()

    def hoverEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def boundingRect(self) -> QRectF:
        # Sized for the lit outline whether or not it is currently lit: the
        # rect has to be stable, because the item is cached into a pixmap of
        # exactly this size and hovering must not have to resize it.
        edge = DROPLET_OUTLINE_WIDTH * HOVER_BOLD
        pad = SHADOW_OFFSET + SHADOW_LAYERS[0][0]
        rect = self.polygon.boundingRect().adjusted(-edge, -edge, pad + edge, pad + edge)
        if self.is_water:
            rect.setRight(rect.right() + self._label_width())
        return rect

    def _label_width(self) -> float:
        font = _font(DROPLET_CODE_PX, DROPLET_LETTER_SPACING)
        return WATER_LABEL_GAP + QFontMetricsF(font).horizontalAdvance(
            self.residue.ref.label_lines[1]) + DROPLET_CODE_PX

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.fill is not None:
            self._paint_flat(painter)
            return
        for width, alpha in SHADOW_LAYERS:
            color = QColor(60, 60, 60, alpha)
            painter.setPen(QPen(color, width) if width else Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawPolygon(self._shadow)

        if self.is_water:
            _paint_sphere(painter, QPointF(0, 0), self.radius, "water")
            painter.setFont(_font(DROPLET_CODE_PX, DROPLET_LETTER_SPACING))
            painter.setPen(QColor(self.style.text))
            painter.drawText(
                QRectF(self.radius + WATER_LABEL_GAP, -DROPLET_CODE_PX,
                       self._label_width(), 2 * DROPLET_CODE_PX),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self.residue.ref.label_lines[1])
            return

        # Light at the back of the body, saturating towards the point: a radial
        # gradient anchored on the back rim reproduces the near-linear ramp
        # measured across VAL A:87 in 4ps5.png.
        back = QPointF(-self.radius * math.cos(self.tip_angle),
                       -self.radius * math.sin(self.tip_angle))
        grad = QRadialGradient(back, self.radius * (1.0 + TIP_REACH))
        grad.setColorAt(0.0, QColor(self.style.light))
        grad.setColorAt(1.0, QColor(self.style.base))
        painter.setBrush(QBrush(grad))
        pen = QPen(QColor(self.style.outline),
                   DROPLET_OUTLINE_WIDTH * (HOVER_BOLD if self.lit else 1.0))
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPolygon(self.polygon)

        name, code = self.residue.ref.label_lines
        painter.setPen(QColor(self.style.text))
        for text, px, dy in ((name, DROPLET_NAME_PX, -DROPLET_LINE_GAP / 2),
                             (code, DROPLET_CODE_PX, DROPLET_LINE_GAP / 2)):
            font = _font(px, DROPLET_LETTER_SPACING)
            painter.setFont(font)
            width = QFontMetricsF(font).horizontalAdvance(text)
            # Qt appends the letter spacing after the last glyph too.
            box = QRectF(-width / 2 - DROPLET_LETTER_SPACING / 2, dy - px, width + px, 2 * px)
            painter.drawText(box, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, text)

    def _paint_flat(self, painter: QPainter) -> None:
        """Matte glyph: one soft offset shadow, flat or two-tone fill, caption inside."""
        fill = self.fill
        offset, alpha = GLYPH_SHADOW
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(40, 48, 58, alpha))
        painter.drawPolygon(self.polygon.translated(offset, offset))

        if fill.second:
            box = self.polygon.boundingRect()
            grad = QLinearGradient(box.topLeft(), box.bottomRight())
            grad.setColorAt(0.0, QColor(fill.color))
            grad.setColorAt(0.5, QColor(fill.color))
            grad.setColorAt(1.0, QColor(fill.second))
            painter.setBrush(QBrush(grad))
        else:
            painter.setBrush(QColor(fill.color))
        pen = QPen(QColor(fill.outline), GLYPH_OUTLINE_WIDTH * (HOVER_BOLD if self.lit else 1.0))
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPolygon(self.polygon)

        name, code = self.residue.ref.label_lines
        painter.setPen(QColor(fill.text))
        if self.is_water:
            painter.setFont(_font(DROPLET_CODE_PX))
            painter.drawText(
                QRectF(self.radius + WATER_LABEL_GAP, -DROPLET_CODE_PX,
                       self._label_width(), 2 * DROPLET_CODE_PX),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, code)
            return
        for text, bold, dy in ((name, True, -DROPLET_LINE_GAP / 2),
                               (code, False, DROPLET_LINE_GAP / 2)):
            font = _font(DROPLET_NAME_PX)
            font.setBold(bold)
            painter.setFont(font)
            painter.drawText(QRectF(-DROPLET_RADIUS, self.text_dy + dy - DROPLET_NAME_PX,
                                    2 * DROPLET_RADIUS, 2 * DROPLET_NAME_PX),
                             Qt.AlignmentFlag.AlignCenter, text)


# ---------------------------------------------------------------------------
# Layer 7 -- legend
# ---------------------------------------------------------------------------

class Legend(QGraphicsObject):
    """Maestro's key, in its wording and order, restricted to what is drawn.

    Maestro prints the whole key on every diagram, so ``data/4ps5.png``
    advertises metals, waters, hydration sites and six line styles it never
    uses.  Copying that verbatim would explain colours the reader cannot find,
    so the rows come from the diagram: the residue classes present and the
    interaction kinds that produced a line.  Pass ``full=True`` for the literal
    Maestro key.
    """

    def __init__(
        self,
        diagram: Diagram | None = None,
        full: bool = False,
        rows: int = 3,
        single_column: bool = False,
    ):
        super().__init__()
        if rows not in (2, 3, 4):
            raise ValueError(f"legend rows must be 2, 3, or 4; got {rows}")
        classes = [c for c, _ in LEGEND_RESIDUE_ROWS]
        kinds = list(LEGEND_LINE_ROWS)
        halo = True
        warn = diagram is not None and diagram.metadata.get("bond_orders_known") is False
        if diagram is not None and not full:
            present = {r.ref.residue_class for r in diagram.residues}
            drawn = {i.kind for i in diagram.interactions} - {"hydrophobic"}
            classes = [c for c in classes if c in present]
            kinds = [k for k in kinds if k in drawn]
            halo = bool(diagram.exposure)

        entries = [("sphere", c, label) for c, label in LEGEND_RESIDUE_ROWS if c in classes]
        if full:
            entries += [("blank", None, LEGEND_EXTRA_ROWS[0]), ("cross", None, LEGEND_EXTRA_ROWS[1])]
        lines = [("line", k, INTERACTION_STYLES[k].label) for k in LEGEND_LINE_ROWS if k in kinds]
        if halo:
            lines.append(("halo", None, "Solvent exposure"))

        entries += lines
        if warn:
            entries.append(("warning", None, LEGEND_BOND_ORDER_WARNING))
        self._text_width = max(
            (QFontMetricsF(_font(LEGEND_FONT_PX)).horizontalAdvance(label) for *_, label in entries),
            default=0.0,
        )
        self._columns = [entries] if single_column and entries else _legend_columns(entries, rows)
        self.setZValue(Z_LEGEND)

    @property
    def row_count(self) -> int:
        return max((len(column) for column in self._columns), default=0)

    @property
    def column_count(self) -> int:
        return len(self._columns)

    def boundingRect(self) -> QRectF:
        cols = max(1, len(self._columns))
        rows = max((len(c) for c in self._columns), default=1)
        width = max(260.0, LEGEND_TEXT_OFFSET + self._text_width + 16)
        return QRectF(-8, -LEGEND_SPHERE_RADIUS - 4,
                      LEGEND_COL_PITCH * (cols - 1) + width, LEGEND_ROW_PITCH * rows + 12)

    def width(self) -> float:
        return self.boundingRect().width()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = _font(LEGEND_FONT_PX)
        metrics = QFontMetricsF(font)
        for col, rows in enumerate(self._columns):
            x = col * LEGEND_COL_PITCH
            for row, (kind, key, label) in enumerate(rows):
                y = row * LEGEND_ROW_PITCH
                cx = x + LEGEND_SPHERE_RADIUS
                if kind == "sphere":
                    _paint_sphere(painter, QPointF(cx, y), LEGEND_SPHERE_RADIUS, key)
                elif kind == "cross":
                    pen = QPen(LEGEND_CROSS_COLOR, 2.8)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(pen)
                    r = LEGEND_SPHERE_RADIUS * 0.62
                    painter.drawLine(QPointF(cx - r, y - r), QPointF(cx + r, y + r))
                    painter.drawLine(QPointF(cx - r, y + r), QPointF(cx + r, y - r))
                elif kind == "halo":
                    painter.setPen(Qt.PenStyle.NoPen)
                    r = LEGEND_SPHERE_RADIUS * 1.15
                    painter.setBrush(QBrush(halo_gradient(QPointF(cx, y), r)))
                    painter.drawEllipse(QPointF(cx, y), r, r)
                elif kind == "line":
                    _paint_line_sample(painter, x, y, key)
                elif kind == "warning":
                    r = LEGEND_SPHERE_RADIUS
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(LEGEND_WARNING_COLOR)
                    painter.drawPolygon(QPolygonF([
                        QPointF(cx, y - r), QPointF(cx + r * 1.1, y + r * 0.85),
                        QPointF(cx - r * 1.1, y + r * 0.85),
                    ]))
                    bang = _font(LEGEND_FONT_PX * 0.9)
                    bang.setBold(True)
                    painter.setFont(bang)
                    painter.setPen(QColor("#ffffff"))
                    painter.drawText(QRectF(cx - r, y - r * 0.55, 2 * r, r * 1.4),
                                     Qt.AlignmentFlag.AlignCenter, "!")

                painter.setPen(QColor("#101010"))
                painter.setFont(font)
                painter.drawText(QPointF(x + LEGEND_TEXT_OFFSET, y + metrics.capHeight() / 2), label)


class LegendTable(QGraphicsObject):
    """Boxed key for the styled look: shapes, colours, lines, surface.

    Sections sit side by side for a top/bottom legend and stacked for a side
    one.  Only what the diagram actually draws is listed.
    """

    def __init__(self, diagram: Diagram, look: DiagramStyle, stacked: bool = True,
                 has_surface: bool = False, has_lines: set[str] | None = None):
        super().__init__()
        self._look = look
        self._stacked = stacked
        self._sections = [s for s in self._collect(diagram, look, has_surface,
                                                   has_lines or set()) if s[1]]
        self._title_font = _font(TABLE_TITLE_PX)
        self._title_font.setBold(True)
        self._text_font = _font(TABLE_TEXT_PX)
        text = QFontMetricsF(self._text_font)
        title = QFontMetricsF(self._title_font)
        self._widths = [
            max([title.horizontalAdvance(name)]
                + [TABLE_TEXT_OFFSET + text.horizontalAdvance(label) for *_, label in rows])
            for name, rows in self._sections
        ]
        self.setZValue(Z_LEGEND)

    @staticmethod
    def _collect(diagram: Diagram, look: DiagramStyle, has_surface: bool,
                 lines: set[str]) -> list[tuple[str, list]]:
        names = sorted({r.ref.name.upper() for r in diagram.residues})
        natures = {r.ref.residue_class for r in diagram.residues}

        shapes = []
        if look.glyphs == "shapes":
            present = {st.shape_of(n, look) for n in names
                       if classify_residue(n) not in ("water", "metal")}
            if look.metals == "star" and "metal" in natures:
                present.add("star")
            shapes = [("shape", shape, label) for shape, label in st.SHAPE_ROWS
                      if shape in present]

        colors = [("swatch", st.nature_color(key, look.palette), label)
                  for key, label in st.NATURE_ROWS if key in natures]
        if look.coloring == "residue":
            by_color: dict[str, list[str]] = {}
            for n in names:
                if classify_residue(n) not in ("water", "metal"):
                    by_color.setdefault(st.fill_for(n, look).color, []).append(n)
            colors = [("swatch", color, ", ".join(group)) for color, group in by_color.items()]
            colors += [("swatch", st.nature_color(key, look.palette), label)
                       for key, label in st.NATURE_ROWS
                       if key in natures and key in ("water", "metal", "unspecified")]
        elif look.coloring == "blend":
            for n in names:
                if n in st.BLEND_NATURES:
                    fill = st.fill_for(n, look)
                    main, second = st.BLEND_NATURES[n]
                    colors.append(("swatch", (fill.color, fill.second),
                                   f"{n}  ({main} → {second.replace('charged_', '')})"))

        interactions = [("line", kind, st.LINE_STYLES[kind].label)
                        for kind in st.LINE_ROWS if kind in lines]

        surface = []
        if has_surface and look.surface != "hidden":
            surface.append(("surface", look.surface, "Pocket surface"))
        if diagram.exposure and look.exposure != "hidden":
            surface.append(("exposure", look.exposure, "Solvent exposure"))
        if diagram.metadata.get("bond_orders_known") is False:
            surface.append(("warning", None, LEGEND_BOND_ORDER_WARNING))

        return [("Residue shape", shapes), ("Residue colour", colors),
                ("Interactions", interactions), ("Surface", surface)]

    def _section_height(self, rows: list) -> float:
        return TABLE_TITLE_PX + 10.0 + TABLE_ROW_PITCH * len(rows)

    def _origins(self) -> list[QPointF]:
        out, x, y = [], TABLE_PAD, TABLE_PAD
        for (_, rows), width in zip(self._sections, self._widths):
            out.append(QPointF(x, y))
            if self._stacked:
                y += self._section_height(rows) + TABLE_SECTION_GAP
            else:
                x += width + TABLE_SECTION_GAP * 1.5
        return out

    def boundingRect(self) -> QRectF:
        if not self._sections:
            return QRectF()
        heights = [self._section_height(rows) for _, rows in self._sections]
        if self._stacked:
            width = max(self._widths)
            height = sum(heights) + TABLE_SECTION_GAP * (len(heights) - 1)
        else:
            width = sum(self._widths) + TABLE_SECTION_GAP * 1.5 * (len(self._widths) - 1)
            height = max(heights)
        return QRectF(0, 0, width + 2 * TABLE_PAD, height + 2 * TABLE_PAD)

    def width(self) -> float:
        return self.boundingRect().width()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if not self._sections:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        box = self.boundingRect().adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(TABLE_BORDER, 1.0))
        painter.setBrush(TABLE_FILL)
        painter.drawRoundedRect(box, 8.0, 8.0)

        metrics = QFontMetricsF(self._text_font)
        for i, ((name, rows), origin) in enumerate(zip(self._sections, self._origins())):
            if i:
                painter.setPen(QPen(TABLE_BORDER, 1.0))
                if self._stacked:
                    y = origin.y() - TABLE_SECTION_GAP / 2
                    painter.drawLine(QPointF(TABLE_PAD, y), QPointF(box.right() - TABLE_PAD, y))
                else:
                    x = origin.x() - TABLE_SECTION_GAP * 0.75
                    painter.drawLine(QPointF(x, TABLE_PAD), QPointF(x, box.bottom() - TABLE_PAD))
            painter.setFont(self._title_font)
            painter.setPen(TABLE_TEXT)
            painter.drawText(QPointF(origin.x(), origin.y() + TABLE_TITLE_PX), name)
            for r, (kind, payload, label) in enumerate(rows):
                y = origin.y() + TABLE_TITLE_PX + 10.0 + TABLE_ROW_PITCH * (r + 0.5)
                self._paint_icon(painter, QPointF(origin.x() + 14.0, y), kind, payload)
                painter.setFont(self._text_font)
                painter.setPen(TABLE_TEXT)
                painter.drawText(QPointF(origin.x() + TABLE_TEXT_OFFSET,
                                         y + metrics.capHeight() / 2), label)

    def _paint_icon(self, painter: QPainter, at: QPointF, kind: str, payload) -> None:
        r = TABLE_ICON_RADIUS
        if kind == "shape":
            polygon, _ = glyph_polygon(payload, r)
            painter.setPen(QPen(TABLE_ICON_OUTLINE, 1.0))
            painter.setBrush(TABLE_ICON_FILL)
            painter.drawPolygon(polygon.translated(at.x(), at.y() - (0.1 * r if payload == "drop" else 0)))
        elif kind == "swatch":
            if isinstance(payload, tuple):
                grad = QLinearGradient(QPointF(at.x() - r, at.y() - r), QPointF(at.x() + r, at.y() + r))
                grad.setColorAt(0.0, QColor(payload[0]))
                grad.setColorAt(0.5, QColor(payload[0]))
                grad.setColorAt(1.0, QColor(payload[1]))
                painter.setBrush(QBrush(grad))
                outline = _darker(payload[0], 0.75)
            else:
                painter.setBrush(QColor(payload))
                outline = _darker(payload, 0.75)
            painter.setPen(QPen(outline, 1.0))
            painter.drawRoundedRect(QRectF(at.x() - r, at.y() - r * 0.8, 2 * r, 1.6 * r), 3, 3)
        elif kind == "line":
            line = st.LINE_STYLES[payload]
            a, b = QPointF(at.x() - 14.0, at.y()), QPointF(at.x() + 14.0, at.y())
            pen = QPen(QColor(line.color), line.width)
            if line.gradient:
                grad = QLinearGradient(a, b)
                grad.setColorAt(0.0, QColor(line.gradient[0]))
                grad.setColorAt(1.0, QColor(line.gradient[1]))
                pen.setBrush(QBrush(grad))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            if line.dash:
                pen.setDashPattern(list(line.dash))
            painter.setPen(pen)
            end = QPointF(b.x() - ARROW_LENGTH * 0.8, b.y()) if line.marker == "arrow" else b
            if payload == "metal_coordination":
                sample = QPainterPath(a)
                sample.lineTo(end)
                painter.drawPath(_wave(sample, WAVE_AMPLITUDE * 0.8, WAVE_LENGTH))
            else:
                painter.drawLine(a, end)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(line.color))
            if line.marker == "arrow":
                painter.drawPolygon(_arrow_head(b, 0.0))
            elif line.marker == "dot":
                painter.drawEllipse(a, DOT_RADIUS * 0.8, DOT_RADIUS * 0.8)
                painter.drawEllipse(b, DOT_RADIUS * 0.8, DOT_RADIUS * 0.8)
        elif kind == "surface":
            color = QColor(st.nature_color("hydrophobic", "medium"))
            if payload == "dots":
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                for k in range(-1, 2):
                    painter.drawEllipse(QPointF(at.x() + k * SURFACE_DOT_STEP, at.y()),
                                        SURFACE_DOT_RADIUS, SURFACE_DOT_RADIUS)
            else:
                pen = QPen(color, SURFACE_LINE_WIDTH)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(QPointF(at.x() - 12, at.y()), QPointF(at.x() + 12, at.y()))
        elif kind == "exposure" and payload == "arc":
            painter.setPen(Qt.PenStyle.NoPen)
            _paint_halo_wedge(painter, QPointF(at.x(), at.y() + r * 0.7), r * 1.7,
                              -math.pi * 0.75, -math.pi * 0.25, gain=1.6)
        elif kind == "exposure" and payload == "trail":
            painter.setBrush(Qt.BrushStyle.NoBrush)
            center = QPointF(at.x(), at.y() + r * 0.6)
            pen = QPen(HALO_COLOR, TRAIL_WIDTH * 0.8)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawPath(_arc_path(center, r * 0.8,
                                       -math.pi * 0.8, -math.pi * 0.2))
        elif kind == "exposure":
            painter.setPen(Qt.PenStyle.NoPen)
            color = QColor(st.EXPOSURE_DOT_COLOR)
            painter.setBrush(color)
            for k in range(7):
                a = math.pi * (0.1 + 0.8 * k / 6)
                painter.drawEllipse(QPointF(at.x() + r * math.cos(a), at.y() + 2 - r * math.sin(a)),
                                    EXPOSURE_DOT_RADIUS, EXPOSURE_DOT_RADIUS)
        elif kind == "warning":
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(LEGEND_WARNING_COLOR)
            painter.drawPolygon(QPolygonF([QPointF(at.x(), at.y() - r),
                                           QPointF(at.x() + r * 1.1, at.y() + r * 0.85),
                                           QPointF(at.x() - r * 1.1, at.y() + r * 0.85)]))


def _legend_columns(
    entries: list[tuple[str, str | None, str]], row_count: int
) -> list[list]:
    """Split entries column-major into columns of at most ``row_count`` rows."""
    if not entries:
        return []
    return [entries[i:i + row_count] for i in range(0, len(entries), row_count)]


def _paint_sphere(painter: QPainter, center: QPointF, radius: float, cls: str) -> None:
    """The legend balls: white specular up-left, base mid, darkened rim."""
    style = RESIDUE_STYLES[cls]
    focus = QPointF(center.x() - radius * 0.38, center.y() - radius * 0.38)
    grad = QRadialGradient(focus, radius * 1.45)
    grad.setColorAt(0.0, _blend(QColor(style.light), QColor("#ffffff"), 0.6))
    grad.setColorAt(0.55, QColor(style.base))
    grad.setColorAt(1.0, _darker(style.base))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(grad))
    painter.drawEllipse(center, radius, radius)


def _paint_line_sample(painter: QPainter, x: float, y: float, kind: str) -> None:
    style = INTERACTION_STYLES[kind]
    a = QPointF(x, y)
    b = QPointF(x + LEGEND_SAMPLE_LENGTH, y)
    if kind == "salt_bridge":
        grad = QLinearGradient(a, b)
        grad.setColorAt(0.0, QColor(SALT_BRIDGE_LEGEND[0]))
        grad.setColorAt(1.0, QColor(SALT_BRIDGE_LEGEND[1]))
        pen = QPen(QBrush(grad), style.width)
    else:
        pen = QPen(QColor(style.color), style.width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    if style.dash:
        pen.setDashPattern(list(style.dash))
    painter.setPen(pen)
    end = QPointF(b.x() - ARROW_LENGTH * 0.8, y) if style.marker == "arrow" else b
    painter.drawLine(a, end)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(style.color))
    if style.marker == "arrow":
        painter.drawPolygon(_arrow_head(b, 0.0))
    elif style.marker == "dot":
        # As in the references: Pi-Pi carries a dot at both ends, Pi-cation one.
        if kind != "pi_cation":
            painter.drawEllipse(a, DOT_RADIUS, DOT_RADIUS)
        painter.drawEllipse(b, DOT_RADIUS, DOT_RADIUS)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@dataclass
class SceneBuild:
    """What :func:`build_scene` hands back to the widget and the exporters."""

    scene: QGraphicsScene
    #: RDKit atom index -> position in scene coordinates.
    atom_coords: dict[int, QPointF] = field(default_factory=dict)
    #: ``Residue.key`` -> the droplet item, which also carries ``residue_key``.
    droplets: dict[str, ResidueDroplet] = field(default_factory=dict)
    ligand: LigandItem | None = None
    #: The layers the widget lets the user switch.  ``None`` when the diagram
    #: has nothing of that kind to draw.
    backbone: BackboneConnectors | None = None
    halos: SolventHalos | None = None
    routes: InteractionRoutes | None = None
    legend: Legend | LegendTable | None = None
    ribbons: Ribbons | None = None


def build_scene(diagram: Diagram, positions: dict[str, tuple[float, float]],
                ligand_coords: list[tuple[float, float]],
                scene: QGraphicsScene | None = None,
                legend_position: str = "left",
                legend_rows: int = 3,
                look: DiagramStyle | None = None,
                bends: dict[str, float] | None = None) -> SceneBuild:
    """Build (or rebuild in place) the whole diagram.

    ``positions`` are droplet body centres and ``ligand_coords`` are per-atom
    ligand positions, both in scene units, both from ``ms_contactmap.layout``.
    Passing an existing ``scene`` clears it first, so the widget can call this
    again after a drag without leaking items.

    ``bends`` maps a route's bend key to a hand-set bow (a fraction of its
    chord), see :class:`InteractionRoutes`.
    """
    if scene is None:
        scene = QGraphicsScene()
    else:
        scene.clear()

    pos = {key: QPointF(*xy) for key, xy in positions.items()}
    ligand_item = _draw_ligand(diagram, ligand_coords)
    atom_coords = ligand_item.atom_coords
    center = QPointF(*centroid(ligand_coords))
    bond_length = _median_bond_length(diagram, ligand_coords)

    spots = _exposure_spots(diagram, atom_coords, ligand_coords)
    halos = None
    if spots:
        halos = SolventHalos(spots, bond_length * HALO_RADIUS_FRAC,
                             "halo" if look is None else look.exposure.replace("hidden", "arc"),
                             atoms=list(atom_coords.values()))
        if look is not None and look.exposure == "hidden":
            halos.setVisible(False)
    if halos is not None:
        scene.addItem(halos)

    runs = _ribbon_runs(diagram, pos, center, ligand_coords, look)
    ribbons = Ribbons(runs, "classic" if look is None else look.surface) if runs else None
    if ribbons is not None:
        ribbons.setVisible(look is None or look.surface != "hidden")
        scene.addItem(ribbons)

    backbone = BackboneConnectors(_backbone_path(diagram, pos, center, ligand_coords),
                                  styled=look is not None)
    scene.addItem(backbone)

    droplets: dict[str, ResidueDroplet] = {}
    for residue in diagram.residues:
        p = pos.get(residue.key)
        if p is None:
            continue
        anchor = atom_coords.get(diagram.nearest_atom.get(residue.key, -1), center)
        coordination = None
        sides = diagram.metal_coordination.get(residue.key)
        if sides:
            sides = max(3, min(8, sides))
            legs = _coordination_legs(diagram, residue.key, p, pos, atom_coords)
            bearings = [math.atan2(q.y() - p.y(), q.x() - p.x()) for _, q in legs]
            coordination = (sides, _vertex_angle(bearings, sides))
        item = ResidueDroplet(
            residue, math.atan2(anchor.y() - p.y(), anchor.x() - p.x()), coordination, look
        )
        item.setPos(p)
        # When two glyphs do end up touching, the outer one goes behind -- the
        # stacked pairs Maestro draws in 4uwh.png read that way, and a fixed
        # rule beats letting insertion order decide which one gets clipped.
        item.setZValue(Z_DROPLETS - math.dist((p.x(), p.y()), (center.x(), center.y())) / 1e4)
        droplets[residue.key] = item

    bonds = [
        (QPointF(*ligand_coords[b.GetBeginAtomIdx()]), QPointF(*ligand_coords[b.GetEndAtomIdx()]))
        for b in diagram.mol.GetBonds()
    ]
    shapes = {key: item.shape_in_scene() for key, item in droplets.items()}
    vertices = _metal_vertices(diagram, pos, atom_coords, droplets)
    routes = _routes(diagram, pos, atom_coords, shapes, bonds, vertices, look, bends)
    route_item = InteractionRoutes(routes, droplets) if routes else None
    if route_item is not None:
        scene.addItem(route_item)

    scene.addItem(ligand_item)
    for item in droplets.values():
        scene.addItem(item)

    if legend_position not in {"left", "right", "top", "bottom"}:
        raise ValueError(f"unknown legend position: {legend_position!r}")
    body = scene.itemsBoundingRect()
    if look is None:
        legend = Legend(
            diagram,
            rows=legend_rows,
            single_column=legend_position in {"left", "right"},
        )
    else:
        legend = LegendTable(
            diagram, look,
            stacked=legend_position in {"left", "right"},
            has_surface=ribbons is not None,
            has_lines={r.kind for r in routes},
        )
    key = legend.boundingRect()
    if legend_position == "left":
        legend.setPos(body.left() - LEGEND_GAP - key.right(),
                      body.center().y() - key.center().y())
    elif legend_position == "right":
        legend.setPos(body.right() + LEGEND_GAP - key.left(),
                      body.center().y() - key.center().y())
    elif legend_position == "top":
        legend.setPos(body.center().x() - key.center().x(),
                      body.top() - LEGEND_GAP - key.bottom())
    else:
        legend.setPos(body.center().x() - key.center().x(),
                      body.bottom() + LEGEND_GAP - key.top())
    scene.addItem(legend)
    scene.setSceneRect(scene.itemsBoundingRect().adjusted(-24, -24, 24, 24))
    return SceneBuild(scene, atom_coords, droplets, ligand_item,
                      backbone, halos, route_item, legend, ribbons)


def _median_bond_length(diagram: Diagram, ligand_coords) -> float:
    lengths = sorted(
        math.dist(ligand_coords[b.GetBeginAtomIdx()], ligand_coords[b.GetEndAtomIdx()])
        for b in diagram.mol.GetBonds()
    )
    return lengths[len(lengths) // 2] if lengths else 38.0
