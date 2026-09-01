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
from PySide6.QtCore import QByteArray, QPointF, QRectF, Qt
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
    QToolTip,
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
    point_segment_distance,
    segments_cross,
)

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
LEGEND_GAP = 42.0
LEGEND_CROSS_COLOR = QColor("#ee0000")
#: The reference legend draws the salt-bridge sample as a red-to-blue blend
#: (negative to positive), unlike the plain blue of ``INTERACTION_STYLES``.
SALT_BRIDGE_LEGEND = ("#fa0014", "#0000ff")

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

    MODES = ("halo", "trail")

    def __init__(self, spots: list[_Exposure], radius: float, mode: str = "halo"):
        super().__init__()
        self._spots = spots
        self._radius = radius
        self._mode = mode
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

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self._mode == "trail":
            self._paint_trails(painter)
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


def halo_gradient(center: QPointF, radius: float) -> QRadialGradient:
    """The ring profile measured off the pyrrolidine halos in 4ps5.png."""
    g = QRadialGradient(center, radius)
    for stop, alpha in ((0.0, 20), (0.30, 15), (0.52, 46), (0.68, 34), (0.86, 12), (1.0, 0)):
        c = QColor(HALO_COLOR)
        c.setAlpha(alpha)
        g.setColorAt(stop, c)
    return g


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

    def __init__(self, ribbons: list[_Ribbon]):
        super().__init__()
        self._ribbons = ribbons
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
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for ribbon in self._ribbons:
            pts, cols = ribbon.points, ribbon.colors
            last = len(pts) - 2
            for i in range(len(pts) - 1):
                pen = QPen(cols[i], RIBBON_WIDTH)
                # Flat caps: consecutive segments abut instead of overlapping,
                # which would bead visibly wherever the alpha is below 255.
                pen.setCapStyle(Qt.PenCapStyle.RoundCap if i in (0, last)
                                else Qt.PenCapStyle.FlatCap)
                painter.setPen(pen)
                painter.drawLine(pts[i], pts[i + 1])


def _ribbon_runs(diagram: Diagram, positions: dict[str, QPointF],
                 center: QPointF, ligand_coords: list[tuple[float, float]]) -> list[_Ribbon]:
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
        classes = [RESIDUE_STYLES[value].base for value in residue_classes]

        # Reconstruct a radial contour after optimization, then roll a probe
        # twice the ligand probe over it.  Unlike an interpolating Catmull-Rom
        # spline, this does not reproduce every local radial wobble or overshoot
        # between two droplets at different depths.
        count = max(2, int(math.ceil((angles[-1] - angles[0]) / RIBBON_SAMPLE_STEP)) + 1)
        sample_angles = np.linspace(angles[0], angles[-1], count)
        raw = np.interp(sample_angles, angles, np.asarray(radii) - RIBBON_DROPLET_INSET)
        step = max(float(sample_angles[1] - sample_angles[0]), 1e-6)
        mean_radius = max(float(np.mean(raw)), 1.0)
        sigma = (RIBBON_SURFACE_PROBE_RADIUS / mean_radius) / step
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
            required = math.hypot(dx, dy) + DROPLET_RADIUS + 10.0
            lift = max(0.0, required - float(smooth[at]))
            if lift > 0.0:
                sigma = 0.14
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

    def __init__(self, path: QPainterPath):
        super().__init__()
        self._path = path
        self.setZValue(Z_BACKBONE)

    def boundingRect(self) -> QRectF:
        return self._path.boundingRect().adjusted(-4, -4, 4, 4)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        pen = QPen(BACKBONE_COLOR, BACKBONE_WIDTH)
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


class InteractionRoutes(QGraphicsObject):
    """Dashed / solid contact lines from ligand atoms to droplet rims.

    All the routes share one item rather than getting one each: they are
    painted in a fixed order and never move independently, and a single item
    is what keeps a busy scene at one pixmap cache instead of forty.  Hover is
    therefore resolved here, against the stroked paths.
    """

    def __init__(self, routes: list[_Route], droplets: dict[str, "ResidueDroplet"] | None = None):
        super().__init__()
        self._routes = routes
        self._droplets = droplets or {}
        self._hot: int | None = None
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
        if self._hot is not None:
            QToolTip.showText(event.screenPos(), self._routes[self._hot].label)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._set_hot(None)
        QToolTip.hideText()
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
        self.update()

    # -- paint -------------------------------------------------------------

    def paint(self, painter: QPainter, option, widget=None) -> None:
        for i, route in enumerate(self._routes):
            style = INTERACTION_STYLES[route.kind]
            color = QColor(style.color)
            hot = i == self._hot
            pen = QPen(color, style.width * (HOVER_BOLD if hot else 1.0))
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


def _sq_dist(a: QPointF, b: QPointF) -> float:
    return (a.x() - b.x()) ** 2 + (a.y() - b.y()) ** 2


def _arrow_head(tip: QPointF, angle: float) -> QPolygonF:
    back = QPointF(tip.x() - ARROW_LENGTH * math.cos(angle),
                   tip.y() - ARROW_LENGTH * math.sin(angle))
    nx, ny = -math.sin(angle) * ARROW_HALF_WIDTH, math.cos(angle) * ARROW_HALF_WIDTH
    return QPolygonF([tip, QPointF(back.x() + nx, back.y() + ny),
                      QPointF(back.x() - nx, back.y() - ny)])


def _ligand_anchor(inter, atom_coords: dict[int, QPointF],
                   target: QPointF) -> tuple[str, QPointF] | None:
    """Where this interaction leaves the ligand, and a stable id for that spot."""
    anchors = [(i, atom_coords[i]) for i in inter.ligand_atoms if i in atom_coords]
    if not anchors:
        return None
    if inter.kind in RING_ANCHORED:
        # Pi systems really do start at the ring centre.
        cx = sum(p.x() for _, p in anchors) / len(anchors)
        cy = sum(p.y() for _, p in anchors) / len(anchors)
        return f"ring:{anchors[0][0]}", QPointF(cx, cy)
    # Detection hands back the whole charged/coordinating group (a carboxylate is
    # C + both O), and the centroid of that group sits in empty space between
    # the atoms -- which is exactly the misalignment seen on the 2gfk salt
    # bridges.  Maestro starts the line on one atom, so pick the group atom
    # facing the residue.
    idx, point = min(
        anchors,
        key=lambda ip: (ip[1].x() - target.x()) ** 2 + (ip[1].y() - target.y()) ** 2,
    )
    return f"atom:{idx}", point


def _coordination_legs(diagram: Diagram, key: str, center: QPointF,
                       positions: dict[str, QPointF],
                       atom_coords: dict[int, QPointF]) -> list[tuple[str, QPointF]]:
    """Every partner in metal ``key``'s sphere: (leg id, where its line comes from)."""
    legs: list[tuple[str, QPointF]] = []
    for inter in diagram.interactions_of(key):
        got = _ligand_anchor(inter, atom_coords, center)
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
        verts = [item.mapToScene(item.polygon.at(i)) for i in range(item.polygon.count())]
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
            vertices: dict[tuple[str, str], QPointF]) -> list[_Route]:
    routes: list[_Route] = []
    seen_water_ligand: set[tuple[str, str]] = set()
    for inter in sorted(diagram.interactions, key=lambda i: i.style.priority):
        # Maestro carries hydrophobic contacts with the residue colour alone.
        if inter.kind == "hydrophobic":
            continue
        target = positions.get(inter.residue_key)
        shape = shapes.get(inter.residue_key)
        if target is None or shape is None:
            continue
        got = _ligand_anchor(inter, atom_coords, target)
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

            path = QPainterPath(start)
            bow = _needed_bow(start, end, positions, key, bonds)
            if bow:
                mid = QPointF((start.x() + end.x()) / 2, (start.y() + end.y()) / 2)
                dx, dy = end.x() - start.x(), end.y() - start.y()
                length = math.hypot(dx, dy) or 1.0
                path.quadTo(QPointF(mid.x() - dy / length * bow, mid.y() + dx / length * bow), end)
            else:
                path.lineTo(end)
            route_arrow = arrow if hop == len(hops) - 1 else 0
            if inter.via_water:
                if branch_from_water or hop == 1:
                    route_arrow = -1 if inter.protein_is_donor else 1
                else:
                    route_arrow = 1 if inter.ligand_is_donor else -1
            routes.append(
                _Route(path, inter.kind, route_arrow,
                       style.marker == "dot",
                       label=_route_label(inter, style),
                       keys=tuple(k for k, _, _ in hops))
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

    def __init__(self, residue, tip_angle: float, coordination: tuple[int, float] | None = None):
        super().__init__()
        self.residue_key: str = residue.key
        self.residue = residue
        self.tip_angle = tip_angle
        self.style = RESIDUE_STYLES[residue.ref.residue_class]
        self.is_water = residue.ref.residue_class == "water"
        self.is_metal = coordination is not None
        self.radius = DROPLET_RADIUS
        if self.is_metal:
            # The coordination polyhedron drawn flat: a metal with four
            # partners is a square, five a pentagon, and each partner's line
            # lands on its own corner.
            sides, angle = coordination
            self.polygon = _ngon(QPointF(0, 0), DROPLET_RADIUS, sides, angle)
        elif self.is_water:
            self.radius = DROPLET_RADIUS * WATER_RADIUS_FRAC
            self.polygon = _circle(QPointF(0, 0), self.radius)
        else:
            self.polygon = _teardrop(QPointF(0, 0), DROPLET_RADIUS, tip_angle)
        self.lit = False
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

    def set_lit(self, lit: bool) -> None:
        """Thicken the outline while a route touching this residue is hovered."""
        if lit != self.lit:
            self.lit = lit
            self.update()

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
        return QRectF(-8, -LEGEND_SPHERE_RADIUS - 4,
                      LEGEND_COL_PITCH * (cols - 1) + 260, LEGEND_ROW_PITCH * rows + 12)

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

                painter.setPen(QColor("#101010"))
                painter.setFont(font)
                painter.drawText(QPointF(x + LEGEND_TEXT_OFFSET, y + metrics.capHeight() / 2), label)


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
    legend: Legend | None = None


def build_scene(diagram: Diagram, positions: dict[str, tuple[float, float]],
                ligand_coords: list[tuple[float, float]],
                scene: QGraphicsScene | None = None,
                legend_position: str = "left",
                legend_rows: int = 3) -> SceneBuild:
    """Build (or rebuild in place) the whole diagram.

    ``positions`` are droplet body centres and ``ligand_coords`` are per-atom
    ligand positions, both in scene units, both from ``ms_contactmap.layout``.
    Passing an existing ``scene`` clears it first, so the widget can call this
    again after a drag without leaking items.
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
    halos = SolventHalos(spots, bond_length * HALO_RADIUS_FRAC) if spots else None
    if halos is not None:
        scene.addItem(halos)

    ribbons = _ribbon_runs(diagram, pos, center, ligand_coords)
    if ribbons:
        scene.addItem(Ribbons(ribbons))

    backbone = BackboneConnectors(_backbone_path(diagram, pos, center, ligand_coords))
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
            residue, math.atan2(anchor.y() - p.y(), anchor.x() - p.x()), coordination
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
    routes = _routes(diagram, pos, atom_coords, shapes, bonds, vertices)
    route_item = InteractionRoutes(routes, droplets) if routes else None
    if route_item is not None:
        scene.addItem(route_item)

    scene.addItem(ligand_item)
    for item in droplets.values():
        scene.addItem(item)

    if legend_position not in {"left", "right", "top", "bottom"}:
        raise ValueError(f"unknown legend position: {legend_position!r}")
    body = scene.itemsBoundingRect()
    legend = Legend(
        diagram,
        rows=legend_rows,
        single_column=legend_position in {"left", "right"},
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
                      backbone, halos, route_item, legend)


def _median_bond_length(diagram: Diagram, ligand_coords) -> float:
    lengths = sorted(
        math.dist(ligand_coords[b.GetBeginAtomIdx()], ligand_coords[b.GetEndAtomIdx()])
        for b in diagram.mol.GetBonds()
    )
    return lengths[len(lengths) // 2] if lengths else 38.0
