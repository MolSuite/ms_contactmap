"""Layout geometry: where every residue droplet goes.

The input is a :class:`~ms_contactmap.model.Diagram` (RDKit 2D coordinates plus a
list of residues and interactions), the output is a :class:`LayoutResult` with
one scene position per residue and the ligand orientation that was chosen.
Nothing here imports RDKit or Qt: it is pure geometry over numpy arrays, so it
runs headless and fast.

The scene frame is centred on the ligand centroid and measured in pixels, with
the ligand scaled so that a bond is :attr:`Weights.bond_px` long.  The y axis
is whatever RDKit handed us (maths-up); flipping it for Qt is the renderer's
job.

The public solver uses the bounded discrete/position-based implementation in
:mod:`ms_contactmap.fast_layout`.  The smooth energy model remains here as a
diagnostic, a source of vectorized geometry primitives and an explicitly named
legacy solver for comparisons; importing the normal path does not import its
SciPy minimizer.
"""
from __future__ import annotations

import math
import sys
from dataclasses import astuple, dataclass, field

import numpy as np

from .model import Diagram

_EPS = 1e-9

#: Orientations from the scan that get a full solve, best energy first, and
#: how far apart they have to be so the restarts are actually different layouts.
_RESTARTS = 4
_RESTART_SPACING = math.radians(45.0)

#: The orientation scan runs in two passes: everything gets ``_SCAN_COARSE``
#: iterations, and only the cheapest ``_SCAN_FINALISTS`` are relaxed properly.
#: A rotation that is going to lose declares itself within a dozen iterations,
#: and the scan is the largest single cost of a full solve.
#: 30 is a threshold, not a taste: below ~25 the coarse ranking misorders 6wak
#: badly enough that the finalists no longer contain the basin the full scan
#: found, and its energy comes out 2.6x worse.
_SCAN_COARSE = 30
_SCAN_FINALISTS = 24

#: Orientations tried per candidate ligand depiction while choosing between
#: them.  Only enough to tell a projection with room from one without.
_PROJECTION_ROTATIONS = 8

#: Where along a line E_span checks that it has cleared the ligand.  An
#: interaction route *starts* on a ligand atom, so its first half is inside the
#: hull by construction and sampling it would only fight the anchor spring;
#: from 0.45 out the term asks the useful question, "does this arrow leave the
#: ligand promptly, or does it lie across it?".  Backbone connectors join two
#: free glyphs and are sampled symmetrically.
_ROUTE_SAMPLES = np.array([0.45, 0.65, 0.85])
_EDGE_SAMPLES = np.array([0.25, 0.5, 0.75])

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

@dataclass
class Weights:
    """Energy weights and the handful of lengths they are balanced against.

    Shape floats live next to the weights because the fidelity loop tunes the
    two together -- changing ``d0_far`` without changing ``anchor`` just moves
    the same compromise around.
    """

    # --- weights, one per energy term -------------------------------------
    anchor: float = 1.0
    ligand: float = 40.0
    overlap: float = 40.0
    backbone: float = 1.6
    spread: float = 50.0
    frame: float = 260.0
    #: Flattens the rim into one smooth surface.  Tuned by sweeping against the
    #: mean second difference of the hull standoff around the rim: on the four
    #: reference systems it falls 20-40% between 0 and 40 and stops improving
    #: after that, while the solve gets *faster* (a smoother landscape means
    #: fewer crossings for the repair pass to chase).
    shell: float = 40.0
    #: Deliberately modest.  E_anchor is what drags a line across the ligand,
    #: and a span weight strong enough to always win that argument (60+) fixes
    #: the last synthetic metal case at the cost of flinging real glyphs to the
    #: rim: on the four reference systems it makes 6wak and 4uwh visibly worse
    #: and reshuffles which ligand projection scores best.  At 14 the term
    #: clears the lines it can and defers where it cannot.
    span: float = 14.0
    #: Smooth line-versus-glyph clearance for interaction routes and backbone
    #: connectors.  It prevents a line from being laid over a foreign water or
    #: residue before the discrete crossing repair runs.
    route_overlap: float = 24.0
    #: Candidate-ranking preference for a ligand's principal axis to read
    #: horizontally.  Scaled by depiction anisotropy, so round ligands are not
    #: assigned an arbitrary direction.  This ranks orientations only; it does
    #: not distort the optimized residue positions or reported energy.
    horizontal_preference: float = 180_000.0

    # --- lengths, in scene pixels -----------------------------------------
    d0_far: float = 150.0       #: stand-off of a residue with no interactions
    d0_near: float = 100.0      #: stand-off a heavily interacting residue tends to
    #: Stand-off of a water bridging the ligand to a residue.  Structural
    #: waters use their own small radius and margin.  Fifty pixels keeps them
    #: in an inner shell without pinning the sphere against the ligand drawing.
    d0_water: float = 50.0
    #: Radius of the virtual 2D probe rolled around the ligand depiction before
    #: residues are placed.  It is half one drawn bond, a useful approximation
    #: to the visual atom diameter at this scale.
    ligand_probe_radius: float = 24.5
    #: The residue shell is filtered with a probe twice as large as the ligand
    #: probe.  This is a geometric length, not a fixed angular blur, so long and
    #: compact ligands receive comparable visual smoothing.
    shell_probe_radius: float = 34.0
    #: A bridge water samples the ligand contour with half the regular probe;
    #: it can occupy a local pocket without changing the contour followed by
    #: the protein residues.
    water_probe_scale: float = 0.5
    #: Structural water is a waypoint, not a residue glyph.  Its visual and
    #: collision radius use this fraction of the normal droplet radius.
    water_radius_fraction: float = 0.52
    #: Spring between a bridging water and the residue it bridges, relative to
    #: a sequence connector.  Stiff: this is the term that puts the water
    #: *between* the ligand and the residue rather than merely near both.
    bridge_spring: float = 15.0
    #: Centre-to-centre rest length of water -> protein in a water bridge.  It
    #: deliberately exceeds one normal glyph pitch: the dashed leg needs room
    #: to read between the small sphere and the residue droplet.
    bridge_rest: float = 96.0
    anchor_decay: float = 2.0   #: interactions needed to close half the gap above
    anchor_stiffen: float = 0.5 #: extra spring stiffness per interaction
    #: Anchor-stiffness multiplier for the contacts that take three glyphs to
    #: read -- a metal with its sphere, a water with the residue it bridges.
    #: They lead, the rest of the pocket follows.
    conductor: float = 3.0
    #: Additional anchor stiffness for a structural water.  The protein
    #: partner may negotiate with the smooth shell, but the water must remain
    #: attached to the local ligand atom that defines the bridge.
    water_anchor: float = 3.0
    #: What removing one crossing is worth to the repair pass, in the same
    #: weighted glyph-to-target pixels it trades against.  Two glyph pitches:
    #: shifting a residue a couple of slots to untangle a diagram is a good
    #: deal, sending it to the far rim -- and dragging its bond across the
    #: ligand to follow -- is not.  The 3D view is there for the precision, so
    #: a short line beats a clean one.
    crossing_cost: float = 170.0
    #: Multiplier on the backbone spring for a metal coordination leg.  The
    #: partners already share the metal's anchor, so this only has to keep them
    #: within a glyph of it rather than fight the rest of the energy.
    metal_leg: float = 3.0
    anchor_soft: float = 60.0   #: displacement past which the anchor pull stops growing
    #: Clearance between the ligand hull and a glyph rim.  This is the padding
    #: that keeps the drawing readable: at 28 the closest glyph sat ~28 px off
    #: the hull, which reads as touching once the teardrop tip points inwards.
    ligand_margin: float = 46.0
    #: Waters are small inner-shell nodes, not full residue droplets.  This is
    #: added to their actual glyph radius instead of ``ligand_margin``.  It is
    #: intentionally smaller than the 46 px residue margin, but large enough
    #: for the bridge line and atom label to remain readable.
    water_ligand_margin: float = 24.0
    #: Pair clearance whenever at least one endpoint is a structural water.
    #: A full residue-residue gap would force the bridge open visually.
    water_gap: float = 8.0
    #: Clearance a line (interaction route, backbone connector, metal leg) tries
    #: to keep from the ligand hull.  Smaller than ``ligand_margin`` on purpose:
    #: a line only has to miss the drawing, not stand off it like a glyph.
    span_margin: float = 12.0
    line_glyph_gap: float = 9.0
    #: Clearance between two glyph rims.  The body radius is 24 but the
    #: teardrop tip reaches 1.62x that, so two glyphs whose rims merely touch
    #: still look stacked; this is the knob that buys the diagram air.
    glyph_gap: float = 32.0
    spread_arc: float = 76.0    #: tangential clearance two glyphs try to keep
    #: How far along the rim E_shell still expects two glyphs to agree on a
    #: radius, as arc length on the reference circle.  At ~1.5 x spread_arc a
    #: glyph is tied to its immediate neighbours and barely to the next ones
    #: out, which is the span over which "smooth surface" means anything.
    shell_span: float = 115.0
    #: Standoff disagreement past which E_shell stops pulling harder.  Well
    #: inside one glyph pitch (2r + glyph_gap = 84) on purpose: a residue
    #: crowded off the front row has to be able to drop a full row behind and
    #: settle there at a flat price, instead of being dragged back and
    #: deforming the surface for everyone around it.
    shell_soft: float = 32.0
    #: On-screen length of a ligand bond.  Measured on data/4ps5.png: the
    #: depiction is 250 x 215 reference px and the same RDKit conformer is
    #: 10.23 x 8.69 bond lengths across, so a bond is ~24.5 reference px, which
    #: is 49 scene units.  At the old 24 the ligand read as a small blob inside
    #: a wide ring of glyphs -- the single largest departure from Maestro.
    bond_px: float = 49.0
    canvas_aspect: float = 1.75 #: width / height of the soft canvas ellipse
    canvas_slack: float = 0.72  #: canvas radius as a fraction of ligand radius + d0_far
    hull_beta: float = 0.4      #: smoothing of the hull's signed distance, 1/px


WEIGHTS = Weights()


@dataclass
class LayoutResult:
    positions: dict[str, tuple[float, float]]
    ligand_coords: list[tuple[float, float]]
    rotation: float
    mirror: bool
    energy: float
    energy_terms: dict[str, float]
    crossings: int
    #: Index into ``[diagram.coords_2d, *diagram.coords_alt]`` of the depiction
    #: this layout was built around.  Feed it back to :func:`solve_layout` to
    #: keep the same drawing when re-solving.
    projection: int = 0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull of ``points``, as indices (monotone chain)."""
    order = np.lexsort((points[:, 1], points[:, 0]))
    if len(order) < 3:
        return order

    def half(seq: np.ndarray) -> list[int]:
        out: list[int] = []
        for i in seq:
            while len(out) >= 2:
                a, b = points[out[-2]], points[out[-1]]
                if (b[0] - a[0]) * (points[i][1] - a[1]) - (b[1] - a[1]) * (points[i][0] - a[0]) > 1e-9:
                    break
                out.pop()
            out.append(int(i))
        return out

    lower = half(order)
    upper = half(order[::-1])
    return np.array(lower[:-1] + upper[:-1], dtype=int)


def _halfplanes(hull: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Outward unit normals and offsets of a CCW hull polygon.

    ``n_e . p - c_e`` is then positive outside edge ``e``, and the maximum over
    the edges is the signed distance to the polygon (exact inside, a slight
    underestimate outside).  A degenerate hull (one atom, or a straight line)
    is replaced by a small circumscribed polygon so the term never blows up.
    """
    if len(hull) >= 3:
        # A mirrored ligand hands us the same vertices wound clockwise, which
        # would point every normal inwards.
        nxt = np.roll(hull, -1, axis=0)
        if np.sum(hull[:, 0] * nxt[:, 1] - nxt[:, 0] * hull[:, 1]) < 0.0:
            hull = hull[::-1]
        edge = np.roll(hull, -1, axis=0) - hull
        length = np.hypot(edge[:, 0], edge[:, 1])
        keep = length > 1e-9
        if keep.any():
            normals = np.stack([edge[keep, 1], -edge[keep, 0]], axis=1) / length[keep, None]
            return normals, np.einsum("ij,ij->i", normals, hull[keep])

    centre = hull.mean(axis=0)
    radius = float(np.max(np.hypot(*(hull - centre).T))) if len(hull) else 0.0
    ang = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    normals = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return normals, normals @ centre + radius


def _outward(points: np.ndarray, normals: np.ndarray, offsets: np.ndarray, beta: float) -> np.ndarray:
    """Unit vector pointing out of the ligand at each point.

    A blend of the hull normals, weighted by how close the point is to each
    edge.  For an elongated ligand this beats "away from the centroid", which
    would send an atom in the middle of the chain off along the long axis
    instead of out to the side where the residue actually sits.
    """
    g = points @ normals.T - offsets
    ex = np.exp(beta * (g - g.max(axis=1, keepdims=True)))
    out = (ex / ex.sum(axis=1, keepdims=True)) @ normals
    length = np.hypot(out[:, 0], out[:, 1])
    flat = length < 1e-6                      # a point at the centre of a symmetric hull
    if flat.any():
        out[flat] = points[flat]
        length = np.hypot(out[:, 0], out[:, 1])
    return out / np.maximum(length, _EPS)[:, None]


def _orient(coords: np.ndarray, rotation: float, mirror: bool) -> np.ndarray:
    """Mirror about x, then rotate by ``rotation`` radians."""
    out = coords * np.array([-1.0, 1.0]) if mirror else coords
    cos, sin = math.cos(rotation), math.sin(rotation)
    return out @ np.array([[cos, sin], [-sin, cos]])


def _bond_length(coords: np.ndarray) -> float:
    """Median nearest-neighbour distance, i.e. the bond length without RDKit."""
    if len(coords) < 2:
        return 1.5
    d = np.hypot(*(coords[:, None, :] - coords[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1))) or 1.5


# ---------------------------------------------------------------------------
# The smooth problem
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    """Everything the smooth objective needs, for one fixed ligand orientation.

    ``objective`` returns ``(energy, gradient)`` over the flattened ``(n, 2)``
    glyph positions and is what L-BFGS-B and ``check_grad`` are handed.
    """

    keys: list[str]
    ligand: np.ndarray          # (m, 2) oriented, scaled, centred ligand atoms
    ligand_bonds: np.ndarray    # (b, 2) ligand atom-index pairs
    anchors: np.ndarray         # (n, 2)
    target: np.ndarray          # (n, 2) anchor pushed d0 out along the hull normal
    normals: np.ndarray         # (h, 2) outward hull normals
    offsets: np.ndarray         # (h,)
    stiff: np.ndarray           # (n,) anchor spring stiffness
    pinned: np.ndarray          # (n,) bool, conductors the crossing repair may not move
    on_shell: np.ndarray        # (n,) bool, glyphs that belong on the rim
    water_mask: np.ndarray      # (n,) bool, local concave ligand clearance
    metal_mask: np.ndarray      # (n,) bool, local outer coordination nodes
    has_line: np.ndarray        # (n,) bool, residues that draw an interaction line
    backbone: np.ndarray        # (e, 2) index pairs
    edge_k: np.ndarray          # (e,) per-edge stiffness multiplier
    span_base: np.ndarray       # (k, 2) the fixed (anchor) share of each E_span sample
    span_map: np.ndarray        # (k, n) the glyph share: sample = base + map @ p
    span_owner: np.ndarray      # (k, n) glyphs belonging to each sampled line
    spread_mask: np.ndarray     # (n, n) 1.0 for pairs that repel angularly
    shell_kernel: np.ndarray    # (n, n) how strongly a pair shares a radius
    ref_radius: float           # circle the spread arc is measured on
    frame: np.ndarray           # (2,) canvas ellipse semi-axes
    radius: float               # glyph radius
    radii: np.ndarray           # (n,) actual collision radius per glyph
    ligand_clearance: np.ndarray # (n,) required centre clearance from ligand hull
    pair_floor: np.ndarray      # (n, n) minimum centre separation
    edge_rest: np.ndarray       # (e,) target length per backbone/bridge edge
    line_start_base: np.ndarray # (l, 2) fixed share of drawn segment starts
    line_start_map: np.ndarray  # (l, n) free-position share of segment starts
    line_end_base: np.ndarray   # (l, 2) fixed share of drawn segment ends
    line_end_map: np.ndarray    # (l, n) free-position share of segment ends
    line_owner: np.ndarray      # (l, n) endpoints that own each drawn segment
    w: Weights = field(default_factory=Weights)

    @property
    def n(self) -> int:
        return len(self.keys)

    # -- energy ------------------------------------------------------------

    def objective(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        e, g, _ = self._eval(x, terms=False)
        return e, g

    def terms(self, x: np.ndarray) -> dict[str, float]:
        return self._eval(x, terms=True)[2]

    def _hull_distance(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Smoothed signed distance from each point to the ligand hull, and d/dpts.

        A log-sum-exp over the hull half-planes; the ``log(h)/beta`` correction
        keeps it an underestimate, so anything built on it errs towards more
        clearance rather than less.
        """
        beta = self.w.hull_beta
        g = pts @ self.normals.T - self.offsets
        top = g.max(axis=1, keepdims=True)
        ex = np.exp(beta * (g - top))
        tot = ex.sum(axis=1, keepdims=True)
        dist = top[:, 0] + np.log(tot[:, 0]) / beta - math.log(len(self.offsets)) / beta
        return dist, (ex / tot) @ self.normals

    def _hull_push(self, pts: np.ndarray, clearance: float) -> tuple[float, np.ndarray]:
        """Quadratic hinge on :meth:`_hull_distance`: energy and gradient."""
        dist, ddist = self._hull_distance(pts)
        pen = np.maximum(0.0, clearance - dist)
        return 0.5 * float(pen @ pen), -pen[:, None] * ddist

    def _molecule_distance(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distance and gradient to the actual 2D atom/bond geometry.

        Unlike the convex hull, this metric preserves concave pockets.  The
        soft minimum is conservative (it slightly underestimates clearance)
        and differentiable where the nearest primitive changes.
        """
        if len(pts) == 0:
            return np.zeros(0), np.zeros((0, 2))
        vectors = [pts[:, None, :] - self.ligand[None, :, :]]
        if len(self.ligand_bonds):
            a = self.ligand[self.ligand_bonds[:, 0]]
            b = self.ligand[self.ligand_bonds[:, 1]]
            edge = b - a
            length2 = np.maximum(np.einsum("ij,ij->i", edge, edge), _EPS)
            rel = pts[:, None, :] - a[None, :, :]
            t = np.clip(
                np.einsum("pbi,bi->pb", rel, edge) / length2[None, :],
                0.0, 1.0,
            )
            closest = a[None, :, :] + t[:, :, None] * edge[None, :, :]
            vectors.append(pts[:, None, :] - closest)
        vector = np.concatenate(vectors, axis=1)
        distance = np.maximum(np.hypot(vector[:, :, 0], vector[:, :, 1]), _EPS)
        beta = self.w.hull_beta
        low = distance.min(axis=1, keepdims=True)
        weight = np.exp(-beta * (distance - low))
        total = weight.sum(axis=1, keepdims=True)
        soft = low[:, 0] - np.log(total[:, 0]) / beta
        gradient = np.sum(
            (weight / total)[:, :, None] * vector / distance[:, :, None],
            axis=1,
        )
        return soft, gradient

    def _molecule_push(
        self, pts: np.ndarray, clearance: np.ndarray | float
    ) -> tuple[float, np.ndarray]:
        distance, ddistance = self._molecule_distance(pts)
        pen = np.maximum(0.0, np.asarray(clearance) - distance)
        return 0.5 * float(pen @ pen), -pen[:, None] * ddistance

    def _eval(self, x: np.ndarray, terms: bool) -> tuple[float, np.ndarray, dict[str, float]]:
        w = self.w
        p = x.reshape(-1, 2)
        grad = np.zeros_like(p)
        out: dict[str, float] = {}

        # E_anchor: spring to the anchor, held d0 away from it and on the
        # outward side of the ligand.  A bare distance spring leaves the glyph
        # free to slide tangentially, which is what sends its interaction line
        # sweeping sideways across the neighbouring droplets.
        #
        # Pseudo-Huber, not a plain quadratic: past anchor_soft pixels the pull
        # stops growing, so a residue can be shouldered aside by its crowded
        # neighbours.  A quadratic here simply wins every argument, and a busy
        # patch of ligand ends up with its droplets stacked on top of each
        # other rather than fanned out along the rim.
        soft = w.anchor_soft
        v = p - self.target
        bend = np.sqrt(1.0 + np.einsum("ij,ij->i", v, v) / (soft * soft))
        e_anchor = soft * soft * float(np.sum(self.stiff * (bend - 1.0)))
        grad += w.anchor * (self.stiff / bend)[:, None] * v

        # E_ligand: protein glyphs clear the smooth global contour.  Structural
        # waters clear the actual atoms/bonds instead, so they may occupy a
        # genuine concavity that the convex hull necessarily fills in.
        standoff, d_standoff = self._hull_distance(p)
        lig_pen = np.maximum(0.0, self.ligand_clearance - standoff)
        lig_pen[self.water_mask] = 0.0
        e_ligand = 0.5 * float(lig_pen @ lig_pen)
        grad -= w.ligand * lig_pen[:, None] * d_standoff
        if np.any(self.water_mask):
            water = np.flatnonzero(self.water_mask)
            local_e, local_g = self._molecule_push(
                p[water], self.ligand_clearance[water]
            )
            e_ligand += local_e
            grad[water] += w.ligand * local_g

        # E_overlap: pairwise repulsion below 2r + gap.
        diff = p[:, None, :] - p[None, :, :]
        sep = np.hypot(diff[:, :, 0], diff[:, :, 1])
        np.fill_diagonal(sep, np.inf)
        over = np.maximum(0.0, self.pair_floor - sep)
        e_overlap = 0.25 * float(np.sum(over * over))
        grad -= w.overlap * np.einsum("ij,ijk->ik", over / np.maximum(sep, _EPS), diff)

        # E_backbone: spring between sequence neighbours, one glyph diameter
        # apart, which is what chains them into the reference necklaces.
        e_backbone = 0.0
        if len(self.backbone):
            i, j = self.backbone[:, 0], self.backbone[:, 1]
            bv = p[i] - p[j]
            bd = np.maximum(np.hypot(bv[:, 0], bv[:, 1]), _EPS)
            bdr = bd - self.edge_rest
            e_backbone = 0.5 * float(np.sum(self.edge_k * bdr * bdr))
            pull = w.backbone * (self.edge_k * bdr / bd)[:, None] * bv
            np.add.at(grad, i, pull)
            np.add.at(grad, j, -pull)

        # E_spread: repulsion between the glyphs' *bearings* from the ligand
        # centre, so a patch of ligand carrying many interactions fans its
        # residues out along an arc instead of stacking them radially (which
        # is all E_overlap would ask for).  The separation is measured as arc
        # length on a fixed reference circle, in pixels, so this weight is on
        # the same footing as the springs; an angle would have been divided by
        # the radius and come out far too weak to matter.
        rad = np.maximum(np.hypot(p[:, 0], p[:, 1]), _EPS)
        unit = p / rad[:, None]
        udiff = unit[:, None, :] - unit[None, :, :]
        chord = np.hypot(udiff[:, :, 0], udiff[:, :, 1])
        np.fill_diagonal(chord, np.inf)
        fan = np.maximum(0.0, w.spread_arc - self.ref_radius * chord) * self.spread_mask
        e_spread = 0.25 * float(np.sum(fan * fan))
        pull = -self.ref_radius * np.einsum("ij,ijk->ik", fan / np.maximum(chord, _EPS), udiff)
        grad += w.spread * (pull - np.einsum("ij,ij->i", unit, pull)[:, None] * unit) / rad[:, None]

        # E_shell: glyphs at nearby bearings have to agree on how far they
        # stand off the ligand, so the rim reads as one smooth surface instead
        # of a zigzag.  Nothing else in the energy has an opinion about this:
        # E_anchor parks each glyph at its own d0, which varies with the
        # interaction count, so the rim oscillates in and out residue by
        # residue.
        #
        # Measured on the hull standoff, not on the radius from the ligand
        # centre.  A circle is only the right surface for a round ligand: on
        # ANP (6wak), equalising the radius pushed the glyphs off the long
        # axis outwards and dragged the ones off the short axis straight onto
        # the phosphates.  The standoff makes the term follow the drawing.
        #
        # Pseudo-Huber, like E_anchor and for the same reason: below
        # shell_soft the term is quadratic and flattens the surface hard, past
        # it the pull stops growing so a glyph with no room on the front row
        # can drop a row behind and settle there.  That is Maestro's rule --
        # the surface is defined by the residues in front, and the crowded-out
        # one goes behind it rather than everyone shuffling to make space.
        # Normalised so the saturated pull is w.shell per neighbour pair, not
        # w.shell * shell_soft: otherwise the one knob that says "let a glyph
        # drop behind" is also the knob that decides how hard the surface is
        # held flat, and the two cannot be set independently.
        s = w.shell_soft
        dd = standoff[:, None] - standoff[None, :]
        sbend = np.sqrt(1.0 + dd * dd / (s * s))
        e_shell = 0.5 * s * float(np.sum(self.shell_kernel * (sbend - 1.0)))
        grad += w.shell * np.sum(self.shell_kernel * (dd / s) / sbend,
                                 axis=1)[:, None] * d_standoff

        # E_frame: soft elliptical canvas with the references' landscape
        # aspect.  This is the only anisotropic term, and therefore the only
        # reason one of the 36 rotations scores better than another.
        q = np.einsum("ij,ij->i", p / self.frame, p / self.frame) - 1.0
        fpen = np.maximum(0.0, q)
        e_frame = 0.5 * float(np.sum(fpen * fpen))
        grad += w.frame * (2.0 * fpen)[:, None] * p / (self.frame * self.frame)

        # E_span: the same hinge, applied to points *along* the lines the
        # renderer will draw -- interaction routes and backbone/metal-leg
        # connectors -- so an arrow reaches its residue around the ligand
        # instead of over it.  Without this the only thing keeping a route off
        # the drawing is the glyph's own clearance, which says nothing about
        # the 150 px of line behind it.
        # Every sample is an affine function of the glyph positions, so one
        # matmul builds them all and its transpose distributes the gradient
        # back to the endpoints by the interpolation weights.
        e_span = 0.0
        e_route_overlap = 0.0
        if len(self.span_map):
            samples = self.span_base + self.span_map @ p
            local_route = np.any(self.span_owner[:, self.water_mask], axis=1)
            global_route = ~local_route
            g_span = np.zeros_like(samples)
            e_span = 0.0
            if np.any(global_route):
                part_e, part_g = self._hull_push(
                    samples[global_route], w.span_margin
                )
                e_span += part_e
                g_span[global_route] = part_g
            if np.any(local_route):
                part_e, part_g = self._molecule_push(
                    samples[local_route], w.span_margin
                )
                e_span += part_e
                g_span[local_route] = part_g
            grad += w.span * self.span_map.T @ g_span

            # E_route_overlap: line samples clear every glyph that does not
            # own the segment.  The sample is affine in its endpoints, so its
            # gradient distributes exactly through ``span_map``; the foreign
            # glyph receives the equal and opposite push.
            lv = samples[:, None, :] - p[None, :, :]
            ld = np.maximum(np.hypot(lv[:, :, 0], lv[:, :, 1]), _EPS)
            lpen = np.maximum(0.0, self.radii[None, :] + w.line_glyph_gap - ld)
            lpen[self.span_owner] = 0.0
            e_route_overlap = 0.5 * float(np.sum(lpen * lpen))
            pair_grad = -(lpen / ld)[:, :, None] * lv
            grad += w.route_overlap * self.span_map.T @ np.sum(pair_grad, axis=1)
            grad -= w.route_overlap * np.sum(pair_grad, axis=0)

        total = (
            w.anchor * e_anchor
            + w.ligand * e_ligand
            + w.overlap * e_overlap
            + w.backbone * e_backbone
            + w.spread * e_spread
            + w.shell * e_shell
            + w.frame * e_frame
            + w.span * e_span
            + w.route_overlap * e_route_overlap
        )
        if terms:
            out = {
                "anchor": w.anchor * e_anchor,
                "ligand": w.ligand * e_ligand,
                "overlap": w.overlap * e_overlap,
                "backbone": w.backbone * e_backbone,
                "spread": w.spread * e_spread,
                "shell": w.shell * e_shell,
                "frame": w.frame * e_frame,
                "span": w.span * e_span,
                "route_overlap": w.route_overlap * e_route_overlap,
            }
        return total, grad.ravel(), out

    # -- seed --------------------------------------------------------------

    def seed(self) -> np.ndarray:
        """Glyphs on the canvas ellipse, in the angular order of their anchors.

        Keeping the ellipse order equal to the anchor order is what makes most
        layouts crossing-free before the optimiser has done anything at all.

        Off-shell glyphs -- the structural waters, which belong between the
        ligand and the residue they bridge -- start on their target instead.
        Seeded out on the rim with everyone else a water has to squeeze back in
        through the ring of residues, and E_anchor saturates (pseudo-Huber), so
        it never does: the orientation scan hid that by keeping whichever of its
        many tries happened to let the water through, and "Reshuffle", which pins
        the orientation and gets exactly one try, did not have that luxury.
        """
        n = self.n
        if n == 0:
            return np.zeros(0)
        seeded = np.array(self.target, dtype=float)
        rim = np.flatnonzero(self.on_shell)
        if len(rim) == 0:
            return seeded.ravel()
        bearing = np.arctan2(self.anchors[rim, 1], self.anchors[rim, 0])
        order = rim[np.argsort(bearing)]

        ax, ay = self.frame
        # Grow the ellipse until its perimeter can hold the rim glyphs side by side.
        need = len(rim) * (2.0 * self.radius + self.w.glyph_gap)
        have = math.pi * (3.0 * (ax + ay) / 2.0 - math.sqrt(ax * ay))
        if have > 0 and need > have:
            ax *= need / have
            ay *= need / have

        slot = bearing.min() + np.arange(len(rim)) * (2.0 * math.pi / len(rim))
        seeded[order, 0] = ax * np.cos(slot)
        seeded[order, 1] = ay * np.sin(slot)
        return seeded.ravel()


# ---------------------------------------------------------------------------
# Crossings (discrete, deliberately outside the smooth objective)
# ---------------------------------------------------------------------------

def _count_crossings(problem: Problem, p: np.ndarray, radius: float, *, culprits: bool = False):
    """Proper crossings between interaction lines, plus lines grazing a glyph.

    Same semantics as :func:`model.segments_cross` and
    :func:`model.point_segment_distance`, vectorised because the swap search
    calls this a few hundred times per repair round.

    With ``culprits`` the glyphs involved in a crossing come back too, as a set
    of indices into ``p``; the repair loop uses it to skip the moves that cannot
    possibly help.
    """
    if len(problem.line_start_map) == 0:
        return (0, set()) if culprits else 0
    a = problem.line_start_base + problem.line_start_map @ p
    b = problem.line_end_base + problem.line_end_map @ p

    # Segment x segment.  All four orientation determinants must be non-zero
    # and pair up with opposite signs; that is the "shared endpoints do not
    # count" rule, and it also keeps residues sharing an anchor atom honest.
    def orient(p0, p1, q):
        return ((p1[:, None, 0] - p0[:, None, 0]) * (q[None, :, 1] - p0[:, None, 1])
                - (p1[:, None, 1] - p0[:, None, 1]) * (q[None, :, 0] - p0[:, None, 0]))

    oa, ob = orient(a, b, a), orient(a, b, b)   # [i, j]: end of segment j vs segment i
    tol = 1e-9
    hit = (
        (np.abs(oa) > tol) & (np.abs(ob) > tol)
        & (np.abs(oa.T) > tol) & (np.abs(ob.T) > tol)
        & ((oa > 0) != (ob > 0))                # segment j straddles the line of i
        & ((oa.T > 0) != (ob.T > 0))            # and segment i straddles the line of j
    )
    np.fill_diagonal(hit, False)
    crossings = int(hit.sum()) // 2

    # Segment passing within a glyph radius of a glyph it does not belong to.
    seg = b - a
    len2 = np.maximum(np.einsum("ij,ij->i", seg, seg), _EPS)
    rel = p[None, :, :] - a[:, None, :]
    t = np.clip(np.einsum("ijk,ik->ij", rel, seg) / len2[:, None], 0.0, 1.0)
    gap = rel - t[:, :, None] * seg[:, None, :]
    near = np.hypot(gap[:, :, 0], gap[:, :, 1]) < problem.radii[None, :]
    near[problem.line_owner] = False
    total = crossings + int(near.sum())
    if not culprits:
        return total
    rows, cols = np.nonzero(near)
    crossing_lines = np.nonzero(hit.any(axis=1))[0]
    blame = set(np.nonzero(problem.line_owner[crossing_lines].any(axis=0))[0].tolist())
    blame.update(np.nonzero(problem.line_owner[rows].any(axis=0))[0].tolist())
    blame.update(cols.tolist())
    return total, blame


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------

def _anchor_weights(diagram: Diagram, keys: list[str], n_atoms: int) -> np.ndarray:
    """Row-stochastic ``(n_residues, n_atoms)`` map from ligand atoms to anchors."""
    mat = np.zeros((len(keys), n_atoms))
    counts: dict[str, dict[int, int]] = {k: {} for k in keys}
    for inter in diagram.interactions:
        # A bridging water has no contact of its own, but the ligand half of the
        # bridge is its contact: anchoring it on that atom is what puts it right
        # beside the group it touches instead of somewhere on the residue's ray.
        buckets = [counts.get(inter.residue_key), counts.get(inter.via_water)]
        for bucket in buckets:
            if bucket is None:
                continue
            for idx in inter.ligand_atoms:
                if 0 <= idx < n_atoms:
                    bucket[idx] = bucket.get(idx, 0) + 1
    for row, key in enumerate(keys):
        bucket = counts[key]
        if bucket:
            for idx, count in bucket.items():
                mat[row, idx] = count
        else:
            idx = diagram.nearest_atom.get(key)
            if idx is None or not (0 <= idx < n_atoms):
                mat[row, :] = 1.0            # no information: aim at the centroid
            else:
                mat[row, idx] = 1.0
    return mat / mat.sum(axis=1, keepdims=True)


def build_problem(
    diagram: Diagram,
    *,
    glyph_radius: float = 26.0,
    weights: Weights = WEIGHTS,
    rotation: float = 0.0,
    mirror: bool = False,
    _cache: dict | None = None,
) -> Problem:
    """Assemble the smooth problem for one ligand orientation.

    ``_cache`` carries the orientation-independent parts (scaled coordinates,
    hull indices, anchor weight matrix, per-residue constants) across the
    orientation trials.
    """
    if _cache is None:
        _cache = _static(diagram, glyph_radius, weights)

    coords = _orient(_cache["coords"], rotation, mirror)
    normals, offsets = _halfplanes(coords[_cache["hull"]])
    anchors = _cache["amat"] @ coords
    target = _contour_targets(
        coords, anchors, _cache["d0"], _cache["on_shell"],
        _cache["water_mask"], _cache["ligand_bonds"],
        _cache["anchor_atoms"], weights,
    )
    return Problem(
        keys=_cache["keys"],
        ligand=coords,
        ligand_bonds=_cache["ligand_bonds"],
        anchors=anchors,
        target=target,
        normals=normals,
        offsets=offsets,
        stiff=_cache["stiff"],
        pinned=_cache["pinned"],
        on_shell=_cache["on_shell"],
        water_mask=_cache["water_mask"],
        metal_mask=_cache["metal_mask"],
        has_line=_cache["has_line"],
        backbone=_cache["backbone"],
        edge_k=_cache["edge_k"],
        span_base=_cache["span_anchor"] @ anchors,
        span_map=_cache["span_map"],
        span_owner=_cache["span_owner"],
        spread_mask=_cache["spread_mask"],
        shell_kernel=_shell_kernel(target, _cache["on_shell"],
                                   _cache["ref_radius"], weights),
        ref_radius=_cache["ref_radius"],
        frame=_cache["frame"],
        radius=glyph_radius,
        radii=_cache["radii"],
        ligand_clearance=_cache["ligand_clearance"],
        pair_floor=_cache["pair_floor"],
        edge_rest=_cache["edge_rest"],
        line_start_base=_cache["line_start_anchor"] @ anchors,
        line_start_map=_cache["line_start_map"],
        line_end_base=_cache["line_end_anchor"] @ anchors,
        line_end_map=_cache["line_end_map"],
        line_owner=_cache["line_owner"],
        w=weights,
    )


def _shell_kernel(target: np.ndarray, on_shell: np.ndarray,
                  ref_radius: float, w: Weights) -> np.ndarray:
    """How strongly each pair of glyphs has to agree on a standoff, by bearing.

    A Gaussian in arc length on the reference circle: neighbours on the rim
    hold each other flat, glyphs on opposite sides of the ligand ignore each
    other entirely.

    # ponytail: frozen from the seed bearings rather than recomputed from the
    # live positions.  A live kernel would have to carry the bearing
    # derivative into E_shell's gradient for no visible gain -- E_spread
    # already fixes the angular order, so the neighbour sets do not change
    # during a relaxation.  Revisit if a glyph is ever seen crossing sides
    # mid-solve.
    """
    rad = np.maximum(np.hypot(target[:, 0], target[:, 1]), _EPS)
    unit = target / rad[:, None]
    d = unit[:, None, :] - unit[None, :, :]
    arc = ref_radius * np.hypot(d[:, :, 0], d[:, :, 1])
    kernel = np.exp(-((arc / w.shell_span) ** 2))
    # Glyphs that are not on the rim neither define the surface nor answer to
    # it; zeroing both their row and their column is what keeps them out.
    kernel *= on_shell[:, None] * on_shell[None, :]
    np.fill_diagonal(kernel, 0.0)
    return kernel


def _gaussian_kernel(sigma: float) -> np.ndarray:
    """Compact, normalized Gaussian kernel with ``sigma`` in sample units."""
    if sigma <= 0.35:
        return np.ones(1)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _probe_contour_radius(
    coords: np.ndarray, angles: np.ndarray, probe_radius: float, samples: int = 360
) -> np.ndarray:
    """Radial reach of a smooth 2D rolling-probe contour.

    The support function of the atom centres gives the convex molecular
    silhouette.  Rolling a circular probe around that silhouette is an offset
    operation; filtering its support over the probe's own arc length removes
    atom-to-atom corners without tying smoothness to a particular ligand size.
    """
    if len(coords) == 0:
        return np.full(len(angles), probe_radius, dtype=float)
    grid = np.linspace(-math.pi, math.pi, samples, endpoint=False)
    unit = np.stack([np.cos(grid), np.sin(grid)], axis=1)
    support = np.max(coords @ unit.T, axis=0)
    molecular_radius = max(float(np.mean(support)), probe_radius, _EPS)
    step = 2.0 * math.pi / samples
    sigma = (probe_radius / molecular_radius) / step
    kernel = _gaussian_kernel(sigma)
    pad = len(kernel) // 2
    wrapped = np.pad(support, (pad, pad), mode="wrap")
    smooth = np.convolve(wrapped, kernel, mode="valid")
    # Periodic interpolation needs the first sample repeated one turn later.
    query = (np.asarray(angles) + math.pi) % (2.0 * math.pi) - math.pi
    return np.interp(query, np.r_[grid, math.pi], np.r_[smooth, smooth[0]]) + probe_radius


def _contour_targets(
    coords: np.ndarray,
    anchors: np.ndarray,
    d0: np.ndarray,
    on_shell: np.ndarray,
    water_mask: np.ndarray,
    ligand_bonds: np.ndarray,
    anchor_atoms: np.ndarray,
    w: Weights,
) -> np.ndarray:
    """Stage-one ligand contour followed by a stage-two smooth residue shell."""
    if len(anchors) == 0:
        return np.zeros((0, 2), dtype=float)
    angles = np.arctan2(anchors[:, 1], anchors[:, 0])
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    regular_probe = w.ligand_probe_radius
    contour = _probe_contour_radius(coords, angles, regular_probe)
    radius = contour + np.maximum(0.0, d0 - regular_probe)

    # Water follows the local chemistry, not a radial molecular silhouette.
    # For a terminal hydroxyl this is exactly the C->O direction, allowing the
    # sphere to enter a concavity even if another remote group lies at the same
    # bearing from the ligand centroid.
    if np.any(water_mask):
        for i in np.flatnonzero(water_mask):
            atom = int(anchor_atoms[i])
            neighbor_ids = ligand_bonds[
                np.any(ligand_bonds == atom, axis=1)
            ].ravel()
            neighbor_ids = neighbor_ids[neighbor_ids != atom]
            if len(neighbor_ids):
                direction = coords[atom] - coords[np.unique(neighbor_ids)].mean(axis=0)
            else:
                direction = anchors[i]
            length = max(float(np.hypot(direction[0], direction[1])), _EPS)
            local = direction / length
            # Half-probe is retained as a small visual buffer around the atom;
            # d0 controls the centre distance of the hydration sphere.
            target = anchors[i] + local * max(
                d0[i], regular_probe * w.water_probe_scale
            )
            radius[i] = np.hypot(target[0], target[1])
            unit[i] = target / max(radius[i], _EPS)

    # Position protein residues on one smooth line before any interaction or
    # overlap optimization.  A Gaussian over circular bearing is the 2D
    # equivalent of rolling the larger shell probe around the provisional rim.
    shell = np.flatnonzero(on_shell)
    if len(shell) > 1:
        a = angles[shell]
        delta = (a[:, None] - a[None, :] + math.pi) % (2.0 * math.pi) - math.pi
        mean_radius = max(float(np.mean(radius[shell])), _EPS)
        sigma = w.shell_probe_radius / mean_radius
        blend = np.exp(-0.5 * (delta / max(sigma, 1e-3)) ** 2)
        blend[np.abs(delta) > 3.0 * sigma] = 0.0
        radius[shell] = (blend @ radius[shell]) / np.maximum(blend.sum(axis=1), _EPS)
    return unit * radius[:, None]


def _static(diagram: Diagram, glyph_radius: float, w: Weights, coords_2d=None) -> dict:
    """Orientation-independent precomputation, shared by all scan trials."""
    coords = np.asarray(
        diagram.coords_2d if coords_2d is None else coords_2d, dtype=float
    ).reshape(-1, 2)
    coords = (coords - coords.mean(axis=0)) * (w.bond_px / _bond_length(coords))
    if diagram.mol is not None:
        ligand_bonds = np.array([
            [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            for bond in diagram.mol.GetBonds()
        ], dtype=int).reshape(-1, 2)
    else:
        # Pure-geometry callers/tests need no RDKit molecule.  Recover the
        # obvious bond graph from distances after scaling.
        delta = coords[:, None, :] - coords[None, :, :]
        distance = np.hypot(delta[:, :, 0], delta[:, :, 1])
        ligand_bonds = np.argwhere(
            np.triu((distance > _EPS) & (distance <= 1.25 * w.bond_px), 1)
        ).astype(int).reshape(-1, 2)

    keys = [r.key for r in diagram.residues]
    index = {k: i for i, k in enumerate(keys)}
    n = len(keys)

    n_inter = np.zeros(n)
    n_drawn = np.zeros(n)
    for inter in diagram.interactions:
        if inter.residue_key in index:
            i = index[inter.residue_key]
            n_inter[i] += 1.0
            n_drawn[i] += float(inter.kind != "hydrophobic")
        # The ligand half of a water bridge is a drawn line too, so the water
        # counts as an interacting glyph: it earns the stiff anchor and the
        # E_span sampling that keep it on its spot rather than letting it drift
        # out on the weakest spring in the diagram.
        if inter.via_water and inter.via_water in index:
            i = index[inter.via_water]
            n_inter[i] += 1.0
            n_drawn[i] += 1.0

    # d0 falls from d0_far to d0_near as interactions accumulate: a residue
    # with many contacts hugs the ligand, a context residue sits out.
    close = n_inter / (n_inter + w.anchor_decay)
    d0 = w.d0_far - (w.d0_far - w.d0_near) * close

    water_mask = np.array([
        residue.ref.residue_class == "water" for residue in diagram.residues
    ], dtype=bool)
    metal_mask = np.array([
        residue.ref.residue_class == "metal" for residue in diagram.residues
    ], dtype=bool)
    radii = np.full(n, glyph_radius, dtype=float)
    radii[water_mask] *= w.water_radius_fraction
    gap = np.full((n, n), w.glyph_gap, dtype=float)
    gap[water_mask[:, None] | water_mask[None, :]] = w.water_gap
    pair_floor = radii[:, None] + radii[None, :] + gap
    ligand_clearance = radii + w.ligand_margin
    ligand_clearance[water_mask] = radii[water_mask] + w.water_ligand_margin

    # A residue that only appears because it completes a metal's coordination
    # sphere gets the *metal's* anchor, one glyph pitch further out.  Left on
    # its own (usually very weak) ligand anchor it lands somewhere else on the
    # rim entirely and its coordination line then rakes across the diagram; no
    # spring stiff enough to drag it back survives contact with the rest of the
    # energy.
    amat = _anchor_weights(diagram, keys, len(coords))
    anchor_atoms = np.argmax(amat, axis=1) if n else np.zeros(0, dtype=int)
    pitch = 2.0 * glyph_radius + w.glyph_gap
    sphere = []  # legs whose partner is drawn *only* to complete the sphere
    for leg in diagram.metal_legs:
        if leg.metal_key not in index or leg.partner_key not in index:
            continue
        i, j = index[leg.partner_key], index[leg.metal_key]
        if n_inter[i] == 0.0:
            amat[i] = amat[j]
            d0[i] = d0[j] + pitch
            sphere.append(leg)

    # A water that bridges the ligand to a residue is not a member of the rim.
    # It belongs *between* the two things it bridges, tucked inside the surface
    # the other glyphs define -- drawn out on the rim next to them, the two
    # halves of the bridge stop reading as one contact.  Its anchor is already
    # the ligand atom it touches (see :func:`_anchor_weights`), so all it needs
    # here is to sit one glyph off that atom instead of at rim distance, and to
    # leave the shell term, which would drag it back out to the common standoff.
    bridges = list(dict.fromkeys(
        (i.via_water, i.residue_key) for i in diagram.interactions
        if i.via_water and i.via_water in index and i.residue_key in index
    ))
    on_shell = np.ones(n, dtype=bool)
    # Metals are compact coordination waypoints, like structural waters, not
    # amino-acid droplets defining the protein surface.  They remain outside
    # the ligand contour but must not pull the smooth residue rim outwards.
    on_shell[metal_mask] = False
    stiff = 1.0 + w.anchor_stiffen * n_inter
    for water, residue in bridges:
        d0[index[water]] = w.d0_water
        on_shell[index[water]] = False
        # The protein partner still belongs to the same smooth pocket surface
        # as every other residue.  Only the water occupies the inner shell.

    # Not every contact carries the same weight.  The ones that need three
    # glyphs to read as one interaction -- a metal and its coordination sphere,
    # a water and the residue it bridges -- are the diagram's conductors: they
    # are placed first and the rest of the pocket settles around them, which is
    # what a stiffer anchor buys.  Left on the same footing as a lone
    # hydrophobic contact they lose the tug-of-war and the interaction reads as
    # three unrelated glyphs.
    conductors = {k for leg in diagram.metal_legs for k in (leg.metal_key, leg.partner_key)}
    conductors |= set(diagram.metal_coordination)
    conductors |= {k for pair in bridges for k in pair}
    lead = np.array([index[k] for k in conductors if k in index], dtype=int)
    pinned = np.zeros(n, dtype=bool)
    if len(lead):
        stiff[lead] *= w.conductor
        pinned[lead] = True
    for water, _ in bridges:
        stiff[index[water]] *= w.water_anchor

    # The legs themselves join the sequence connectors on the same spring.
    edges = [(a, b, 1.0, "backbone") for a, b in diagram.backbone_edges()]
    # Only the sphere-completing legs get a spring.  A coordinating residue that
    # has contacts of its own is anchored by those, and since the leg itself is
    # no longer drawn, dragging the metal towards it buys nothing and costs the
    # coordination lines, which then rake across the ligand to reach it.
    edges += [
        (leg.metal_key, leg.partner_key, w.metal_leg, "metal") for leg in sphere
    ]
    edges += [(water, res, w.bridge_spring, "bridge") for water, res in bridges]
    edges = [e for e in edges if e[0] in index and e[1] in index]
    backbone = np.array(
        [[index[a], index[b]] for a, b, _, _ in edges], dtype=int
    ).reshape(-1, 2)
    edge_k = np.array([k for _, _, k, _ in edges], dtype=float)
    edge_rest = np.array([
        max(pair_floor[index[a], index[b]], w.bridge_rest)
        if kind == "bridge" else pair_floor[index[a], index[b]]
        for a, b, _, kind in edges
    ], dtype=float)

    # Sequence neighbours and coordination partners are meant to touch, so they
    # are exempt from the angular spread term; everything else fans out.
    spread_mask = 1.0 - np.eye(n)
    for i, j in backbone:
        spread_mask[i, j] = spread_mask[j, i] = 0.0

    # Describe the lines exactly as the renderer does.  In particular, a water
    # bridge is ligand -> water -> residue, never ligand -> residue.  Keeping
    # these affine endpoint maps lets both E_span and the crossing repair use
    # the same geometry while ligand rotation changes only ``anchors``.
    line_start_anchor: list[np.ndarray] = []
    line_start_map: list[np.ndarray] = []
    line_end_anchor: list[np.ndarray] = []
    line_end_map: list[np.ndarray] = []
    line_owner: list[np.ndarray] = []
    line_samples: list[np.ndarray] = []

    def add_line(*, start_anchor=None, start_pos=None, end_anchor=None,
                 end_pos=None, owners=(), samples=_EDGE_SAMPLES):
        sa, sm, ea, em = (np.zeros(n) for _ in range(4))
        if start_anchor is not None:
            sa[start_anchor] = 1.0
        if start_pos is not None:
            sm[start_pos] = 1.0
        if end_anchor is not None:
            ea[end_anchor] = 1.0
        if end_pos is not None:
            em[end_pos] = 1.0
        own = np.zeros(n, dtype=bool)
        own[list(owners)] = True
        line_start_anchor.append(sa)
        line_start_map.append(sm)
        line_end_anchor.append(ea)
        line_end_map.append(em)
        line_owner.append(own)
        line_samples.append(samples)

    seen_routes: set[tuple] = set()
    for inter in diagram.interactions:
        if inter.kind == "hydrophobic" or inter.residue_key not in index:
            continue
        residue_i = index[inter.residue_key]
        if inter.via_water and inter.via_water in index:
            water_i = index[inter.via_water]
            first = ("bridge_ligand", water_i)
            second = ("bridge_protein", water_i, residue_i)
            if first not in seen_routes:
                add_line(start_anchor=water_i, end_pos=water_i,
                         owners=(water_i,), samples=_ROUTE_SAMPLES)
                seen_routes.add(first)
            if second not in seen_routes:
                add_line(start_pos=water_i, end_pos=residue_i,
                         owners=(water_i, residue_i))
                seen_routes.add(second)
        else:
            direct = ("direct", residue_i)
            if direct not in seen_routes:
                add_line(start_anchor=residue_i, end_pos=residue_i,
                         owners=(residue_i,), samples=_ROUTE_SAMPLES)
                seen_routes.add(direct)

    def rows(values, dtype=float):
        return np.asarray(values, dtype=dtype).reshape(-1, n)

    lsa, lsm = rows(line_start_anchor), rows(line_start_map)
    lea, lem = rows(line_end_anchor), rows(line_end_map)
    owners = rows(line_owner, dtype=bool)
    span_anchor_rows, span_map_rows, span_owner_rows = [], [], []
    for sa, sm, ea, em, own, samples in zip(
        lsa, lsm, lea, lem, owners, line_samples
    ):
        for t in samples:
            span_anchor_rows.append((1.0 - t) * sa + t * ea)
            span_map_rows.append((1.0 - t) * sm + t * em)
            span_owner_rows.append(own)
    for i, j in backbone:
        for t in _EDGE_SAMPLES:
            row = np.zeros(n)
            row[int(i)], row[int(j)] = 1.0 - t, t
            own = np.zeros(n, dtype=bool)
            own[[int(i), int(j)]] = True
            span_anchor_rows.append(np.zeros(n))
            span_map_rows.append(row)
            span_owner_rows.append(own)
    span_anchor = rows(span_anchor_rows)
    span_map = rows(span_map_rows)
    span_owner = rows(span_owner_rows, dtype=bool)

    lig_r = float(np.max(np.hypot(coords[:, 0], coords[:, 1]))) if len(coords) else 0.0
    span = lig_r + w.canvas_slack * w.d0_far
    frame = np.array([span * math.sqrt(w.canvas_aspect), span / math.sqrt(w.canvas_aspect)])

    return {
        "coords": coords,
        "hull": _convex_hull(coords),
        "keys": keys,
        "amat": amat,
        "anchor_atoms": anchor_atoms,
        "ligand_bonds": ligand_bonds,
        "d0": d0,
        "stiff": stiff,
        "pinned": pinned,
        "has_line": n_drawn > 0,
        "backbone": backbone,
        "edge_k": edge_k,
        "edge_rest": edge_rest,
        "span_anchor": span_anchor,
        "span_map": span_map,
        "span_owner": span_owner,
        "line_start_anchor": lsa,
        "line_start_map": lsm,
        "line_end_anchor": lea,
        "line_end_map": lem,
        "line_owner": owners,
        "spread_mask": spread_mask,
        "on_shell": on_shell,
        "water_mask": water_mask,
        "metal_mask": metal_mask,
        "radii": radii,
        "ligand_clearance": ligand_clearance,
        "pair_floor": pair_floor,
        "frame": frame,
        "ref_radius": float(frame.mean()),
    }


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def _pick_projection(
    diagram: Diagram, glyph_radius: float, weights: Weights
) -> tuple[int, dict]:
    """Choose which of the ligand's candidate depictions to lay out around.

    A folded depiction does not look wrong on its own -- it looks wrong once
    the residues have to fit around it, which is a question only the layout
    energy can answer.  ``chem._depictions`` prunes the illegible ones and
    hands over the rest; each gets a coarse scan here and the cheapest wins.

    # The scan is deliberately short, but it includes both mirrors: the angular
    # seed and the frame make mirrored depictions land in different basins.
    """
    # The scan is deterministic and costs ~0.6 s of the ~4 s solve, so the
    # widget's "Recalculate" would spend a seventh of its time re-deriving an
    # answer it already has.  Memoised on the diagram, keyed on the two inputs
    # that can change it, so it dies with the diagram and never goes stale.
    memo_key = (glyph_radius, astuple(weights))
    memo = getattr(diagram, "_projection_memo", None)
    if memo is not None and memo[0] == memo_key:
        return memo[1]

    caches = [_static(diagram, glyph_radius, weights, view)
              for view in [diagram.coords_2d, *diagram.coords_alt]]
    if len(caches) == 1:
        diagram._projection_memo = (memo_key, (0, caches[0]))
        return 0, caches[0]

    scores = []
    for cache in caches:
        best = math.inf
        for flip in (False, True):
            for step in range(_PROJECTION_ROTATIONS):
                trial = build_problem(
                    diagram, glyph_radius=glyph_radius, weights=weights,
                    rotation=2.0 * math.pi * step / _PROJECTION_ROTATIONS,
                    mirror=flip, _cache=cache,
                )
                energy = _relax(trial, trial.seed(), 30)[1]
                best = min(best, _candidate_score(trial, energy))
        scores.append(best)
    pick = int(np.argmin(scores))
    print("[ms_contactmap.layout] projection " + " ".join(
        f"{'*' if i == pick else ' '}{s:.0f}" for i, s in enumerate(scores)),
        file=sys.stderr)
    diagram._projection_memo = (memo_key, (pick, caches[pick]))
    return pick, caches[pick]


def _relax(problem: Problem, x0: np.ndarray, maxiter: int, bounds=None) -> tuple[np.ndarray, float]:
    from scipy.optimize import minimize

    if problem.n == 0:
        return x0, 0.0
    res = minimize(
        problem.objective, x0, jac=True, method="L-BFGS-B", bounds=bounds,
        # Ten correction vectors are ample for the 20-60 variables in these
        # diagrams.  Twenty doubled the expensive L-BFGS matrix work without a
        # measurable improvement in final energy or crossing repair.
        options={"maxiter": maxiter, "maxcor": 10, "ftol": 1e-9, "gtol": 1e-6},
    )
    return res.x, float(res.fun)


def _candidate_score(problem: Problem, energy: float) -> float:
    """Rank a solved basin while keeping the reported physical layout energy pure."""
    coords = problem.ligand
    if len(coords) < 2:
        return energy
    covariance = np.cov(coords.T, bias=True)
    values, vectors = np.linalg.eigh(covariance)
    total = max(float(values.sum()), _EPS)
    anisotropy = max(0.0, float(values[-1] - values[0]) / total)
    major = vectors[:, -1]
    vertical = float(major[1] * major[1])
    return energy + problem.w.horizontal_preference * anisotropy * vertical


def solve_layout_legacy(
    diagram: Diagram,
    *,
    glyph_radius: float = 26.0,
    weights: Weights = WEIGHTS,
    rotations: int = 36,
    seed_positions: dict[str, tuple[float, float]] | None = None,
    pinned: set[str] | None = None,
    max_swaps: int = 30,
    orientation: tuple[float, bool] | None = None,
    projection: int | None = None,
) -> LayoutResult:
    """Place every residue glyph and pick the ligand orientation.

    ``seed_positions`` starts the optimiser from the given scene coordinates
    instead of the angular seed -- used by the tests and by the widget when the
    user has dragged a glyph and wants the rest re-settled around it.  It is a
    hint, not a constraint: if it leads somewhere tangled the orientation scan
    still gets its turn.

    ``orientation`` pins the ligand to one ``(rotation, mirror)`` and skips the
    scan, so the residues re-solve around a rotation the caller chose -- what
    the widget's "rotate ligand" and "reshuffle" actions do.

    ``pinned`` names residues whose ``seed_positions`` are a constraint rather
    than a hint: the solver settles everything else around them and leaves them
    exactly where the caller put them.  That is what the widget's "rotate this
    metal, re-accommodate the neighbours" action needs, because nothing in the
    energy model cares which way a coordination polygon faces, so an ordinary
    re-solve would simply undo the rotation.

    ``projection`` picks one of the ligand's candidate depictions by index
    instead of letting the coarse scan choose -- pass ``result.projection``
    back in so a re-solve keeps drawing the same ligand.
    """
    if projection is not None:
        views = [diagram.coords_2d, *diagram.coords_alt]
        cache = _static(diagram, glyph_radius, weights, views[projection])
    else:
        projection, cache = _pick_projection(diagram, glyph_radius, weights)
    keys = cache["keys"]

    if not keys:
        return LayoutResult(
            {}, [tuple(p) for p in cache["coords"]], 0.0, False, 0.0, {}, 0, projection
        )

    attempts: list[tuple[float, bool, Problem, np.ndarray]] = []

    # ``orientation`` and ``pinned`` both fix things the scan would undo, so
    # they short-circuit it.  Given together the caller's coordinates are the
    # start and the caller's rotation is the ligand.
    theta, flip = orientation if orientation is not None else (0.0, False)
    if orientation is not None or pinned:
        fixed = build_problem(
            diagram, glyph_radius=glyph_radius, weights=weights,
            rotation=theta, mirror=flip, _cache=cache,
        )
        seed = fixed.seed().reshape(-1, 2)
        for i, key in enumerate(keys):
            if seed_positions and key in seed_positions:
                seed[i] = seed_positions[key]
        # L-BFGS-B already takes box bounds, so a pin is just a degenerate box.
        bounds = None if not pinned else [
            (v, v) if key in pinned else (None, None)
            for i, key in enumerate(keys)
            for v in seed[i]
        ]
        return _finish(
            [(theta, flip, fixed, seed.ravel())], keys, glyph_radius, max_swaps,
            projection, bounds,
        )

    if seed_positions is not None:
        upright = build_problem(diagram, glyph_radius=glyph_radius, weights=weights, _cache=cache)
        seed = upright.seed().reshape(-1, 2)
        for i, key in enumerate(keys):
            if key in seed_positions:
                seed[i] = seed_positions[key]
        attempts.append((0.0, False, upright, seed.ravel()))

    # Score every orientation with a short relaxation, then relax the finalists
    # properly from where the coarse pass left them.  The reflected half of the
    # scan is not redundant: the seed orders the glyphs by bearing, so a mirror
    # reverses that order and the relaxation lands in a different basin --
    # measured at up to 3x the energy of the same angle unmirrored.
    coarse: list[tuple[float, float, bool, Problem, np.ndarray]] = []
    for flip in (False, True):
        for step in range(rotations):
            theta = 2.0 * math.pi * step / rotations
            trial = build_problem(
                diagram, glyph_radius=glyph_radius, weights=weights,
                rotation=theta, mirror=flip, _cache=cache,
            )
            xc, energy = _relax(trial, trial.seed(), _SCAN_COARSE)
            coarse.append((_candidate_score(trial, energy), theta, flip, trial, xc))
    coarse.sort(key=lambda row: row[0])
    scan: list[tuple[float, float, bool, Problem, np.ndarray]] = []
    for _, theta, flip, trial, xc in coarse[:_SCAN_FINALISTS]:
        xt, energy = _relax(trial, xc, 45)
        scan.append((_candidate_score(trial, energy), theta, flip, trial, xt))
    # Take the best orientations, but keep them apart: energy varies smoothly
    # with the angle, so the top few by score alone are the same layout nudged
    # a few degrees and all fall into the same tangle.
    scan.sort(key=lambda row: row[0])
    chosen: list[tuple[float, bool]] = []
    for _, theta, flip, trial, xt in scan:
        if len(chosen) >= _RESTARTS:
            break
        gap = min(
            (abs((theta - t + math.pi) % (2.0 * math.pi) - math.pi) for t, f in chosen if f == flip),
            default=math.inf,
        )
        if gap >= _RESTART_SPACING:
            chosen.append((theta, flip))
            attempts.append((theta, flip, trial, xt))

    return _finish(attempts, keys, glyph_radius, max_swaps, projection)


def _finish(attempts, keys, glyph_radius, max_swaps, projection, bounds=None) -> LayoutResult:
    # Take the cheapest orientation first and stop as soon as one comes out
    # clean; a tangle is usually a property of the starting arrangement, not
    # of the diagram, so the next orientation down the list often just works.
    best: tuple[int, float, float, bool, Problem, np.ndarray] | None = None
    for rotation, mirror, problem, start in attempts:
        x, _ = _relax(problem, start, 400, bounds)
        if bounds is not None:
            # Crossing repair swaps whole glyphs between slots, which would
            # move the very residues the caller pinned.  Report the crossings
            # and leave the arrangement alone.
            crossings = _count_crossings(problem, x.reshape(-1, 2), glyph_radius)
        else:
            x, crossings = _repair_crossings(problem, x, glyph_radius, max_swaps)
        energy = problem.objective(x)[0]
        if best is None or (crossings, energy) < (best[0], best[1]):
            best = (crossings, energy, rotation, mirror, problem, x)
        if crossings == 0:
            break

    crossings, energy, rotation, mirror, problem, x = best
    p = x.reshape(-1, 2)
    return LayoutResult(
        positions={k: (float(p[i, 0]), float(p[i, 1])) for i, k in enumerate(keys)},
        ligand_coords=[(float(a), float(b)) for a, b in problem.ligand],
        rotation=rotation,
        mirror=mirror,
        energy=float(energy),
        energy_terms=problem.terms(x),
        crossings=crossings,
        projection=projection,
    )


def _repair_crossings(
    problem: Problem, x: np.ndarray, radius: float, max_swaps: int
) -> tuple[np.ndarray, int]:
    """Greedily reassign glyphs to their own positions until crossings go away.

    The move that helps most is applied, then the continuous phase runs again
    from there -- a move that looks good on frozen straight lines still has to
    survive the springs.  When it does not, it goes on a tabu list and the
    next-best candidate is tried instead of giving up.

    "Helps most" is not the crossing count alone.  Counting only crossings, the
    pass will pay any price for one: on 4uwh it bought a crossing by sending
    ILE685 to the far rim, which drew its hydrogen bond straight across the
    ligand -- a worse drawing by every measure a reader applies.  So a move is
    scored on crossings *and* on how far it leaves each glyph from its anchor
    target, converted into the same units by ``Weights.crossing_cost``.

    # ponytail: greedy 2-opt only moves downhill, so a tangle that needs two
    # simultaneous moves to improve is a local minimum this cannot leave, and
    # the tabu list only papers over the single-move case.  It starts to bite
    # above ~25 residues.  The upgrade is simulated annealing over the same
    # neighbourhood.
    """
    def score(q: np.ndarray, n_cross: int) -> float:
        reach = np.hypot(*(q - problem.target).T)
        return n_cross * problem.w.crossing_cost + float(problem.stiff @ reach)

    p = x.reshape(-1, 2).copy()
    crossings, blame = _count_crossings(problem, p, radius, culprits=True)
    cost = score(p, crossings)
    best_p, best_crossings, best_cost = p, crossings, cost
    tabu: set[tuple[int, int, int]] = set()
    for _ in range(max_swaps):
        if best_crossings == 0:
            break
        pick, pick_q, pick_cost = None, None, cost
        for move, q in _moves(p, blame, problem):
            if move in tabu:
                continue
            got = score(q, _count_crossings(problem, q, radius))
            if got < pick_cost:
                pick, pick_q, pick_cost = move, q, got
        if pick is None:
            break
        q, _ = _relax(problem, pick_q.ravel(), 250)
        q = q.reshape(-1, 2)
        got, got_blame = _count_crossings(problem, q, radius, culprits=True)
        got_cost = score(q, got)
        if got_cost < cost:
            p, crossings, blame, cost, tabu = q, got, got_blame, got_cost, set()
        else:
            tabu.add(pick)
        if got_cost < best_cost:
            best_p, best_crossings, best_cost = q, got, got_cost
    return best_p.ravel(), best_crossings


def _moves(p: np.ndarray, blame: set[int], problem: Problem):
    """Candidate reassignments of the glyphs to the slots they already occupy.

    Two neighbourhoods: transpositions of any pair, and 2-opt reversals of a
    contiguous run of the bearing-sorted order.  The reversals are what makes
    this work -- undoing a tangle by transpositions alone needs a long run of
    individually useless moves, whereas one reversal uncrosses a whole arc, in
    the same way 2-opt uncrosses a travelling-salesman tour.

    ``blame`` holds the glyphs currently involved in a crossing.  A move that
    leaves all of them where they are cannot lower the count, so it is not
    offered: on 24 residues that is ~530 candidates down to a few dozen, and
    each one costs a full crossing count.

    Ordinary residues are free to be shuffled -- :func:`_repair_crossings`
    prices what that costs them.  Conductors (``problem.pinned`` -- metals,
    their sphere, structural waters and the residues they bridge) are not up
    for negotiation at all: a move that leaves one further from its anchor
    target than it already is is never offered.  They lead, everyone else
    accommodates.
    """
    n = len(p)
    pinned, target = problem.pinned, problem.target
    home = np.hypot(*(p[pinned] - target[pinned]).T)

    def keeps(q: np.ndarray) -> bool:
        return bool(np.all(np.hypot(*(q[pinned] - target[pinned]).T) <= home + _EPS))

    for i in range(n):
        for j in range(i + 1, n):
            if i not in blame and j not in blame:
                continue
            q = p.copy()
            q[[i, j]] = q[[j, i]]
            if not keeps(q):
                continue
            yield (0, i, j), q
    order = np.argsort(np.arctan2(p[:, 1], p[:, 0]))
    # Prefix count of blamed glyphs along the sorted order, so testing whether a
    # run contains one is a subtraction rather than a scan.
    seen = np.cumsum([0] + [int(i in blame) for i in order])
    for a in range(n - 1):
        for b in range(a + 2, n):
            if seen[b + 1] == seen[a]:
                continue
            block = order[a:b + 1]
            q = p.copy()
            q[block] = p[block[::-1]]
            if not keeps(q):
                continue
            yield (1, a, b), q


def solve_layout(
    diagram: Diagram,
    *,
    glyph_radius: float = 26.0,
    weights: Weights = WEIGHTS,
    rotations: int = 16,
    seed_positions: dict[str, tuple[float, float]] | None = None,
    pinned: set[str] | None = None,
    max_swaps: int = 12,
    orientation: tuple[float, bool] | None = None,
    projection: int | None = None,
    variant: int = 0,
) -> LayoutResult:
    """Fast deterministic layout used by the application.

    The former L-BFGS implementation remains available as
    :func:`solve_layout_legacy` for numerical comparisons.  The interactive
    path now evaluates discrete ligand views and settles local constraints in
    bounded vector operations; it does not need a worker process.
    """
    from .fast_layout import solve_layout_fast

    return solve_layout_fast(
        diagram,
        glyph_radius=glyph_radius,
        weights=weights,
        rotations=rotations,
        seed_positions=seed_positions,
        pinned=pinned,
        max_swaps=max_swaps,
        orientation=orientation,
        projection=projection,
        variant=variant,
    )
