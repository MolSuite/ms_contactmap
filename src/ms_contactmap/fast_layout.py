"""Bounded, discrete layout for interactive 2D interaction diagrams.

The ligand depictions are candidates, not promises about the 3D pose.  This
module ranks them by whether their *important interaction graph* can be read,
places strong/multi-anchor residues first, and resolves local constraints with
a small fixed number of position-based iterations.  There is no global
minimizer, no unbounded convergence loop and no need for multiprocessing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import RING_ANCHORED, Diagram


@dataclass
class _Placed:
    index: int
    position: np.ndarray


def _interaction_atoms(diagram: Diagram, key: str, *, strong_only: bool = False) -> list[int]:
    atoms: list[int] = []
    for inter in diagram.interactions:
        owns = inter.residue_key == key or inter.via_water == key
        if not owns or (strong_only and inter.kind == "hydrophobic"):
            continue
        atoms.extend(inter.ligand_atoms)
    return list(dict.fromkeys(atoms))


def _priority(diagram: Diagram, key: str) -> tuple[int, int, int, str]:
    residue = diagram.residue(key)
    interactions = [
        inter for inter in diagram.interactions
        if inter.residue_key == key or inter.via_water == key
    ]
    strong = [inter for inter in interactions if inter.kind != "hydrophobic"]
    distinct = len({atom for inter in strong for atom in inter.ligand_atoms})
    if residue.ref.residue_class == "metal":
        conductor_rank = 0
    elif residue.ref.residue_class == "water":
        conductor_rank = 1
    elif any(key in (leg.metal_key, leg.partner_key) for leg in diagram.metal_legs):
        conductor_rank = 2
    elif any(key in (inter.via_water, inter.residue_key)
             for inter in diagram.interactions if inter.via_water):
        conductor_rank = 3
    else:
        conductor_rank = 4
    best = min((inter.style.priority for inter in strong), default=99)
    # Conductors first, then the residues that have to reconcile several
    # anchors.  Context residues deliberately come last and may use row two.
    return (conductor_rank, -distinct, best, key)


def _placed_water(problem, inter, placed_mask) -> int | None:
    """Index of the placed water ``inter`` reaches ``key`` through, if any."""
    try:
        w = problem.keys.index(inter.via_water)
    except ValueError:
        return None
    return w if placed_mask[w] else None


def _angle_delta(a: np.ndarray | float, b: float) -> np.ndarray:
    return (np.asarray(a) - b + math.pi) % (2.0 * math.pi) - math.pi


#: Slot-score price of one strong line drawn across one ligand bond, in the
#: pixels of glyph-to-anchor distance it trades against.  Several glyph
#: pitches: a residue may move round the ligand to keep its line clear, but
#: not to the far side of the canvas.
BOND_CROSS_COST = 700.0


def _place_discrete(problem, cache: dict, diagram: Diagram, glyph_radius: float,
                    variant: int, fixed: dict[str, tuple[float, float]] | None,
                    pinned: set[str], lines: _Lines
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Place residues on sampled contour slots, important interactions first."""
    n = len(problem.keys)
    if n == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))

    # A 5 degree grid is much finer than a residue glyph at the normal shell
    # radius but still tiny: 72 candidates x ~25 residues.
    count = 72
    phase = (variant * 0.6180339887498948) % 1.0
    angles = np.linspace(-math.pi, math.pi, count, endpoint=False) + phase * 2 * math.pi / count
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    from .layout import _probe_contour_radius

    contour = _probe_contour_radius(
        problem.ligand, angles, problem.w.ligand_probe_radius, samples=180
    )
    base_radius = contour[:, None] + np.maximum(
        0.0, cache["d0"][None, :] - problem.w.ligand_probe_radius
    )
    pitch = 2.0 * glyph_radius + problem.w.glyph_gap

    p = np.zeros((n, 2), dtype=float)
    target = np.zeros((n, 2), dtype=float)
    placed: list[_Placed] = []
    placed_mask = np.zeros(n, dtype=bool)
    ligand_gap = _Clearance(problem)
    room_limit = float(np.max(problem.ligand_clearance, initial=0.0))
    concave_clearance = _concave_clearance(problem)
    key_index = {key: i for i, key in enumerate(problem.keys)}
    backbone_neighbours: dict[int, list[tuple[int, float]]] = {
        i: [] for i in range(n)
    }
    for edge, rest in zip(problem.backbone, problem.edge_rest):
        left, right = map(int, edge)
        backbone_neighbours[left].append((right, float(rest)))
        backbone_neighbours[right].append((left, float(rest)))

    # User-pinned positions are constraints and occupy space before anything
    # else is assigned a slot.
    for key in sorted(pinned):
        if fixed is None or key not in fixed or key not in key_index:
            continue
        i = key_index[key]
        p[i] = target[i] = fixed[key]
        placed.append(_Placed(i, p[i].copy()))
        placed_mask[i] = True

    order = sorted(
        (i for i, key in enumerate(problem.keys) if key not in pinned),
        key=lambda i: _priority(diagram, problem.keys[i]),
    )
    for i in order:
        key = problem.keys[i]
        atom_ids = _interaction_atoms(diagram, key, strong_only=True)
        if not atom_ids:
            atom_ids = _interaction_atoms(diagram, key)
        atom_ids = [a for a in atom_ids if 0 <= a < len(problem.ligand)]
        anchors = problem.ligand[atom_ids] if atom_ids else problem.anchors[i:i + 1]
        # A water-bridged line starts at the water, already placed: aiming at
        # the ligand atom instead left the two legs as a fork off the water.
        own = [inter for inter in diagram.interactions
               if inter.residue_key == key and inter.kind != "hydrophobic"]
        waters = [_placed_water(problem, inter, placed_mask) for inter in own if inter.via_water]
        waters = [w for w in waters if w is not None]
        if waters:
            direct = [a for inter in own if not inter.via_water
                      for a in inter.ligand_atoms if 0 <= a < len(problem.ligand)]
            anchors = np.concatenate([problem.ligand[direct].reshape(-1, 2), p[waters]])

        residue = diagram.residue(key)
        is_water = residue.ref.residue_class == "water"
        is_metal = residue.ref.residue_class == "metal"
        is_context = not residue.has_interactions and not atom_ids
        ideal = math.atan2(problem.target[i, 1], problem.target[i, 0])

        # Water follows its local atom/bond geometry and may enter a concavity;
        # ordinary residues sample the global contour.  Context residues can
        # use a second row without pulling the surface out with them.
        if is_water or is_metal:
            anchor_center = anchors.mean(axis=0)
            local = problem.target[i] - anchor_center
            local_bearing = math.atan2(local[1], local[0]) \
                if float(np.hypot(local[0], local[1])) > 1e-8 else ideal
            # A metal with several ligand legs belongs immediately outside
            # their local group.  Its wider fan lets the minimax term choose a
            # clean side, while remaining far smaller than the full contour
            # search that used to send Mg hundreds of pixels away.
            fan = 0.58 if is_metal else 0.42
            local_angles = local_bearing + np.linspace(-fan, fan, 21 if is_metal else 17)
            local_unit = np.stack([np.cos(local_angles), np.sin(local_angles)], axis=1)
            if is_metal:
                # ``d0`` is measured from the coordinating group, not from the
                # global ligand silhouette.  The final hard hull clearance can
                # still move this outward when the group lies in a concavity.
                local_distance = max(82.0, min(float(cache["d0"][i]), 116.0))
            else:
                local_distance = max(
                    float(np.linalg.norm(problem.target[i] - anchor_center)),
                    float(cache["d0"][i]),
                )
            candidates = anchor_center + local_unit * local_distance
            layer = np.zeros(len(candidates), dtype=int)
            candidate_angles = np.arctan2(candidates[:, 1], candidates[:, 0])
        else:
            layers = 2 if is_context else 1
            candidates = np.concatenate([
                unit * (base_radius[:, i] + layer_i * pitch)[:, None]
                for layer_i in range(layers)
            ])
            layer = np.repeat(np.arange(layers), count)
            if atom_ids and not is_context:
                # Right beside the anchors, a concavity of the drawing included:
                # the global contour alone keeps a residue off a bay its atom
                # faces into.
                reach = concave_clearance[i] * np.array([1.05, 1.35])
                local = (anchors.mean(axis=0) + unit[:, None, :] * reach[None, :, None]).reshape(-1, 2)
                candidates = np.concatenate([candidates, local])
                layer = np.concatenate([layer, np.zeros(len(local), dtype=int)])
            candidate_angles = np.arctan2(candidates[:, 1], candidates[:, 0])

        # Minimax anchor distance is the decisive term for a residue connected
        # to atoms on different parts of the ligand.  It avoids the misleading
        # centroid solution that is excellent for one line and terrible for the
        # other.  A smaller bearing term breaks ties consistently.
        distances = np.linalg.norm(candidates[:, None, :] - anchors[None, :, :], axis=2)
        score = distances.max(axis=1) + 0.18 * distances.mean(axis=1)
        score += 18.0 * np.abs(_angle_delta(candidate_angles, ideal))
        score += layer * (30.0 if is_context else 240.0)
        # Solvent exposure wins over the residue: a protein glyph stays out of
        # the bearings the surface leaves open and settles at a side of it.
        if len(problem.open_angle) and not (is_water or is_metal):
            delta = np.abs(_angle_delta(candidate_angles[:, None], problem.open_angle[None, :]))
            distance = np.linalg.norm(candidates, axis=1)
            reach = problem.open_half[None, :] + np.arctan2(problem.radii[i], distance)[:, None]
            arc = np.maximum(0.0, reach - delta).max(axis=1) * distance
            score += OPEN_GAP_COST * arc ** 2
        # A glyph may enter a bay of the drawing but never sit on it.
        room = ligand_gap(candidates, room_limit)
        score += 600.0 * np.maximum(0.0, np.minimum(concave_clearance[i], problem.ligand_clearance[i]) - room) ** 2

        for other in placed:
            sep = np.linalg.norm(candidates - other.position[None, :], axis=1)
            floor = problem.pair_floor[i, other.index]
            score += 600.0 * np.maximum(0.0, floor - sep) ** 2

        # Sequence neighbours define the smooth residue trace.  In particular,
        # a context glyph between two interacting residues should follow them
        # around the ligand instead of accepting its unrelated 3D bearing and
        # drawing a backbone chord through the molecule.
        placed_by_index = {other.index: other for other in placed}
        for neighbour, rest in backbone_neighbours[i]:
            other = placed_by_index.get(neighbour)
            if other is None:
                continue
            sep = np.linalg.norm(candidates - other.position[None, :], axis=1)
            score += 1.8 * np.square(sep - rest)
            for t in (0.45, 0.65, 0.85):
                samples = (1.0 - t) * other.position + t * candidates
                clearance = problem._hull_distance(samples)[0]
                score += 120.0 * np.maximum(0.0, 8.0 - clearance) ** 2

        # The line terms below cost far more per candidate than the ones above
        # and only reorder near-equal slots: score the best few only.
        keep = np.argsort(score)[:SHORTLIST]
        candidates, score = candidates[keep], score[keep]

        # Inside a bay the nearby atoms must leave an open side, so the glyph
        # reads as beside the ligand, not enclosed by it.
        inside = problem._hull_distance(candidates)[0] < problem.ligand_clearance[i]
        if np.any(inside):
            score[inside] += 4000.0 * np.maximum(0.0, OPEN_SIDE - _open_side(problem, candidates[inside]))

        # The outer part of each line keeps a margin from the drawing itself
        # -- atoms and bonds, not the convex hull, so a line may leave through
        # a concavity when that is where its anchor faces.
        steps = np.array([0.50, 0.68, 0.84])[None, :, None, None]
        samples = (1.0 - steps) * anchors[:, None, None, :] + steps * candidates[None, None]
        clearance = ligand_gap(samples.reshape(-1, 2), LINE_CLEARANCE).reshape(samples.shape[:-1])
        score += 40.0 * np.sum(np.maximum(0.0, LINE_CLEARANCE - clearance) ** 2, axis=(0, 1))

        # The whole drawn line, not only its outer half, has to stay off the
        # ligand: an anchor atom inside the drawing needs a slot it can see.
        score += BOND_CROSS_COST * _glyph_bond_hits(
            problem, lines, i, candidates, p, placed_mask)

        choice = int(np.argmin(score))
        p[i] = target[i] = candidates[choice]
        placed.append(_Placed(i, p[i].copy()))
        placed_mask[i] = True

    return p, target


#: Per px^2 of a glyph's arc inside a solvent gap: above every soft term,
#: below a glyph overlap, so a crowded side can still spill into the gap.
OPEN_GAP_COST = 40.0

#: Slots per residue that get the line terms, after the cheap ranking.
SHORTLIST = 48

#: Widest empty bearing, seen from a glyph, that the nearby atoms must leave:
#: below it the glyph reads as enclosed by the ligand rather than beside it.
OPEN_SIDE = math.radians(160.0)


def _open_side(problem, pts: np.ndarray) -> np.ndarray:
    """Largest angular gap between the ligand atoms as seen from each point.

    Every atom counts, not only near ones: the hollow of a macrocycle is
    enclosed however far its far side is.
    """
    if len(problem.ligand) < 2:
        return np.full(len(pts), 2.0 * math.pi)
    rel = problem.ligand[None, :, :] - pts[:, None, :]
    bearing = np.sort(np.arctan2(rel[..., 1], rel[..., 0]), axis=1)
    wrap = bearing[:, :1] + 2.0 * math.pi
    return np.diff(np.concatenate([bearing, wrap], axis=1), axis=1).max(axis=1)


def _concave_clearance(problem) -> np.ndarray:
    """Centre clearance from the drawn atoms/bonds for a glyph in a concavity."""
    return problem.radii + CONCAVE_MARGIN


def _settle_local(problem, p: np.ndarray, target: np.ndarray,
                  pinned: set[str], fixed: dict[str, tuple[float, float]] | None,
                  iterations: int = 24, concave: np.ndarray | None = None) -> np.ndarray:
    """Resolve local clearances with bounded position-based constraints.

    ``concave`` marks glyphs measured against the drawn atoms and bonds rather
    than the convex hull, so they may sit in a bay of the ligand like a water.
    """
    if len(p) == 0:
        return p
    if concave is None:
        concave = np.zeros(len(p), dtype=bool)
    drawing = _Clearance(problem)
    local_nodes = problem.water_mask | problem.metal_mask | concave
    clearance = problem.ligand_clearance.copy()
    soft = concave & ~problem.water_mask & ~problem.metal_mask
    clearance[soft] = _concave_clearance(problem)[soft]
    fixed_mask = np.array([key in pinned for key in problem.keys], dtype=bool)
    fixed_values = p.copy()

    def keep_metals_on_pocket_side() -> None:
        """Keep each metal just inside the protein surface, like a water."""
        metals = np.flatnonzero(problem.metal_mask)
        shell = np.flatnonzero(
            problem.on_shell & ~problem.water_mask & ~problem.metal_mask
        )
        if not len(metals) or not len(shell):
            return
        shell_angle = np.arctan2(p[shell, 1], p[shell, 0])
        shell_radius = np.linalg.norm(p[shell], axis=1)
        # These match the renderer's front-row neighbourhood and droplet
        # inset.  Keeping the constants here avoids importing Qt into layout.
        neighbourhood = math.radians(20.0)
        surface_inset = 46.0
        visual_gap = 10.0
        for i in metals:
            if fixed_mask[i]:
                continue
            radius = max(float(np.linalg.norm(p[i])), 1e-8)
            angle = math.atan2(p[i, 1], p[i, 0])
            delta = np.abs(_angle_delta(shell_angle, angle))
            local = shell_radius[delta <= neighbourhood]
            if not len(local):
                # A slightly wider fallback covers a metal centred in the gap
                # between two adjacent front-row residues.
                local = shell_radius[delta <= 1.55 * neighbourhood]
            if not len(local):
                continue
            surface = float(np.min(local)) - surface_inset
            maximum = surface - float(problem.radii[i]) - visual_gap
            # "Outside the protein surface" means on its exposed pocket side:
            # radially inside the residue ribbon, not behind the residues.
            # Respect ligand clearance even when a very tight surface leaves
            # less room than one complete coordination glyph.
            hull_distance, _ = drawing.away(p[i:i + 1])
            ligand_floor = radius - max(
                0.0, float(hull_distance[0] - problem.ligand_clearance[i])
            )
            maximum = max(ligand_floor, maximum)
            if radius > maximum:
                p[i] *= maximum / radius

    def enforce_water_bridges() -> None:
        """Keep bridge waters local and leave their outer leg readable."""
        water_ids = np.flatnonzero(problem.water_mask)
        for i in water_ids:
            if fixed_mask[i]:
                continue
            delta = p[i] - problem.anchors[i]
            distance = max(float(np.hypot(delta[0], delta[1])), 1e-8)
            maximum = max(68.0, 1.35 * problem.w.d0_water)
            if distance > maximum:
                p[i] = problem.anchors[i] + delta / distance * maximum

        for edge, rest in zip(problem.backbone, problem.edge_rest):
            i, j = map(int, edge)
            floor = float(problem.pair_floor[i, j])
            # Only water bridges have a designed rest length substantially
            # larger than collision clearance.  Backbone and metal edges are
            # left to their ordinary springs.
            if rest < floor + 20.0 or not (problem.water_mask[i] or problem.water_mask[j]):
                continue
            delta = p[j] - p[i]
            distance = max(float(np.hypot(delta[0], delta[1])), 1e-8)
            minimum = 0.82 * float(rest)
            water, residue = (i, j) if problem.water_mask[i] else (j, i)
            outward = delta / distance if residue == j else -delta / distance
            if distance < minimum:
                correction = outward * (minimum - distance + 0.05)
                if not fixed_mask[residue]:
                    p[residue] += 0.85 * correction
                if not fixed_mask[water]:
                    p[water] -= 0.15 * correction
            else:
                maximum = 1.20 * float(rest)
                if distance > maximum:
                    correction = outward * (distance - maximum)
                    if not fixed_mask[residue]:
                        p[residue] -= 0.90 * correction
                    if not fixed_mask[water]:
                        p[water] += 0.10 * correction

        # The bridge correction above gives most motion to the protein, but
        # clamp once more so even its small water share cannot accumulate.
        for i in water_ids:
            if fixed_mask[i]:
                continue
            delta = p[i] - problem.anchors[i]
            distance = max(float(np.hypot(delta[0], delta[1])), 1e-8)
            maximum = max(68.0, 1.35 * problem.w.d0_water)
            if distance > maximum:
                p[i] = problem.anchors[i] + delta / distance * maximum

    for _ in range(iterations):
        before = p.copy()
        # Weak springs retain the discrete chemical assignment.  Conductors
        # are stronger; context nodes remain free to give way.
        alpha = np.where(problem.stiff > 2.0, 0.13, 0.055)
        alpha[fixed_mask] = 0.0
        p += alpha[:, None] * (target - p)

        # Exact ligand minimum.  Waters use the true atom/bond geometry so a
        # concavity remains available; all other glyphs clear the global hull.
        normal = ~local_nodes
        if np.any(normal):
            d, direction = problem._hull_distance(p[normal])
            need = np.maximum(0.0, clearance[normal] - d)
            p[normal] += need[:, None] * direction
        if np.any(local_nodes):
            d, direction = drawing.away(p[local_nodes])
            need = np.maximum(0.0, clearance[local_nodes] - d)
            p[local_nodes] += need[:, None] * direction

        # Pair constraints.  Resolve every overlap symmetrically unless one
        # endpoint was manually pinned.  Only pairs overlapping at the start of
        # the sweep enter it; one pushed into overlap is caught next iteration.
        gap = np.linalg.norm(p[None, :] - p[:, None], axis=-1) - problem.pair_floor
        for i, j in zip(*np.nonzero(np.triu(gap < 0.0, 1))):
            if fixed_mask[i] and fixed_mask[j]:
                continue
            delta = p[j] - p[i]
            distance = float(np.hypot(delta[0], delta[1]))
            floor = float(problem.pair_floor[i, j])
            if distance >= floor:
                continue
            if distance < 1e-8:
                angle = (i * 2.399963229728653) % (2.0 * math.pi)
                direction = np.array([math.cos(angle), math.sin(angle)])
            else:
                direction = delta / distance
            correction = direction * (floor - distance + 0.05)
            if fixed_mask[i]:
                p[j] += correction
            elif fixed_mask[j]:
                p[i] -= correction
            else:
                p[i] -= 0.5 * correction
                p[j] += 0.5 * correction

        # Backbone and bridge relations are local springs.  Their strengths are
        # capped so they can never violate ligand/glyph clearance in one step.
        # Applied all at once (Jacobi); the per-edge step is small enough.
        if len(problem.backbone):
            ei, ej = np.asarray(problem.backbone, dtype=int).reshape(-1, 2).T
            delta = p[ej] - p[ei]
            distance = np.maximum(np.hypot(delta[:, 0], delta[:, 1]), 1e-8)
            error = np.clip(distance - problem.edge_rest, -8.0, 8.0)
            step = np.minimum(0.18, 0.025 * np.asarray(problem.edge_k, dtype=float))
            correction = delta / distance[:, None] * (error * step)[:, None]
            move = np.zeros_like(p)
            np.add.at(move, ei, correction)
            np.add.at(move, ej, -correction)
            move[fixed_mask] = 0.0
            p += move

        # Keep the sampled outer part of interaction/backbone routes outside
        # the ligand.  Move only their free endpoints; anchors are molecular
        # atoms and intentionally fixed.
        if len(problem.span_map):
            samples = problem.span_base + problem.span_map @ p
            # The drawing itself, not the hull: a line may leave via a bay.
            d, direction = drawing.away(samples)
            need = np.maximum(0.0, 12.0 - d)
            for row in np.flatnonzero(need > 0.0):
                owners = np.flatnonzero(problem.span_map[row] > 0.0)
                owners = owners[~fixed_mask[owners]]
                if len(owners):
                    push = direction[row] * min(12.0, float(need[row]))
                    p[owners] += push / len(owners)

        if np.any(fixed_mask):
            p[fixed_mask] = fixed_values[fixed_mask]
        enforce_water_bridges()
        if float(np.max(np.linalg.norm(p - before, axis=1))) < 0.04:
            break

    # Finish with hard constraints, because a late spring must never leave a
    # context residue (MET682 was the visible example) touching the ligand.
    for _ in range(4):
        enforce_water_bridges()
        d, direction = problem._hull_distance(p[~local_nodes])
        need = np.maximum(0.0, clearance[~local_nodes] - d)
        p[~local_nodes] += need[:, None] * direction
        if np.any(local_nodes):
            d, direction = drawing.away(p[local_nodes])
            need = np.maximum(0.0, clearance[local_nodes] - d)
            p[local_nodes] += need[:, None] * direction
        gap = np.linalg.norm(p[None, :] - p[:, None], axis=-1) - problem.pair_floor
        for i, j in zip(*np.nonzero(np.triu(gap + 1e-6 < 0.0, 1))):
            delta = p[j] - p[i]
            distance = max(float(np.hypot(delta[0], delta[1])), 1e-8)
            floor = float(problem.pair_floor[i, j])
            if distance + 1e-6 >= floor or (fixed_mask[i] and fixed_mask[j]):
                continue
            correction = delta / distance * (floor - distance + 0.1)
            if fixed_mask[i]:
                p[j] += correction
            elif fixed_mask[j]:
                p[i] -= correction
            else:
                p[i] -= correction * 0.5
                p[j] += correction * 0.5
        keep_metals_on_pocket_side()
        if np.any(fixed_mask):
            p[fixed_mask] = fixed_values[fixed_mask]
    return p


def _repair_discrete(problem, p: np.ndarray, target: np.ndarray,
                     pinned: set[str], rounds: int,
                     lines: _Lines) -> tuple[np.ndarray, int]:
    """Swap occupied slots only when that strictly improves line readability."""
    from .layout import _count_crossings

    fixed = {i for i, key in enumerate(problem.keys) if key in pinned}

    def score_all(qs: np.ndarray) -> list[tuple[int, float]]:
        bonds = _line_bond_crossings(problem, qs, lines)
        reach = np.sum(problem.stiff * np.linalg.norm(qs - target, axis=-1), axis=-1)
        return [(_count_crossings(problem, q, problem.radius) + int(b), float(r))
                for q, b, r in zip(qs, np.atleast_1d(bonds), np.atleast_1d(reach))]

    def culprits(q: np.ndarray) -> set[int]:
        _, crossed = _count_crossings(problem, q, problem.radius, culprits=True)
        _, bonds = _line_bond_crossings(problem, q, lines, culprits=True)
        return crossed | bonds

    best = score_all(p[None])[0]
    if best[0] == 0:
        return p, 0
    blame = culprits(p)
    for _ in range(max(0, min(rounds, 4))):
        # A swap that leaves every implicated glyph untouched cannot remove a
        # crossing.  Restricting one endpoint to ``blame`` cuts ANP from ~900
        # full crossing counts per round to a few dozen.
        swaps = [(i, j) for i in sorted(blame - fixed) for j in range(len(p))
                 if j != i and j not in fixed and not (j < i and j in blame)]
        if not swaps:
            break
        qs = np.repeat(p[None], len(swaps), axis=0)
        k = np.arange(len(swaps))
        i, j = np.array(swaps).T
        qs[k, i], qs[k, j] = p[j], p[i]
        scores = score_all(qs)
        pick = min(range(len(swaps)), key=scores.__getitem__)
        if not scores[pick] < best:
            break
        p, best = qs[pick], scores[pick]
        if best[0] == 0:
            break
        blame = culprits(p)
    return p, best[0]


@dataclass
class _Lines:
    """Every strong line the diagram draws, one row each, as flat arrays.

    ``start`` is the fixed ligand end; rows whose line starts at a water glyph
    (the water -> residue leg of a bridge) carry NaN there and the water's
    index in ``water``.  ``own``/``own_atoms`` mark the bonds and atoms a line
    legitimately touches at its start: its anchor atom's bonds and its own
    atom group, or its ring.
    """

    owner: np.ndarray       # (L,) glyph index the line ends at
    start: np.ndarray       # (L, 2)
    water: np.ndarray       # (L,) water glyph index, -1 for a ligand start
    own: np.ndarray         # (L, bonds) bool
    own_atoms: np.ndarray   # (L, atoms) bool

    def rows_of(self, glyph: int) -> np.ndarray:
        return np.flatnonzero(self.owner == glyph)

    def starts(self, p: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Line starts for ``rows`` with water legs resolved against ``p``."""
        start = self.start[rows].copy()
        water = self.water[rows]
        start[water >= 0] = p[water[water >= 0]]
        return start


def _line_specs(problem, diagram: Diagram) -> _Lines:
    """Every strong line, with the start rule ``render._ligand_anchor`` draws.

    A ring contact starts at the ring centroid and may cross its own ring; a
    single-atom contact starts at its anchor atom; each metal leg starts at its
    own atom.  A water-bridged contact is two lines: ligand -> water belongs to
    the water, water -> residue to the residue.  Hydrophobic contacts are off
    by default and only a hint when shown, so they never steer the layout.
    """
    index = {key: i for i, key in enumerate(problem.keys)}
    bonds = problem.ligand_bonds.reshape(-1, 2).astype(int)
    n_atoms = len(problem.ligand)
    rows: list[tuple] = []
    seen: set[tuple] = set()

    def add(owner: int, atoms: list[int] | None, ring: bool = False, water: int = -1) -> None:
        own_atoms = np.zeros(n_atoms, dtype=bool)
        if atoms is None:
            mark, start, own = (owner, "water", water), np.full(2, np.nan), \
                np.zeros(len(bonds), dtype=bool)
        elif ring:
            mark, start = (owner, tuple(atoms)), problem.ligand[atoms].mean(axis=0)
            own = np.isin(bonds, atoms).all(axis=1)
            own_atoms[atoms] = True
        else:
            # The line may run over the rest of its own group (both O of a
            # carboxylate), not over anything else.
            mark, start = (owner, atoms[0]), problem.ligand[atoms[0]]
            own = np.isin(bonds, atoms[:1]).any(axis=1) | np.isin(bonds, atoms).all(axis=1)
            own_atoms[atoms] = True
        if mark not in seen:
            seen.add(mark)
            rows.append((owner, start, water, own, own_atoms))

    for inter in diagram.interactions:
        if inter.kind == "hydrophobic" or inter.residue_key not in index:
            continue
        atoms = [a for a in inter.ligand_atoms if 0 <= a < n_atoms]
        if not atoms:
            continue
        ring = inter.kind in RING_ANCHORED
        if not ring:
            atoms = sorted(atoms)
            if inter.anchor_atom in atoms:
                atoms = [inter.anchor_atom] + [a for a in atoms if a != inter.anchor_atom]
        owner = index[inter.residue_key]
        water = index.get(inter.via_water, -1) if inter.via_water else -1
        if water >= 0:
            add(water, atoms, ring)
            add(owner, None, water=water)
        elif inter.kind == "metal_coordination":
            for atom in atoms:
                add(owner, [atom])
        else:
            add(owner, atoms, ring)
    if not rows:
        return _Lines(np.zeros(0, dtype=int), np.zeros((0, 2)), np.zeros(0, dtype=int),
                      np.zeros((0, len(bonds)), dtype=bool), np.zeros((0, n_atoms), dtype=bool))
    owner, start, water, own, own_atoms = zip(*rows)
    return _Lines(np.array(owner), np.array(start, dtype=float), np.array(water),
                  np.array(own, dtype=bool).reshape(len(rows), len(bonds)),
                  np.array(own_atoms, dtype=bool).reshape(len(rows), n_atoms))


#: Rim-to-drawing gap of a residue glyph sitting in a concavity of the ligand,
#: measured to atom centres and bonds: room for an atom label plus air.
CONCAVE_MARGIN = 46.0
#: Margin the outer part of a strong line keeps from the drawn ligand.
LINE_CLEARANCE = 10.0
#: A line passing closer than this to a foreign atom centre reads as drawn
#: through it (over its label, for a heteroatom): about a quarter bond.
ATOM_CLEARANCE = 12.0


def _segment_hits(problem, start: np.ndarray, end: np.ndarray,
                  own: np.ndarray, own_atoms: np.ndarray) -> np.ndarray:
    """Bonds crossed plus foreign atoms grazed by each segment ``start -> end``.

    Everything broadcasts over the leading axes: ``start``/``end`` are
    ``(..., 2)``, ``own`` is ``(..., bonds)`` and ``own_atoms`` ``(..., atoms)``.
    """
    lig = problem.ligand
    seg = end - start
    start, seg = np.broadcast_arrays(start, seg)
    len2 = np.maximum(np.sum(seg * seg, axis=-1), 1e-9)
    rel = lig - start[..., None, :]                                  # (..., atoms, 2)
    along = np.clip(np.sum(rel * seg[..., None, :], axis=-1) / len2[..., None], 0.0, 1.0)
    gap = rel - along[..., None] * seg[..., None, :]
    hits = np.sum((np.hypot(gap[..., 0], gap[..., 1]) < ATOM_CLEARANCE) & ~own_atoms, axis=-1)
    bonds = problem.ligand_bonds.reshape(-1, 2).astype(int)
    if not len(bonds):
        return hits
    a, b = lig[bonds[:, 0]], lig[bonds[:, 1]]

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    s = start[..., None, :]
    o1 = cross(seg[..., None, :], a - s)
    o2 = cross(seg[..., None, :], b - s)
    o3 = cross(b - a, s - a)
    o4 = cross(b - a, (start + seg)[..., None, :] - a)
    crossed = (o1 * o2 < -1e-9) & (o3 * o4 < -1e-9) & ~own
    return hits + np.sum(crossed, axis=-1)


def _glyph_bond_hits(problem, lines: _Lines, glyph: int, ends: np.ndarray,
                     p: np.ndarray, placed: np.ndarray) -> np.ndarray:
    """Bond crossings of one glyph's lines for each candidate position in ``ends``."""
    rows = lines.rows_of(glyph)
    rows = rows[(lines.water[rows] < 0) | placed[np.maximum(lines.water[rows], 0)]]
    if not len(rows):
        return np.zeros(len(ends), dtype=int)
    start = lines.starts(p, rows)[:, None, :]
    hits = _segment_hits(problem, start, ends[None, :, :],
                         lines.own[rows][:, None, :], lines.own_atoms[rows][:, None, :])
    return hits.sum(axis=0)


def _line_bond_crossings(problem, p: np.ndarray, lines: _Lines, *, culprits: bool = False):
    """Strong lines drawn across the ligand, counted per crossed bond or grazed atom."""
    if not len(lines.owner):
        if culprits:
            return 0, set()
        return np.zeros(len(p)) if p.ndim == 3 else 0
    rows = np.arange(len(lines.owner))
    if p.ndim == 3:     # a stack of layouts -> one count each
        start = np.where((lines.water >= 0)[:, None], p[:, np.maximum(lines.water, 0)],
                         lines.start)
        return _segment_hits(problem, start, p[:, lines.owner],
                             lines.own, lines.own_atoms).sum(axis=-1)
    hits = _segment_hits(problem, lines.starts(p, rows), p[lines.owner],
                         lines.own, lines.own_atoms)
    total = int(hits.sum())
    if not culprits:
        return total
    bad = hits > 0
    blame = set(lines.owner[bad].tolist()) | set(lines.water[bad & (lines.water >= 0)].tolist())
    return total, blame


#: Spacing of the points sampled along each drawn bond for ``_Clearance``.
#: Half of it is the worst distance error, far below the margins it checks.
_CLEARANCE_STEP = 3.0


class _Clearance:
    """Distance from points to the drawn ligand, capped at ``limit``.

    The atoms and bonds are sampled every ``_CLEARANCE_STEP`` px into a
    KD-tree once per candidate; a query is then a bounded nearest-neighbour
    search instead of a points x primitives distance matrix.
    """

    def __init__(self, problem) -> None:
        from scipy.spatial import cKDTree

        lig = problem.ligand
        bonds = problem.ligand_bonds.reshape(-1, 2).astype(int)
        a, b = lig[bonds[:, 0]], lig[bonds[:, 1]]
        longest = float(np.max(np.hypot(*(b - a).T), initial=0.0))
        u = np.linspace(0.0, 1.0, max(2, int(longest / _CLEARANCE_STEP)) + 1)[1:-1, None, None]
        cloud = [lig, (a + u * (b - a)).reshape(-1, 2)]
        self._tree = cKDTree(np.concatenate(cloud))

    def __call__(self, pts: np.ndarray, limit: float) -> np.ndarray:
        d, _ = self._tree.query(pts, distance_upper_bound=limit)
        return np.minimum(d, limit)

    def away(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distance to the drawing and the unit direction away from it."""
        d, k = self._tree.query(pts)
        vector = pts - self._tree.data[k]
        return d, vector / np.maximum(d, 1e-8)[:, None]


def _legalize_pins(problem, fixed: dict[str, tuple[float, float]] | None,
                   pinned: set[str]) -> dict[str, tuple[float, float]] | None:
    """Push pinned spots out of the ligand and off each other.

    A pin says where the user wants a glyph, not that it may sit on the
    ligand: rotating a metal swings its partners wherever the arc takes them,
    and the settling never moves a pinned glyph again.  So the hard
    clearances are applied here, once, before the pins become constraints.
    """
    if not fixed or not pinned:
        return fixed
    idx = [i for i, key in enumerate(problem.keys) if key in pinned and key in fixed]
    if not idx:
        return fixed
    pts = np.array([fixed[problem.keys[i]] for i in idx], dtype=float)
    local = problem.water_mask[idx] | problem.metal_mask[idx]
    clearance = problem.ligand_clearance[idx]
    radii = problem.radii[idx]
    center = problem.ligand.mean(axis=0)
    overlapping = np.zeros(len(idx), dtype=bool)
    for _ in range(40):
        for mask, measure in ((~local, problem._hull_distance),
                              (local, problem._molecule_distance)):
            if not np.any(mask):
                continue
            d, direction = measure(pts[mask])
            # Only a glyph that actually overlaps the ligand is moved, and then
            # to the normal standoff; a pin merely close to it is respected.
            overlapping[mask] |= d < radii[mask] - 0.5
            need = np.where(overlapping[mask], np.maximum(0.0, clearance[mask] - d), 0.0)
            # The smoothed gradient vanishes deep inside (a pin on the ligand
            # centre); fall back to the radial direction there.
            radial = pts[mask] - center
            radial[np.hypot(*radial.T) < 1e-6] = (1.0, 0.0)
            weak = np.hypot(*direction.T) < 0.2
            direction[weak] = radial[weak]
            direction /= np.hypot(*direction.T)[:, None]
            # Small steps: the fallback direction is not the hull normal.
            pts[mask] += np.minimum(need + 0.5 * (need > 0), 12.0)[:, None] * direction
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                delta = pts[b] - pts[a]
                distance = max(float(np.hypot(*delta)), 1e-8)
                floor = float(problem.pair_floor[idx[a], idx[b]])
                if distance < floor:
                    push = delta / distance * (floor - distance) * 0.5
                    pts[a] -= push
                    pts[b] += push
    out = dict(fixed)
    for i, point in zip(idx, pts):
        out[problem.keys[i]] = (float(point[0]), float(point[1]))
    return out


def _one_candidate(diagram: Diagram, view: list[tuple[float, float]], projection: int,
                   rotation: float, mirror: bool, glyph_radius: float, weights,
                   variant: int, seed_positions, pinned: set[str], max_swaps: int,
                   cache: dict | None = None):
    from .layout import _static, build_problem

    cache = cache or _static(diagram, glyph_radius, weights, view)
    problem = build_problem(
        diagram, glyph_radius=glyph_radius, weights=weights,
        rotation=rotation, mirror=mirror, _cache=cache,
    )
    seed_positions = _legalize_pins(problem, seed_positions, pinned)
    lines = _line_specs(problem, diagram)
    p, target = _place_discrete(
        problem, cache, diagram, glyph_radius, variant, seed_positions, pinned, lines
    )
    concave = np.zeros(len(p), dtype=bool)
    concave[lines.owner] = True
    p = _settle_local(problem, p, target, pinned, seed_positions, iterations=8, concave=concave)
    p, crossings = _repair_discrete(problem, p, target, pinned, max_swaps, lines)
    p = _settle_local(problem, p, p.copy(), pinned, seed_positions, iterations=2, concave=concave)
    # Recount after the final hard-clearance pass.
    from .layout import _count_crossings
    crossings = _count_crossings(problem, p, glyph_radius)
    crossings += _line_bond_crossings(problem, p, lines)
    energy = problem.objective(p.ravel())[0]
    # Readability dominates: a crossed strong line is never traded for a
    # slightly smaller shell.  Horizontal ligand presentation breaks close ties.
    covariance = np.cov(problem.ligand.T, bias=True) if len(problem.ligand) > 1 else np.eye(2)
    values, vectors = np.linalg.eigh(covariance)
    anisotropy = max(0.0, float(values[-1] - values[0]) / max(float(values.sum()), 1e-9))
    vertical = float(vectors[1, -1] ** 2)
    rank = crossings * 10_000_000.0 + energy + 45_000.0 * anisotropy * vertical
    return (
        rank, problem, p, float(energy), int(crossings), projection,
        float(rotation), bool(mirror),
    )


def solve_layout_fast(
    diagram: Diagram,
    *,
    glyph_radius: float,
    weights,
    seed_positions: dict[str, tuple[float, float]] | None,
    pinned: set[str] | None,
    max_swaps: int,
    orientation: tuple[float, bool] | None,
    projection: int | None,
    variant: int,
):
    """Evaluate bounded candidates and return the most legible arrangement."""
    from .layout import LayoutResult

    views = [diagram.coords_2d, *diagram.coords_alt]
    indices = [int(projection)] if projection is not None else list(range(len(views)))
    pinned = set(pinned or ())

    best = None
    from .layout import _static

    for projection_i in indices:
        cache = _static(diagram, glyph_radius, weights, views[projection_i])
        if orientation is not None:
            candidate_orientations = [orientation]
        else:
            raw = np.asarray(views[projection_i], dtype=float)
            raw = raw - raw.mean(axis=0)
            # Major axis horizontal and both mirrors.  A 180-degree global
            # turn is the same radial layout with every bearing shifted; the
            # old scan paid twice for that visually equivalent candidate.  A
            # mirror changes the major-axis bearing, so compute its rotation
            # *after* reflecting instead of reusing the unmirrored angle.
            candidate_orientations = []
            for mirror in (False, True):
                oriented = raw * np.array([-1.0, 1.0]) if mirror else raw
                if len(oriented) > 1:
                    _, vectors = np.linalg.eigh(np.cov(oriented.T, bias=True))
                    major = vectors[:, -1]
                    horizontal = -math.atan2(float(major[1]), float(major[0]))
                else:
                    horizontal = 0.0
                candidate_orientations.append((horizontal, mirror))
        for rotation, mirror in candidate_orientations:
            candidate = _one_candidate(
                diagram, views[projection_i], projection_i, rotation, mirror,
                glyph_radius, weights, variant, seed_positions, pinned, max_swaps,
                cache,
            )
            if best is None or candidate[0] < best[0]:
                best = candidate

    _, problem, p, energy, crossings, projection_i, rotation, mirror = best
    return LayoutResult(
        positions={key: (float(p[i, 0]), float(p[i, 1]))
                   for i, key in enumerate(problem.keys)},
        ligand_coords=[(float(x), float(y)) for x, y in problem.ligand],
        rotation=rotation,
        mirror=mirror,
        energy=energy,
        energy_terms=problem.terms(p.ravel()),
        crossings=crossings,
        projection=projection_i,
    )
