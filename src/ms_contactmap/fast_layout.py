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

from .model import Diagram


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


def _angle_delta(a: np.ndarray | float, b: float) -> np.ndarray:
    return (np.asarray(a) - b + math.pi) % (2.0 * math.pi) - math.pi


def _place_discrete(problem, cache: dict, diagram: Diagram, glyph_radius: float,
                    variant: int, fixed: dict[str, tuple[float, float]] | None,
                    pinned: set[str]) -> tuple[np.ndarray, np.ndarray]:
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
            candidate_angles = np.tile(angles, layers)

        # Minimax anchor distance is the decisive term for a residue connected
        # to atoms on different parts of the ligand.  It avoids the misleading
        # centroid solution that is excellent for one line and terrible for the
        # other.  A smaller bearing term breaks ties consistently.
        distances = np.linalg.norm(candidates[:, None, :] - anchors[None, :, :], axis=2)
        score = distances.max(axis=1) + 0.18 * distances.mean(axis=1)
        score += 18.0 * np.abs(_angle_delta(candidate_angles, ideal))
        score += layer * (30.0 if is_context else 240.0)

        # A route has to leave the convex ligand promptly.  Test the useful
        # outer half of every line; this sees an arrow laid across a ring even
        # when it happens not to intersect an individual bond segment.
        for anchor in anchors:
            for t in (0.50, 0.68, 0.84):
                samples = (1.0 - t) * anchor + t * candidates
                clearance = problem._hull_distance(samples)[0]
                score += 40.0 * np.maximum(0.0, 7.0 - clearance) ** 2

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

        choice = int(np.argmin(score))
        p[i] = target[i] = candidates[choice]
        placed.append(_Placed(i, p[i].copy()))

    return p, target


def _settle_local(problem, p: np.ndarray, target: np.ndarray,
                  pinned: set[str], fixed: dict[str, tuple[float, float]] | None,
                  iterations: int = 24) -> np.ndarray:
    """Resolve local clearances with bounded position-based constraints."""
    if len(p) == 0:
        return p
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
            hull_distance, _ = problem._molecule_distance(p[i:i + 1])
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
        local_nodes = problem.water_mask | problem.metal_mask
        normal = ~local_nodes
        if np.any(normal):
            d, direction = problem._hull_distance(p[normal])
            need = np.maximum(0.0, problem.ligand_clearance[normal] - d)
            p[normal] += need[:, None] * direction
        if np.any(local_nodes):
            d, direction = problem._molecule_distance(p[local_nodes])
            need = np.maximum(
                0.0, problem.ligand_clearance[local_nodes] - d
            )
            p[local_nodes] += need[:, None] * direction

        # Pair constraints.  Resolve every overlap symmetrically unless one
        # endpoint was manually pinned.
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
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
                if fixed_mask[i] and fixed_mask[j]:
                    continue
                if fixed_mask[i]:
                    p[j] += correction
                elif fixed_mask[j]:
                    p[i] -= correction
                else:
                    p[i] -= 0.5 * correction
                    p[j] += 0.5 * correction

        # Backbone and bridge relations are local springs.  Their strengths are
        # capped so they can never violate ligand/glyph clearance in one step.
        for edge, k, rest in zip(problem.backbone, problem.edge_k, problem.edge_rest):
            i, j = map(int, edge)
            delta = p[j] - p[i]
            distance = max(float(np.hypot(delta[0], delta[1])), 1e-8)
            error = np.clip(distance - rest, -8.0, 8.0)
            correction = delta / distance * error * min(0.18, 0.025 * float(k))
            if not fixed_mask[i]:
                p[i] += correction
            if not fixed_mask[j]:
                p[j] -= correction

        # Keep the sampled outer part of interaction/backbone routes outside
        # the ligand.  Move only their free endpoints; anchors are molecular
        # atoms and intentionally fixed.
        if len(problem.span_map):
            samples = problem.span_base + problem.span_map @ p
            d, direction = problem._hull_distance(samples)
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
        local_nodes = problem.water_mask | problem.metal_mask
        d, direction = problem._hull_distance(p[~local_nodes])
        need = np.maximum(0.0, problem.ligand_clearance[~local_nodes] - d)
        p[~local_nodes] += need[:, None] * direction
        if np.any(local_nodes):
            d, direction = problem._molecule_distance(p[local_nodes])
            need = np.maximum(0.0, problem.ligand_clearance[local_nodes] - d)
            p[local_nodes] += need[:, None] * direction
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                delta = p[j] - p[i]
                distance = max(float(np.hypot(delta[0], delta[1])), 1e-8)
                floor = float(problem.pair_floor[i, j])
                if distance + 1e-6 >= floor:
                    continue
                direction = delta / distance
                correction = direction * (floor - distance + 0.1)
                if fixed_mask[i] and fixed_mask[j]:
                    continue
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
                     pinned: set[str], rounds: int) -> tuple[np.ndarray, int]:
    """Swap occupied slots only when that strictly improves line readability."""
    from .layout import _count_crossings

    fixed = {i for i, key in enumerate(problem.keys) if key in pinned}

    def score(q: np.ndarray) -> tuple[int, float]:
        crossings = _count_crossings(problem, q, problem.radius)
        reach = float(np.sum(problem.stiff * np.linalg.norm(q - target, axis=1)))
        return crossings, reach

    best = score(p)
    if best[0] == 0:
        return p, 0
    _, blame = _count_crossings(problem, p, problem.radius, culprits=True)
    for _ in range(max(0, min(rounds, 4))):
        choice = None
        choice_score = best
        # A swap that leaves every implicated glyph untouched cannot remove a
        # crossing.  Restricting one endpoint to ``blame`` cuts ANP from ~900
        # full crossing counts per round to a few dozen.
        for i in sorted(blame - fixed):
            for j in range(len(p)):
                if j == i or j in fixed:
                    continue
                if j < i and j in blame:
                    continue
                q = p.copy()
                q[[i, j]] = q[[j, i]]
                got = score(q)
                if got < choice_score:
                    choice, choice_score = q, got
        if choice is None:
            break
        p, best = choice, choice_score
        if best[0] == 0:
            break
        _, blame = _count_crossings(problem, p, problem.radius, culprits=True)
    return p, best[0]


def _metal_route_crossings(problem, p: np.ndarray, diagram: Diagram) -> int:
    """Count every ligand-metal leg separately against the ligand bond graph."""
    index = {key: i for i, key in enumerate(problem.keys)}

    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) \
            - (b[1] - a[1]) * (c[0] - a[0])

    total = 0
    for interaction in diagram.interactions:
        if interaction.kind != "metal_coordination" \
                or interaction.residue_key not in index:
            continue
        metal = p[index[interaction.residue_key]]
        for atom in interaction.ligand_atoms:
            if not 0 <= atom < len(problem.ligand):
                continue
            anchor = problem.ligand[atom]
            for left, right in problem.ligand_bonds:
                left, right = int(left), int(right)
                if atom in (left, right):
                    continue
                a, b = problem.ligand[left], problem.ligand[right]
                total += int(
                    side(anchor, metal, a) * side(anchor, metal, b) < -1e-8
                    and side(a, b, anchor) * side(a, b, metal) < -1e-8
                )
    return total


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
    p, target = _place_discrete(
        problem, cache, diagram, glyph_radius, variant, seed_positions, pinned
    )
    p = _settle_local(problem, p, target, pinned, seed_positions, iterations=8)
    p, crossings = _repair_discrete(problem, p, target, pinned, max_swaps)
    p = _settle_local(problem, p, p.copy(), pinned, seed_positions, iterations=2)
    # Recount after the final hard-clearance pass.
    from .layout import _count_crossings
    crossings = _count_crossings(problem, p, glyph_radius)
    crossings += _metal_route_crossings(problem, p, diagram)
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
    rotations: int,
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
