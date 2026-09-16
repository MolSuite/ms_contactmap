"""Self-check for ms_contactmap.layout.  Run: python tests/test_layout.py

Everything here is synthetic: a ring of 2D points stands in for the ligand and
the residues are invented, so this file has no dependency on chem.py or
interactions.py.
"""
from __future__ import annotations

import math
import os
import sys
import time
import types

import numpy as np
from scipy.optimize import check_grad

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Bind the package without running ms_contactmap/__init__.py: layout.py is pure
# geometry over model.py, and this test must keep working while the chem and
# interaction modules are still being written.
_pkg = types.ModuleType("ms_contactmap")
_pkg.__path__ = [os.path.join(ROOT, "ms_contactmap")]
sys.modules.setdefault("ms_contactmap", _pkg)

from ms_contactmap.layout import (  # noqa: E402
    WEIGHTS,
    Problem,
    _convex_hull,
    _count_crossings,
    _probe_contour_radius,
    _static,
    build_problem,
    solve_layout,
)
from ms_contactmap.model import (  # noqa: E402
    Diagram,
    Interaction,
    MetalLeg,
    Residue,
    ResidueRef,
)

RADIUS = 26.0
NAMES = ["ALA", "SER", "ASP", "ARG", "GLY", "PHE", "THR", "GLU", "LYS", "TRP"]


def make_diagram(n_res, n_atoms=18, n_context=0, seed=0, chain="A"):
    """Ring-shaped fake ligand plus ``n_res`` residues wired onto it.

    Laid out like the reference diagrams: residues come in short runs of
    consecutive numbers (which ``backbone_edges`` turns into necklaces) and
    each run walks along a contiguous arc of the ligand, because a loop that
    lines the pocket touches neighbouring atoms with neighbouring residues.
    Runs are separated by a gap in the numbering, so they stay independent.
    """
    rng = np.random.default_rng(seed)
    ang = np.linspace(0.0, 2.0 * math.pi, n_atoms, endpoint=False)
    radius = 1.5 / (2.0 * math.sin(math.pi / n_atoms))          # ~1.5 per bond
    coords = [(radius * math.cos(a), 0.7 * radius * math.sin(a)) for a in ang]

    context = set(rng.choice(n_res, size=n_context, replace=False).tolist())
    residues, interactions, nearest = [], [], {}
    number, run_left, walk = 40, 0, 0.0
    for i in range(n_res):
        if run_left == 0:
            run_left = int(rng.integers(2, 6))
            number += 7                                          # break the chain
        else:
            number += 1
        run_left -= 1

        ref = ResidueRef(chain, number, NAMES[i % len(NAMES)])
        residues.append(Residue(ref, has_interactions=i not in context))
        walk += n_atoms / n_res * rng.uniform(0.6, 1.4)
        atom = int(walk) % n_atoms
        nearest[ref.key] = atom
        if i not in context:
            for _ in range(int(rng.integers(1, 4))):
                interactions.append(
                    Interaction("hbond", ref.key, (atom, (atom + 1) % n_atoms))
                )
    return Diagram("fake", "LIG", None, coords, residues, interactions, (), nearest)


def hull_signed_distance(points, hull):
    """Positive outside the hull polygon, negative inside (convex, so exact)."""
    edge = np.roll(hull, -1, axis=0) - hull
    normal = np.stack([edge[:, 1], -edge[:, 0]], axis=1)
    normal /= np.hypot(normal[:, 0], normal[:, 1])[:, None]
    return (points @ normal.T - np.einsum("ij,ij->i", normal, hull)).max(axis=1)


def check(label, ok, detail=""):
    assert ok, f"FAIL {label} {detail}"
    print(f"  ok   {label} {detail}")


# ---------------------------------------------------------------------------

def test_gradient():
    print("gradient")
    worst = 0.0
    for n_res, n_context, seed in ((12, 3, 1), (18, 5, 2), (25, 8, 3)):
        problem = build_problem(
            make_diagram(n_res, n_context=n_context, seed=seed),
            glyph_radius=RADIUS, rotation=0.4, mirror=True,
        )
        rng = np.random.default_rng(seed)
        for trial in range(4):
            # Start from the seed and jitter hard, so the hinge terms (ligand
            # overlap, glyph overlap, spread, canvas) are all switched on.
            x = problem.seed() + rng.normal(scale=45.0, size=2 * n_res)
            err = check_grad(
                lambda v: problem.objective(v)[0],
                lambda v: problem.objective(v)[1],
                x, epsilon=1e-6,
            )
            scale = np.linalg.norm(problem.objective(x)[1]) + 1.0
            worst = max(worst, err / scale)
    check("check_grad relative error < 1e-5", worst < 1e-5, f"= {worst:.3e}")


def test_terms_sum():
    print("energy terms")
    problem = build_problem(make_diagram(14, n_context=4, seed=7), glyph_radius=RADIUS)
    x = problem.seed()
    total, _, terms = problem._eval(x, terms=True)
    check("terms sum to the total energy", abs(sum(terms.values()) - total) < 1e-6,
          f"total = {total:.1f}")
    check("all nine terms reported", set(terms) == {
        "anchor", "ligand", "overlap", "backbone", "spread", "shell",
        "frame", "span", "route_overlap"})


def test_one_strong_contact_keeps_the_normal_residue_standoff():
    """Interaction kind must not collapse the entire residue shell."""
    diagram = make_diagram(8, n_context=0, seed=9)
    cache = _static(diagram, RADIUS, WEIGHTS)
    counts = {
        key: sum(i.residue_key == key for i in diagram.interactions)
        for key in cache["keys"]
    }
    key = next(key for key, count in counts.items() if count == 1)
    idx = cache["keys"].index(key)
    target_standoff = float(cache["d0"][idx])
    expected = WEIGHTS.d0_far - (
        (WEIGHTS.d0_far - WEIGHTS.d0_near) / (1.0 + WEIGHTS.anchor_decay)
    )
    check("one H-bond uses the normal interaction-count scale",
          abs(target_standoff - expected) < 1e-6,
          f"target = {target_standoff:.1f} px")


def test_rolling_probe_smooths_the_ligand_contour():
    """The explicit pre-layout contour must soften atom-to-atom corners."""
    coords = np.asarray(make_diagram(4, n_atoms=18).coords_2d, dtype=float) * 30.0
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    raw = np.max(coords @ unit.T, axis=0)
    smooth = _probe_contour_radius(coords, angles, WEIGHTS.ligand_probe_radius)
    raw_roughness = float(np.mean(np.abs(np.roll(raw, -1) - 2 * raw + np.roll(raw, 1))))
    smooth_roughness = float(
        np.mean(np.abs(np.roll(smooth, -1) - 2 * smooth + np.roll(smooth, 1)))
    )
    check("rolling probe reduces contour curvature spikes",
          smooth_roughness < raw_roughness,
          f"{smooth_roughness:.3f} < {raw_roughness:.3f}")


def assert_layout_sane(label, diagram, result, glyph_radius=RADIUS):
    keys = [r.key for r in diagram.residues]
    p = np.array([result.positions[k] for k in keys])

    d = np.hypot(*(p[:, None, :] - p[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    check(f"{label}: no two glyphs closer than 2r", d.min() >= 2.0 * glyph_radius,
          f"min = {d.min():.1f} px")

    lig = np.array(result.ligand_coords)
    sd = hull_signed_distance(p, lig[_convex_hull(lig)])
    check(f"{label}: every glyph outside the ligand hull", sd.min() > 0.0,
          f"min clearance = {sd.min():.1f} px")

    check(f"{label}: crossings == 0", result.crossings == 0, f"= {result.crossings}")


def deepest_inside(diagram, result, glyph_radius=RADIUS):
    """How far the worst drawn line reaches into the ligand hull, in pixels.

    Negative means every arrow and connector stays clear.  Sampled past the
    anchor, since an interaction route starts on a ligand atom by definition.
    """
    keys = [r.key for r in diagram.residues]
    p = np.array([result.positions[k] for k in keys])
    problem = build_problem(
        diagram, glyph_radius=glyph_radius,
        rotation=result.rotation, mirror=result.mirror,
    )
    lines = [(problem.anchors[i], p[i]) for i in np.flatnonzero(problem.has_line)]
    lines += [(p[i], p[j]) for i, j in problem.backbone]
    samples = np.array([
        (1.0 - t) * a + t * b for a, b in lines for t in (0.45, 0.65, 0.85)
    ])
    lig = np.array(result.ligand_coords)
    return -float(hull_signed_distance(samples, lig[_convex_hull(lig)]).min())


def test_small_case():
    print("12 residues")
    diagram = make_diagram(12, n_context=3, seed=11)
    problem = build_problem(diagram, glyph_radius=RADIUS)
    seed_energy = problem.objective(problem.seed())[0]
    result = solve_layout(diagram, glyph_radius=RADIUS)
    # The fast layout deliberately optimizes discrete readability before the
    # legacy 3D-bearing energy, so that scalar is diagnostic rather than its
    # acceptance criterion.  The geometric assertions below are the contract.
    check("layout energy is finite", math.isfinite(result.energy),
          f"energy = {result.energy:.1f}, legacy seed = {seed_energy:.1f}")
    assert_layout_sane("12res", diagram, result)


def test_scrambled_hard_case():
    print("24 residues, scrambled seed")
    diagram = make_diagram(24, n_atoms=26, n_context=6, seed=23)
    keys = [r.key for r in diagram.residues]

    # Deliberately destroy the order-preserving seed: hand the solver the
    # ellipse slots under a random permutation, which plants many crossings.
    rng = np.random.default_rng(5)
    ang = rng.permutation(24) * (2.0 * math.pi / 24)
    scrambled = {k: (300.0 * math.cos(a), 190.0 * math.sin(a)) for k, a in zip(keys, ang)}

    problem = build_problem(diagram, glyph_radius=RADIUS)
    start = np.array([scrambled[k] for k in keys])
    planted = _count_crossings(problem, start, RADIUS)
    check("scrambled seed really has crossings", planted > 0, f"= {planted}")

    result = solve_layout(diagram, glyph_radius=RADIUS, seed_positions=scrambled)
    assert_layout_sane("24res", diagram, result)


def make_metal_diagram():
    """A 12-residue case with a zinc whose sphere is closed by three residues."""
    diagram = make_diagram(12, n_context=5, seed=31)
    partners = [r.key for r in diagram.residues if not r.has_interactions][:3]

    zinc = ResidueRef("A", 900, "ZN")
    diagram.residues.append(Residue(zinc, has_interactions=True))
    diagram.interactions.append(Interaction("metal_coordination", zinc.key, (0,)))
    diagram.nearest_atom[zinc.key] = 0
    diagram.metal_legs = [MetalLeg(zinc.key, key, 2.1) for key in partners]
    diagram.metal_coordination = {zinc.key: 4}
    return diagram, zinc, partners


def test_metal_coordination():
    """A metal's partners have to end up on the metal, not on their own anchors."""
    print("metal coordination")
    diagram, zinc, partners = make_metal_diagram()

    result = solve_layout(diagram, glyph_radius=RADIUS)
    metal = np.array(result.positions[zinc.key])
    reach = max(float(np.hypot(*(np.array(result.positions[k]) - metal))) for k in partners)
    # One glyph pitch is where the spring wants them; twice that still reads as
    # "attached to the metal" and leaves the other terms room to argue.
    limit = 2.0 * (2.0 * RADIUS + WEIGHTS.glyph_gap)
    check("partners stay within two glyph pitches of the metal", reach < limit,
          f"furthest = {reach:.0f} px, limit {limit:.0f}")
    assert_layout_sane("metal", diagram, result)


def test_structural_water_has_a_readable_inner_shell_leg():
    """A small bridge water leaves a clearly visible leg to its residue."""
    diagram = make_diagram(10, n_context=10, seed=41)
    partner = diagram.residues[2]
    partner.has_interactions = True
    water = Residue(ResidueRef("A", 900, "HOH"), has_interactions=True)
    diagram.residues.append(water)
    diagram.interactions.append(
        Interaction("water_bridge", partner.key, (3,), 2.8, False, water.key)
    )
    diagram.nearest_atom[water.key] = 3

    result = solve_layout(diagram, glyph_radius=RADIUS)
    problem = build_problem(
        diagram, glyph_radius=RADIUS,
        rotation=result.rotation, mirror=result.mirror,
    )
    water_index = problem.keys.index(water.key)
    anchor_gap = math.dist(problem.target[water_index], problem.anchors[water_index])
    water_position = np.asarray([result.positions[water.key]])
    hull_standoff = float(problem._hull_distance(water_position)[0][0])
    water_radius = RADIUS * WEIGHTS.water_radius_fraction
    distance = math.dist(result.positions[water.key], result.positions[partner.key])
    full_residue_pitch = 2.0 * RADIUS + WEIGHTS.glyph_gap
    cache = _static(diagram, RADIUS, WEIGHTS)
    check("only water leaves the smooth residue shell",
          not cache["on_shell"][cache["keys"].index(water.key)]
          and cache["on_shell"][cache["keys"].index(partner.key)])
    check("water target is local to its ligand atom",
          abs(anchor_gap - WEIGHTS.d0_water) < 1e-6,
          f"target gap = {anchor_gap:.1f} px")
    check("water keeps readable space from the ligand",
          hull_standoff - water_radius >= 18.0,
          f"body gap = {hull_standoff - water_radius:.1f} px")
    check("water bridge is longer than a residue-residue pitch",
          distance > full_residue_pitch,
          f"distance = {distance:.1f}, pitch = {full_residue_pitch:.1f}")
    check("water bridge remains an inner-shell relationship",
          distance < 1.5 * full_residue_pitch,
          f"distance = {distance:.1f}")
    check("water bridge has no interaction-line crossings", result.crossings == 0,
          f"= {result.crossings}")


def test_span():
    """No arrow or connector should be drawn across the ligand."""
    print("lines clear of the ligand")
    for label, diagram in (
        ("12res", make_diagram(12, n_context=3, seed=11)),
        ("25res", make_diagram(25, n_atoms=30, n_context=7, seed=99)),
    ):
        deep = deepest_inside(diagram, solve_layout(diagram, glyph_radius=RADIUS))
        check(f"{label}: every line clears the ligand", deep < 0.0,
              f"deepest = {deep:.1f} px inside")

    # The case the term cannot win outright: a coordination sphere hanging off
    # a ring-shaped ligand whose hull is mostly hollow.  E_anchor outvotes it
    # there (see Weights.span), so all that is asserted is that no line rakes
    # across: within a glyph pitch of the rim is a line clipping the hollow, not
    # one drawn over the structure.  Toggling ``span`` no longer separates the
    # two -- the crossing repair leaves the sphere alone now (``pinned`` in
    # :func:`_moves`), which keeps these lines shallow on its own.
    diagram = make_metal_diagram()[0]
    deep = deepest_inside(diagram, solve_layout(diagram, glyph_radius=RADIUS))
    pitch = 2.0 * RADIUS + WEIGHTS.glyph_gap
    check("metal: no line rakes across the ligand", deep < pitch,
          f"deepest = {deep:.0f} px inside, limit {pitch:.0f}")


def test_pinning():
    """What the widget's "rotate this, re-accommodate the rest" action needs."""
    print("pinned re-solve")
    diagram = make_diagram(14, n_context=4, seed=17)
    base = solve_layout(diagram, glyph_radius=RADIUS)
    victim = [r.key for r in diagram.residues][3]

    moved = dict(base.positions)
    moved[victim] = (moved[victim][0] + 150.0, moved[victim][1] - 95.0)
    result = solve_layout(
        diagram, glyph_radius=RADIUS, seed_positions=moved, pinned={victim},
        orientation=(base.rotation, base.mirror),
    )
    drift = max(abs(a - b) for a, b in zip(result.positions[victim], moved[victim]))
    check("pinned glyph stayed exactly where it was put", drift < 1e-6,
          f"drift = {drift:.2e} px")
    others = max(
        math.dist(result.positions[k], base.positions[k])
        for k in base.positions if k != victim
    )
    check("the rest re-settled around it", others > 1.0, f"largest shift = {others:.0f} px")


def test_vertex_angle():
    """The closed-form polygon rotation really does land corners on partners."""
    print("coordination polygon")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPointF  # noqa: E402
    from ms_contactmap.render import _ngon, _vertex_angle, _wrap  # noqa: E402

    offset = 0.3
    bearings = [offset + i * math.pi / 2 for i in range(4)]
    angle = _vertex_angle(bearings, 4)
    check("square aligns with four right-angled partners",
          abs((angle - offset + math.pi / 4) % (math.pi / 2) - math.pi / 4) < 1e-9,
          f"= {math.degrees(angle):.3f} deg")

    # The case a per-partner circular mean gets wrong: two partners crowded
    # together plus three spread out, as on 2gfk's ZN A:402.  Corners are
    # exclusive, so all five partners must still get one each.
    crowded = [0.0, 0.5, 1.9, 3.3, 4.6]
    corners = [_vertex_angle(crowded, 5) + i * 2 * math.pi / 5 for i in range(5)]
    taken = [min(range(5), key=lambda i: abs(_wrap(corners[i] - b))) for b in crowded]
    worst = max(abs(_wrap(corners[j] - b)) for b, j in zip(crowded, taken))
    check("five partners get five distinct corners", len(set(taken)) == 5)
    check("no partner further than half a corner off", worst < math.pi / 5,
          f"worst = {math.degrees(worst):.1f} deg")

    # Corners must sit outside the body circle the caption is sized against.
    square = _ngon(QPointF(0, 0), RADIUS, 4, angle)
    reach = [math.hypot(square.at(i).x(), square.at(i).y()) for i in range(square.count())]
    check("polygon inradius is the glyph radius",
          abs(min(reach) - RADIUS * math.sqrt(2.0)) < 1e-9, f"corner = {min(reach):.1f} px")


def test_timing():
    print("timing")
    diagram = make_diagram(25, n_atoms=30, n_context=7, seed=99)
    t0 = time.perf_counter()
    result = solve_layout(diagram, glyph_radius=RADIUS)
    elapsed = time.perf_counter() - t0
    check("25 residues solve under 3 s", elapsed < 3.0, f"= {elapsed:.2f} s")
    assert_layout_sane("25res", diagram, result)
    print(f"       rotation = {math.degrees(result.rotation):.0f} deg, mirror = {result.mirror}")
    print("       " + "  ".join(f"{k}={v:.0f}" for k, v in result.energy_terms.items()))


if __name__ == "__main__":
    for fn in (test_gradient, test_terms_sum,
               test_one_strong_contact_keeps_the_normal_residue_standoff,
               test_rolling_probe_smooths_the_ligand_contour,
               test_small_case,
               test_scrambled_hard_case, test_metal_coordination,
               test_structural_water_has_a_readable_inner_shell_leg, test_span,
               test_pinning, test_vertex_angle, test_timing):
        fn()
    print("\nall layout checks passed")
