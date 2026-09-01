"""In-process detection of protein-ligand interactions.

The package's only interaction detector, written from the published criteria
rather than ported from any implementation -- see :mod:`ms_contactmap.detect.roles`
for why that distinction is the point.

    from ms_contactmap.detect import detect_interactions
    interactions, metal_legs, coordination, skipped = detect_interactions(
        pdb_path, geom, serial_to_atom)
"""
from .engine import detect_interactions
from .roles import Group, Hit, Perception, Site

__all__ = ["detect_interactions", "Group", "Hit", "Perception", "Site"]
