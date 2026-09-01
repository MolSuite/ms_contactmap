"""Stable JSON document for analysis, layout and visualization.

The document is deliberately self-contained.  It stores the typed ligand as a
Mol block, the native interaction report, solvent exposure and (when present)
the solved layout.  A viewer therefore needs neither the source PDB nor a
detector to reproduce the diagram.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdkit import Chem

from .detect.geometry import reference_parameters
from .layout import LayoutResult
from .model import Diagram, Interaction, MetalLeg, Residue, ResidueRef

SCHEMA_NAME = "ms_contactmap.interaction-diagram"
SCHEMA_VERSION = 1


def _points(values) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in values]


def _read_points(values, field: str) -> list[tuple[float, float]]:
    try:
        return [(float(point[0]), float(point[1])) for point in values]
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"invalid 2D coordinates in {field}") from exc


def diagram_to_dict(diagram: Diagram) -> dict[str, Any]:
    """Serialize detector output without requiring a solved layout."""
    mol = Chem.Mol(diagram.mol)
    for conformer in mol.GetConformers():
        conformer.Set3D(False)
    return {
        "name": diagram.name,
        "ligand": {
            "name": diagram.ligand_name,
            "mol_block": Chem.MolToMolBlock(mol),
            "coordinates_2d": _points(diagram.coords_2d),
            "alternative_coordinates_2d": [
                _points(view) for view in diagram.coords_alt
            ],
        },
        "residues": [
            {
                "key": residue.key,
                "chain": residue.ref.chain,
                "number": residue.ref.number,
                "name": residue.ref.name,
                "insertion": residue.ref.insertion,
                "has_interactions": residue.has_interactions,
                "center_3d": list(residue.center_3d)
                if residue.center_3d is not None else None,
            }
            for residue in diagram.residues
        ],
        "interactions": [
            {
                "type": row.kind,
                "residue": row.residue_key,
                "ligand_atoms": list(row.ligand_atoms),
                "distance_angstrom": float(row.distance),
                "ligand_is_donor": bool(row.ligand_is_donor),
                "via_water": row.via_water,
                "protein_atom": row.protein_atom,
                "angle_degree": float(row.angle) if row.angle is not None else None,
                "protein_distance_angstrom": (
                    float(row.protein_distance)
                    if row.protein_distance is not None else None
                ),
                "protein_is_donor": row.protein_is_donor,
            }
            for row in diagram.interactions
        ],
        "exposure": {str(atom): float(value) for atom, value in diagram.exposure.items()},
        "nearest_ligand_atom": {
            key: int(atom) for key, atom in diagram.nearest_atom.items()
        },
        "metal_legs": [
            {
                "metal": leg.metal_key,
                "partner": leg.partner_key,
                "distance_angstrom": float(leg.distance),
            }
            for leg in diagram.metal_legs
        ],
        "metal_coordination": {
            key: int(number) for key, number in diagram.metal_coordination.items()
        },
        "metadata": dict(diagram.metadata),
    }


def diagram_from_dict(data: dict[str, Any]) -> Diagram:
    """Rebuild the render model from a serialized detector result."""
    ligand = data.get("ligand", {})
    mol = Chem.MolFromMolBlock(
        str(ligand.get("mol_block", "")), sanitize=True, removeHs=False
    )
    if mol is None:
        raise ValueError("JSON contains an invalid ligand mol_block")

    residues = []
    for row in data.get("residues", []):
        center = row.get("center_3d")
        ref = ResidueRef(
            str(row["chain"]), int(row["number"]), str(row["name"]),
            str(row.get("insertion", "")),
        )
        if row.get("key") not in (None, ref.key):
            raise ValueError(f"residue key does not match its fields: {row.get('key')}")
        residues.append(Residue(
            ref=ref,
            has_interactions=bool(row.get("has_interactions", False)),
            center_3d=tuple(float(value) for value in center) if center is not None else None,
        ))

    interactions = [
        Interaction(
            kind=str(row["type"]),
            residue_key=str(row["residue"]),
            ligand_atoms=tuple(int(atom) for atom in row.get("ligand_atoms", [])),
            distance=float(row.get("distance_angstrom", 0.0)),
            ligand_is_donor=bool(row.get("ligand_is_donor", True)),
            via_water=row.get("via_water"),
            protein_atom=row.get("protein_atom"),
            angle=(float(row["angle_degree"])
                   if row.get("angle_degree") is not None else None),
            protein_distance=(float(row["protein_distance_angstrom"])
                              if row.get("protein_distance_angstrom") is not None
                              else None),
            protein_is_donor=row.get("protein_is_donor"),
        )
        for row in data.get("interactions", [])
    ]
    coords = _read_points(ligand.get("coordinates_2d", []), "ligand.coordinates_2d")
    if len(coords) != mol.GetNumAtoms():
        raise ValueError(
            "ligand.coordinates_2d atom count does not match the mol_block "
            f"({len(coords)} != {mol.GetNumAtoms()})"
        )
    return Diagram(
        name=str(data.get("name", "Interaction diagram")),
        ligand_name=str(ligand.get("name", "LIG")),
        mol=mol,
        coords_2d=coords,
        coords_alt=[
            _read_points(view, "ligand.alternative_coordinates_2d")
            for view in ligand.get("alternative_coordinates_2d", [])
        ],
        residues=residues,
        interactions=interactions,
        exposure={int(atom): float(value) for atom, value in data.get("exposure", {}).items()},
        nearest_atom={
            str(key): int(atom) for key, atom in data.get("nearest_ligand_atom", {}).items()
        },
        metal_legs=[
            MetalLeg(str(row["metal"]), str(row["partner"]),
                     float(row.get("distance_angstrom", 0.0)))
            for row in data.get("metal_legs", [])
        ],
        metal_coordination={
            str(key): int(value) for key, value in data.get("metal_coordination", {}).items()
        },
        metadata=dict(data.get("metadata", {})),
    )


def layout_to_dict(layout: LayoutResult) -> dict[str, Any]:
    return {
        "positions": {
            key: [float(point[0]), float(point[1])]
            for key, point in layout.positions.items()
        },
        "ligand_coordinates": _points(layout.ligand_coords),
        "rotation_radians": float(layout.rotation),
        "mirrored": bool(layout.mirror),
        "projection": int(layout.projection),
        "energy": float(layout.energy),
        "energy_terms": {
            key: float(value) for key, value in layout.energy_terms.items()
        },
        "crossings": int(layout.crossings),
    }


def layout_from_dict(data: dict[str, Any], diagram: Diagram) -> LayoutResult:
    positions = {
        str(key): (float(point[0]), float(point[1]))
        for key, point in data.get("positions", {}).items()
    }
    expected = {residue.key for residue in diagram.residues}
    if set(positions) != expected:
        missing = sorted(expected - set(positions))
        extra = sorted(set(positions) - expected)
        raise ValueError(f"layout positions do not match residues; missing={missing}, extra={extra}")
    ligand_coords = _read_points(data.get("ligand_coordinates", []), "layout.ligand_coordinates")
    if len(ligand_coords) != diagram.mol.GetNumAtoms():
        raise ValueError("layout.ligand_coordinates atom count does not match ligand")
    return LayoutResult(
        positions=positions,
        ligand_coords=ligand_coords,
        rotation=float(data.get("rotation_radians", 0.0)),
        mirror=bool(data.get("mirrored", False)),
        projection=int(data.get("projection", 0)),
        energy=float(data.get("energy", 0.0)),
        energy_terms={
            str(key): float(value) for key, value in data.get("energy_terms", {}).items()
        },
        crossings=int(data.get("crossings", 0)),
    )


def document_to_dict(
    diagram: Diagram,
    layout: LayoutResult | None = None,
    *,
    view: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a self-contained, versioned JSON-compatible document."""
    detector = str(diagram.metadata.get("detector", "unknown"))
    return {
        "schema": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "analysis": {
            "detector": detector,
            "parameters": reference_parameters() if detector == "native" else {},
        },
        "diagram": diagram_to_dict(diagram),
        "layout": layout_to_dict(layout) if layout is not None else None,
        "view": dict(view or {}),
    }


def document_from_dict(
    document: dict[str, Any],
) -> tuple[Diagram, LayoutResult | None, dict[str, Any]]:
    if document.get("schema") != SCHEMA_NAME:
        raise ValueError(f"unsupported JSON schema: {document.get('schema')!r}")
    if int(document.get("version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported JSON version: {document.get('version')!r}")
    diagram = diagram_from_dict(document.get("diagram", {}))
    raw_layout = document.get("layout")
    layout = layout_from_dict(raw_layout, diagram) if raw_layout is not None else None
    return diagram, layout, dict(document.get("view", {}))


def save_json(
    path: str | Path,
    diagram: Diagram,
    layout: LayoutResult | None = None,
    *,
    view: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.write_text(
        json.dumps(document_to_dict(diagram, layout, view=view), indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def load_json(path: str | Path) -> tuple[Diagram, LayoutResult | None, dict[str, Any]]:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("interaction-diagram JSON root must be an object")
    return document_from_dict(document)
