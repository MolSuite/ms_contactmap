"""Maestro-style 2D protein-ligand interaction diagrams for PySide6.

Public objects are imported lazily to reduce startup time and keep analysis,
layout and Qt drawing independently usable.
"""
from __future__ import annotations

from importlib import import_module

_PUBLIC = {
    "Diagram": ("model", "Diagram"),
    "DiagramView": ("widget", "DiagramView"),
    "Interaction": ("model", "Interaction"),
    "InteractionDiagramWidget": ("widget", "InteractionDiagramWidget"),
    "LayoutResult": ("layout", "LayoutResult"),
    "LigandGeometry": ("chem", "LigandGeometry"),
    "Residue": ("model", "Residue"),
    "ResidueRef": ("model", "ResidueRef"),
    "build_diagram": ("interactions", "build_diagram"),
    "build_scene": ("render", "build_scene"),
    "document_from_dict": ("json_io", "document_from_dict"),
    "document_to_dict": ("json_io", "document_to_dict"),
    "export_png": ("export", "export_png"),
    "export_svg": ("export", "export_svg"),
    "load_json": ("json_io", "load_json"),
    "load_ligand": ("chem", "load_ligand"),
    "save_json": ("json_io", "save_json"),
    "solve_layout": ("layout", "solve_layout"),
}

__all__ = list(_PUBLIC)


def __getattr__(name: str):
    try:
        module_name, attribute = _PUBLIC[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value
