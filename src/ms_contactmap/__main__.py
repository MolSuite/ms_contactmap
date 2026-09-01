"""Command line entry point.

    python -m ms_contactmap data/4ps5.pdb --ligand 2TA --json out.json
    python -m ms_contactmap --from-json out.json --png out.png
    python -m ms_contactmap --from-json out.json --show

With no ``--smiles`` the code is looked up in ``data/ligands.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_LIGANDS = Path(__file__).resolve().parent.parent / "data" / "ligands.json"


def resolve_smiles(code: str, explicit: str | None, catalog_path: Path) -> str:
    if explicit:
        return explicit
    if not catalog_path.exists():
        raise SystemExit(
            f"no --smiles given and no catalog at {catalog_path}; "
            f"pass --smiles for ligand {code}"
        )
    catalog = json.loads(catalog_path.read_text())
    entry = catalog.get(code) or catalog.get(code.upper())
    if not entry:
        raise SystemExit(
            f"ligand {code} is not in {catalog_path}; pass --smiles explicitly"
        )
    return entry["smiles"] if isinstance(entry, dict) else entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ms_contactmap", description=__doc__)
    parser.add_argument("pdb", nargs="?", type=Path, help="PDB file of the complex")
    parser.add_argument("--from-json", type=Path, default=None,
                        help="visualize a previously computed diagram JSON")
    parser.add_argument("--ligand", help="ligand residue name, e.g. 2TA")
    parser.add_argument("--chain", default=None, help="restrict to this chain")
    parser.add_argument("--resnum", type=int, default=None, help="restrict to this residue number")
    parser.add_argument("--smiles", default=None, help="ligand SMILES (else from --ligands)")
    parser.add_argument("--ligands", type=Path, default=DEFAULT_LIGANDS, help="SMILES catalog")
    parser.add_argument("--no-exposure", action="store_true",
                        help="skip SASA halos for a faster first analysis")
    parser.add_argument("--json", dest="json_output", type=Path, default=None,
                        help="write reusable analysis and layout JSON here")
    parser.add_argument("--png", type=Path, default=None, help="write a PNG here")
    parser.add_argument("--svg", type=Path, default=None, help="write an SVG here")
    parser.add_argument("--scale", type=float, default=2.0, help="PNG scale factor")
    parser.add_argument("--transparent", action="store_true", help="PNG without white background")
    parser.add_argument("--show", action="store_true", help="open the interactive window")
    args = parser.parse_args(argv)

    if not args.png and not args.svg and not args.json_output and not args.show:
        parser.error("nothing to do: pass --json, --png, --svg or --show")
    if args.from_json and args.pdb:
        parser.error("choose either a PDB input or --from-json, not both")
    if not args.from_json and not args.pdb:
        parser.error("pass a PDB input or --from-json")
    if not args.from_json and not args.ligand:
        parser.error("--ligand is required with a PDB input")

    from .export import ensure_app
    app = ensure_app()
    from .widget import InteractionDiagramWidget

    if args.from_json:
        widget = InteractionDiagramWidget.from_json(args.from_json)
        diagram = widget.diagram
    else:
        from .interactions import build_diagram

        smiles = resolve_smiles(args.ligand, args.smiles, args.ligands)
        diagram = build_diagram(
            args.pdb,
            args.ligand,
            smiles,
            chain=args.chain,
            resnum=args.resnum,
            name=args.pdb.stem,
            compute_exposure=not args.no_exposure,
        )
        # Batch exports need a solved scene immediately.  A show-only command
        # displays the window first and solves in the responsive Qt thread.
        widget = InteractionDiagramWidget(
            diagram if (args.png or args.svg or args.json_output) else None
        )
    assert diagram is not None
    print(
        f"{diagram.name}: {diagram.ligand_name}, "
        f"{len(diagram.residues)} residues, {len(diagram.interactions)} interactions",
        file=sys.stderr,
    )

    if args.json_output:
        widget.export_json(args.json_output)
        print(f"wrote {args.json_output}", file=sys.stderr)
    if args.png:
        widget.export_png(args.png, scale=args.scale, background=None if args.transparent else "#ffffff")
        print(f"wrote {args.png}", file=sys.stderr)
    if args.svg:
        widget.export_svg(args.svg)
        print(f"wrote {args.svg}", file=sys.stderr)
    if args.show:
        widget.resize(1100, 780)
        widget.setWindowTitle(f"{diagram.name} - {diagram.ligand_name}")
        widget.show()
        if widget.layout_result is None:
            widget.set_diagram_async(diagram)
        return app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
