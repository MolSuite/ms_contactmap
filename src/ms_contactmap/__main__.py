"""Command line entry point.

    ms_contactmap complex.pdb                         # finds the ligand, opens the diagram
    ms_contactmap complex.pdb --gui diagram,3d,table  # diagram, PyMOL and the interaction table
    ms_contactmap complex.pdb -o contacts.png -o contacts.json
    ms_contactmap complex.pdb --ligand STI --chain B --smiles "..." -o out.svg
    ms_contactmap contacts.json                       # reopen a saved diagram

The ligand is found on its own when the complex holds one small molecule;
--ligand/--chain/--resnum narrow the search when it holds several.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

OUTPUTS = (".json", ".png", ".svg")


class LigandError(ValueError):
    pass


def _use_own_settings(widget, *, apply: bool) -> None:
    """Standalone, the window is its own host: it applies and saves MS-ContactMap's settings.

    An embedding app (AMDockVS) never comes through here; it edits the same file from
    its settings dialog.  Batch exports ignore it too, so scripts stay reproducible.
    """
    from .settings import load_view_settings, save_view_settings

    if apply:
        try:
            widget.apply_view_settings(load_view_settings())
        except ValueError:  # retired values: start from the built-in look
            pass
    widget.set_configure_action("Save as default", lambda: save_view_settings(widget.view_settings()))


def find_site(pdb: Path, ligand=None, chain=None, resnum=None, *, ask=None):
    """The ligand to draw as ``(resname, chain, resnum)``.

    One candidate is taken as is.  Several are resolved by ``ask(sites)`` when
    there is a window to ask in; in batch, copies of one ligand resolve to the
    first and different ligands are an error.  No small molecule at all
    (protein-protein, peptide, nucleic acid) is not supported yet.
    """
    from .chem import find_ligands, read_pdb_atoms

    atoms = read_pdb_atoms(pdb)
    sites = find_ligands(atoms, ligand, chain, resnum)
    if not sites and ligand:  # named by the user: trust it even if it looks like an additive
        return ligand, chain, resnum
    if not sites:
        chains = sorted({a.chain for a in atoms if not a.hetatm})
        raise LigandError(
            f"no small-molecule ligand found in {pdb.name} ({len(chains)} chain(s): {', '.join(chains)}). "
            "Protein-protein, peptide and nucleic-acid interfaces are not supported yet; "
            "pass --ligand RESNAME if the ligand is there under another name.")
    if len(sites) > 1:
        if ask is not None:
            site = ask(sites)
            if site is None:
                raise LigandError("no ligand chosen")
            return site.resname, site.chain, site.resnum
        listing = "\n  ".join(str(s) for s in sites)
        if len({s.resname for s in sites}) > 1:
            raise LigandError(f"several ligands in {pdb.name}; choose one with --ligand/--chain/--resnum:"
                              f"\n  {listing}")
        print(f"ms_contactmap: {len(sites)} copies of {sites[0].resname}, drawing {sites[0]}; "
              "--chain/--resnum picks another", file=sys.stderr)
    site = sites[0]
    return site.resname, site.chain, site.resnum


def _panels(value: str) -> list[str]:
    from .gui import PANELS

    panels = [p.strip().lower() for p in value.split(",") if p.strip()]
    unknown = [p for p in panels if p not in PANELS]
    if unknown or not panels:
        raise argparse.ArgumentTypeError(f"panels are {', '.join(PANELS)}; got {value!r}")
    return panels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ms_contactmap", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="PDB file of the complex, or a saved diagram .json")
    parser.add_argument("-o", "--output", type=Path, action="append", default=[],
                        help="write .json, .png or .svg (by suffix); repeat for several")
    parser.add_argument("--gui", type=_panels, nargs="?", const=["diagram"], default=None,
                        metavar="PANELS",
                        help="open the window with these comma-separated panels: diagram, 3d, table "
                             "(default: diagram; the default action when no --output is given)")
    pick = parser.add_argument_group("ligand (found automatically; these narrow the search)")
    pick.add_argument("--ligand", help="residue name, e.g. STI")
    pick.add_argument("--chain", default=None)
    pick.add_argument("--resnum", type=int, default=None)
    chemistry = parser.add_mutually_exclusive_group()
    chemistry.add_argument("--smiles", default=None,
                           help="ligand SMILES, for bond orders; without it or --ligand-file "
                                "the diagram is flagged as unreliable")
    chemistry.add_argument("--ligand-file", type=Path, default=None,
                           help="the same pose as .sdf/.mol/.mol2; its bonds are used as they are")
    parser.add_argument("--no-exposure", action="store_true",
                        help="skip SASA halos for a faster first analysis")
    image = parser.add_argument_group("PNG")
    image.add_argument("--scale", type=float, default=2.0, help="scale factor")
    image.add_argument("--dpi", type=float, default=None,
                       help="resolution at the SVG's physical size; overrides --scale")
    image.add_argument("--transparent", action="store_true", help="no white background")
    args = parser.parse_args(argv)

    for path in args.output:
        if path.suffix.lower() not in OUTPUTS:
            parser.error(f"{path}: output must end in {', '.join(OUTPUTS)}")
    panels = args.gui if args.gui is not None else ([] if args.output else ["diagram"])
    from_json = args.input.suffix.lower() == ".json"

    from .export import ensure_app
    app = ensure_app()
    from .widget import InteractionDiagramWidget

    site = None
    if from_json:
        widget = InteractionDiagramWidget.from_json(args.input)
        diagram = widget.diagram
    else:
        from .interactions import build_diagram

        ask = None
        if panels:
            from .gui import pick_ligand
            ask = pick_ligand
        try:
            site = find_site(args.input, args.ligand, args.chain, args.resnum, ask=ask)
        except LigandError as error:
            if panels:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(None, "MS-ContactMap", str(error))
            print(f"ms_contactmap: {error}", file=sys.stderr)
            return 2
        diagram = build_diagram(
            args.input, site[0], args.smiles, chain=site[1], resnum=site[2], name=args.input.stem,
            compute_exposure=not args.no_exposure, ligand=args.ligand_file,
        )
        # Batch exports need a solved scene immediately.  A window-only command
        # shows first and solves in the responsive Qt thread.
        widget = InteractionDiagramWidget(diagram if args.output else None)
    assert diagram is not None
    print(f"{diagram.name}: {diagram.ligand_name}, "
          f"{len(diagram.residues)} residues, {len(diagram.interactions)} interactions", file=sys.stderr)

    for path in args.output:
        suffix = path.suffix.lower()
        if suffix == ".json":
            widget.export_json(path)
        elif suffix == ".png":
            widget.export_png(path, scale=args.scale, dpi=args.dpi,
                              background=None if args.transparent else "#ffffff")
        else:
            widget.export_svg(path)
        print(f"wrote {path}", file=sys.stderr)
    if not panels:
        return 0

    from .gui import ContactMapWindow

    selection = None
    if site is not None:
        selection = f"resn {site[0]} and resi {site[2]}" if site[2] is not None else f"resn {site[0]}"
        if site[1] and site[1] != "_":
            selection += f" and chain {site[1]}"
    _use_own_settings(widget, apply=not from_json)  # a saved document keeps its own look
    window = ContactMapWindow(diagram, panels, widget=widget, pdb=None if from_json else args.input,
                              ligand_selection=selection, ligand_file=args.ligand_file)
    window.resize(1100 if panels == ["diagram"] else 1500, 780)
    window.setWindowTitle(f"{diagram.name} - {diagram.ligand_name}")
    window.show()
    if "diagram" in panels and widget.layout_result is None:
        widget.set_diagram_async(diagram)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
