# ms_contactmap

`ms_contactmap` detects protein-ligand interactions from a PDB complex and
renders an editable two-dimensional contact map. Analysis, layout, JSON
serialization, Qt presentation, and image export are separate layers, so the
same result can be used in a batch process or an interactive application.

## Install

```bash
pip install ms_contactmap
```

The package requires Python 3.12, RDKit, Biopython, SciPy, NumPy, and PySide6.

## Command line

Open the diagram of a complex; the ligand is found on its own:

```bash
ms_contactmap complex.pdb
```

`--gui` chooses the panels, comma-separated: `diagram` (the default), `3d` (the
complex in PyMOL with the contacts drawn, needs `pip install "ms_contactmap[3d]"`)
and `table` (one row per contact). Selecting a residue in the diagram or a row
in the table shows it in 3D.

```bash
ms_contactmap complex.pdb --gui diagram,3d,table
```

Write files instead with `-o`, once per file; the suffix picks the format
(`.json`, `.png`, `.svg`):

```bash
ms_contactmap complex.pdb --smiles "CCO" -o contacts.json -o contacts.png
```

The ligand is the one small-molecule HETATM residue that is not water, a metal,
a buffer, a glycan or a lipid. When there are several, `--ligand`, `--chain` and
`--resnum` narrow the search; without them the window asks, and a batch run
draws the first copy of a single ligand or stops on different ligands.
Protein-protein, peptide and nucleic-acid interfaces are not supported yet.

`--smiles` supplies the bond orders for the ligand's HETATM block. When the same
pose exists as `.sdf`, `.mol` or `.mol2`, pass `--ligand-file pose.sdf` instead:
its bonds and hydrogens are used as they are, and its atoms are matched to the
HETATM records by position. With neither, bonds are perceived from geometry
and left single; the legend then warns that the bond orders, and the chemistry
derived from them, are not reliable.

Open a saved analysis without detecting the interactions again:

```bash
ms_contactmap contacts.json
```

Use `--no-exposure` for a faster first pass without solvent-exposure halos.

## Python entry point

```python
from ms_contactmap import build_diagram

diagram = build_diagram(
    "complex.pdb",
    "LIG",
    "CCO",
    chain="A",
    resnum=401,
)
print(len(diagram.interactions))
```

See the [Python API guide](api.md), [complete examples](examples.md), and
[generated reference](api_reference.md).

## License

`ms_contactmap` is released under the MIT License.
