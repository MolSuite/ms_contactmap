# ms_contactmap

`ms_contactmap` detects protein-ligand interactions from a PDB complex and
renders an editable two-dimensional contact map. Analysis, layout, JSON
serialization, Qt presentation, and image export are separate layers, so the
same result can be used in a batch process or an interactive application.

## Install

```bash
pip install "ms_contactmap @ git+https://github.com/MolSuite/ms_contactmap"
```

The package requires Python 3.12, RDKit, Biopython, SciPy, NumPy, and PySide6.

## Command line

Build a diagram directly from a complex:

```bash
ms_contactmap complex.pdb \
  --ligand LIG \
  --smiles "CCO" \
  --json contacts.json \
  --png contacts.png
```

Open a saved analysis without detecting the interactions again:

```bash
ms_contactmap --from-json contacts.json --show
```

Use `--chain` and `--resnum` when the PDB contains more than one copy of the
ligand. Use `--no-exposure` for a faster first pass without solvent-exposure
halos.

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
