# ms_contactmap

MolSuite Contact Map: computes protein–ligand contact maps and renders
2D interaction diagrams.

Part of the [MolSuite](https://molsuite.github.io/) stack.

## Install

```bash
pip install ms_contactmap
```

## Usage

```bash
ms_contactmap --help
ms_contactmap complex.pdb                          # finds the ligand, opens the diagram
ms_contactmap complex.pdb --gui diagram,3d,table   # plus PyMOL and the interaction table
ms_contactmap complex.pdb --smiles "CCO" -o contacts.json -o contacts.png
```

The `3d` panel needs `pip install "ms_contactmap[3d]"`.

Reference structures and ligand catalogs in `data/` are development and test
fixtures; the installed package does not depend on them.

## License

MIT — see [LICENSE](LICENSE).
