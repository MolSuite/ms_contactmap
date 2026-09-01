# ms_contactmap

MolSuite Contact Map: computes protein–ligand contact maps and renders
2D interaction diagrams.

Part of the [MolSuite](https://molsuite.github.io/) stack.

## Install

```bash
pip install git+https://github.com/MolSuite/ms_contactmap
```

## Usage

```bash
ms_contactmap --help
ms_contactmap complex.pdb --ligand LIG --smiles "CCO" --json contacts.json
```

Reference structures and ligand catalogs in `data/` are development and test
fixtures; the installed package does not depend on them.

## License

MIT — see [LICENSE](LICENSE).
