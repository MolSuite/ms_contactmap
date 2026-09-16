"""Legend layout checks.  Run with ``python tests/test_legend.py``."""
from __future__ import annotations

import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Qt must be loaded before RDKit in the same process (see render.py).
from PySide6.QtWidgets import QApplication  # noqa: F401
from rdkit import Chem
from rdkit.Chem import AllChem

from ms_contactmap.export import ensure_app
from ms_contactmap.model import Diagram, Interaction, Residue, ResidueRef
from ms_contactmap.render import build_scene


def sample_diagram() -> tuple[Diagram, dict[str, tuple[float, float]], list[tuple[float, float]]]:
    mol = Chem.MolFromSmiles("CCO")
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    ligand = [
        (conf.GetAtomPosition(i).x * 38.0, -conf.GetAtomPosition(i).y * 38.0)
        for i in range(mol.GetNumAtoms())
    ]
    names = ("ASP", "LYS", "PHE", "SER")
    residues = [Residue(ResidueRef("A", 10 + i, name), True) for i, name in enumerate(names)]
    positions = {
        residue.key: (150.0 * math.cos(i * math.pi / 2), 150.0 * math.sin(i * math.pi / 2))
        for i, residue in enumerate(residues)
    }
    interactions = [Interaction("hbond", residues[0].key, (2,), 2.9, False)]
    diagram = Diagram(
        "legend", "LIG", mol, ligand, residues, interactions,
        exposure={2: 1.0}, nearest_atom={residue.key: i % 3 for i, residue in enumerate(residues)},
    )
    return diagram, positions, ligand


def main() -> None:
    ensure_app()
    diagram, positions, ligand = sample_diagram()
    for side in ("left", "right"):
        build = build_scene(diagram, positions, ligand, legend_position=side)
        assert build.legend.column_count == 1
        key = build.legend.sceneBoundingRect()
        body = None
        for item in build.scene.items():
            if item is build.legend:
                continue
            body = item.sceneBoundingRect() if body is None else body.united(item.sceneBoundingRect())
        if side == "left":
            assert key.right() < body.left()
        else:
            assert key.left() > body.right()

    for side in ("top", "bottom"):
        for rows in (2, 3, 4):
            build = build_scene(
                diagram, positions, ligand, legend_position=side, legend_rows=rows
            )
            assert build.legend.row_count == rows
            key = build.legend.sceneBoundingRect()
            body = None
            for item in build.scene.items():
                if item is build.legend:
                    continue
                body = item.sceneBoundingRect() if body is None else body.united(item.sceneBoundingRect())
            if side == "top":
                assert key.bottom() < body.top()
            else:
                assert key.top() > body.bottom()
    print("legend checks passed")


if __name__ == "__main__":
    main()
