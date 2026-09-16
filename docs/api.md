# Python API

The package root exposes the supported high-level API through lazy imports.

## Analysis pipeline

`build_diagram()` is the normal entry point. It selects the requested ligand,
detects interactions, computes pocket context and exposure, and returns a
`Diagram` independent of Qt rendering.

```python
from ms_contactmap import build_diagram

diagram = build_diagram(
    pdb_path="complex.pdb",
    resname="LIG",
    smiles="CCO",
    name="screening hit 17",
    chain="A",
    resnum=401,
    compute_exposure=True,
)

for interaction in diagram.interactions:
    print(interaction.kind, interaction.residue_key, interaction.distance)
```

Pass `ligand=` instead of `smiles=` to take the chemistry from the pose itself:
a `.sdf`/`.mol`/`.mol2` path or an RDKit `Mol` with a conformer. Its heavy atoms
must sit on the HETATM records (within 0.05 Å); a different pose raises
`ValueError`.

For a docked pose there is no complex file to hand over: pass the receptor and
the pose, and the complex is assembled internally.

```python
from ms_contactmap import build_pose_diagram

diagram = build_pose_diagram("receptor.pdbqt", "pose.sdf")  # or an RDKit Mol
```

The receptor may be `.pdb`, `.pdbqt` (AutoDock types are mapped to elements) or
`.cif`. A `.pdb` pose has no bond orders: pass `smiles=` for them, or the
legend warns that the chemistry is not reliable (the same happens with
`build_diagram` when neither `smiles` nor `ligand` is given).

Set `compute_exposure=False` when solvent accessibility is unnecessary. This
does not change interaction detection.

## Layout and persistence

Analysis can be serialized before or after solving its layout:

```python
from ms_contactmap import load_json, save_json, solve_layout

layout = solve_layout(diagram)
save_json("hit-17.json", diagram, layout)

restored_diagram, restored_layout, view = load_json("hit-17.json")
```

The JSON document includes a schema version. Prefer `save_json()` and
`load_json()` over depending on its internal dictionary layout.

## Qt widget

```python
from PySide6.QtWidgets import QApplication
from ms_contactmap import InteractionDiagramWidget

app = QApplication([])
widget = InteractionDiagramWidget(diagram)
widget.resize(1100, 780)
widget.show()
app.exec()
```

`set_diagram_async()` solves larger layouts in the Qt thread pool and emits
`layoutStarted`, `layoutFinished`, or `layoutFailed`. Use the synchronous
constructor for batch export and the asynchronous method for a responsive UI.

## Low-level detection

`ms_contactmap.detect.detect_interactions()` is available for integrations that
already own a `LigandGeometry`. Most callers should use `build_diagram()` so
residue context, exposure, and metadata remain consistent.
