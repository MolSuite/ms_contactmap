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
