# Examples

## Batch JSON, PNG, and SVG export

```python
from PySide6.QtWidgets import QApplication

from ms_contactmap import InteractionDiagramWidget, build_diagram

app = QApplication.instance() or QApplication([])
diagram = build_diagram(
    "complex.pdb",
    "LIG",
    "CCO",
    chain="A",
    resnum=401,
)
widget = InteractionDiagramWidget(diagram)
widget.export_json("contacts.json")
widget.export_png("contacts.png", scale=2.0)
widget.export_svg("contacts.svg")
```

The widget owns the solved scene used by the exporters. A Qt event loop does
not need to be started for this synchronous batch case, but a `QApplication`
must exist.

## Reopen and edit a saved diagram

```python
from PySide6.QtWidgets import QApplication
from ms_contactmap import InteractionDiagramWidget

app = QApplication([])
widget = InteractionDiagramWidget.from_json("contacts.json")
widget.resize(1100, 780)
widget.show()
app.exec()
```

The interactive widget supports moving residues, rotating or mirroring the
ligand, recalculating the layout, changing the legend position, and exporting
the edited result.
