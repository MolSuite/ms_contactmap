"""Export checks: a fully vector SVG and a PNG that records its dpi."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage

from ms_contactmap.export import _target_rect, ensure_app, export_png, export_svg
from ms_contactmap.render import build_scene
from test_legend import sample_diagram


def test_svg_has_no_raster_and_png_keeps_its_physical_size(tmp_path):
    ensure_app()
    scene = build_scene(*sample_diagram()).scene
    svg = export_svg(scene, tmp_path / "d.svg").read_text()
    assert "<image" not in svg

    image = QImage(str(export_png(scene, tmp_path / "d.png", dpi=300)))
    assert round(image.dotsPerMeterX() * 0.0254) == 300
    assert abs(image.width() - _target_rect(scene).width() * 300 / 96) <= 1
