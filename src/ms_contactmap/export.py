"""Headless PNG/SVG export of a diagram scene.

The same :class:`QGraphicsScene` feeds the interactive widget and these
functions, so what you export is exactly what you see.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtSvg import QSvgGenerator
from PySide6.QtSvgWidgets import QGraphicsSvgItem
from PySide6.QtWidgets import QApplication, QGraphicsItem, QGraphicsScene

#: Blank border kept around the drawing, in scene units.
EXPORT_MARGIN = 16.0


@contextmanager
def _uncached(scene: QGraphicsScene):
    """Draw ``scene`` without the pixmap caches the widget adds for dragging.

    Rendering through a ``DeviceCoordinateCache`` exports the cached bitmap
    instead of the item, which costs resolution in a scaled PNG and turns an
    SVG into embedded raster.  ``QGraphicsSvgItem`` -- the ligand -- is left
    alone: Qt caches it by default, so that is what the export has always
    drawn, and removing it here would silently change every reference PNG.
    """
    saved = [(item, item.cacheMode()) for item in scene.items()
             if not isinstance(item, QGraphicsSvgItem)]
    for item, _ in saved:
        item.setCacheMode(QGraphicsItem.CacheMode.NoCache)
    try:
        yield
    finally:
        for item, mode in saved:
            item.setCacheMode(mode)


def ensure_app() -> QApplication:
    """Return the running QApplication, creating an offscreen one if needed.

    Rendering a scene needs a QApplication even with no window on screen, so
    scripts and tests can call this instead of setting one up themselves.
    """
    app = QApplication.instance()
    if app is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QApplication([])
    return app


def _target_rect(scene: QGraphicsScene) -> QRectF:
    rect = scene.itemsBoundingRect()
    return rect.adjusted(-EXPORT_MARGIN, -EXPORT_MARGIN, EXPORT_MARGIN, EXPORT_MARGIN)


def export_png(
    scene: QGraphicsScene,
    path: str | Path,
    scale: float = 2.0,
    background: str | None = "#ffffff",
) -> Path:
    """Render ``scene`` to a PNG at ``scale`` times its scene units.

    ``background=None`` keeps the alpha channel transparent, which is how the
    Maestro reference images in ``data/`` are stored.
    """
    ensure_app()
    rect = _target_rect(scene)
    size = QSize(max(1, round(rect.width() * scale)), max(1, round(rect.height() * scale)))
    image = QImage(size, QImage.Format_ARGB32_Premultiplied)
    image.fill(QColor(background) if background else Qt.transparent)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    with _uncached(scene):
        scene.render(painter, QRectF(0, 0, size.width(), size.height()), rect)
    painter.end()

    path = Path(path)
    if not image.save(str(path)):
        raise OSError(f"could not write {path}")
    return path


def export_svg(scene: QGraphicsScene, path: str | Path, title: str = "") -> Path:
    """Render ``scene`` to a vector SVG at 1:1 scene units."""
    ensure_app()
    rect = _target_rect(scene)
    path = Path(path)

    generator = QSvgGenerator()
    generator.setFileName(str(path))
    generator.setSize(QSize(round(rect.width()), round(rect.height())))
    generator.setViewBox(QRectF(0, 0, rect.width(), rect.height()))
    if title:
        generator.setTitle(title)

    painter = QPainter(generator)
    painter.setRenderHint(QPainter.Antialiasing, True)
    with _uncached(scene):
        scene.render(painter, QRectF(0, 0, rect.width(), rect.height()), rect)
    painter.end()
    return path
