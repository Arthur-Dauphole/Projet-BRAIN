"""
ARC-AGI Geometric Detection System (legacy wrapper).

Ce module conserve l'API historique en réexportant les classes et utilitaires
depuis la nouvelle arborescence `src.arc_brain`.
"""

from src.arc_brain.core.models import BoundingBox, Direction, GeometricShape, Point
from src.arc_brain.perception.detectors import LineDetector, RectangleDetector, SymmetryDetector
from src.arc_brain.perception.engine import GeometricDetectionEngine
from src.arc_brain.perception.grid_utils import GridUtils
from src.arc_brain.perception.visualize import GeometricVisualizer

__all__ = [
    "BoundingBox",
    "Direction",
    "GeometricShape",
    "Point",
    "GridUtils",
    "RectangleDetector",
    "LineDetector",
    "SymmetryDetector",
    "GeometricDetectionEngine",
    "GeometricVisualizer",
]
