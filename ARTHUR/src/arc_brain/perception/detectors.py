"""
Détecteurs de formes géométriques pour ARC-AGI.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.arc_brain.core.models import BoundingBox, GeometricShape, Point
from src.arc_brain.perception.grid_utils import GridUtils


class RectangleDetector:
    """Détecteur de rectangles (pleins ou creux)."""

    @staticmethod
    def is_rectangle(pixels: Set[Point], bbox: BoundingBox, tolerance: float = 0.05) -> bool:
        """Vérifie si les pixels forment un rectangle."""
        expected_area = bbox.area
        actual_area = len(pixels)

        if expected_area == 0:
            return False

        # Rectangle plein : aire proche de la bounding box
        if abs(actual_area - expected_area) / expected_area <= tolerance:
            return True

        # Rectangle creux : comparer au périmètre
        border_area = 2 * (bbox.width + bbox.height) - 4
        if border_area > 0 and abs(actual_area - border_area) / border_area <= tolerance:
            return True

        return False

    @staticmethod
    def detect_rectangles(grid: np.ndarray, background_color: int = 0) -> List[GeometricShape]:
        """Détecte tous les rectangles dans la grille."""
        components = GridUtils.extract_connected_components(grid, background_color)
        rectangles: List[GeometricShape] = []

        for component in components:
            if len(component) < 4:
                continue

            bbox = GridUtils.compute_bounding_box(component)

            if RectangleDetector.is_rectangle(component, bbox):
                sample_point = next(iter(component))
                color = grid[sample_point.y, sample_point.x]

                properties: Dict = {
                    "width": bbox.width,
                    "height": bbox.height,
                    "aspect_ratio": bbox.width / bbox.height if bbox.height else 0,
                    "is_square": abs(bbox.width - bbox.height) <= 1,
                    "perimeter": 2 * (bbox.width + bbox.height),
                }

                border = GridUtils.get_border_pixels(component, grid.shape)
                properties["is_filled"] = len(component) > len(border)
                properties["border_pixels"] = border
                properties["interior_pixels"] = component - border

                rectangles.append(
                    GeometricShape(
                        shape_type="rectangle",
                        pixels=component,
                        color=color,
                        bounding_box=bbox,
                        properties=properties,
                    )
                )

        return rectangles


class LineDetector:
    """Détecteur de segments (horizontaux, verticaux, diagonaux)."""

    @staticmethod
    def is_line(pixels: Set[Point], tolerance: float = 0.1) -> Tuple[bool, Optional[str]]:
        """Vérifie si les pixels forment une ligne."""
        if len(pixels) < 2:
            return False, None

        points_list = list(pixels)
        xs = {p.x for p in points_list}
        ys = {p.y for p in points_list}

        if len(ys) == 1:
            return True, "horizontal"

        if len(xs) == 1:
            return True, "vertical"

        if len(xs) == len(pixels) and len(ys) == len(pixels):
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            if x_range == y_range:
                return True, "diagonal"

        return False, None

    @staticmethod
    def detect_lines(grid: np.ndarray, background_color: int = 0, min_length: int = 2) -> List[GeometricShape]:
        """Détecte les segments dans la grille."""
        components = GridUtils.extract_connected_components(grid, background_color)
        lines: List[GeometricShape] = []

        for component in components:
            if len(component) < min_length:
                continue

            is_line, direction = LineDetector.is_line(component)

            if is_line and direction:
                sample_point = next(iter(component))
                color = grid[sample_point.y, sample_point.x]
                bbox = GridUtils.compute_bounding_box(component)

                points_list = list(component)
                if direction == "horizontal":
                    endpoints = [
                        min(points_list, key=lambda p: p.x),
                        max(points_list, key=lambda p: p.x),
                    ]
                elif direction == "vertical":
                    endpoints = [
                        min(points_list, key=lambda p: p.y),
                        max(points_list, key=lambda p: p.y),
                    ]
                else:  # diagonal
                    endpoints = [
                        min(points_list, key=lambda p: (p.x + p.y)),
                        max(points_list, key=lambda p: (p.x + p.y)),
                    ]

                properties: Dict = {
                    "direction": direction,
                    "length": len(component),
                    "endpoints": endpoints,
                    "thickness": 1,
                }

                lines.append(
                    GeometricShape(
                        shape_type="line",
                        pixels=component,
                        color=color,
                        bounding_box=bbox,
                        properties=properties,
                    )
                )

        return lines


class SymmetryDetector:
    """Détecteur de symétries (verticale, horizontale)."""

    @staticmethod
    def has_vertical_symmetry(pixels: Set[Point], tolerance: int = 0) -> bool:
        if not pixels:
            return False

        bbox = GridUtils.compute_bounding_box(pixels)
        center_x = (bbox.min_x + bbox.max_x) / 2

        for pixel in pixels:
            mirror_x = int(2 * center_x - pixel.x)
            mirror = Point(mirror_x, pixel.y)

            if mirror not in pixels:
                found_close = any(Point(mirror_x + dx, pixel.y) in pixels for dx in range(-tolerance, tolerance + 1))
                if not found_close:
                    return False

        return True

    @staticmethod
    def has_horizontal_symmetry(pixels: Set[Point], tolerance: int = 0) -> bool:
        if not pixels:
            return False

        bbox = GridUtils.compute_bounding_box(pixels)
        center_y = (bbox.min_y + bbox.max_y) / 2

        for pixel in pixels:
            mirror_y = int(2 * center_y - pixel.y)
            mirror = Point(pixel.x, mirror_y)

            if mirror not in pixels:
                found_close = any(Point(pixel.x, mirror_y + dy) in pixels for dy in range(-tolerance, tolerance + 1))
                if not found_close:
                    return False

        return True

    @staticmethod
    def detect_symmetries(shape: GeometricShape) -> Dict[str, bool]:
        """Retourne un dict des symétries détectées pour une forme."""
        return {
            "vertical_symmetry": SymmetryDetector.has_vertical_symmetry(shape.pixels),
            "horizontal_symmetry": SymmetryDetector.has_horizontal_symmetry(shape.pixels),
        }


__all__ = ["RectangleDetector", "LineDetector", "SymmetryDetector"]

