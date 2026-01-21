"""
Moteur de détection géométrique ARC-AGI.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.arc_brain.core.models import GeometricShape
from src.arc_brain.perception.detectors import LineDetector, RectangleDetector, SymmetryDetector


class GeometricDetectionEngine:
    """Interface haut niveau pour détecter les formes dans une grille ARC."""

    def __init__(self, background_color: int = 0):
        self.background_color = background_color
        self.detectors = {
            "rectangles": RectangleDetector(),
            "lines": LineDetector(),
        }

    def detect_all_shapes(self, grid: np.ndarray) -> Dict[str, List[GeometricShape]]:
        """Lance tous les détecteurs et résout les ambiguïtés simples."""
        detected_rectangles = self.detectors["rectangles"].detect_rectangles(grid, self.background_color)
        detected_lines = self.detectors["lines"].detect_lines(grid, self.background_color)

        line_pixel_sets = {frozenset(line.pixels) for line in detected_lines}
        final_rectangles = [rect for rect in detected_rectangles if frozenset(rect.pixels) not in line_pixel_sets]

        results = {"rectangles": final_rectangles, "lines": detected_lines}

        for shapes in results.values():
            for shape in shapes:
                symmetries = SymmetryDetector.detect_symmetries(shape)
                shape.properties.update(symmetries)

        return results

    def analyze_grid(self, grid: np.ndarray, verbose: bool = True) -> Dict:
        """Analyse complète d'une grille ARC."""
        results = self.detect_all_shapes(grid)
        total_shapes = sum(len(shapes) for shapes in results.values())

        analysis = {
            "grid_shape": grid.shape,
            "detected_shapes": results,
            "statistics": {
                "total_shapes": total_shapes,
                "rectangles": len(results["rectangles"]),
                "lines": len(results["lines"]),
            },
        }

        if verbose:
            print(f"\n{'='*60}")
            print("GRID ANALYSIS")
            print(f"{'='*60}")
            print(f"Grid size: {grid.shape[1]} x {grid.shape[0]}")
            print(f"Total shapes detected: {total_shapes}")
            print(f"  - Rectangles: {len(results['rectangles'])}")
            print(f"  - Lines: {len(results['lines'])}")

            for shape_type, shapes in results.items():
                if shapes:
                    print(f"\n{shape_type.upper()}:")
                    for i, shape in enumerate(shapes):
                        print(f"\n  Shape {i+1}:")
                        print(f"    Color: {shape.color}")
                        print(f"    Size: {shape.bounding_box.width}x{shape.bounding_box.height}")
                        print(f"    Density: {shape.properties.get('density', 0):.2%}")
                        if "vertical_symmetry" in shape.properties:
                            print(f"    Vertical symmetry: {shape.properties['vertical_symmetry']}")
                        if "horizontal_symmetry" in shape.properties:
                            print(f"    Horizontal symmetry: {shape.properties['horizontal_symmetry']}")

        return analysis


__all__ = ["GeometricDetectionEngine"]

