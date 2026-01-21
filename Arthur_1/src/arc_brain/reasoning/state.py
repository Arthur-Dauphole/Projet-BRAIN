"""
Représentation d’état de grille et métriques de distance pour la recherche.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple

import numpy as np

from src.arc_brain.core.models import GeometricShape


@dataclass
class GridState:
    """
    État complet d’une grille : liste de formes + taille + couleur de fond.
    """

    shapes: List[GeometricShape]
    grid_size: Tuple[int, int]  # (width, height)
    background_color: int = 0

    def to_grid(self) -> np.ndarray:
        """Convertit l’état abstrait en grille de pixels numpy."""
        height, width = self.grid_size[1], self.grid_size[0]
        grid = np.full((height, width), self.background_color, dtype=int)

        for shape in self.shapes:
            for pixel in shape.pixels:
                if 0 <= pixel.x < width and 0 <= pixel.y < height:
                    grid[pixel.y, pixel.x] = shape.color

        return grid

    def is_valid(self) -> bool:
        """Vérifie que toutes les formes sont dans les bornes de la grille."""
        width, height = self.grid_size

        for shape in self.shapes:
            for pixel in shape.pixels:
                if not (0 <= pixel.x < width and 0 <= pixel.y < height):
                    return False
        return True

    def __hash__(self) -> int:
        """Hash de l’état pour utilisation dans un set (visités)."""
        return hash(
            tuple(
                (
                    shape.color,
                    shape.bounding_box.min_x,
                    shape.bounding_box.min_y,
                    shape.bounding_box.max_x,
                    shape.bounding_box.max_y,
                    len(shape.pixels),
                )
                for shape in sorted(
                    self.shapes,
                    key=lambda s: (s.bounding_box.min_x, s.bounding_box.min_y),
                )
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridState):
            return False
        return hash(self) == hash(other)


class DistanceMetric:
    """
    Métriques de distance entre deux états (heuristiques pour A*).
    """

    @staticmethod
    def pixel_based_distance(current: GridState, goal: GridState) -> float:
        """
        Distance pixel-level : proportion de pixels différents.
        """
        current_grid = current.to_grid()
        goal_grid = goal.to_grid()

        if current_grid.shape != goal_grid.shape:
            return float("inf")

        mismatches = np.sum(current_grid != goal_grid)
        total_pixels = current_grid.size
        return mismatches / total_pixels

    @staticmethod
    def shape_based_distance(current: GridState, goal: GridState) -> float:
        """
        Distance shape-level : compare propriétés de formes (couleur, position, taille, type).
        """
        if len(current.shapes) != len(goal.shapes):
            return 1000.0 + abs(len(current.shapes) - len(goal.shapes)) * 100

        current_shapes = sorted(
            current.shapes, key=lambda s: (s.bounding_box.min_x, s.bounding_box.min_y)
        )
        goal_shapes = sorted(
            goal.shapes, key=lambda s: (s.bounding_box.min_x, s.bounding_box.min_y)
        )

        total_distance = 0.0

        for curr_shape, goal_shape in zip(current_shapes, goal_shapes):
            if curr_shape.color != goal_shape.color:
                total_distance += 10.0

            curr_center = curr_shape.bounding_box.center
            goal_center = goal_shape.bounding_box.center
            position_dist = curr_center.manhattan_distance(goal_center)
            total_distance += position_dist

            size_diff = abs(len(curr_shape.pixels) - len(goal_shape.pixels))
            total_distance += size_diff * 0.1

            if curr_shape.shape_type != goal_shape.shape_type:
                total_distance += 5.0

        return total_distance

    @staticmethod
    def combined_distance(
        current: GridState,
        goal: GridState,
        shape_weight: float = 0.7,
    ) -> float:
        """
        Combine distance shape-level et pixel-level.
        """
        shape_dist = DistanceMetric.shape_based_distance(current, goal)
        pixel_dist = DistanceMetric.pixel_based_distance(current, goal)
        return shape_weight * shape_dist + (1 - shape_weight) * pixel_dist


HeuristicFn = Callable[[GridState, GridState], float]


__all__ = ["GridState", "DistanceMetric", "HeuristicFn"]

