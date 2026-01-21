"""
Utilitaires de grille pour l'analyse géométrique ARC-AGI.
"""

from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np

from src.arc_brain.core.models import BoundingBox, Direction, Point


class GridUtils:
    """Fonctions utilitaires pour la manipulation et l'analyse de grilles."""

    @staticmethod
    def get_neighbors(point: Point, grid_shape: Tuple[int, int], connectivity: int = 4) -> List[Point]:
        """
        Retourne les voisins valides d'un point dans la grille.
        """
        height, width = grid_shape
        directions = Direction.cardinal() if connectivity == 4 else Direction.all_directions()

        neighbors: List[Point] = []
        for direction in directions:
            dx, dy = direction.value
            new_point = Point(point.x + dx, point.y + dy)

            if 0 <= new_point.x < width and 0 <= new_point.y < height:
                neighbors.append(new_point)

        return neighbors

    @staticmethod
    def flood_fill(grid: np.ndarray, start: Point, target_color: int, connectivity: int = 4) -> Set[Point]:
        """
        Flood fill pour extraire une composante connectée.
        """
        if grid[start.y, start.x] != target_color:
            return set()

        visited: Set[Point] = set()
        stack = [start]
        component: Set[Point] = set()

        while stack:
            point = stack.pop()

            if point in visited:
                continue

            visited.add(point)

            if grid[point.y, point.x] == target_color:
                component.add(point)
                neighbors = GridUtils.get_neighbors(point, grid.shape, connectivity)
                stack.extend(neighbors)

        return component

    @staticmethod
    def extract_connected_components(
        grid: np.ndarray, background_color: int = 0, connectivity: int = 8
    ) -> List[Set[Point]]:
        """
        Extrait toutes les composantes connectées d'une grille (hors fond).
        """
        visited = np.zeros_like(grid, dtype=bool)
        components: List[Set[Point]] = []
        height, width = grid.shape

        for y in range(height):
            for x in range(width):
                point = Point(x, y)

                if visited[y, x] or grid[y, x] == background_color:
                    continue

                color = grid[y, x]
                component = GridUtils.flood_fill(grid, point, color, connectivity)

                for p in component:
                    visited[p.y, p.x] = True

                components.append(component)

        return components

    @staticmethod
    def compute_bounding_box(pixels: Set[Point]) -> BoundingBox:
        """Calcule la bounding box axis-alignée pour un ensemble de pixels."""
        if not pixels:
            return BoundingBox(0, 0, 0, 0)

        xs = [p.x for p in pixels]
        ys = [p.y for p in pixels]

        return BoundingBox(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))

    @staticmethod
    def get_border_pixels(pixels: Set[Point], grid_shape: Tuple[int, int]) -> Set[Point]:
        """
        Retourne les pixels du bord (au moins un voisin hors de la forme).
        """
        border: Set[Point] = set()

        for pixel in pixels:
            neighbors = GridUtils.get_neighbors(pixel, grid_shape, connectivity=4)

            if any(n not in pixels for n in neighbors):
                border.add(pixel)

        return border

    @staticmethod
    def get_interior_pixels(pixels: Set[Point], grid_shape: Tuple[int, int]) -> Set[Point]:
        """Retourne les pixels intérieurs (tous les voisins dans la forme)."""
        border = GridUtils.get_border_pixels(pixels, grid_shape)
        return pixels - border


__all__ = ["GridUtils"]

