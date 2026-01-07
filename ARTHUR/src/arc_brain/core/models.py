"""
Structures de base pour la perception géométrique ARC-AGI.

Contient les types partagés (points, bounding boxes, formes géométriques)
utilisés par les modules de perception, de raisonnement et de pont LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set

import numpy as np


class Direction(Enum):
    """Directions cardinales et intercardinales pour le raisonnement spatial."""

    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)
    NORTHEAST = (1, -1)
    NORTHWEST = (-1, -1)
    SOUTHEAST = (1, 1)
    SOUTHWEST = (-1, 1)

    @classmethod
    def cardinal(cls) -> list["Direction"]:
        """Retourne uniquement les directions cardinales (N, S, E, O)."""
        return [cls.NORTH, cls.SOUTH, cls.EAST, cls.WEST]

    @classmethod
    def all_directions(cls) -> list["Direction"]:
        """Retourne les 8 directions."""
        return list(cls)


@dataclass(frozen=True)
class Point:
    """Point 2D en coordonnées de grille."""

    x: int
    y: int

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def to_tuple(self) -> tuple[int, int]:
        return self.x, self.y

    def manhattan_distance(self, other: "Point") -> int:
        """Distance de Manhattan vers un autre point."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def euclidean_distance(self, other: "Point") -> float:
        """Distance euclidienne vers un autre point."""
        return float(np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2))


@dataclass
class BoundingBox:
    """Boîte englobante axis-alignée."""

    min_x: int
    min_y: int
    max_x: int
    max_y: int

    @property
    def width(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def height(self) -> int:
        return self.max_y - self.min_y + 1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return Point((self.min_x + self.max_x) // 2, (self.min_y + self.max_y) // 2)

    def contains(self, point: Point) -> bool:
        """Vérifie si un point est dans la boîte."""
        return self.min_x <= point.x <= self.max_x and self.min_y <= point.y <= self.max_y

    def corners(self) -> list[Point]:
        """Retourne les quatre coins de la boîte."""
        return [
            Point(self.min_x, self.min_y),
            Point(self.max_x, self.min_y),
            Point(self.max_x, self.max_y),
            Point(self.min_x, self.max_y),
        ]


@dataclass
class GeometricShape:
    """
    Forme géométrique détectée et ses propriétés.

    Combine les pixels bruts et les attributs sémantiques extraits.
    """

    shape_type: str  # 'rectangle', 'line', 'blob', etc.
    pixels: Set[Point]
    color: int
    bounding_box: BoundingBox
    properties: Dict

    def __post_init__(self) -> None:
        """Complète les propriétés dérivées après initialisation."""
        if "area" not in self.properties:
            self.properties["area"] = len(self.pixels)
        if "density" not in self.properties:
            self.properties["density"] = len(self.pixels) / self.bounding_box.area

    def is_filled(self, threshold: float = 0.9) -> bool:
        """Retourne True si la densité dépasse le seuil (forme remplie)."""
        return self.properties.get("density", 0.0) >= threshold

    def is_hollow(self, threshold: float = 0.5) -> bool:
        """Retourne True si la densité est faible (forme creuse)."""
        return self.properties.get("density", 1.0) <= threshold


__all__ = [
    "Direction",
    "Point",
    "BoundingBox",
    "GeometricShape",
]
