"""
Actions atomiques de transformation de formes géométriques (DSL ARC-AGI).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from src.arc_brain.core.models import BoundingBox, GeometricShape, Point
from src.arc_brain.perception.grid_utils import GridUtils


class ActionType(Enum):
    """Types d’actions atomiques."""

    MOVE = "move"
    ROTATE = "rotate"
    SCALE = "scale"
    CHANGE_COLOR = "change_color"
    FLIP = "flip"
    COPY = "copy"
    DELETE = "delete"
    FILL = "fill"
    HOLLOW = "hollow"


class Action(ABC):
    """
    Base abstraite pour toutes les transformations.

    Chaque action représente une opération élémentaire (idéalement réversible)
    appliquée à une `GeometricShape`.
    """

    @abstractmethod
    def apply(self, shape: GeometricShape) -> GeometricShape:
        """Applique la transformation et retourne une nouvelle forme."""

    @abstractmethod
    def to_dict(self) -> Dict:
        """Sérialise l’action pour log / replay."""

    @abstractmethod
    def __repr__(self) -> str:  # pragma: no cover - représentation simple
        """Représentation lisible pour le debug."""

    @property
    @abstractmethod
    def action_type(self) -> ActionType:
        """Type symbolique de cette action."""


@dataclass
class MoveAction(Action):
    """Déplacement d’une forme par un vecteur (dx, dy)."""

    dx: int
    dy: int

    @property
    def action_type(self) -> ActionType:
        return ActionType.MOVE

    def apply(self, shape: GeometricShape) -> GeometricShape:
        new_pixels = {Point(p.x + self.dx, p.y + self.dy) for p in shape.pixels}
        new_bbox = BoundingBox(
            shape.bounding_box.min_x + self.dx,
            shape.bounding_box.min_y + self.dy,
            shape.bounding_box.max_x + self.dx,
            shape.bounding_box.max_y + self.dy,
        )
        return GeometricShape(
            shape_type=shape.shape_type,
            pixels=new_pixels,
            color=shape.color,
            bounding_box=new_bbox,
            properties=deepcopy(shape.properties),
        )

    def to_dict(self) -> Dict:
        return {"type": "move", "dx": self.dx, "dy": self.dy}

    def __repr__(self) -> str:
        return f"Move(dx={self.dx}, dy={self.dy})"


@dataclass
class RotateAction(Action):
    """Rotation de 90, 180 ou 270 degrés autour du centre de la forme."""

    angle: int

    def __post_init__(self) -> None:
        if self.angle not in (90, 180, 270):
            raise ValueError("Angle must be 90, 180, or 270 degrees")

    @property
    def action_type(self) -> ActionType:
        return ActionType.ROTATE

    def apply(self, shape: GeometricShape) -> GeometricShape:
        center = shape.bounding_box.center
        new_pixels = set()
        rotations = self.angle // 90

        for pixel in shape.pixels:
            x, y = pixel.x - center.x, pixel.y - center.y
            for _ in range(rotations):
                x, y = -y, x
            new_pixels.add(Point(x + center.x, y + center.y))

        new_bbox = GridUtils.compute_bounding_box(new_pixels)
        return GeometricShape(
            shape_type=shape.shape_type,
            pixels=new_pixels,
            color=shape.color,
            bounding_box=new_bbox,
            properties=deepcopy(shape.properties),
        )

    def to_dict(self) -> Dict:
        return {"type": "rotate", "angle": self.angle}

    def __repr__(self) -> str:
        return f"Rotate({self.angle}°)"


@dataclass
class FlipAction(Action):
    """Symétrie horizontale ou verticale d’une forme."""

    axis: str  # 'horizontal' ou 'vertical'

    def __post_init__(self) -> None:
        if self.axis not in ("horizontal", "vertical"):
            raise ValueError("Axis must be 'horizontal' or 'vertical'")

    @property
    def action_type(self) -> ActionType:
        return ActionType.FLIP

    def apply(self, shape: GeometricShape) -> GeometricShape:
        bbox = shape.bounding_box
        new_pixels = set()

        if self.axis == "horizontal":
            center_y = (bbox.min_y + bbox.max_y) / 2
            for pixel in shape.pixels:
                new_y = int(2 * center_y - pixel.y)
                new_pixels.add(Point(pixel.x, new_y))
        else:
            center_x = (bbox.min_x + bbox.max_x) / 2
            for pixel in shape.pixels:
                new_x = int(2 * center_x - pixel.x)
                new_pixels.add(Point(new_x, pixel.y))

        new_bbox = GridUtils.compute_bounding_box(new_pixels)
        return GeometricShape(
            shape_type=shape.shape_type,
            pixels=new_pixels,
            color=shape.color,
            bounding_box=new_bbox,
            properties=deepcopy(shape.properties),
        )

    def to_dict(self) -> Dict:
        return {"type": "flip", "axis": self.axis}

    def __repr__(self) -> str:
        return f"Flip({self.axis})"


@dataclass
class ChangeColorAction(Action):
    """Changement de couleur d’une forme."""

    new_color: int

    @property
    def action_type(self) -> ActionType:
        return ActionType.CHANGE_COLOR

    def apply(self, shape: GeometricShape) -> GeometricShape:
        return GeometricShape(
            shape_type=shape.shape_type,
            pixels=shape.pixels.copy(),
            color=self.new_color,
            bounding_box=shape.bounding_box,
            properties=deepcopy(shape.properties),
        )

    def to_dict(self) -> Dict:
        return {"type": "change_color", "new_color": self.new_color}

    def __repr__(self) -> str:
        return f"ChangeColor({self.new_color})"


@dataclass
class ScaleAction(Action):
    """Mise à l’échelle (facteurs entiers) d’une forme depuis son coin haut-gauche."""

    scale_x: int
    scale_y: int

    def __post_init__(self) -> None:
        if self.scale_x < 1 or self.scale_y < 1:
            raise ValueError("Scale factors must be >= 1")

    @property
    def action_type(self) -> ActionType:
        return ActionType.SCALE

    def apply(self, shape: GeometricShape) -> GeometricShape:
        bbox = shape.bounding_box
        anchor = Point(bbox.min_x, bbox.min_y)
        new_pixels = set()

        for pixel in shape.pixels:
            rel_x = pixel.x - anchor.x
            rel_y = pixel.y - anchor.y
            for dx in range(self.scale_x):
                for dy in range(self.scale_y):
                    new_x = anchor.x + rel_x * self.scale_x + dx
                    new_y = anchor.y + rel_y * self.scale_y + dy
                    new_pixels.add(Point(new_x, new_y))

        new_bbox = GridUtils.compute_bounding_box(new_pixels)
        return GeometricShape(
            shape_type=shape.shape_type,
            pixels=new_pixels,
            color=shape.color,
            bounding_box=new_bbox,
            properties=deepcopy(shape.properties),
        )

    def to_dict(self) -> Dict:
        return {"type": "scale", "scale_x": self.scale_x, "scale_y": self.scale_y}

    def __repr__(self) -> str:
        return f"Scale({self.scale_x}x, {self.scale_y}y)"


class ActionGenerator:
    """
    Générateur d’actions possibles pour un état.

    Définit essentiellement le facteur de branchement de l’arbre de recherche.
    """

    def __init__(
        self,
        max_move_distance: int = 5,
        allow_colors: Optional[List[int]] = None,
        allow_rotations: bool = True,
        allow_flips: bool = True,
        allow_scales: bool = False,
    ) -> None:
        self.max_move_distance = max_move_distance
        self.allow_colors = allow_colors if allow_colors is not None else list(range(10))
        self.allow_rotations = allow_rotations
        self.allow_flips = allow_flips
        self.allow_scales = allow_scales

    def generate_actions(self, shape: GeometricShape) -> List[Action]:
        """Génère toutes les actions valides pour une forme donnée."""
        actions: List[Action] = []

        for dx in range(-self.max_move_distance, self.max_move_distance + 1):
            for dy in range(-self.max_move_distance, self.max_move_distance + 1):
                if dx != 0 or dy != 0:
                    actions.append(MoveAction(dx, dy))

        for color in self.allow_colors:
            if color != shape.color:
                actions.append(ChangeColorAction(color))

        if self.allow_rotations:
            actions.extend([RotateAction(90), RotateAction(180), RotateAction(270)])

        if self.allow_flips:
            actions.extend([FlipAction("horizontal"), FlipAction("vertical")])

        if self.allow_scales:
            actions.extend([ScaleAction(2, 1), ScaleAction(1, 2), ScaleAction(2, 2)])

        return actions


__all__ = [
    "ActionType",
    "Action",
    "MoveAction",
    "RotateAction",
    "FlipAction",
    "ChangeColorAction",
    "ScaleAction",
    "ActionGenerator",
]

