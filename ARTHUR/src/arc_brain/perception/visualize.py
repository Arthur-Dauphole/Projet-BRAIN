"""
Outils de visualisation pour le debug de la détection géométrique.
"""

from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle as MPLRect

from src.arc_brain.core.models import GeometricShape


class GeometricVisualizer:
    """Visualisation basique des grilles et formes détectées."""

    @staticmethod
    def plot_grid(grid: np.ndarray, title: str = "ARC Grid", figsize: Tuple[int, int] = (8, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(grid, cmap="tab10", interpolation="nearest")
        ax.set_title(title)
        ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.3)
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_shapes(grid: np.ndarray, shapes: List[GeometricShape], title: str = "Detected Shapes", figsize: Tuple[int, int] = (10, 10)):
        fig, ax = GeometricVisualizer.plot_grid(grid, title, figsize)

        for i, shape in enumerate(shapes):
            GeometricVisualizer._add_shape_overlay(ax, shape, i)

        return fig, ax

    @staticmethod
    def _add_shape_overlay(ax: plt.Axes, shape: GeometricShape, index: int):
        """Helper to add a bounding box and label for a shape to an axes."""
        bbox = shape.bounding_box
        rect = MPLRect(
            (bbox.min_x - 0.5, bbox.min_y - 0.5),
            bbox.width,
            bbox.height,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

        label = f"{shape.shape_type} #{index+1}"
        ax.text(
            bbox.min_x,
            bbox.min_y - 0.7,
            label,
            color="red",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    @staticmethod
    def plot_solver_debug(
        input_grid: np.ndarray,
        target_grid: np.ndarray,
        detected_shapes: List[GeometricShape],
        title: str = "ARC Solver Debug",
        figsize: Tuple[int, int] = (18, 6)
    ):
        """Generates a debug figure with Raw Input, Target Output, and Perception View."""
        fig, axs = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(title, fontsize=16)

        # 1. Raw Input
        axs[0].imshow(input_grid, cmap="tab10", interpolation="nearest")
        axs[0].set_title("Raw Input")
        axs[0].grid(True, which="both", color="gray", linewidth=0.5, alpha=0.3)

        # 2. Target Output
        axs[1].imshow(target_grid, cmap="tab10", interpolation="nearest")
        axs[1].set_title("Target Output")
        axs[1].grid(True, which="both", color="gray", linewidth=0.5, alpha=0.3)

        # 3. Perception View (Input + Bounding Boxes)
        axs[2].imshow(input_grid, cmap="tab10", interpolation="nearest")
        axs[2].set_title("Perception View")
        axs[2].grid(True, which="both", color="gray", linewidth=0.5, alpha=0.3)

        for i, shape in enumerate(detected_shapes):
            GeometricVisualizer._add_shape_overlay(axs[2], shape, i)

        for ax in axs:
            ax.set_xticks(np.arange(-0.5, input_grid.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, input_grid.shape[0], 1), minor=True)

        plt.tight_layout()
        return fig, axs

    @staticmethod
    def print_shape_info(shape: GeometricShape) -> None:
        """Affiche des infos détaillées pour une forme."""
        print(f"\n{'='*60}")
        print(f"Shape Type: {shape.shape_type.upper()}")
        print(f"{'='*60}")
        print(f"Color: {shape.color}")
        print(f"Number of pixels: {len(shape.pixels)}")
        print(f"Bounding box: ({shape.bounding_box.min_x}, {shape.bounding_box.min_y}) "
              f"to ({shape.bounding_box.max_x}, {shape.bounding_box.max_y})")
        print(f"Dimensions: {shape.bounding_box.width} x {shape.bounding_box.height}")
        print(f"Area: {shape.bounding_box.area}")
        print(f"Density: {shape.properties.get('density', 0):.2%}")

        print(f"\nProperties:")
        for key, value in shape.properties.items():
            if key not in ["border_pixels", "interior_pixels"]:
                print(f"  {key}: {value}")


__all__ = ["GeometricVisualizer"]

