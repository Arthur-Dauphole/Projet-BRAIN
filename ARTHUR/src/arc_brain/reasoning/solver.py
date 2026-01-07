"""
Orchestrateur haut niveau du moteur symbolique ARC-AGI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.arc_brain.core.models import GeometricShape, Point
from src.arc_brain.perception.grid_utils import GridUtils
from src.arc_brain.reasoning.actions import ActionGenerator
from src.arc_brain.reasoning.search import SolverResult, SymbolicSolver
from src.arc_brain.reasoning.state import GridState


@dataclass
class ArcReasoningEngine:
    """
    Moteur de haut niveau pour raisonner sur les transformations ARC.

    Il encapsule :
    - la construction d’états `GridState` à partir de formes,
    - la configuration du générateur d’actions,
    - l’appel au `SymbolicSolver`.
    """

    max_depth: int = 10
    max_nodes: int = 10_000
    timeout: float = 60.0

    def solve_shapes(
        self,
        initial_shapes: List[GeometricShape],
        goal_shapes: List[GeometricShape],
        grid_size: tuple[int, int],
        background_color: int = 0,
        verbose: bool = False,
    ) -> SolverResult:
        """
        Résout une transformation entre deux listes de formes sur une même grille.
        """
        initial_state = GridState(
            shapes=initial_shapes,
            grid_size=grid_size,
            background_color=background_color,
        )
        goal_state = GridState(
            shapes=goal_shapes,
            grid_size=grid_size,
            background_color=background_color,
        )

        action_gen = ActionGenerator()
        solver = SymbolicSolver(
            action_generator=action_gen,
            max_depth=self.max_depth,
            max_nodes=self.max_nodes,
            timeout=self.timeout,
        )

        return solver.solve(initial_state=initial_state, goal_state=goal_state, verbose=verbose)


def demo_simple_move() -> None:
    """
    Démonstration : déplacer un rectangle de 2 cases vers la droite.
    """
    print("\n" + "=" * 70)
    print("DEMO: Simple Move Transformation")
    print("=" * 70 + "\n")

    initial_pixels = {Point(1, 1), Point(2, 1), Point(1, 2), Point(2, 2)}
    initial_bbox = GridUtils.compute_bounding_box(initial_pixels)
    initial_shape = GeometricShape(
        shape_type="rectangle",
        pixels=initial_pixels,
        color=1,
        bounding_box=initial_bbox,
        properties={"width": 2, "height": 2},
    )

    initial_state = GridState(shapes=[initial_shape], grid_size=(8, 8), background_color=0)

    goal_pixels = {Point(3, 1), Point(4, 1), Point(3, 2), Point(4, 2)}
    goal_bbox = GridUtils.compute_bounding_box(goal_pixels)
    goal_shape = GeometricShape(
        shape_type="rectangle",
        pixels=goal_pixels,
        color=1,
        bounding_box=goal_bbox,
        properties={"width": 2, "height": 2},
    )

    goal_state = GridState(shapes=[goal_shape], grid_size=(8, 8), background_color=0)

    print("Initial grid:")
    print(initial_state.to_grid())
    print("\nGoal grid:")
    print(goal_state.to_grid())

    action_gen = ActionGenerator(
        max_move_distance=3,
        allow_rotations=False,
        allow_flips=False,
        allow_scales=False,
    )
    solver = SymbolicSolver(
        action_generator=action_gen,
        max_depth=3,
        max_nodes=1000,
        timeout=10.0,
    )

    result = solver.solve(initial_state, goal_state, verbose=True)

    print("\n" + "=" * 70)
    if result.success:
        print("✓ SOLUTION FOUND!")
        print(f"\nAction sequence ({len(result.actions)} steps):")
        for i, action in enumerate(result.actions, 1):
            print(f"  {i}. {action}")
        if result.final_state:
            print("\nFinal grid:")
            print(result.final_state.to_grid())
    else:
        print("✗ NO SOLUTION FOUND")
        print(f"Nodes explored: {result.nodes_explored}")
        print(f"Time: {result.time_elapsed:.2f}s")
    print("=" * 70)


def demo_color_change() -> None:
    """
    Démonstration : changement de couleur d’un rectangle.
    """
    print("\n" + "=" * 70)
    print("DEMO: Color Change Transformation")
    print("=" * 70 + "\n")

    pixels = {Point(2, 2), Point(3, 2), Point(2, 3), Point(3, 3)}
    bbox = GridUtils.compute_bounding_box(pixels)

    initial_shape = GeometricShape(
        shape_type="rectangle",
        pixels=pixels,
        color=1,
        bounding_box=bbox,
        properties={},
    )
    initial_state = GridState(shapes=[initial_shape], grid_size=(6, 6))

    goal_shape = GeometricShape(
        shape_type="rectangle",
        pixels=pixels.copy(),
        color=3,
        bounding_box=bbox,
        properties={},
    )
    goal_state = GridState(shapes=[goal_shape], grid_size=(6, 6))

    print("Initial (color=1):")
    print(initial_state.to_grid())
    print("\nGoal (color=3):")
    print(goal_state.to_grid())

    action_gen = ActionGenerator(
        max_move_distance=0,
        allow_colors=[1, 2, 3],
        allow_rotations=False,
    )
    solver = SymbolicSolver(action_gen, max_depth=2, max_nodes=500)
    result = solver.solve(initial_state, goal_state, verbose=True)

    if result.success and result.actions:
        print(f"\n✓ Solution: {result.actions[0]}")


__all__ = ["ArcReasoningEngine", "demo_simple_move", "demo_color_change"]

