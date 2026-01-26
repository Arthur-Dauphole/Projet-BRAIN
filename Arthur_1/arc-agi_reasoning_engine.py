"""
ARC-AGI Reasoning Engine (legacy wrapper).

Ce module conserve l’API historique en réexportant les classes du nouveau
package `arc_brain.reasoning`.
"""

from src.arc_brain.reasoning.actions import (
    ActionType,
    Action,
    MoveAction,
    RotateAction,
    FlipAction,
    ChangeColorAction,
    ScaleAction,
    ActionGenerator,
)
from src.arc_brain.reasoning.state import GridState, DistanceMetric
from src.arc_brain.reasoning.search import SearchNode, SolverResult, SymbolicSolver
from src.arc_brain.reasoning.solver import ArcReasoningEngine, demo_simple_move, demo_color_change

__all__ = [
    "ActionType",
    "Action",
    "MoveAction",
    "RotateAction",
    "FlipAction",
    "ChangeColorAction",
    "ScaleAction",
    "ActionGenerator",
    "GridState",
    "DistanceMetric",
    "SearchNode",
    "SolverResult",
    "SymbolicSolver",
    "ArcReasoningEngine",
    "demo_simple_move",
    "demo_color_change",
]
