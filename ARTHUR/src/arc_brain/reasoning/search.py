"""
Structures de nœuds et moteur de recherche symbolique (A*).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Callable
import heapq
import time

from src.arc_brain.reasoning.actions import Action, ActionGenerator
from src.arc_brain.reasoning.state import GridState, DistanceMetric, HeuristicFn


@dataclass
class SearchNode:
    """Nœud dans l’arbre de recherche."""

    state: GridState
    parent: Optional["SearchNode"]
    action: Optional[Action]
    g_cost: float  # coût depuis la racine
    h_cost: float  # heuristique vers le but
    depth: int

    @property
    def f_cost(self) -> float:
        """Coût total estimé (A*)."""
        return self.g_cost + self.h_cost

    def __lt__(self, other: "SearchNode") -> bool:  # pragma: no cover - ordre PQ
        return self.f_cost < other.f_cost

    def get_action_sequence(self) -> List[Action]:
        """Reconstruit la séquence d’actions depuis la racine."""
        actions: List[Action] = []
        node: Optional["SearchNode"] = self

        while node and node.parent is not None:
            if node.action:
                actions.append(node.action)
            node = node.parent

        return list(reversed(actions))


@dataclass
class SolverResult:
    """Résultat d’une recherche."""

    success: bool
    actions: List[Action]
    final_state: Optional[GridState]
    nodes_explored: int
    time_elapsed: float
    distance_to_goal: float


class SymbolicSolver:
    """
    Solveur A* pour trouver une séquence d’actions transformant un état initial en état but.
    """

    def __init__(
        self,
        action_generator: ActionGenerator,
        distance_metric: Optional[HeuristicFn] = None,
        max_depth: int = 10,
        max_nodes: int = 10000,
        timeout: float = 60.0,
    ) -> None:
        self.action_generator = action_generator
        self.distance_metric: HeuristicFn = distance_metric or DistanceMetric.combined_distance
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.timeout = timeout

    def solve(self, initial_state: GridState, goal_state: GridState, verbose: bool = False) -> SolverResult:
        """
        Lance la recherche A* entre `initial_state` et `goal_state`.
        """
        start_time = time.time()

        initial_h = self.distance_metric(initial_state, goal_state)
        start_node = SearchNode(
            state=initial_state,
            parent=None,
            action=None,
            g_cost=0.0,
            h_cost=initial_h,
            depth=0,
        )

        open_set: List[SearchNode] = [start_node]
        closed_set: Set[GridState] = set()
        nodes_explored = 0

        if verbose:
            print("Starting A* search...")
            print(f"Initial distance to goal: {initial_h:.2f}")
            print(f"Max depth: {self.max_depth}, Max nodes: {self.max_nodes}")

        while open_set and nodes_explored < self.max_nodes:
            if time.time() - start_time > self.timeout:
                if verbose:
                    print("TIMEOUT reached!")
                return SolverResult(
                    success=False,
                    actions=[],
                    final_state=None,
                    nodes_explored=nodes_explored,
                    time_elapsed=time.time() - start_time,
                    distance_to_goal=float("inf"),
                )

            current_node = heapq.heappop(open_set)
            nodes_explored += 1

            if verbose and nodes_explored % 100 == 0:
                print(
                    f"Explored: {nodes_explored}, "
                    f"Queue: {len(open_set)}, "
                    f"Best f-cost: {current_node.f_cost:.2f}, "
                    f"Depth: {current_node.depth}"
                )

            distance = self.distance_metric(current_node.state, goal_state)
            if distance < 0.01:
                if verbose:
                    print("\n✓ SOLUTION FOUND!")
                    print(f"Depth: {current_node.depth}")
                    print(f"Nodes explored: {nodes_explored}")
                    print(f"Time: {time.time() - start_time:.2f}s")

                return SolverResult(
                    success=True,
                    actions=current_node.get_action_sequence(),
                    final_state=current_node.state,
                    nodes_explored=nodes_explored,
                    time_elapsed=time.time() - start_time,
                    distance_to_goal=distance,
                )

            if current_node.state in closed_set:
                continue

            closed_set.add(current_node.state)

            if current_node.depth >= self.max_depth:
                continue

            if current_node.state.shapes:
                shape = current_node.state.shapes[0]
                actions = self.action_generator.generate_actions(shape)

                for action in actions:
                    new_shape = action.apply(shape)
                    new_shapes = [new_shape] + current_node.state.shapes[1:]
                    new_state = GridState(
                        shapes=new_shapes,
                        grid_size=current_node.state.grid_size,
                        background_color=current_node.state.background_color,
                    )

                    if not new_state.is_valid():
                        continue

                    if new_state in closed_set:
                        continue

                    new_g = current_node.g_cost + 1.0
                    new_h = self.distance_metric(new_state, goal_state)

                    child_node = SearchNode(
                        state=new_state,
                        parent=current_node,
                        action=action,
                        g_cost=new_g,
                        h_cost=new_h,
                        depth=current_node.depth + 1,
                    )

                    heapq.heappush(open_set, child_node)

        if verbose:
            print("\n✗ No solution found within limits")

        return SolverResult(
            success=False,
            actions=[],
            final_state=None,
            nodes_explored=nodes_explored,
            time_elapsed=time.time() - start_time,
            distance_to_goal=float("inf"),
        )


__all__ = ["SearchNode", "SolverResult", "SymbolicSolver"]

