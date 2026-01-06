"""
ARC-AGI Reasoning Engine
========================

A symbolic solver using state-space search to find transformation sequences
that solve ARC-AGI tasks by manipulating detected geometric shapes.

Author: BRAIN Project Team
Date: 2025
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import heapq
from copy import deepcopy
import time

# Import from the perception module
from arc_geometric_detection import (
    GeometricShape, Point, BoundingBox, GridUtils, Direction
)


# ============================================================================
# 1. DSL (DOMAIN SPECIFIC LANGUAGE) - ACTION SPACE
# ============================================================================

class ActionType(Enum):
    """Types of atomic transformations."""
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
    Abstract base class for all atomic transformations.
    
    Each action represents a single, reversible operation on a shape.
    """
    
    @abstractmethod
    def apply(self, shape: GeometricShape) -> GeometricShape:
        """
        Apply transformation to a shape and return a new shape.
        
        Args:
            shape: Input shape to transform
            
        Returns:
            New transformed shape (original is not modified)
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict:
        """Serialize action to dictionary for logging/replay."""
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """Human-readable representation."""
        pass
    
    @property
    @abstractmethod
    def action_type(self) -> ActionType:
        """Return the type of this action."""
        pass


@dataclass
class MoveAction(Action):
    """Move a shape by a displacement vector."""
    dx: int
    dy: int
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.MOVE
    
    def apply(self, shape: GeometricShape) -> GeometricShape:
        """Move all pixels of the shape."""
        new_pixels = {Point(p.x + self.dx, p.y + self.dy) for p in shape.pixels}
        
        new_bbox = BoundingBox(
            shape.bounding_box.min_x + self.dx,
            shape.bounding_box.min_y + self.dy,
            shape.bounding_box.max_x + self.dx,
            shape.bounding_box.max_y + self.dy
        )
        
        return GeometricShape(
            shape_type=shape.shape_type,
            pixels=new_pixels,
            color=shape.color,
            bounding_box=new_bbox,
            properties=deepcopy(shape.properties)
        )
    
    def to_dict(self) -> Dict:
        return {'type': 'move', 'dx': self.dx, 'dy': self.dy}
    
    def __repr__(self) -> str:
        return f"Move(dx={self.dx}, dy={self.dy})"


@dataclass
class RotateAction(Action):
    """Rotate a shape by 90, 180, or 270 degrees clockwise."""
    angle: int  # Must be 90, 180, or 270
    
    def __post_init__(self):
        if self.angle not in [90, 180, 270]:
            raise ValueError("Angle must be 90, 180, or 270 degrees")
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.ROTATE
    
    def apply(self, shape: GeometricShape) -> GeometricShape:
        """Rotate shape around its center."""
        center = shape.bounding_box.center
        new_pixels = set()
        
        # Rotation matrix for 90° increments
        rotations = self.angle // 90
        
        for pixel in shape.pixels:
            # Translate to origin
            x, y = pixel.x - center.x, pixel.y - center.y
            
            # Apply rotation (rotations times)
            for _ in range(rotations):
                x, y = -y, x  # 90° clockwise rotation
            
            # Translate back
            new_pixels.add(Point(x + center.x, y + center.y))
        
        new_bbox = GridUtils.compute_bounding_box(new_pixels)
        
        return GeometricShape(
            shape_type=shape.shape_type,
            pixels=new_pixels,
            color=shape.color,
            bounding_box=new_bbox,
            properties=deepcopy(shape.properties)
        )
    
    def to_dict(self) -> Dict:
        return {'type': 'rotate', 'angle': self.angle}
    
    def __repr__(self) -> str:
        return f"Rotate({self.angle}°)"


@dataclass
class FlipAction(Action):
    """Flip shape along horizontal or vertical axis."""
    axis: str  # 'horizontal' or 'vertical'
    
    def __post_init__(self):
        if self.axis not in ['horizontal', 'vertical']:
            raise ValueError("Axis must be 'horizontal' or 'vertical'")
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.FLIP
    
    def apply(self, shape: GeometricShape) -> GeometricShape:
        """Flip shape along specified axis."""
        bbox = shape.bounding_box
        new_pixels = set()
        
        if self.axis == 'horizontal':
            # Flip around horizontal center line
            center_y = (bbox.min_y + bbox.max_y) / 2
            for pixel in shape.pixels:
                new_y = int(2 * center_y - pixel.y)
                new_pixels.add(Point(pixel.x, new_y))
        else:  # vertical
            # Flip around vertical center line
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
            properties=deepcopy(shape.properties)
        )
    
    def to_dict(self) -> Dict:
        return {'type': 'flip', 'axis': self.axis}
    
    def __repr__(self) -> str:
        return f"Flip({self.axis})"


@dataclass
class ChangeColorAction(Action):
    """Change the color of a shape."""
    new_color: int
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.CHANGE_COLOR
    
    def apply(self, shape: GeometricShape) -> GeometricShape:
        """Change color while keeping all other properties."""
        return GeometricShape(
            shape_type=shape.shape_type,
            pixels=shape.pixels.copy(),
            color=self.new_color,
            bounding_box=shape.bounding_box,
            properties=deepcopy(shape.properties)
        )
    
    def to_dict(self) -> Dict:
        return {'type': 'change_color', 'new_color': self.new_color}
    
    def __repr__(self) -> str:
        return f"ChangeColor({self.new_color})"


@dataclass
class ScaleAction(Action):
    """Scale a shape by integer factors."""
    scale_x: int
    scale_y: int
    
    def __post_init__(self):
        if self.scale_x < 1 or self.scale_y < 1:
            raise ValueError("Scale factors must be >= 1")
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.SCALE
    
    def apply(self, shape: GeometricShape) -> GeometricShape:
        """Scale shape from its top-left corner."""
        bbox = shape.bounding_box
        anchor = Point(bbox.min_x, bbox.min_y)
        new_pixels = set()
        
        for pixel in shape.pixels:
            # Relative position from anchor
            rel_x = pixel.x - anchor.x
            rel_y = pixel.y - anchor.y
            
            # Scale and add all pixels in the scaled block
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
            properties=deepcopy(shape.properties)
        )
    
    def to_dict(self) -> Dict:
        return {'type': 'scale', 'scale_x': self.scale_x, 'scale_y': self.scale_y}
    
    def __repr__(self) -> str:
        return f"Scale({self.scale_x}x, {self.scale_y}y)"


class ActionGenerator:
    """
    Generates all possible actions for a given state.
    
    This defines the branching factor of our search tree.
    """
    
    def __init__(self, 
                 max_move_distance: int = 5,
                 allow_colors: List[int] = None,
                 allow_rotations: bool = True,
                 allow_flips: bool = True,
                 allow_scales: bool = False):
        """
        Initialize action generator with constraints.
        
        Args:
            max_move_distance: Maximum cells to move in any direction
            allow_colors: List of allowed colors (None = all 0-9)
            allow_rotations: Whether to generate rotation actions
            allow_flips: Whether to generate flip actions
            allow_scales: Whether to generate scale actions
        """
        self.max_move_distance = max_move_distance
        self.allow_colors = allow_colors if allow_colors else list(range(10))
        self.allow_rotations = allow_rotations
        self.allow_flips = allow_flips
        self.allow_scales = allow_scales
    
    def generate_actions(self, shape: GeometricShape) -> List[Action]:
        """
        Generate all valid actions for a shape.
        
        Returns:
            List of possible actions
        """
        actions = []
        
        # Movement actions (including diagonals)
        for dx in range(-self.max_move_distance, self.max_move_distance + 1):
            for dy in range(-self.max_move_distance, self.max_move_distance + 1):
                if dx != 0 or dy != 0:  # Skip no-op
                    actions.append(MoveAction(dx, dy))
        
        # Color change actions
        for color in self.allow_colors:
            if color != shape.color:
                actions.append(ChangeColorAction(color))
        
        # Rotation actions
        if self.allow_rotations:
            actions.extend([
                RotateAction(90),
                RotateAction(180),
                RotateAction(270)
            ])
        
        # Flip actions
        if self.allow_flips:
            actions.extend([
                FlipAction('horizontal'),
                FlipAction('vertical')
            ])
        
        # Scale actions (conservative - only 2x for now)
        if self.allow_scales:
            actions.extend([
                ScaleAction(2, 1),
                ScaleAction(1, 2),
                ScaleAction(2, 2)
            ])
        
        return actions


# ============================================================================
# 2. STATE MANAGER & DISTANCE METRIC
# ============================================================================

@dataclass
class GridState:
    """
    Represents a complete state of the grid.
    
    A state is defined by the set of shapes present and their properties.
    """
    shapes: List[GeometricShape]
    grid_size: Tuple[int, int]  # (width, height)
    background_color: int = 0
    
    def to_grid(self) -> np.ndarray:
        """
        Convert abstract state to concrete pixel grid.
        
        Returns:
            NumPy array representing the grid
        """
        height, width = self.grid_size[1], self.grid_size[0]
        grid = np.full((height, width), self.background_color, dtype=int)
        
        for shape in self.shapes:
            for pixel in shape.pixels:
                if 0 <= pixel.x < width and 0 <= pixel.y < height:
                    grid[pixel.y, pixel.x] = shape.color
        
        return grid
    
    def is_valid(self) -> bool:
        """Check if all shapes are within grid bounds."""
        width, height = self.grid_size
        
        for shape in self.shapes:
            for pixel in shape.pixels:
                if not (0 <= pixel.x < width and 0 <= pixel.y < height):
                    return False
        return True
    
    def __hash__(self):
        """Hash state for visited set in search."""
        # Simple hash based on shape positions and colors
        return hash(tuple(
            (shape.color, shape.bounding_box.min_x, shape.bounding_box.min_y,
             shape.bounding_box.max_x, shape.bounding_box.max_y, len(shape.pixels))
            for shape in sorted(self.shapes, key=lambda s: (s.bounding_box.min_x, s.bounding_box.min_y))
        ))
    
    def __eq__(self, other):
        """Equality check for states."""
        if not isinstance(other, GridState):
            return False
        return self.__hash__() == other.__hash__()


class DistanceMetric:
    """
    Computes distance between current state and goal state.
    
    This is the heuristic for A* search. Lower is better.
    """
    
    @staticmethod
    def pixel_based_distance(current: GridState, goal: GridState) -> float:
        """
        Pixel-level distance: count mismatched pixels.
        
        This is the most accurate but can be slow.
        """
        current_grid = current.to_grid()
        goal_grid = goal.to_grid()
        
        if current_grid.shape != goal_grid.shape:
            return float('inf')
        
        # Count mismatched pixels
        mismatches = np.sum(current_grid != goal_grid)
        total_pixels = current_grid.size
        
        return mismatches / total_pixels
    
    @staticmethod
    def shape_based_distance(current: GridState, goal: GridState) -> float:
        """
        Shape-level distance: compare shape properties.
        
        Faster heuristic that works at abstraction level.
        """
        if len(current.shapes) != len(goal.shapes):
            # Penalize different number of shapes heavily
            return 1000.0 + abs(len(current.shapes) - len(goal.shapes)) * 100
        
        # Try to match shapes optimally (greedy matching)
        current_shapes = sorted(current.shapes, key=lambda s: (s.bounding_box.min_x, s.bounding_box.min_y))
        goal_shapes = sorted(goal.shapes, key=lambda s: (s.bounding_box.min_x, s.bounding_box.min_y))
        
        total_distance = 0.0
        
        for curr_shape, goal_shape in zip(current_shapes, goal_shapes):
            # Color mismatch
            if curr_shape.color != goal_shape.color:
                total_distance += 10.0
            
            # Position difference (center of bounding box)
            curr_center = curr_shape.bounding_box.center
            goal_center = goal_shape.bounding_box.center
            position_dist = curr_center.manhattan_distance(goal_center)
            total_distance += position_dist
            
            # Size difference
            size_diff = abs(len(curr_shape.pixels) - len(goal_shape.pixels))
            total_distance += size_diff * 0.1
            
            # Shape type mismatch
            if curr_shape.shape_type != goal_shape.shape_type:
                total_distance += 5.0
        
        return total_distance
    
    @staticmethod
    def combined_distance(current: GridState, goal: GridState, 
                         shape_weight: float = 0.7) -> float:
        """
        Combine both metrics for balanced heuristic.
        
        Args:
            shape_weight: Weight for shape-based distance (0-1)
        """
        shape_dist = DistanceMetric.shape_based_distance(current, goal)
        pixel_dist = DistanceMetric.pixel_based_distance(current, goal)
        
        return shape_weight * shape_dist + (1 - shape_weight) * pixel_dist


# ============================================================================
# 3. SEARCH ENGINE (THE SOLVER)
# ============================================================================

@dataclass
class SearchNode:
    """
    Node in the search tree.
    
    Contains state, path cost, heuristic, and action history.
    """
    state: GridState
    parent: Optional['SearchNode']
    action: Optional[Action]
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic to goal
    depth: int
    
    @property
    def f_cost(self) -> float:
        """Total estimated cost (for A*)."""
        return self.g_cost + self.h_cost
    
    def __lt__(self, other):
        """Comparison for priority queue."""
        return self.f_cost < other.f_cost
    
    def get_action_sequence(self) -> List[Action]:
        """Reconstruct action sequence from start to this node."""
        actions = []
        node = self
        
        while node.parent is not None:
            if node.action:
                actions.append(node.action)
            node = node.parent
        
        return list(reversed(actions))


@dataclass
class SolverResult:
    """Result of a solver run."""
    success: bool
    actions: List[Action]
    final_state: Optional[GridState]
    nodes_explored: int
    time_elapsed: float
    distance_to_goal: float


class SymbolicSolver:
    """
    Main solver using A* search.
    
    Finds a sequence of actions that transforms input to output.
    """
    
    def __init__(self,
                 action_generator: ActionGenerator,
                 distance_metric: Callable[[GridState, GridState], float] = None,
                 max_depth: int = 10,
                 max_nodes: int = 10000,
                 timeout: float = 60.0):
        """
        Initialize solver.
        
        Args:
            action_generator: Generates possible actions
            distance_metric: Function to compute state distance
            max_depth: Maximum search depth
            max_nodes: Maximum nodes to explore
            timeout: Maximum time in seconds
        """
        self.action_generator = action_generator
        self.distance_metric = distance_metric or DistanceMetric.combined_distance
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.timeout = timeout
    
    def solve(self, initial_state: GridState, goal_state: GridState,
              verbose: bool = False) -> SolverResult:
        """
        Find action sequence to transform initial state to goal state.
        
        Args:
            initial_state: Starting configuration
            goal_state: Target configuration
            verbose: Print search progress
            
        Returns:
            SolverResult with solution or failure info
        """
        start_time = time.time()
        
        # Initialize search
        initial_h = self.distance_metric(initial_state, goal_state)
        start_node = SearchNode(
            state=initial_state,
            parent=None,
            action=None,
            g_cost=0.0,
            h_cost=initial_h,
            depth=0
        )
        
        # Priority queue for A*
        open_set = [start_node]
        closed_set: Set[GridState] = set()
        nodes_explored = 0
        
        if verbose:
            print(f"Starting A* search...")
            print(f"Initial distance to goal: {initial_h:.2f}")
            print(f"Max depth: {self.max_depth}, Max nodes: {self.max_nodes}")
        
        while open_set and nodes_explored < self.max_nodes:
            # Check timeout
            if time.time() - start_time > self.timeout:
                if verbose:
                    print("TIMEOUT reached!")
                return SolverResult(
                    success=False,
                    actions=[],
                    final_state=None,
                    nodes_explored=nodes_explored,
                    time_elapsed=time.time() - start_time,
                    distance_to_goal=float('inf')
                )
            
            # Get best node
            current_node = heapq.heappop(open_set)
            nodes_explored += 1
            
            if verbose and nodes_explored % 100 == 0:
                print(f"Explored: {nodes_explored}, "
                      f"Queue: {len(open_set)}, "
                      f"Best f-cost: {current_node.f_cost:.2f}, "
                      f"Depth: {current_node.depth}")
            
            # Goal check
            distance = self.distance_metric(current_node.state, goal_state)
            if distance < 0.01:  # Essentially zero
                if verbose:
                    print(f"\n✓ SOLUTION FOUND!")
                    print(f"Depth: {current_node.depth}")
                    print(f"Nodes explored: {nodes_explored}")
                    print(f"Time: {time.time() - start_time:.2f}s")
                
                return SolverResult(
                    success=True,
                    actions=current_node.get_action_sequence(),
                    final_state=current_node.state,
                    nodes_explored=nodes_explored,
                    time_elapsed=time.time() - start_time,
                    distance_to_goal=distance
                )
            
            # Skip if already visited
            if current_node.state in closed_set:
                continue
            
            closed_set.add(current_node.state)
            
            # Don't expand beyond max depth
            if current_node.depth >= self.max_depth:
                continue
            
            # Expand node (try all actions on first shape only for now)
            if current_node.state.shapes:
                shape = current_node.state.shapes[0]  # Simplification: act on first shape
                actions = self.action_generator.generate_actions(shape)
                
                for action in actions:
                    # Apply action
                    new_shape = action.apply(shape)
                    new_shapes = [new_shape] + current_node.state.shapes[1:]
                    new_state = GridState(
                        shapes=new_shapes,
                        grid_size=current_node.state.grid_size,
                        background_color=current_node.state.background_color
                    )
                    
                    # Skip invalid states
                    if not new_state.is_valid():
                        continue
                    
                    # Skip visited
                    if new_state in closed_set:
                        continue
                    
                    # Create child node
                    new_g = current_node.g_cost + 1.0  # Unit cost per action
                    new_h = self.distance_metric(new_state, goal_state)
                    
                    child_node = SearchNode(
                        state=new_state,
                        parent=current_node,
                        action=action,
                        g_cost=new_g,
                        h_cost=new_h,
                        depth=current_node.depth + 1
                    )
                    
                    heapq.heappush(open_set, child_node)
        
        # No solution found
        if verbose:
            print(f"\n✗ No solution found within limits")
        
        return SolverResult(
            success=False,
            actions=[],
            final_state=None,
            nodes_explored=nodes_explored,
            time_elapsed=time.time() - start_time,
            distance_to_goal=float('inf')
        )


# ============================================================================
# DEMONSTRATION & TESTING
# ============================================================================

def demo_simple_move():
    """
    Demonstration: Move a rectangle 2 cells to the right.
    """
    print("\n" + "="*70)
    print("DEMO: Simple Move Transformation")
    print("="*70 + "\n")
    
    # Create initial state: rectangle at (1,1)
    initial_pixels = {Point(1, 1), Point(2, 1), Point(1, 2), Point(2, 2)}
    initial_bbox = GridUtils.compute_bounding_box(initial_pixels)
    initial_shape = GeometricShape(
        shape_type='rectangle',
        pixels=initial_pixels,
        color=1,
        bounding_box=initial_bbox,
        properties={'width': 2, 'height': 2}
    )
    
    initial_state = GridState(
        shapes=[initial_shape],
        grid_size=(8, 8),
        background_color=0
    )
    
    # Create goal state: same rectangle at (3,1) - moved 2 right
    goal_pixels = {Point(3, 1), Point(4, 1), Point(3, 2), Point(4, 2)}
    goal_bbox = GridUtils.compute_bounding_box(goal_pixels)
    goal_shape = GeometricShape(
        shape_type='rectangle',
        pixels=goal_pixels,
        color=1,
        bounding_box=goal_bbox,
        properties={'width': 2, 'height': 2}
    )
    
    goal_state = GridState(
        shapes=[goal_shape],
        grid_size=(8, 8),
        background_color=0
    )
    
    print("Initial grid:")
    print(initial_state.to_grid())
    print("\nGoal grid:")
    print(goal_state.to_grid())
    
    # Setup solver with limited search space
    action_gen = ActionGenerator(
        max_move_distance=3,
        allow_rotations=False,
        allow_flips=False,
        allow_scales=False
    )
    
    solver = SymbolicSolver(
        action_generator=action_gen,
        max_depth=3,
        max_nodes=1000,
        timeout=10.0
    )
    
    # Solve
    result = solver.solve(initial_state, goal_state, verbose=True)
    
    # Display results
    print("\n" + "="*70)
    if result.success:
        print("✓ SOLUTION FOUND!")
        print(f"\nAction sequence ({len(result.actions)} steps):")
        for i, action in enumerate(result.actions, 1):
            print(f"  {i}. {action}")
        
        print("\nFinal grid:")
        print(result.final_state.to_grid())
    else:
        print("✗ NO SOLUTION FOUND")
        print(f"Nodes explored: {result.nodes_explored}")
        print(f"Time: {result.time_elapsed:.2f}s")
    
    print("="*70)


def demo_color_change():
    """
    Demonstration: Change color of a shape.
    """
    print("\n" + "="*70)
    print("DEMO: Color Change Transformation")
    print("="*70 + "\n")
    
    # Initial: red rectangle
    pixels = {Point(2, 2), Point(3, 2), Point(2, 3), Point(3, 3)}
    bbox = GridUtils.compute_bounding_box(pixels)
    
    initial_shape = GeometricShape(
        shape_type='rectangle',
        pixels=pixels,
        color=1,
        bounding_box=bbox,
        properties={}
    )
    
    initial_state = GridState(shapes=[initial_shape], grid_size=(6, 6))
    
    # Goal: blue rectangle (same position)
    goal_shape = GeometricShape(
        shape_type='rectangle',
        pixels=pixels.copy(),
        color=3,
        bounding_box=bbox,
        properties={}
    )
    
    goal_state = GridState(shapes=[goal_shape], grid_size=(6, 6))
    
    print("Initial (color=1):")
    print(initial_state.to_grid())
    print("\nGoal (color=3):")
    print(goal_state.to_grid())
    
    # Solve
    action_gen = ActionGenerator(
        max_move_distance=0,  # No movement needed
        allow_colors=[1, 2, 3],
        allow_rotations=False
    )
    
    solver = SymbolicSolver(action_gen, max_depth=2, max_nodes=500)
    result = solver.solve(initial_state, goal_state, verbose=True)
    
    if result.success:
        print(f"\n✓ Solution: {result.actions[0]}")


if __name__ == "__main__":
    # Run demonstrations
    demo_simple_move()
    print("\n" + "="*70 + "\n")
    demo_color_change()