"""
types.py - Data Classes for BRAIN Project
==========================================
Contains the core data structures used throughout the pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
import numpy as np


@dataclass
class GeometricObject:
    """
    Represents a detected geometric object/symbol in a grid.
    
    Attributes:
        object_id: Unique identifier for the object
        object_type: Type of object (e.g., 'rectangle', 'line', 'point', 'pattern')
        color: Color value (0-9 in ARC format)
        pixels: List of (row, col) coordinates belonging to this object
        bounding_box: (min_row, min_col, max_row, max_col) enclosing the object
        properties: Additional properties (symmetry, repetition, etc.)
    """
    object_id: int
    object_type: str
    color: int
    pixels: List[Tuple[int, int]] = field(default_factory=list)
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    properties: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate bounding box if not provided."""
        if self.pixels and self.bounding_box is None:
            rows = [p[0] for p in self.pixels]
            cols = [p[1] for p in self.pixels]
            self.bounding_box = (min(rows), min(cols), max(rows), max(cols))
    
    @property
    def width(self) -> int:
        """Width of the bounding box."""
        if self.bounding_box:
            return self.bounding_box[3] - self.bounding_box[1] + 1
        return 0
    
    @property
    def height(self) -> int:
        """Height of the bounding box."""
        if self.bounding_box:
            return self.bounding_box[2] - self.bounding_box[0] + 1
        return 0
    
    @property
    def area(self) -> int:
        """Number of pixels in the object."""
        return len(self.pixels)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "color": self.color,
            "pixels": self.pixels,
            "bounding_box": self.bounding_box,
            "properties": self.properties,
            "width": self.width,
            "height": self.height,
            "area": self.area,
        }


@dataclass
class Grid:
    """
    Represents an ARC-AGI grid with its detected objects.
    
    Attributes:
        data: 2D numpy array of the grid (values 0-9)
        objects: List of detected GeometricObjects
        metadata: Additional metadata about the grid
    """
    data: np.ndarray
    objects: List[GeometricObject] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def from_list(cls, grid_list: List[List[int]]) -> "Grid":
        """Create a Grid from a 2D list (JSON format)."""
        return cls(data=np.array(grid_list, dtype=np.int8))
    
    @property
    def height(self) -> int:
        """Number of rows in the grid."""
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        """Number of columns in the grid."""
        return self.data.shape[1]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the grid (height, width)."""
        return self.data.shape
    
    @property
    def unique_colors(self) -> List[int]:
        """List of unique colors in the grid (excluding background 0)."""
        colors = np.unique(self.data).tolist()
        return [c for c in colors if c != 0]
    
    def get_color_mask(self, color: int) -> np.ndarray:
        """Return a boolean mask for a specific color."""
        return self.data == color
    
    def to_list(self) -> List[List[int]]:
        """Convert to 2D list for JSON serialization."""
        return self.data.tolist()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "data": self.to_list(),
            "height": self.height,
            "width": self.width,
            "unique_colors": self.unique_colors,
            "objects": [obj.to_dict() for obj in self.objects],
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        """String representation of the grid."""
        return f"Grid({self.height}x{self.width}, colors={self.unique_colors}, objects={len(self.objects)})"


@dataclass
class TaskPair:
    """
    Represents a single input-output pair in an ARC task.
    
    Attributes:
        input_grid: The input Grid
        output_grid: The expected output Grid (None for test pairs)
    """
    input_grid: Grid
    output_grid: Optional[Grid] = None


@dataclass
class ARCTask:
    """
    Represents a complete ARC-AGI task.
    
    Attributes:
        task_id: Unique identifier for the task
        train_pairs: List of training input-output pairs
        test_pairs: List of test pairs (output may be None)
    """
    task_id: str
    train_pairs: List[TaskPair] = field(default_factory=list)
    test_pairs: List[TaskPair] = field(default_factory=list)
    
    @classmethod
    def from_json(cls, task_id: str, json_data: dict) -> "ARCTask":
        """Create an ARCTask from JSON data."""
        task = cls(task_id=task_id)
        
        # Parse training pairs
        for pair in json_data.get("train", []):
            task.train_pairs.append(TaskPair(
                input_grid=Grid.from_list(pair["input"]),
                output_grid=Grid.from_list(pair["output"])
            ))
        
        # Parse test pairs
        for pair in json_data.get("test", []):
            task.test_pairs.append(TaskPair(
                input_grid=Grid.from_list(pair["input"]),
                output_grid=Grid.from_list(pair["output"]) if "output" in pair else None
            ))
        
        return task
    
    def __str__(self) -> str:
        return f"ARCTask({self.task_id}, train={len(self.train_pairs)}, test={len(self.test_pairs)})"
