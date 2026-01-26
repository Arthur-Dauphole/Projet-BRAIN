"""
detector.py - Symbol Detector (Perception Module)
=================================================
Step 1 of the pipeline: Perception

Analyzes input grids to detect and extract geometric objects,
patterns, and symbolic representations.
"""

from typing import List, Tuple, Optional
import numpy as np
from collections import deque

from .types import Grid, GeometricObject


class SymbolDetector:
    """
    Perception module that detects geometric objects and patterns in ARC grids.
    
    Responsibilities:
        - Connected component detection
        - Shape classification (rectangle, line, point, etc.)
        - Pattern recognition (repetition, symmetry)
        - Object property extraction
    """
    
    # ARC color palette names (for reference)
    COLOR_NAMES = {
        0: "black",
        1: "blue",
        2: "red",
        3: "green",
        4: "yellow",
        5: "grey",
        6: "magenta",
        7: "orange",
        8: "azure",
        9: "brown"
    }
    
    def __init__(self, connectivity: int = 4):
        """
        Initialize the detector.
        
        Args:
            connectivity: 4 for orthogonal neighbors, 8 for diagonal included
        """
        self.connectivity = connectivity
        self._object_counter = 0
    
    def detect(self, grid: Grid) -> Grid:
        """
        Main detection method. Analyzes the grid and populates its objects list.
        
        Args:
            grid: The Grid to analyze
            
        Returns:
            The same Grid with detected objects populated
        """
        self._object_counter = 0
        grid.objects = []
        
        # Step 1: Detect connected components for each color
        for color in grid.unique_colors:
            components = self._find_connected_components(grid.data, color)
            for pixels in components:
                obj = self._create_object(pixels, color, grid)
                grid.objects.append(obj)
        
        # Step 2: Analyze global patterns
        grid.metadata["symmetry"] = self._detect_symmetry(grid)
        grid.metadata["background_color"] = self._detect_background(grid)
        
        return grid
    
    def _find_connected_components(
        self, 
        data: np.ndarray, 
        color: int
    ) -> List[List[Tuple[int, int]]]:
        """
        Find all connected components of a given color using BFS.
        
        Args:
            data: The grid data
            color: The color to search for
            
        Returns:
            List of components, each component is a list of (row, col) tuples
        """
        mask = (data == color)
        visited = np.zeros_like(mask, dtype=bool)
        components = []
        
        # Define neighbor offsets
        if self.connectivity == 4:
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:  # 8-connectivity
            neighbors = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),           (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]
        
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if mask[i, j] and not visited[i, j]:
                    # BFS to find component
                    component = []
                    queue = deque([(i, j)])
                    visited[i, j] = True
                    
                    while queue:
                        r, c = queue.popleft()
                        component.append((r, c))
                        
                        for dr, dc in neighbors:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < data.shape[0] and 
                                0 <= nc < data.shape[1] and
                                mask[nr, nc] and 
                                not visited[nr, nc]):
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                    
                    components.append(component)
        
        return components
    
    def _create_object(
        self, 
        pixels: List[Tuple[int, int]], 
        color: int,
        grid: Grid
    ) -> GeometricObject:
        """
        Create a GeometricObject from a list of pixels.
        
        Args:
            pixels: List of (row, col) coordinates
            color: The color of the object
            grid: The parent grid (for context)
            
        Returns:
            A new GeometricObject with classified type and properties
        """
        self._object_counter += 1
        
        obj = GeometricObject(
            object_id=self._object_counter,
            object_type=self._classify_shape(pixels),
            color=color,
            pixels=pixels
        )
        
        # Add additional properties
        obj.properties["color_name"] = self.COLOR_NAMES.get(color, "unknown")
        obj.properties["is_filled"] = self._is_filled(pixels, obj.bounding_box)
        obj.properties["density"] = obj.area / (obj.width * obj.height) if obj.width * obj.height > 0 else 0
        
        return obj
    
    def _classify_shape(self, pixels: List[Tuple[int, int]]) -> str:
        """
        Classify the shape type based on pixel arrangement.
        
        Args:
            pixels: List of (row, col) coordinates
            
        Returns:
            Shape type string
        """
        if len(pixels) == 1:
            return "point"
        
        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]
        
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        area = len(pixels)
        
        # Check if it's a horizontal line
        if height == 1 and width > 1:
            return "horizontal_line"
        
        # Check if it's a vertical line
        if width == 1 and height > 1:
            return "vertical_line"
        
        # Check if it's a filled rectangle
        if area == height * width:
            return "rectangle"
        
        # Check if it's an L-shape, T-shape, etc.
        # TODO: Add more shape classifications
        
        return "irregular"
    
    def _is_filled(
        self, 
        pixels: List[Tuple[int, int]], 
        bbox: Tuple[int, int, int, int]
    ) -> bool:
        """Check if the object completely fills its bounding box."""
        if bbox is None:
            return False
        height = bbox[2] - bbox[0] + 1
        width = bbox[3] - bbox[1] + 1
        return len(pixels) == height * width
    
    def _detect_symmetry(self, grid: Grid) -> dict:
        """
        Detect symmetry properties of the grid.
        
        Args:
            grid: The grid to analyze
            
        Returns:
            Dictionary with symmetry information
        """
        data = grid.data
        
        symmetry = {
            "horizontal": np.array_equal(data, np.flipud(data)),
            "vertical": np.array_equal(data, np.fliplr(data)),
            "diagonal": False  # TODO: Implement diagonal symmetry check
        }
        
        return symmetry
    
    def _detect_background(self, grid: Grid) -> int:
        """
        Detect the background color (most common color, usually 0).
        
        Args:
            grid: The grid to analyze
            
        Returns:
            The background color value
        """
        unique, counts = np.unique(grid.data, return_counts=True)
        return int(unique[np.argmax(counts)])
    
    def describe_objects(self, grid: Grid) -> str:
        """
        Generate a natural language description of detected objects.
        
        Args:
            grid: The grid with detected objects
            
        Returns:
            Human-readable description string
        """
        if not grid.objects:
            return "No objects detected in the grid."
        
        descriptions = []
        for obj in grid.objects:
            desc = (f"Object {obj.object_id}: {obj.properties.get('color_name', 'unknown')} "
                   f"{obj.object_type} at position {obj.bounding_box}, "
                   f"size {obj.width}x{obj.height}")
            descriptions.append(desc)
        
        return "\n".join(descriptions)
