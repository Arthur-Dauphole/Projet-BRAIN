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
        obj.properties["is_convex"] = self._is_convex(pixels, obj.bounding_box)
        obj.properties["has_hole"] = self._has_hole(pixels, obj.bounding_box)
        obj.properties["centroid"] = self._compute_centroid(pixels)
        
        return obj
    
    def _is_convex(self, pixels: List[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        """
        Check if the object is convex (no concavities).
        A simple approximation: convex if density > 0.5 and no holes.
        """
        if bbox is None:
            return False
        height = bbox[2] - bbox[0] + 1
        width = bbox[3] - bbox[1] + 1
        density = len(pixels) / (height * width)
        return density > 0.5 and not self._has_hole(pixels, bbox)
    
    def _has_hole(self, pixels: List[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        """
        Check if the object has internal holes.
        Uses flood fill from outside to detect unreachable interior regions.
        """
        if bbox is None or len(pixels) < 4:
            return False
        
        min_r, min_c, max_r, max_c = bbox
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        
        # Create a padded grid (1 pixel border)
        grid = np.zeros((height + 2, width + 2), dtype=bool)
        pixel_set = set(pixels)
        
        for r, c in pixels:
            grid[r - min_r + 1, c - min_c + 1] = True
        
        # Flood fill from outside (0,0)
        visited = np.zeros_like(grid, dtype=bool)
        queue = deque([(0, 0)])
        visited[0, 0] = True
        
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height + 2 and 0 <= nc < width + 2:
                    if not visited[nr, nc] and not grid[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        
        # Check for unvisited empty cells (holes)
        for r in range(1, height + 1):
            for c in range(1, width + 1):
                if not grid[r, c] and not visited[r, c]:
                    return True
        
        return False
    
    def _compute_centroid(self, pixels: List[Tuple[int, int]]) -> Tuple[float, float]:
        """Compute the centroid (center of mass) of the object."""
        if not pixels:
            return (0.0, 0.0)
        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]
        return (sum(rows) / len(rows), sum(cols) / len(cols))
    
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
        pixel_set = set(pixels)
        
        # === LINES ===
        # Horizontal line
        if height == 1 and width > 1:
            return "horizontal_line"
        
        # Vertical line
        if width == 1 and height > 1:
            return "vertical_line"
        
        # Diagonal line (check if all pixels are on a diagonal)
        if self._is_diagonal_line(pixels, min_r, min_c, height, width):
            return "diagonal_line"
        
        # === FILLED SHAPES ===
        if area == height * width:
            if height == width:
                return "square"
            return "rectangle"
        
        # === HOLLOW SHAPES ===
        # Hollow rectangle (frame): perimeter pixels only
        perimeter = 2 * (height + width) - 4
        if area == perimeter and height >= 3 and width >= 3:
            if self._is_hollow_rectangle(pixel_set, min_r, min_c, max_r, max_c):
                return "hollow_rectangle"
        
        # === SPECIAL SHAPES ===
        # Plus shape (+)
        if self._is_plus_shape(pixel_set, min_r, min_c, max_r, max_c, height, width):
            return "plus_shape"
        
        # T shape
        if self._is_t_shape(pixel_set, min_r, min_c, max_r, max_c, height, width, area):
            return "T_shape"
        
        # L shape
        if self._is_l_shape(pixel_set, min_r, min_c, max_r, max_c, height, width, area):
            return "L_shape"
        
        # === DEFAULT: BLOB ===
        return "blob"
    
    def _is_diagonal_line(self, pixels: List[Tuple[int, int]], 
                          min_r: int, min_c: int, height: int, width: int) -> bool:
        """Check if pixels form a diagonal line."""
        if height != width or height < 2:
            return False
        
        n = height
        pixel_set = set(pixels)
        
        # Check main diagonal (top-left to bottom-right)
        main_diag = all((min_r + i, min_c + i) in pixel_set for i in range(n))
        if main_diag and len(pixels) == n:
            return True
        
        # Check anti-diagonal (top-right to bottom-left)
        anti_diag = all((min_r + i, min_c + n - 1 - i) in pixel_set for i in range(n))
        if anti_diag and len(pixels) == n:
            return True
        
        return False
    
    def _is_hollow_rectangle(self, pixel_set: set, 
                             min_r: int, min_c: int, max_r: int, max_c: int) -> bool:
        """Check if pixels form a hollow rectangle (frame)."""
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                is_border = (r == min_r or r == max_r or c == min_c or c == max_c)
                if is_border and (r, c) not in pixel_set:
                    return False
                if not is_border and (r, c) in pixel_set:
                    return False
        return True
    
    def _is_plus_shape(self, pixel_set: set, min_r: int, min_c: int, 
                       max_r: int, max_c: int, height: int, width: int) -> bool:
        """Check if pixels form a plus (+) shape."""
        if height < 3 or width < 3:
            return False
        if height != width:  # Plus is typically symmetric
            return False
        
        # Find center
        center_r = (min_r + max_r) // 2
        center_c = (min_c + max_c) // 2
        
        # Check horizontal and vertical bars through center
        expected = set()
        for r in range(min_r, max_r + 1):
            expected.add((r, center_c))
        for c in range(min_c, max_c + 1):
            expected.add((center_r, c))
        
        return pixel_set == expected
    
    def _is_t_shape(self, pixel_set: set, min_r: int, min_c: int,
                    max_r: int, max_c: int, height: int, width: int, area: int) -> bool:
        """Check if pixels form a T shape (any orientation)."""
        if height < 2 or width < 2:
            return False
        
        # T shape area: width + height - 1 (approximate for various T sizes)
        expected_area = width + height - 1
        if area != expected_area:
            return False
        
        # Check various T orientations
        # T pointing down: top row full, center column continues down
        if self._check_t_down(pixel_set, min_r, min_c, max_r, max_c, height, width):
            return True
        # T pointing up: bottom row full, center column continues up
        if self._check_t_up(pixel_set, min_r, min_c, max_r, max_c, height, width):
            return True
        # T pointing right: left column full, center row continues right
        if self._check_t_right(pixel_set, min_r, min_c, max_r, max_c, height, width):
            return True
        # T pointing left: right column full, center row continues left
        if self._check_t_left(pixel_set, min_r, min_c, max_r, max_c, height, width):
            return True
        
        return False
    
    def _check_t_down(self, pixel_set, min_r, min_c, max_r, max_c, height, width):
        """T with bar at top, stem going down."""
        center_c = (min_c + max_c) // 2
        expected = set()
        for c in range(min_c, max_c + 1):
            expected.add((min_r, c))
        for r in range(min_r, max_r + 1):
            expected.add((r, center_c))
        return pixel_set == expected
    
    def _check_t_up(self, pixel_set, min_r, min_c, max_r, max_c, height, width):
        """T with bar at bottom, stem going up."""
        center_c = (min_c + max_c) // 2
        expected = set()
        for c in range(min_c, max_c + 1):
            expected.add((max_r, c))
        for r in range(min_r, max_r + 1):
            expected.add((r, center_c))
        return pixel_set == expected
    
    def _check_t_right(self, pixel_set, min_r, min_c, max_r, max_c, height, width):
        """T with bar at left, stem going right."""
        center_r = (min_r + max_r) // 2
        expected = set()
        for r in range(min_r, max_r + 1):
            expected.add((r, min_c))
        for c in range(min_c, max_c + 1):
            expected.add((center_r, c))
        return pixel_set == expected
    
    def _check_t_left(self, pixel_set, min_r, min_c, max_r, max_c, height, width):
        """T with bar at right, stem going left."""
        center_r = (min_r + max_r) // 2
        expected = set()
        for r in range(min_r, max_r + 1):
            expected.add((r, max_c))
        for c in range(min_c, max_c + 1):
            expected.add((center_r, c))
        return pixel_set == expected
    
    def _is_l_shape(self, pixel_set: set, min_r: int, min_c: int,
                    max_r: int, max_c: int, height: int, width: int, area: int) -> bool:
        """Check if pixels form an L shape (any orientation)."""
        if height < 2 or width < 2:
            return False
        
        # L shape area: height + width - 1
        expected_area = height + width - 1
        if area != expected_area:
            return False
        
        # Check 4 L orientations
        corners = [
            (min_r, min_c),  # top-left corner
            (min_r, max_c),  # top-right corner
            (max_r, min_c),  # bottom-left corner
            (max_r, max_c),  # bottom-right corner
        ]
        
        for corner_r, corner_c in corners:
            expected = set()
            # Vertical part from corner
            for r in range(min_r, max_r + 1):
                expected.add((r, corner_c))
            # Horizontal part from corner
            for c in range(min_c, max_c + 1):
                expected.add((corner_r, c))
            
            if pixel_set == expected:
                return True
        
        return False
    
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
            "diagonal_main": self._check_diagonal_symmetry(data, main=True),
            "diagonal_anti": self._check_diagonal_symmetry(data, main=False),
            "rotational_90": self._check_rotational_symmetry(data, 90),
            "rotational_180": self._check_rotational_symmetry(data, 180),
        }
        
        return symmetry
    
    def _check_diagonal_symmetry(self, data: np.ndarray, main: bool = True) -> bool:
        """
        Check diagonal symmetry.
        
        Args:
            data: Grid data
            main: True for main diagonal (top-left to bottom-right),
                  False for anti-diagonal
        """
        if data.shape[0] != data.shape[1]:
            return False  # Must be square for diagonal symmetry
        
        if main:
            # Main diagonal: data[i,j] == data[j,i]
            return np.array_equal(data, data.T)
        else:
            # Anti-diagonal: data[i,j] == data[n-1-j, n-1-i]
            n = data.shape[0]
            flipped = np.flip(np.flip(data, 0), 1).T
            return np.array_equal(data, flipped)
    
    def _check_rotational_symmetry(self, data: np.ndarray, angle: int) -> bool:
        """
        Check rotational symmetry.
        
        Args:
            data: Grid data
            angle: Rotation angle (90, 180, 270)
        """
        if angle == 90:
            rotated = np.rot90(data, k=1)
        elif angle == 180:
            rotated = np.rot90(data, k=2)
        elif angle == 270:
            rotated = np.rot90(data, k=3)
        else:
            return False
        
        return np.array_equal(data, rotated)
    
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
