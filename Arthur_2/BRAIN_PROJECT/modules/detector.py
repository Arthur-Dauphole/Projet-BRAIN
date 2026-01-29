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
        
        # Basic properties
        obj.properties["color_name"] = self.COLOR_NAMES.get(color, "unknown")
        obj.properties["is_filled"] = self._is_filled(pixels, obj.bounding_box)
        obj.properties["density"] = obj.area / (obj.width * obj.height) if obj.width * obj.height > 0 else 0
        obj.properties["is_convex"] = self._is_convex(pixels, obj.bounding_box)
        obj.properties["has_hole"] = self._has_hole(pixels, obj.bounding_box)
        obj.properties["centroid"] = self._compute_centroid(pixels)
        
        # Advanced blob properties (useful for transformation detection)
        obj.properties["perimeter"] = self._compute_perimeter(pixels)
        obj.properties["compactness"] = self._compute_compactness(pixels)
        obj.properties["shape_signature"] = self._compute_shape_signature(pixels, obj.bounding_box)
        obj.properties["corner_count"] = self._count_corners(pixels)
        obj.properties["orientation"] = self._compute_orientation(pixels)
        obj.properties["aspect_ratio"] = obj.width / obj.height if obj.height > 0 else 1.0
        
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
    
    def _compute_perimeter(self, pixels: List[Tuple[int, int]]) -> int:
        """
        Compute the perimeter of the object (boundary pixels count).
        A pixel is on the boundary if it has at least one empty neighbor.
        """
        pixel_set = set(pixels)
        perimeter = 0
        
        for r, c in pixels:
            # Check 4-neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r + dr, c + dc) not in pixel_set:
                    perimeter += 1
        
        return perimeter
    
    def _compute_compactness(self, pixels: List[Tuple[int, int]]) -> float:
        """
        Compute compactness (circularity) of the shape.
        Compactness = 4π * Area / Perimeter²
        A perfect circle has compactness = 1, more complex shapes have lower values.
        """
        if len(pixels) < 2:
            return 1.0
        
        perimeter = self._compute_perimeter(pixels)
        if perimeter == 0:
            return 1.0
        
        area = len(pixels)
        compactness = (4 * 3.14159 * area) / (perimeter ** 2)
        return min(1.0, compactness)  # Cap at 1.0
    
    def _compute_shape_signature(
        self, 
        pixels: List[Tuple[int, int]], 
        bbox: Tuple[int, int, int, int]
    ) -> str:
        """
        Compute a unique signature for the shape that is translation-invariant.
        This helps identify if two blobs have the same shape.
        
        Returns a normalized binary string representation.
        """
        if not pixels or bbox is None:
            return ""
        
        min_r, min_c, max_r, max_c = bbox
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        
        # Create normalized grid
        pixel_set = set(pixels)
        signature = []
        
        for r in range(min_r, max_r + 1):
            row_sig = ""
            for c in range(min_c, max_c + 1):
                row_sig += "1" if (r, c) in pixel_set else "0"
            signature.append(row_sig)
        
        return "|".join(signature)
    
    def _count_corners(self, pixels: List[Tuple[int, int]]) -> int:
        """
        Count the number of corner pixels in the shape.
        A corner is a pixel where exactly 2 orthogonal neighbors are present.
        """
        pixel_set = set(pixels)
        corners = 0
        
        for r, c in pixels:
            # Get orthogonal neighbors
            neighbors = [
                (r - 1, c) in pixel_set,  # up
                (r + 1, c) in pixel_set,  # down
                (r, c - 1) in pixel_set,  # left
                (r, c + 1) in pixel_set,  # right
            ]
            
            neighbor_count = sum(neighbors)
            
            # Check for L-shaped corner patterns (exactly 2 adjacent neighbors that form a corner)
            if neighbor_count == 2:
                # Check if the 2 neighbors are adjacent (forming an L)
                up, down, left, right = neighbors
                if (up and left) or (up and right) or (down and left) or (down and right):
                    corners += 1
            elif neighbor_count == 1:
                # End point (tip of a shape)
                corners += 1
        
        return corners
    
    def _compute_orientation(self, pixels: List[Tuple[int, int]]) -> str:
        """
        Compute the principal orientation of the shape.
        Returns "horizontal", "vertical", "diagonal", or "symmetric".
        """
        if len(pixels) < 2:
            return "symmetric"
        
        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]
        
        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1
        
        if width > height * 1.5:
            return "horizontal"
        elif height > width * 1.5:
            return "vertical"
        elif abs(width - height) <= 1:
            return "symmetric"
        else:
            # Check for diagonal orientation using covariance
            centroid = self._compute_centroid(pixels)
            
            # Compute covariance
            cov_sum = sum((r - centroid[0]) * (c - centroid[1]) for r, c in pixels)
            
            if abs(cov_sum) > len(pixels) * 0.3:
                return "diagonal"
            else:
                return "symmetric"
    
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
        
        # === DEFAULT: BLOB (with sub-classification) ===
        return self._classify_blob(pixel_set, min_r, min_c, max_r, max_c, height, width, area)
    
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
    
    def _classify_blob(self, pixel_set: set, min_r: int, min_c: int,
                       max_r: int, max_c: int, height: int, width: int, area: int) -> str:
        """
        Classify a blob into more specific sub-types based on its properties.
        
        Returns a descriptive blob type like:
        - "blob_compact" for round-ish shapes
        - "blob_elongated" for long shapes
        - "blob_sparse" for shapes with low density
        - "blob_complex" for shapes with holes or many corners
        - "blob" for generic irregular shapes
        """
        pixels = list(pixel_set)
        
        # Calculate properties
        density = area / (height * width) if height * width > 0 else 0
        aspect_ratio = max(width / height, height / width) if min(height, width) > 0 else 1
        compactness = self._compute_compactness(pixels)
        has_hole = self._has_hole(pixels, (min_r, min_c, max_r, max_c))
        corner_count = self._count_corners(pixels)
        
        # Classify based on properties
        if has_hole:
            return "blob_with_hole"
        
        if compactness > 0.7 and density > 0.7:
            return "blob_compact"
        
        if aspect_ratio > 2.5:
            return "blob_elongated"
        
        if density < 0.4:
            return "blob_sparse"
        
        if corner_count > 6:
            return "blob_complex"
        
        return "blob"
    
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
    
    # ==================== BLOB COMPARISON METHODS ====================
    
    def compare_shapes(self, obj1: GeometricObject, obj2: GeometricObject) -> dict:
        """
        Compare two objects to determine if they have the same shape.
        
        Args:
            obj1: First object
            obj2: Second object
            
        Returns:
            Dictionary with comparison results:
            - same_shape: bool - True if identical shape
            - is_rotated: int - Rotation angle if rotated (0, 90, 180, 270), None if not
            - is_reflected: str - Reflection axis if reflected, None if not
            - is_scaled: float - Scale factor if scaled, None if not
        """
        result = {
            "same_shape": False,
            "is_rotated": None,
            "is_reflected": None,
            "is_scaled": None,
        }
        
        if not obj1.pixels or not obj2.pixels:
            return result
        
        # Normalize both shapes to origin
        norm1 = self._normalize_pixels(obj1.pixels)
        norm2 = self._normalize_pixels(obj2.pixels)
        
        # Check if identical
        if norm1 == norm2:
            result["same_shape"] = True
            return result
        
        # Check for rotations
        for angle in [90, 180, 270]:
            rotated = self._rotate_normalized_pixels(norm1, angle)
            if rotated == norm2:
                result["same_shape"] = True
                result["is_rotated"] = angle
                return result
        
        # Check for reflections
        for axis, reflected in [
            ("horizontal", self._reflect_normalized_pixels(norm1, "horizontal")),
            ("vertical", self._reflect_normalized_pixels(norm1, "vertical")),
        ]:
            if reflected == norm2:
                result["same_shape"] = True
                result["is_reflected"] = axis
                return result
        
        # Check for scaling (if areas are proportional)
        if obj1.area > 0 and obj2.area > 0:
            scale = (obj2.area / obj1.area) ** 0.5
            if abs(scale - round(scale)) < 0.1:
                # Integer scale factor - could be scaled
                result["is_scaled"] = round(scale)
        
        return result
    
    def _normalize_pixels(self, pixels: List[Tuple[int, int]]) -> frozenset:
        """
        Normalize pixels to origin (0,0) for shape comparison.
        """
        if not pixels:
            return frozenset()
        
        min_r = min(p[0] for p in pixels)
        min_c = min(p[1] for p in pixels)
        
        return frozenset((p[0] - min_r, p[1] - min_c) for p in pixels)
    
    def _rotate_normalized_pixels(self, pixels: frozenset, angle: int) -> frozenset:
        """
        Rotate normalized pixels by the given angle and re-normalize.
        """
        if not pixels:
            return frozenset()
        
        pixels_list = list(pixels)
        h = max(p[0] for p in pixels_list) + 1
        w = max(p[1] for p in pixels_list) + 1
        
        rotated = []
        for r, c in pixels_list:
            if angle == 90:
                new_r, new_c = c, h - 1 - r
            elif angle == 180:
                new_r, new_c = h - 1 - r, w - 1 - c
            elif angle == 270:
                new_r, new_c = w - 1 - c, r
            else:
                new_r, new_c = r, c
            rotated.append((new_r, new_c))
        
        # Re-normalize
        min_r = min(p[0] for p in rotated)
        min_c = min(p[1] for p in rotated)
        
        return frozenset((p[0] - min_r, p[1] - min_c) for p in rotated)
    
    def _reflect_normalized_pixels(self, pixels: frozenset, axis: str) -> frozenset:
        """
        Reflect normalized pixels along the given axis and re-normalize.
        """
        if not pixels:
            return frozenset()
        
        pixels_list = list(pixels)
        h = max(p[0] for p in pixels_list) + 1
        w = max(p[1] for p in pixels_list) + 1
        
        reflected = []
        for r, c in pixels_list:
            if axis == "horizontal":
                new_r, new_c = h - 1 - r, c
            elif axis == "vertical":
                new_r, new_c = r, w - 1 - c
            else:
                new_r, new_c = r, c
            reflected.append((new_r, new_c))
        
        # Re-normalize
        min_r = min(p[0] for p in reflected)
        min_c = min(p[1] for p in reflected)
        
        return frozenset((p[0] - min_r, p[1] - min_c) for p in reflected)
    
    def find_matching_object(
        self, 
        target: GeometricObject, 
        candidates: List[GeometricObject],
        match_color: bool = True
    ) -> Optional[Tuple[GeometricObject, dict]]:
        """
        Find an object in candidates that matches the target shape.
        
        Args:
            target: The object to match
            candidates: List of potential matches
            match_color: If True, only match objects of the same color
            
        Returns:
            Tuple of (matched_object, comparison_result) or None
        """
        for candidate in candidates:
            if match_color and candidate.color != target.color:
                continue
            
            comparison = self.compare_shapes(target, candidate)
            if comparison["same_shape"]:
                return (candidate, comparison)
        
        return None
    
    # ==================== PATTERN DETECTION ====================
    
    def detect_repeating_pattern(self, grid: Grid) -> Optional[dict]:
        """
        Detect if the grid contains a repeating pattern (tile/motif).
        
        This finds the smallest rectangular pattern that, when tiled,
        recreates the grid (or a significant portion of it).
        
        Args:
            grid: The grid to analyze
            
        Returns:
            Dictionary with pattern info or None if no pattern found:
            {
                "pattern": 2D array of the repeating unit,
                "tile_height": height of the pattern,
                "tile_width": width of the pattern,
                "repetitions_h": number of horizontal repetitions,
                "repetitions_v": number of vertical repetitions,
                "coverage": percentage of grid covered by the pattern
            }
        """
        data = grid.data
        h, w = data.shape
        
        # Try different tile sizes (from small to large)
        # Maximum tile size is half the grid dimension
        best_pattern = None
        best_score = 0
        
        for tile_h in range(1, h // 2 + 1):
            for tile_w in range(1, w // 2 + 1):
                # Skip if tile doesn't divide grid evenly (less likely to be a pattern)
                if h % tile_h != 0 or w % tile_w != 0:
                    continue
                
                # Extract the candidate tile from top-left
                tile = data[0:tile_h, 0:tile_w]
                
                # Skip if tile is all background
                if np.all(tile == 0):
                    continue
                
                # Check if this tile repeats across the grid
                match_count = 0
                total_tiles = (h // tile_h) * (w // tile_w)
                
                for row_idx in range(h // tile_h):
                    for col_idx in range(w // tile_w):
                        row_start = row_idx * tile_h
                        col_start = col_idx * tile_w
                        
                        region = data[row_start:row_start + tile_h, 
                                     col_start:col_start + tile_w]
                        
                        if np.array_equal(region, tile):
                            match_count += 1
                
                # Calculate coverage score
                coverage = match_count / total_tiles
                
                # We want high coverage with the smallest possible tile
                # Score favors smaller tiles and higher coverage
                score = coverage * (1.0 / (tile_h * tile_w))
                
                if coverage >= 0.9 and score > best_score:  # At least 90% coverage
                    best_score = score
                    best_pattern = {
                        "pattern": tile.copy(),
                        "tile_height": tile_h,
                        "tile_width": tile_w,
                        "repetitions_h": w // tile_w,
                        "repetitions_v": h // tile_h,
                        "coverage": coverage
                    }
        
        return best_pattern
    
    def detect_subgrids(self, grid: Grid) -> List[dict]:
        """
        Detect rectangular subgrids within the main grid.
        
        Subgrids are identified by:
        - Regular rectangular regions
        - Separated by grid lines or borders
        - Consistent dimensions
        
        Args:
            grid: The grid to analyze
            
        Returns:
            List of subgrid info dictionaries:
            {
                "row": starting row,
                "col": starting column,
                "height": subgrid height,
                "width": subgrid width,
                "data": 2D array of subgrid content
            }
        """
        data = grid.data
        h, w = data.shape
        subgrids = []
        
        # Method 1: Find subgrids separated by a specific color (grid lines)
        # Look for horizontal and vertical separating lines
        
        # Find potential horizontal separators (rows with single color)
        h_separators = []
        for row in range(h):
            unique_in_row = np.unique(data[row, :])
            if len(unique_in_row) == 1 and unique_in_row[0] != 0:
                h_separators.append((row, unique_in_row[0]))
        
        # Find potential vertical separators (columns with single color)
        v_separators = []
        for col in range(w):
            unique_in_col = np.unique(data[:, col])
            if len(unique_in_col) == 1 and unique_in_col[0] != 0:
                v_separators.append((col, unique_in_col[0]))
        
        # If we have separators, extract subgrids
        if h_separators and v_separators:
            # Get the separator color (should be consistent)
            sep_color = h_separators[0][1]
            
            # Build list of row and column boundaries
            row_bounds = [0] + [s[0] for s in h_separators] + [h]
            col_bounds = [0] + [s[0] for s in v_separators] + [w]
            
            # Extract each subgrid
            for i in range(len(row_bounds) - 1):
                for j in range(len(col_bounds) - 1):
                    r1, r2 = row_bounds[i], row_bounds[i + 1]
                    c1, c2 = col_bounds[j], col_bounds[j + 1]
                    
                    # Skip separator rows/columns themselves
                    if r2 - r1 <= 1 or c2 - c1 <= 1:
                        continue
                    
                    # Adjust to exclude separator lines
                    if r1 > 0:
                        r1 += 1
                    if c1 > 0:
                        c1 += 1
                    
                    if r2 - r1 > 0 and c2 - c1 > 0:
                        subgrid_data = data[r1:r2, c1:c2]
                        subgrids.append({
                            "row": r1,
                            "col": c1,
                            "height": r2 - r1,
                            "width": c2 - c1,
                            "data": subgrid_data.copy()
                        })
        
        # Method 2: Look for regular partitioning without explicit separators
        if not subgrids:
            # Try common partition sizes
            for num_rows in [2, 3, 4]:
                for num_cols in [2, 3, 4]:
                    if h % num_rows == 0 and w % num_cols == 0:
                        sub_h = h // num_rows
                        sub_w = w // num_cols
                        
                        temp_subgrids = []
                        for i in range(num_rows):
                            for j in range(num_cols):
                                r1, c1 = i * sub_h, j * sub_w
                                subgrid_data = data[r1:r1 + sub_h, c1:c1 + sub_w]
                                temp_subgrids.append({
                                    "row": r1,
                                    "col": c1,
                                    "height": sub_h,
                                    "width": sub_w,
                                    "data": subgrid_data.copy()
                                })
                        
                        # Verify these are meaningful subgrids (not all same content)
                        unique_contents = set()
                        for sg in temp_subgrids:
                            unique_contents.add(tuple(sg["data"].flatten()))
                        
                        # If we have some variation, these are valid subgrids
                        if len(unique_contents) > 1:
                            subgrids = temp_subgrids
                            break
                if subgrids:
                    break
        
        return subgrids
    
    def detect_bordered_objects(self, grid: Grid) -> List[dict]:
        """
        Detect objects that have a border/contour of a different color.
        
        These are shapes where:
        - An inner region has one color
        - The outer border/contour has a different color
        
        Args:
            grid: The grid to analyze
            
        Returns:
            List of bordered object info:
            {
                "inner_color": color of the interior,
                "border_color": color of the border,
                "inner_pixels": set of (row, col) for interior,
                "border_pixels": set of (row, col) for border,
                "bounding_box": (min_row, min_col, max_row, max_col)
            }
        """
        data = grid.data
        h, w = data.shape
        bordered_objects = []
        
        # For each non-background color, check if it forms a filled interior
        non_bg_colors = [c for c in np.unique(data) if c != 0]
        
        for inner_color in non_bg_colors:
            # Get all pixels of this color
            inner_positions = set(zip(*np.where(data == inner_color)))
            
            if not inner_positions:
                continue
            
            # Find the bounding box
            rows, cols = zip(*inner_positions)
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            
            # Get pixels adjacent to this region (potential border)
            adjacent_positions = set()
            for r, c in inner_positions:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if (nr, nc) not in inner_positions and data[nr, nc] != 0:
                            adjacent_positions.add((nr, nc))
            
            if not adjacent_positions:
                continue
            
            # Check if adjacent pixels form a consistent border (single color)
            adjacent_colors = set(data[r, c] for r, c in adjacent_positions)
            
            if len(adjacent_colors) == 1:
                border_color = adjacent_colors.pop()
                
                # Verify this is a "surrounding" border
                # The border should be on multiple sides of the inner region
                border_on_top = any(r < min_r for r, c in adjacent_positions)
                border_on_bottom = any(r > max_r for r, c in adjacent_positions)
                border_on_left = any(c < min_c for r, c in adjacent_positions)
                border_on_right = any(c > max_c for r, c in adjacent_positions)
                
                sides_with_border = sum([border_on_top, border_on_bottom, 
                                        border_on_left, border_on_right])
                
                # At least 2 sides should have border to be considered bordered
                if sides_with_border >= 2:
                    # Find all border pixels (not just adjacent)
                    border_pixels = set(zip(*np.where(data == border_color)))
                    
                    # Filter to only border pixels near this object
                    expanded_box = (min_r - 2, min_c - 2, max_r + 2, max_c + 2)
                    border_pixels = {
                        (r, c) for r, c in border_pixels
                        if expanded_box[0] <= r <= expanded_box[2] and 
                           expanded_box[1] <= c <= expanded_box[3]
                    }
                    
                    bordered_objects.append({
                        "inner_color": int(inner_color),
                        "border_color": int(border_color),
                        "inner_pixels": inner_positions,
                        "border_pixels": border_pixels,
                        "bounding_box": (min_r, min_c, max_r, max_c)
                    })
        
        return bordered_objects