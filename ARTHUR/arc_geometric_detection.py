"""
ARC-AGI Geometric Detection System
===================================

A modular system for geometric shape detection and property extraction
from ARC-AGI grids, designed for reasoning-based AI approaches.

Author: BRAIN Project Team
Date: 2025
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MPLRect
from scipy import ndimage
from collections import defaultdict


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class Direction(Enum):
    """Cardinal and intercardinal directions for spatial reasoning."""
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)
    NORTHEAST = (1, -1)
    NORTHWEST = (-1, -1)
    SOUTHEAST = (1, 1)
    SOUTHWEST = (-1, 1)
    
    @classmethod
    def cardinal(cls):
        """Return only cardinal directions (N, S, E, W)."""
        return [cls.NORTH, cls.SOUTH, cls.EAST, cls.WEST]
    
    @classmethod
    def all_directions(cls):
        """Return all 8 directions."""
        return list(cls)


@dataclass
class Point:
    """2D point in grid coordinates."""
    x: int
    y: int
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def to_tuple(self):
        return (self.x, self.y)
    
    def manhattan_distance(self, other: 'Point') -> int:
        """Calculate Manhattan distance to another point."""
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def euclidean_distance(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    
    @property
    def width(self) -> int:
        return self.max_x - self.min_x + 1
    
    @property
    def height(self) -> int:
        return self.max_y - self.min_y + 1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Point:
        return Point(
            (self.min_x + self.max_x) // 2,
            (self.min_y + self.max_y) // 2
        )
    
    def contains(self, point: Point) -> bool:
        """Check if point is inside bounding box."""
        return (self.min_x <= point.x <= self.max_x and 
                self.min_y <= point.y <= self.max_y)
    
    def corners(self) -> List[Point]:
        """Return the four corners of the bounding box."""
        return [
            Point(self.min_x, self.min_y),
            Point(self.max_x, self.min_y),
            Point(self.max_x, self.max_y),
            Point(self.min_x, self.max_y)
        ]


@dataclass
class GeometricShape:
    """
    Represents a detected geometric shape with its properties.
    
    This is the core output structure for geometric detection.
    It contains both the raw pixels and extracted semantic properties.
    """
    shape_type: str  # 'rectangle', 'line', 'blob', etc.
    pixels: Set[Point]
    color: int
    bounding_box: BoundingBox
    properties: Dict  # Flexible dict for shape-specific properties
    
    def __post_init__(self):
        """Compute derived properties after initialization."""
        if 'area' not in self.properties:
            self.properties['area'] = len(self.pixels)
        if 'density' not in self.properties:
            self.properties['density'] = len(self.pixels) / self.bounding_box.area
    
    def is_filled(self, threshold: float = 0.9) -> bool:
        """Check if shape is filled (high density)."""
        return self.properties.get('density', 0) >= threshold
    
    def is_hollow(self, threshold: float = 0.5) -> bool:
        """Check if shape is hollow (low density)."""
        return self.properties.get('density', 1) <= threshold


# ============================================================================
# GRID UTILITIES
# ============================================================================

class GridUtils:
    """Utility functions for grid manipulation and analysis."""
    
    @staticmethod
    def get_neighbors(point: Point, grid_shape: Tuple[int, int], 
                     connectivity: int = 4) -> List[Point]:
        """
        Get valid neighbors of a point in the grid.
        
        Args:
            point: The point to get neighbors for
            grid_shape: (height, width) of the grid
            connectivity: 4 (cardinal) or 8 (all directions)
        
        Returns:
            List of valid neighbor points
        """
        height, width = grid_shape
        directions = Direction.cardinal() if connectivity == 4 else Direction.all_directions()
        
        neighbors = []
        for direction in directions:
            dx, dy = direction.value
            new_point = Point(point.x + dx, point.y + dy)
            
            if 0 <= new_point.x < width and 0 <= new_point.y < height:
                neighbors.append(new_point)
        
        return neighbors
    
    @staticmethod
    def flood_fill(grid: np.ndarray, start: Point, target_color: int,
                   connectivity: int = 4) -> Set[Point]:
        """
        Perform flood fill to extract connected component.
        
        Args:
            grid: The input grid
            start: Starting point for flood fill
            target_color: Color to match
            connectivity: 4 or 8 connectivity
        
        Returns:
            Set of points in the connected component
        """
        if grid[start.y, start.x] != target_color:
            return set()
        
        visited = set()
        stack = [start]
        component = set()
        
        while stack:
            point = stack.pop()
            
            if point in visited:
                continue
            
            visited.add(point)
            
            if grid[point.y, point.x] == target_color:
                component.add(point)
                neighbors = GridUtils.get_neighbors(
                    point, grid.shape, connectivity
                )
                stack.extend(neighbors)
        
        return component
    
    @staticmethod
    def extract_connected_components(grid: np.ndarray, 
                                    background_color: int = 0,
                                    connectivity: int = 8) -> List[Set[Point]]:
        """
        Extract all connected components from grid (excluding background).
        
        Args:
            grid: The input grid
            background_color: Color to treat as background
            connectivity: 4 or 8 connectivity
        
        Returns:
            List of connected components (each a set of Points)
        """
        visited = np.zeros_like(grid, dtype=bool)
        components = []
        height, width = grid.shape
        
        for y in range(height):
            for x in range(width):
                point = Point(x, y)
                
                if visited[y, x] or grid[y, x] == background_color:
                    continue
                
                # Found new component
                color = grid[y, x]
                component = GridUtils.flood_fill(grid, point, color, connectivity)
                
                # Mark as visited
                for p in component:
                    visited[p.y, p.x] = True
                
                components.append(component)
        
        return components
    
    @staticmethod
    def compute_bounding_box(pixels: Set[Point]) -> BoundingBox:
        """Compute axis-aligned bounding box for a set of pixels."""
        if not pixels:
            return BoundingBox(0, 0, 0, 0)
        
        xs = [p.x for p in pixels]
        ys = [p.y for p in pixels]
        
        return BoundingBox(
            min_x=min(xs),
            min_y=min(ys),
            max_x=max(xs),
            max_y=max(ys)
        )
    
    @staticmethod
    def get_border_pixels(pixels: Set[Point], grid_shape: Tuple[int, int]) -> Set[Point]:
        """
        Extract border pixels of a shape (pixels with at least one non-shape neighbor).
        
        Args:
            pixels: Set of pixels in the shape
            grid_shape: Shape of the grid
        
        Returns:
            Set of border pixels
        """
        border = set()
        
        for pixel in pixels:
            neighbors = GridUtils.get_neighbors(pixel, grid_shape, connectivity=4)
            
            # If any neighbor is not in the shape, this is a border pixel
            if any(n not in pixels for n in neighbors):
                border.add(pixel)
        
        return border
    
    @staticmethod
    def get_interior_pixels(pixels: Set[Point], grid_shape: Tuple[int, int]) -> Set[Point]:
        """Extract interior pixels (all neighbors are in the shape)."""
        border = GridUtils.get_border_pixels(pixels, grid_shape)
        return pixels - border


# ============================================================================
# SHAPE DETECTORS
# ============================================================================

class RectangleDetector:
    """
    Detector for rectangular shapes.
    
    Detects both filled and hollow rectangles, extracts properties like
    aspect ratio, orientation, and fill pattern.
    """
    
    @staticmethod
    def is_rectangle(pixels: Set[Point], bbox: BoundingBox, 
                     tolerance: float = 0.05) -> bool:
        """
        Check if a set of pixels forms a rectangle.
        
        Args:
            pixels: Set of pixels to check
            bbox: Bounding box of the pixels
            tolerance: Allowed deviation from perfect rectangle
        
        Returns:
            True if pixels form a rectangle
        """
        expected_area = bbox.area
        actual_area = len(pixels)
        
        # For filled rectangle: area should match bounding box
        if abs(actual_area - expected_area) / expected_area <= tolerance:
            return True
        
        # For hollow rectangle: check if pixels form border
        border_area = 2 * (bbox.width + bbox.height) - 4  # Perimeter
        if border_area > 0 and abs(actual_area - border_area) / border_area <= tolerance:
            return True
        
        return False
    
    @staticmethod
    def detect_rectangles(grid: np.ndarray, 
                         background_color: int = 0) -> List[GeometricShape]:
        """
        Detect all rectangles in the grid.
        
        Args:
            grid: Input grid
            background_color: Color to treat as background
        
        Returns:
            List of detected rectangular shapes
        """
        components = GridUtils.extract_connected_components(grid, background_color)
        rectangles = []
        
        for component in components:
            if len(component) < 4:  # Too small to be a rectangle
                continue
            
            bbox = GridUtils.compute_bounding_box(component)
            
            if RectangleDetector.is_rectangle(component, bbox):
                # Extract color
                sample_point = next(iter(component))
                color = grid[sample_point.y, sample_point.x]
                
                # Compute properties
                properties = {
                    'width': bbox.width,
                    'height': bbox.height,
                    'aspect_ratio': bbox.width / bbox.height,
                    'is_square': abs(bbox.width - bbox.height) <= 1,
                    'perimeter': 2 * (bbox.width + bbox.height),
                }
                
                # Check if filled or hollow
                border = GridUtils.get_border_pixels(component, grid.shape)
                properties['is_filled'] = len(component) > len(border)
                properties['border_pixels'] = border
                properties['interior_pixels'] = component - border
                
                shape = GeometricShape(
                    shape_type='rectangle',
                    pixels=component,
                    color=color,
                    bounding_box=bbox,
                    properties=properties
                )
                
                rectangles.append(shape)
        
        return rectangles


class LineDetector:
    """
    Detector for line segments.
    
    Detects straight lines (horizontal, vertical, diagonal) and extracts
    their direction, length, and endpoints.
    """
    
    @staticmethod
    def is_line(pixels: Set[Point], tolerance: float = 0.1) -> Tuple[bool, Optional[str]]:
        """
        Check if pixels form a straight line using a more robust method.
        """
        if len(pixels) < 2:
            return False, None

        points_list = list(pixels)
        xs = {p.x for p in points_list}
        ys = {p.y for p in points_list}
        
        # Horizontal line: one unique y-coordinate
        if len(ys) == 1:
            return True, 'horizontal'
        
        # Vertical line: one unique x-coordinate
        if len(xs) == 1:
            return True, 'vertical'
        
        # Diagonal line: number of unique x's and y's must equal the total number of pixels
        # This prevents blocks like 2x2 from being classified as diagonal
        if len(xs) == len(pixels) and len(ys) == len(pixels):
            # Further check: the range of x and y should be the same
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            if x_range == y_range:
                return True, 'diagonal'
        
        return False, None
    
    @staticmethod
    def detect_lines(grid: np.ndarray, 
                    background_color: int = 0,
                    min_length: int = 2) -> List[GeometricShape]:
        """
        Detect all line segments in the grid.
        
        Args:
            grid: Input grid
            background_color: Color to treat as background
            min_length: Minimum length for a valid line
        
        Returns:
            List of detected line shapes
        """
        components = GridUtils.extract_connected_components(grid, background_color)
        lines = []
        
        for component in components:
            if len(component) < min_length:
                continue
            
            is_line, direction = LineDetector.is_line(component)
            
            if is_line:
                sample_point = next(iter(component))
                color = grid[sample_point.y, sample_point.x]
                bbox = GridUtils.compute_bounding_box(component)
                
                # Find endpoints
                points_list = list(component)
                if direction == 'horizontal':
                    endpoints = [
                        min(points_list, key=lambda p: p.x),
                        max(points_list, key=lambda p: p.x)
                    ]
                elif direction == 'vertical':
                    endpoints = [
                        min(points_list, key=lambda p: p.y),
                        max(points_list, key=lambda p: p.y)
                    ]
                else:  # diagonal
                    endpoints = [
                        min(points_list, key=lambda p: (p.x + p.y)),
                        max(points_list, key=lambda p: (p.x + p.y))
                    ]
                
                properties = {
                    'direction': direction,
                    'length': len(component),
                    'endpoints': endpoints,
                    'thickness': 1,  # Could be extended to detect thick lines
                }
                
                shape = GeometricShape(
                    shape_type='line',
                    pixels=component,
                    color=color,
                    bounding_box=bbox,
                    properties=properties
                )
                
                lines.append(shape)
        
        return lines


class SymmetryDetector:
    """
    Detector for symmetries in shapes and grids.
    
    Detects reflection symmetry (horizontal, vertical, diagonal) and
    rotational symmetry.
    """
    
    @staticmethod
    def has_vertical_symmetry(pixels: Set[Point], tolerance: int = 0) -> bool:
        """Check if shape has vertical (left-right) symmetry."""
        if not pixels:
            return False
        
        bbox = GridUtils.compute_bounding_box(pixels)
        center_x = (bbox.min_x + bbox.max_x) / 2
        
        for pixel in pixels:
            # Mirror point across vertical axis
            mirror_x = int(2 * center_x - pixel.x)
            mirror = Point(mirror_x, pixel.y)
            
            # Check if mirror point exists in shape
            if mirror not in pixels:
                # Allow some tolerance
                found_close = False
                for dx in range(-tolerance, tolerance + 1):
                    if Point(mirror_x + dx, pixel.y) in pixels:
                        found_close = True
                        break
                
                if not found_close:
                    return False
        
        return True
    
    @staticmethod
    def has_horizontal_symmetry(pixels: Set[Point], tolerance: int = 0) -> bool:
        """Check if shape has horizontal (top-bottom) symmetry."""
        if not pixels:
            return False
        
        bbox = GridUtils.compute_bounding_box(pixels)
        center_y = (bbox.min_y + bbox.max_y) / 2
        
        for pixel in pixels:
            mirror_y = int(2 * center_y - pixel.y)
            mirror = Point(pixel.x, mirror_y)
            
            if mirror not in pixels:
                found_close = False
                for dy in range(-tolerance, tolerance + 1):
                    if Point(pixel.x, mirror_y + dy) in pixels:
                        found_close = True
                        break
                
                if not found_close:
                    return False
        
        return True
    
    @staticmethod
    def detect_symmetries(shape: GeometricShape) -> Dict[str, bool]:
        """
        Detect all types of symmetries in a shape.
        
        Returns:
            Dictionary with symmetry types as keys and boolean values
        """
        return {
            'vertical_symmetry': SymmetryDetector.has_vertical_symmetry(shape.pixels),
            'horizontal_symmetry': SymmetryDetector.has_horizontal_symmetry(shape.pixels),
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

class GeometricVisualizer:
    """Visualization tools for debugging and understanding geometric detection."""
    
    @staticmethod
    def plot_grid(grid: np.ndarray, title: str = "ARC Grid", 
                  figsize: Tuple[int, int] = (8, 8)):
        """Plot a basic ARC grid with color coding."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(grid, cmap='tab10', interpolation='nearest')
        ax.set_title(title)
        ax.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_shapes(grid: np.ndarray, shapes: List[GeometricShape],
                   title: str = "Detected Shapes", figsize: Tuple[int, int] = (10, 10)):
        """
        Plot grid with detected shapes highlighted.
        
        Args:
            grid: Original grid
            shapes: List of detected shapes
            title: Plot title
            figsize: Figure size
        """
        fig, ax = GeometricVisualizer.plot_grid(grid, title, figsize)
        
        for i, shape in enumerate(shapes):
            bbox = shape.bounding_box
            
            # Draw bounding box
            rect = MPLRect(
                (bbox.min_x - 0.5, bbox.min_y - 0.5),
                bbox.width, bbox.height,
                linewidth=2, edgecolor='red', facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{shape.shape_type} #{i+1}"
            ax.text(bbox.min_x, bbox.min_y - 0.7, label,
                   color='red', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        return fig, ax
    
    @staticmethod
    def print_shape_info(shape: GeometricShape):
        """Print detailed information about a shape."""
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
            if key not in ['border_pixels', 'interior_pixels']:
                print(f"  {key}: {value}")


# ============================================================================
# MAIN DETECTION ENGINE
# ============================================================================

class GeometricDetectionEngine:
    """
    Main engine that coordinates all geometric detectors.
    
    This is the high-level interface for shape detection in ARC grids.
    """
    
    def __init__(self, background_color: int = 0):
        """
        Initialize the detection engine.
        
        Args:
            background_color: Color to treat as background
        """
        self.background_color = background_color
        self.detectors = {
            'rectangles': RectangleDetector(),
            'lines': LineDetector(),
        }
    
    def detect_all_shapes(self, grid: np.ndarray) -> Dict[str, List[GeometricShape]]:
        """
        Run all detectors on the grid and resolve ambiguities.
        """
        # --- ÉTAPE A : DÉTECTION BRUTE ---
        detected_rectangles = self.detectors['rectangles'].detect_rectangles(
            grid, self.background_color
        )
        detected_lines = self.detectors['lines'].detect_lines(
            grid, self.background_color
        )

        # --- ÉTAPE B : CORRECTION DES AMBIGUÏTÉS ---
        # Crée un set des pixels de chaque ligne pour une recherche rapide
        line_pixel_sets = {frozenset(line.pixels) for line in detected_lines}

        # Filtre les rectangles : ne garde que ceux dont les pixels ne correspondent pas à une ligne
        final_rectangles = [
            rect for rect in detected_rectangles
            if frozenset(rect.pixels) not in line_pixel_sets
        ]

        results = {
            'rectangles': final_rectangles,
            'lines': detected_lines
        }
        
        # Detect symmetries for each shape
        for shape_type, shapes in results.items():
            for shape in shapes:
                symmetries = SymmetryDetector.detect_symmetries(shape)
                shape.properties.update(symmetries)
        
        return results
    
    def analyze_grid(self, grid: np.ndarray, verbose: bool = True) -> Dict:
        """
        Complete analysis of an ARC grid.
        
        Args:
            grid: Input grid
            verbose: Whether to print detailed information
        
        Returns:
            Dictionary with analysis results
        """
        results = self.detect_all_shapes(grid)
        
        # Compute statistics
        total_shapes = sum(len(shapes) for shapes in results.values())
        
        analysis = {
            'grid_shape': grid.shape,
            'detected_shapes': results,
            'statistics': {
                'total_shapes': total_shapes,
                'rectangles': len(results['rectangles']),
                'lines': len(results['lines']),
            }
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"GRID ANALYSIS")
            print(f"{'='*60}")
            print(f"Grid size: {grid.shape[1]} x {grid.shape[0]}")
            print(f"Total shapes detected: {total_shapes}")
            print(f"  - Rectangles: {len(results['rectangles'])}")
            print(f"  - Lines: {len(results['lines'])}")
            
            for shape_type, shapes in results.items():
                if shapes:
                    print(f"\n{shape_type.upper()}:")
                    for i, shape in enumerate(shapes):
                        print(f"\n  Shape {i+1}:")
                        print(f"    Color: {shape.color}")
                        print(f"    Size: {shape.bounding_box.width}x{shape.bounding_box.height}")
                        print(f"    Density: {shape.properties.get('density', 0):.2%}")
                        if 'vertical_symmetry' in shape.properties:
                            print(f"    Vertical symmetry: {shape.properties['vertical_symmetry']}")
                        if 'horizontal_symmetry' in shape.properties:
                            print(f"    Horizontal symmetry: {shape.properties['horizontal_symmetry']}")
        
        return analysis


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create a simple test grid
    test_grid = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 2, 2],
        [0, 1, 1, 1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0, 0, 2, 2],
        [0, 0, 3, 3, 3, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    
    # Initialize detection engine
    engine = GeometricDetectionEngine(background_color=0)
    
    # Analyze the grid
    analysis = engine.analyze_grid(test_grid, verbose=True)
    
    # Visualize results
    all_shapes = []
    for shapes in analysis['detected_shapes'].values():
        all_shapes.extend(shapes)
    
    fig, ax = GeometricVisualizer.plot_shapes(test_grid, all_shapes)
    plt.show()
    
    # Print detailed info for first shape
    if all_shapes:
        GeometricVisualizer.print_shape_info(all_shapes[0])