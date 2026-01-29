"""
transformation_detector.py - Transformation Detection Module
=============================================================
Automatically detects transformations between input and output grids.

Detects:
    - Translation (dx, dy)
    - Rotation (90°, 180°, 270°)
    - Reflection (horizontal, vertical, diagonal)
    - Color changes
    - Scaling
    - Draw line (connecting two points)
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np

from .types import Grid, GeometricObject


@dataclass
class TransformationResult:
    """
    Result of transformation detection.
    
    Attributes:
        transformation_type: Type of transformation detected
        confidence: Confidence score (0.0 to 1.0)
        parameters: Transformation-specific parameters
        objects_matched: List of (input_obj, output_obj) pairs
    """
    transformation_type: str
    confidence: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    objects_matched: List[Tuple[GeometricObject, GeometricObject]] = field(default_factory=list)


class TransformationDetector:
    """
    Detects transformations between input and output grids.
    
    Analyzes pairs of grids to identify:
        - Spatial transformations (translation, rotation, reflection)
        - Color transformations
        - Size transformations (scaling)
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the detector.
        
        Args:
            verbose: Whether to print detection details
        """
        self.verbose = verbose
    
    def detect_all(self, input_grid: Grid, output_grid: Grid) -> List[TransformationResult]:
        """
        Detect all possible transformations between input and output.
        
        Args:
            input_grid: The input grid (with detected objects)
            output_grid: The output grid (with detected objects)
            
        Returns:
            List of detected transformations, sorted by confidence
        """
        results = []
        
        # Try to detect various transformations
        # Order matters! 
        
        # PRIORITY -1: Check for SIZE CHANGE first
        # If grids have different sizes, only certain transformations are possible
        in_h, in_w = input_grid.data.shape
        out_h, out_w = output_grid.data.shape
        size_changed = (in_h != out_h) or (in_w != out_w)
        
        if size_changed:
            # Output is larger - check for tiling first
            if out_h >= in_h and out_w >= in_w:
                tiling = self.detect_tiling(input_grid, output_grid)
                if tiling and tiling.confidence >= 0.95:
                    return [tiling]  # Clear tiling pattern
            
            # Output is smaller or different dimensions - check scaling
            scaling = self.detect_scaling(input_grid, output_grid)
            if scaling:
                return [scaling]
            
            # If no clear size-change transformation found, return empty
            # (most transformations require same-size grids)
            return []
        
        # === SAME SIZE GRIDS - check all transformations ===
        
        # PRIORITY 0: Check draw_line FIRST (very specific: 2 points → line)
        # If detected with 100% confidence, return immediately
        draw_line = self.detect_draw_line(input_grid, output_grid)
        if draw_line and draw_line.confidence >= 1.0:
            return [draw_line]  # Return immediately - this is a clear draw_line task
        
        # PRIORITY 0.5: Check GRID-LEVEL reflection BEFORE anything else
        # This is a very specific pattern: entire grid flipped
        grid_reflection = self._detect_grid_level_reflection(input_grid, output_grid)
        if grid_reflection and grid_reflection.confidence >= 1.0:
            return [grid_reflection]  # Return immediately - this is a clear grid-level reflection
        
        # PRIORITY 0.6: Check GRID-LEVEL rotation
        grid_rotation = self._detect_grid_level_rotation(input_grid, output_grid)
        if grid_rotation and grid_rotation.confidence >= 1.0:
            return [grid_rotation]  # Return immediately - this is a clear grid-level rotation
        
        # PRIORITY 1: Translation (but only if no grid-level transform was found)
        translation = self.detect_translation(input_grid, output_grid)
        if translation:
            results.append(translation)
        
        # PRIORITY 2: Check reflection BEFORE rotation (object-level)
        reflection = self.detect_reflection(input_grid, output_grid)
        if reflection:
            results.append(reflection)
        
        # PRIORITY 3: Check rotation (object-level)
        # Allow rotation even if reflection was found - they might be different
        rotation = self.detect_rotation(input_grid, output_grid)
        if rotation:
            results.append(rotation)
        
        # PRIORITY 4: Color change
        color_change = self.detect_color_change(input_grid, output_grid)
        if color_change:
            results.append(color_change)
        
        # PRIORITY 5: Scaling (for same-size grids where object scales)
        scaling = self.detect_scaling(input_grid, output_grid)
        if scaling:
            results.append(scaling)
        
        # Add draw_line if detected but not with 100% confidence
        if draw_line:
            results.append(draw_line)
        
        # Blob-specific transformation detection (only if nothing else found)
        if not results:
            blob_transform = self.detect_blob_transformation(input_grid, output_grid)
            if blob_transform:
                results.append(blob_transform)
        
        # COMPOSITE DETECTION: If no high-confidence result, try composite transformations
        if not results or (results and results[0].confidence < 0.95):
            composite = self.detect_composite_transformation(input_grid, output_grid)
            if composite and composite.confidence > 0.9:
                # If composite is better than any simple transformation, add it
                if not results or composite.confidence > results[0].confidence:
                    results.insert(0, composite)
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
    
    def _detect_grid_level_reflection(self, input_grid: Grid, output_grid: Grid) -> Optional[TransformationResult]:
        """
        Detect if the entire grid is reflected (flipped).
        
        This checks for exact np.flipud or np.fliplr matches.
        Returns immediately if found with 100% confidence.
        """
        in_data = input_grid.data
        out_data = output_grid.data
        
        if in_data.shape != out_data.shape:
            return None
        
        # Check horizontal reflection (flip up-down)
        if np.array_equal(np.flipud(in_data), out_data):
            return TransformationResult(
                transformation_type="reflection",
                confidence=1.0,
                parameters={"axis": "horizontal", "grid_level": True}
            )
        
        # Check vertical reflection (flip left-right)
        if np.array_equal(np.fliplr(in_data), out_data):
            return TransformationResult(
                transformation_type="reflection",
                confidence=1.0,
                parameters={"axis": "vertical", "grid_level": True}
            )
        
        return None
    
    def _detect_grid_level_rotation(self, input_grid: Grid, output_grid: Grid) -> Optional[TransformationResult]:
        """
        Detect if the entire grid is rotated.
        
        This checks for exact np.rot90 matches at 90, 180, 270 degrees.
        Returns immediately if found with 100% confidence.
        """
        in_data = input_grid.data
        out_data = output_grid.data
        
        # Check each rotation angle
        for angle in [90, 180, 270]:
            rotated = np.rot90(in_data, k=angle // 90)
            if rotated.shape == out_data.shape and np.array_equal(rotated, out_data):
                return TransformationResult(
                    transformation_type="rotation",
                    confidence=1.0,
                    parameters={"angle": angle, "grid_level": True}
                )
        
        return None
    
    def detect_translation(self, input_grid: Grid, output_grid: Grid) -> Optional[TransformationResult]:
        """
        Detect if objects are translated (moved) between input and output.
        
        Returns:
            TransformationResult with dx, dy if translation detected
        """
        if not input_grid.objects or not output_grid.objects:
            return None
        
        translations = []
        matched_pairs = []
        
        for in_obj in input_grid.objects:
            # Find matching object in output (same color, same shape, same size)
            for out_obj in output_grid.objects:
                if self._objects_match(in_obj, out_obj, check_position=False):
                    if in_obj.bounding_box and out_obj.bounding_box:
                        dx = out_obj.bounding_box[1] - in_obj.bounding_box[1]
                        dy = out_obj.bounding_box[0] - in_obj.bounding_box[0]
                        translations.append((dx, dy))
                        matched_pairs.append((in_obj, out_obj))
                    break
        
        if not translations:
            return None
        
        # Check if all translations are consistent
        if len(set(translations)) == 1:
            dx, dy = translations[0]
            # Skip if no actual movement (dx=0 AND dy=0)
            if dx == 0 and dy == 0:
                return None
            return TransformationResult(
                transformation_type="translation",
                confidence=1.0,
                parameters={"dx": dx, "dy": dy},
                objects_matched=matched_pairs
            )
        else:
            # Multiple different translations - return the most common
            from collections import Counter
            most_common = Counter(translations).most_common(1)[0]
            dx, dy = most_common[0]
            # Skip if no actual movement (dx=0 AND dy=0)
            if dx == 0 and dy == 0:
                return None
            confidence = most_common[1] / len(translations)
            return TransformationResult(
                transformation_type="translation",
                confidence=confidence,
                parameters={"dx": dx, "dy": dy},
                objects_matched=matched_pairs
            )
    
    def detect_rotation(self, input_grid: Grid, output_grid: Grid) -> Optional[TransformationResult]:
        """
        Detect if the grid or objects are rotated.
        
        Returns:
            TransformationResult with rotation angle if detected
        """
        in_data = input_grid.data
        out_data = output_grid.data
        
        # Check grid-level rotation
        for angle in [90, 180, 270]:
            rotated = np.rot90(in_data, k=angle // 90)
            if rotated.shape == out_data.shape and np.array_equal(rotated, out_data):
                return TransformationResult(
                    transformation_type="rotation",
                    confidence=1.0,
                    parameters={"angle": angle}
                )
        
        # Check object-level rotation (same color objects)
        if input_grid.objects and output_grid.objects:
            for in_obj in input_grid.objects:
                for out_obj in output_grid.objects:
                    if in_obj.color == out_obj.color:
                        obj_rotation = self._detect_object_rotation(in_obj, out_obj)
                        if obj_rotation:
                            # Check if position also changed (would indicate composite transform)
                            in_center = ((in_obj.bounding_box[0] + in_obj.bounding_box[2]) / 2,
                                        (in_obj.bounding_box[1] + in_obj.bounding_box[3]) / 2)
                            out_center = ((out_obj.bounding_box[0] + out_obj.bounding_box[2]) / 2,
                                         (out_obj.bounding_box[1] + out_obj.bounding_box[3]) / 2)
                            
                            # If centers moved significantly, it's likely composite (rotation + translation)
                            center_moved = abs(in_center[0] - out_center[0]) > 0.5 or abs(in_center[1] - out_center[1]) > 0.5
                            
                            if center_moved:
                                # Lower confidence - might be composite transformation
                                return TransformationResult(
                                    transformation_type="rotation",
                                    confidence=0.7,  # Lower confidence since position changed
                                    parameters={"angle": obj_rotation, "per_object": True, "color": in_obj.color, "position_changed": True},
                                    objects_matched=[(in_obj, out_obj)]
                                )
                            else:
                                return TransformationResult(
                                    transformation_type="rotation",
                                    confidence=1.0,  # High confidence for pure rotation
                                    parameters={"angle": obj_rotation, "per_object": True, "color": in_obj.color},
                                    objects_matched=[(in_obj, out_obj)]
                                )
        
        # Check object-level rotation (ignoring color - for different colored examples)
        # This checks if the shape structure rotates regardless of color
        if input_grid.objects and output_grid.objects:
            for in_obj in input_grid.objects:
                for out_obj in output_grid.objects:
                    # Match by shape type and area, ignore color
                    if in_obj.area == out_obj.area and in_obj.object_type == out_obj.object_type:
                        obj_rotation = self._detect_object_rotation(in_obj, out_obj)
                        if obj_rotation:
                            return TransformationResult(
                                transformation_type="rotation",
                                confidence=0.9,  # Slightly lower since colors differ
                                parameters={"angle": obj_rotation, "per_object": True},
                                objects_matched=[(in_obj, out_obj)]
                            )
        
        return None
    
    def detect_reflection(self, input_grid: Grid, output_grid: Grid) -> Optional[TransformationResult]:
        """
        Detect if the grid or objects are reflected (mirrored).
        
        Returns:
            TransformationResult with axis if reflection detected
        """
        in_data = input_grid.data
        out_data = output_grid.data
        
        if in_data.shape != out_data.shape:
            return None
        
        # First, check OBJECT-LEVEL reflection (before grid-level)
        # This is important for shapes that are reflected within their bounding box
        if input_grid.objects and output_grid.objects:
            for in_obj in input_grid.objects:
                for out_obj in output_grid.objects:
                    # Match by color, area, and same dimensions (reflection keeps dimensions)
                    if (in_obj.color == out_obj.color and 
                        in_obj.area == out_obj.area and
                        in_obj.width == out_obj.width and 
                        in_obj.height == out_obj.height):
                        
                        obj_reflection = self._detect_object_reflection(in_obj, out_obj)
                        if obj_reflection:
                            return TransformationResult(
                                transformation_type="reflection",
                                confidence=1.0,  # High confidence for object reflection
                                parameters={"axis": obj_reflection, "per_object": True, "color": in_obj.color},
                                objects_matched=[(in_obj, out_obj)]
                            )
        
        # Grid-level reflection checks
        # Horizontal reflection (flip up-down)
        if np.array_equal(np.flipud(in_data), out_data):
            return TransformationResult(
                transformation_type="reflection",
                confidence=1.0,
                parameters={"axis": "horizontal"}
            )
        
        # Vertical reflection (flip left-right)
        if np.array_equal(np.fliplr(in_data), out_data):
            return TransformationResult(
                transformation_type="reflection",
                confidence=1.0,
                parameters={"axis": "vertical"}
            )
        
        # Diagonal reflection (transpose)
        if in_data.shape[0] == in_data.shape[1]:  # Square grid
            if np.array_equal(in_data.T, out_data):
                return TransformationResult(
                    transformation_type="reflection",
                    confidence=1.0,
                    parameters={"axis": "diagonal_main"}
                )
            
            # Anti-diagonal reflection
            anti_diag = np.flip(np.flip(in_data, 0), 1).T
            if np.array_equal(anti_diag, out_data):
                return TransformationResult(
                    transformation_type="reflection",
                    confidence=1.0,
                    parameters={"axis": "diagonal_anti"}
                )
        
        return None
    
    def _detect_object_reflection(self, in_obj: GeometricObject, out_obj: GeometricObject) -> Optional[str]:
        """
        Detect if an object has been reflected (mirrored) within its bounding box.
        
        Args:
            in_obj: Input object
            out_obj: Output object
            
        Returns:
            "horizontal" or "vertical" if reflection detected, None otherwise
        """
        if not in_obj.pixels or not out_obj.pixels:
            return None
        
        # Normalize both objects to origin
        in_min_r = min(p[0] for p in in_obj.pixels)
        in_min_c = min(p[1] for p in in_obj.pixels)
        in_normalized = frozenset((p[0] - in_min_r, p[1] - in_min_c) for p in in_obj.pixels)
        
        out_min_r = min(p[0] for p in out_obj.pixels)
        out_min_c = min(p[1] for p in out_obj.pixels)
        out_normalized = frozenset((p[0] - out_min_r, p[1] - out_min_c) for p in out_obj.pixels)
        
        # If identical, no reflection (might be identity or translation)
        if in_normalized == out_normalized:
            return None
        
        # Get dimensions of normalized shape
        in_h = max(p[0] for p in in_normalized) + 1
        in_w = max(p[1] for p in in_normalized) + 1
        
        # Check vertical reflection (flip left-right)
        vertical_reflected = set()
        for r, c in in_normalized:
            vertical_reflected.add((r, in_w - 1 - c))
        # Re-normalize
        if vertical_reflected:
            vr_min_r = min(p[0] for p in vertical_reflected)
            vr_min_c = min(p[1] for p in vertical_reflected)
            vertical_reflected = frozenset((p[0] - vr_min_r, p[1] - vr_min_c) for p in vertical_reflected)
        
        if vertical_reflected == out_normalized:
            return "vertical"
        
        # Check horizontal reflection (flip up-down)
        horizontal_reflected = set()
        for r, c in in_normalized:
            horizontal_reflected.add((in_h - 1 - r, c))
        # Re-normalize
        if horizontal_reflected:
            hr_min_r = min(p[0] for p in horizontal_reflected)
            hr_min_c = min(p[1] for p in horizontal_reflected)
            horizontal_reflected = frozenset((p[0] - hr_min_r, p[1] - hr_min_c) for p in horizontal_reflected)
        
        if horizontal_reflected == out_normalized:
            return "horizontal"
        
        return None
    
    def detect_color_change(self, input_grid: Grid, output_grid: Grid) -> Optional[TransformationResult]:
        """
        Detect if colors are changed between input and output.
        
        Returns:
            TransformationResult with color mapping if detected
        """
        in_data = input_grid.data
        out_data = output_grid.data
        
        if in_data.shape != out_data.shape:
            return None
        
        # Find color mapping
        color_map = {}
        for i in range(in_data.shape[0]):
            for j in range(in_data.shape[1]):
                in_color = int(in_data[i, j])
                out_color = int(out_data[i, j])
                
                if in_color in color_map:
                    if color_map[in_color] != out_color:
                        # Inconsistent mapping - not a simple color change
                        return None
                else:
                    color_map[in_color] = out_color
        
        # Check if any color actually changed
        changes = {k: v for k, v in color_map.items() if k != v}
        
        if not changes:
            return None
        
        return TransformationResult(
            transformation_type="color_change",
            confidence=1.0,
            parameters={"color_map": color_map, "changes": changes}
        )
    
    def detect_scaling(self, input_grid: Grid, output_grid: Grid) -> Optional[TransformationResult]:
        """
        Detect if objects are scaled (enlarged or reduced).
        
        Returns:
            TransformationResult with scale factor if detected
        """
        if not input_grid.objects or not output_grid.objects:
            return None
        
        scale_factors = []
        matched_pairs = []
        
        for in_obj in input_grid.objects:
            for out_obj in output_grid.objects:
                if in_obj.color == out_obj.color:
                    # Check if shapes are similar but different size
                    if in_obj.width > 0 and in_obj.height > 0:
                        scale_x = out_obj.width / in_obj.width
                        scale_y = out_obj.height / in_obj.height
                        
                        if abs(scale_x - scale_y) < 0.1:  # Uniform scaling
                            scale = (scale_x + scale_y) / 2
                            if scale != 1.0:
                                scale_factors.append(scale)
                                matched_pairs.append((in_obj, out_obj))
                    break
        
        if not scale_factors:
            return None
        
        # Check consistency
        avg_scale = sum(scale_factors) / len(scale_factors)
        if all(abs(s - avg_scale) < 0.1 for s in scale_factors):
            return TransformationResult(
                transformation_type="scaling",
                confidence=0.9,
                parameters={"factor": round(avg_scale, 2)},
                objects_matched=matched_pairs
            )
        
        return None
    
    def detect_tiling(self, input_grid: Grid, output_grid: Grid) -> Optional[TransformationResult]:
        """
        Detect if the output is a tiled repetition of the input pattern.
        
        This checks if:
        1. The output is larger than the input
        2. The input pattern tiles perfectly to create the output
        
        Returns:
            TransformationResult with tiling parameters if detected
        """
        in_data = input_grid.data
        out_data = output_grid.data
        
        in_h, in_w = in_data.shape
        out_h, out_w = out_data.shape
        
        # Output should be larger than or equal to input
        if out_h < in_h or out_w < in_w:
            return None
        
        # Check if output dimensions are multiples of input dimensions
        if out_h % in_h != 0 or out_w % in_w != 0:
            return None
        
        reps_v = out_h // in_h  # Vertical repetitions
        reps_h = out_w // in_w  # Horizontal repetitions
        
        # Must have at least some tiling (not just 1x1)
        if reps_v == 1 and reps_h == 1:
            return None
        
        # Check if input tiles perfectly to create output
        match_count = 0
        total_tiles = reps_v * reps_h
        
        for row_idx in range(reps_v):
            for col_idx in range(reps_h):
                row_start = row_idx * in_h
                col_start = col_idx * in_w
                
                region = out_data[row_start:row_start + in_h, 
                                 col_start:col_start + in_w]
                
                if np.array_equal(region, in_data):
                    match_count += 1
        
        coverage = match_count / total_tiles
        
        if coverage >= 0.95:  # At least 95% match
            return TransformationResult(
                transformation_type="tiling",
                confidence=1.0 if coverage == 1.0 else 0.9,
                parameters={
                    "repetitions_horizontal": reps_h,
                    "repetitions_vertical": reps_v,
                    "tile_width": in_w,
                    "tile_height": in_h,
                    "coverage": coverage
                }
            )
        
        return None
    
    def detect_draw_line(self, input_grid: Grid, output_grid: Grid) -> Optional[TransformationResult]:
        """
        Detect if two points in the input are connected by a line in the output.
        
        Checks:
        1. Input has exactly 2 isolated pixels of the same color
        2. Output has a line connecting those 2 points
        
        Returns:
            TransformationResult with line endpoints if detected
        """
        # Analyze per-color to find pairs of points
        for color in input_grid.unique_colors:
            # Get positions of this color in input
            in_positions = list(zip(*np.where(input_grid.data == color)))
            
            # We need exactly 2 pixels (two points)
            if len(in_positions) != 2:
                continue
            
            # Get positions of this color in output
            out_positions = set(zip(*np.where(output_grid.data == color)))
            
            # Check if output has more pixels (a line was drawn)
            if len(out_positions) <= 2:
                continue
            
            # Verify that the original 2 points are in the output
            if not all((r, c) in out_positions for r, c in in_positions):
                continue
            
            # Check if output forms a valid line between the 2 points
            p1, p2 = in_positions[0], in_positions[1]
            expected_line = self._get_line_pixels(p1, p2)
            
            if set(expected_line) == out_positions:
                return TransformationResult(
                    transformation_type="draw_line",
                    confidence=1.0,
                    parameters={
                        "color": color,
                        "point1": {"row": p1[0], "col": p1[1]},
                        "point2": {"row": p2[0], "col": p2[1]},
                        "line_type": self._classify_line_type(p1, p2)
                    }
                )
        
        return None
    
    def _get_line_pixels(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get all pixels that form a line between two points using Bresenham's algorithm.
        
        Args:
            p1: (row, col) of first point
            p2: (row, col) of second point
            
        Returns:
            List of (row, col) tuples forming the line
        """
        r1, c1 = p1
        r2, c2 = p2
        
        pixels = []
        
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        
        sr = 1 if r1 < r2 else -1
        sc = 1 if c1 < c2 else -1
        
        if dc > dr:
            # More horizontal than vertical
            err = dc // 2
            r = r1
            for c in range(c1, c2 + sc, sc):
                pixels.append((r, c))
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
        else:
            # More vertical than horizontal (or diagonal)
            err = dr // 2
            c = c1
            for r in range(r1, r2 + sr, sr):
                pixels.append((r, c))
                err -= dc
                if err < 0:
                    c += sc
                    err += dr
        
        return pixels
    
    def _classify_line_type(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> str:
        """
        Classify the type of line between two points.
        
        Returns:
            "horizontal", "vertical", or "diagonal"
        """
        r1, c1 = p1
        r2, c2 = p2
        
        if r1 == r2:
            return "horizontal"
        elif c1 == c2:
            return "vertical"
        else:
            return "diagonal"
    
    # ==================== BLOB TRANSFORMATION DETECTION ====================
    
    def detect_blob_transformation(
        self, 
        input_grid: Grid, 
        output_grid: Grid
    ) -> Optional[TransformationResult]:
        """
        Detect transformations applied to blob (irregular) shapes.
        
        Blobs are compared using shape signatures to detect:
        - Translation (same shape, different position)
        - Rotation (same shape but rotated)
        - Reflection (same shape but mirrored)
        - Color change (same shape and position, different color)
        
        Returns:
            TransformationResult if a blob transformation is detected
        """
        # Find blob objects in both grids
        in_blobs = [obj for obj in input_grid.objects if "blob" in obj.object_type]
        out_blobs = [obj for obj in output_grid.objects if "blob" in obj.object_type]
        
        if not in_blobs and not out_blobs:
            return None
        
        # If no blobs in either grid, also check for any irregular objects
        if not in_blobs:
            in_blobs = [obj for obj in input_grid.objects 
                       if obj.object_type not in ["square", "rectangle", "horizontal_line", 
                                                   "vertical_line", "point"]]
        if not out_blobs:
            out_blobs = [obj for obj in output_grid.objects 
                        if obj.object_type not in ["square", "rectangle", "horizontal_line", 
                                                    "vertical_line", "point"]]
        
        if not in_blobs or not out_blobs:
            return None
        
        # Try to match blobs
        for in_blob in in_blobs:
            for out_blob in out_blobs:
                transform = self._detect_blob_transform(in_blob, out_blob)
                if transform:
                    return transform
        
        return None
    
    def _detect_blob_transform(
        self, 
        in_obj: GeometricObject, 
        out_obj: GeometricObject
    ) -> Optional[TransformationResult]:
        """
        Detect the transformation between two specific blob objects.
        """
        if not in_obj.pixels or not out_obj.pixels:
            return None
        
        # Normalize both shapes to origin
        in_norm = self._normalize_blob(in_obj.pixels)
        out_norm = self._normalize_blob(out_obj.pixels)
        
        # Check for direct match (translation or color change)
        if in_norm == out_norm:
            # Calculate translation
            in_min_r = min(p[0] for p in in_obj.pixels)
            in_min_c = min(p[1] for p in in_obj.pixels)
            out_min_r = min(p[0] for p in out_obj.pixels)
            out_min_c = min(p[1] for p in out_obj.pixels)
            
            dx = out_min_c - in_min_c
            dy = out_min_r - in_min_r
            
            # Check for color change
            if in_obj.color != out_obj.color:
                if dx == 0 and dy == 0:
                    return TransformationResult(
                        transformation_type="color_change",
                        confidence=0.95,
                        parameters={
                            "from_color": in_obj.color,
                            "to_color": out_obj.color,
                            "shape_type": in_obj.object_type
                        },
                        objects_matched=[(in_obj, out_obj)]
                    )
                else:
                    # Both translation and color change
                    return TransformationResult(
                        transformation_type="translation_and_color",
                        confidence=0.90,
                        parameters={
                            "dx": dx, "dy": dy,
                            "from_color": in_obj.color,
                            "to_color": out_obj.color
                        },
                        objects_matched=[(in_obj, out_obj)]
                    )
            elif dx != 0 or dy != 0:
                return TransformationResult(
                    transformation_type="translation",
                    confidence=0.95,
                    parameters={"dx": dx, "dy": dy, "shape_type": in_obj.object_type},
                    objects_matched=[(in_obj, out_obj)]
                )
        
        # Check for rotations
        for angle in [90, 180, 270]:
            rotated = self._rotate_blob(in_norm, angle)
            if rotated == out_norm:
                return TransformationResult(
                    transformation_type="rotation",
                    confidence=0.95,
                    parameters={
                        "angle": angle, 
                        "per_object": True,
                        "shape_type": in_obj.object_type,
                        "color": in_obj.color
                    },
                    objects_matched=[(in_obj, out_obj)]
                )
        
        # Check for reflections
        for axis in ["horizontal", "vertical"]:
            reflected = self._reflect_blob(in_norm, axis)
            if reflected == out_norm:
                return TransformationResult(
                    transformation_type="reflection",
                    confidence=0.95,
                    parameters={
                        "axis": axis,
                        "shape_type": in_obj.object_type,
                        "color": in_obj.color
                    },
                    objects_matched=[(in_obj, out_obj)]
                )
        
        return None
    
    def _normalize_blob(self, pixels: List[Tuple[int, int]]) -> frozenset:
        """Normalize blob pixels to origin (0,0)."""
        if not pixels:
            return frozenset()
        
        min_r = min(p[0] for p in pixels)
        min_c = min(p[1] for p in pixels)
        
        return frozenset((p[0] - min_r, p[1] - min_c) for p in pixels)
    
    def _rotate_blob(self, pixels: frozenset, angle: int) -> frozenset:
        """Rotate normalized blob pixels and re-normalize."""
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
        if rotated:
            min_r = min(p[0] for p in rotated)
            min_c = min(p[1] for p in rotated)
            return frozenset((p[0] - min_r, p[1] - min_c) for p in rotated)
        return frozenset()
    
    def _reflect_blob(self, pixels: frozenset, axis: str) -> frozenset:
        """Reflect normalized blob pixels and re-normalize."""
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
        if reflected:
            min_r = min(p[0] for p in reflected)
            min_c = min(p[1] for p in reflected)
            return frozenset((p[0] - min_r, p[1] - min_c) for p in reflected)
        return frozenset()
    
    def _objects_match(self, obj1: GeometricObject, obj2: GeometricObject, 
                       check_position: bool = True) -> bool:
        """
        Check if two objects match (same color, shape, size).
        """
        # Same color
        if obj1.color != obj2.color:
            return False
        
        # Same size
        if obj1.width != obj2.width or obj1.height != obj2.height:
            return False
        
        # Same area (number of pixels)
        if obj1.area != obj2.area:
            return False
        
        # Same position (optional)
        if check_position and obj1.bounding_box != obj2.bounding_box:
            return False
        
        return True
    
    def _detect_object_rotation(self, in_obj: GeometricObject, 
                                out_obj: GeometricObject) -> Optional[int]:
        """
        Detect rotation angle between two objects.
        """
        if not in_obj.pixels or not out_obj.pixels:
            return None
        
        # Normalize pixels to origin
        in_min_r = min(p[0] for p in in_obj.pixels)
        in_min_c = min(p[1] for p in in_obj.pixels)
        in_normalized = set((p[0] - in_min_r, p[1] - in_min_c) for p in in_obj.pixels)
        
        out_min_r = min(p[0] for p in out_obj.pixels)
        out_min_c = min(p[1] for p in out_obj.pixels)
        out_normalized = set((p[0] - out_min_r, p[1] - out_min_c) for p in out_obj.pixels)
        
        h = max(p[0] for p in in_normalized) + 1
        w = max(p[1] for p in in_normalized) + 1
        
        # Check various rotations
        for angle in [90, 180, 270]:
            rotated = self._rotate_pixels(in_normalized, h, w, angle)
            
            # Normalize rotated pixels
            if rotated:
                min_r = min(p[0] for p in rotated)
                min_c = min(p[1] for p in rotated)
                rotated_normalized = set((p[0] - min_r, p[1] - min_c) for p in rotated)
                
                if rotated_normalized == out_normalized:
                    return angle
        
        return None
    
    def _rotate_pixels(self, pixels: set, height: int, width: int, angle: int) -> set:
        """Rotate a set of pixels by the given angle."""
        rotated = set()
        
        for r, c in pixels:
            if angle == 90:
                new_r, new_c = c, height - 1 - r
            elif angle == 180:
                new_r, new_c = height - 1 - r, width - 1 - c
            elif angle == 270:
                new_r, new_c = width - 1 - c, r
            else:
                new_r, new_c = r, c
            rotated.add((new_r, new_c))
        
        return rotated
    
    def describe_transformations(self, results: List[TransformationResult]) -> str:
        """
        Generate a human-readable description of detected transformations.
        """
        if not results:
            return "No transformations detected."
        
        descriptions = []
        for result in results:
            if result.transformation_type == "translation":
                dx = result.parameters.get("dx", 0)
                dy = result.parameters.get("dy", 0)
                desc = f"Translation: dx={dx} (right), dy={dy} (down)"
            elif result.transformation_type == "rotation":
                angle = result.parameters.get("angle", 0)
                desc = f"Rotation: {angle}° clockwise"
            elif result.transformation_type == "reflection":
                axis = result.parameters.get("axis", "unknown")
                desc = f"Reflection: {axis} axis"
            elif result.transformation_type == "color_change":
                changes = result.parameters.get("changes", {})
                desc = f"Color change: {changes}"
            elif result.transformation_type == "scaling":
                factor = result.parameters.get("factor", 1.0)
                desc = f"Scaling: factor={factor}x"
            else:
                desc = f"{result.transformation_type}: {result.parameters}"
            
            descriptions.append(f"[{result.confidence:.0%}] {desc}")
        
        return "\n".join(descriptions)
    
    # ==================== MULTI-TRANSFORM DETECTION ====================
    
    def detect_per_color_transformations(self, input_grid: Grid, output_grid: Grid) -> Dict[int, TransformationResult]:
        """
        Detect transformations for each color separately.
        
        This allows detecting different transformations for different objects
        based on their color.
        
        Args:
            input_grid: The input grid (with detected objects)
            output_grid: The output grid (with detected objects)
            
        Returns:
            Dictionary mapping color -> TransformationResult
        """
        per_color_transforms = {}
        
        # Get all colors in input (excluding background 0)
        input_colors = set(input_grid.unique_colors)
        output_colors = set(output_grid.unique_colors)
        
        if self.verbose:
            print(f"  Detecting per-color transformations...")
            print(f"    Input colors: {input_colors}")
            print(f"    Output colors: {output_colors}")
        
        # For each input color, find what transformation applies to it
        for color in input_colors:
            transform = self._detect_single_color_transform(
                input_grid, output_grid, color
            )
            if transform:
                per_color_transforms[color] = transform
                if self.verbose:
                    print(f"    Color {color}: {transform.transformation_type} - {transform.parameters}")
        
        return per_color_transforms
    
    def _detect_single_color_transform(
        self, 
        input_grid: Grid, 
        output_grid: Grid, 
        color: int
    ) -> Optional[TransformationResult]:
        """
        Detect transformation for a single color.
        
        Checks in order:
        1. Translation (same color appears at different position)
        2. Rotation (shape rotates but stays same color)
        3. Color change (same position but different color)
        
        Args:
            input_grid: Input grid
            output_grid: Output grid
            color: The color to analyze
            
        Returns:
            TransformationResult for this color, or None
        """
        in_data = input_grid.data
        out_data = output_grid.data
        
        # Get pixels of this color in input
        in_positions = set(zip(*np.where(in_data == color)))
        
        if not in_positions:
            return None
        
        # Check if same color exists in output at different position (TRANSLATION)
        out_same_color_positions = set(zip(*np.where(out_data == color)))
        
        if out_same_color_positions and out_same_color_positions != in_positions:
            # Same color moved - detect translation
            translation = self._detect_color_translation(in_positions, out_same_color_positions)
            if translation:
                return TransformationResult(
                    transformation_type="translation",
                    confidence=1.0,
                    parameters={"dx": translation[0], "dy": translation[1], "color": color}
                )
        
        # Check if color rotated (same color, rotated shape)
        in_obj = self._get_object_by_color(input_grid, color)
        out_obj = self._get_object_by_color(output_grid, color)
        
        if in_obj and out_obj:
            rotation = self._detect_object_rotation(in_obj, out_obj)
            if rotation:
                return TransformationResult(
                    transformation_type="rotation",
                    confidence=1.0,
                    parameters={"angle": rotation, "color": color}
                )
        
        # Check if color changed (same positions but different color in output)
        for new_color in output_grid.unique_colors:
            if new_color == color:
                continue
            out_positions = set(zip(*np.where(out_data == new_color)))
            if in_positions == out_positions:
                # Exact same positions - color change
                return TransformationResult(
                    transformation_type="color_change",
                    confidence=1.0,
                    parameters={"from_color": color, "to_color": new_color}
                )
        
        # Check if color stayed in place (no transformation)
        if in_positions == out_same_color_positions:
            return TransformationResult(
                transformation_type="identity",
                confidence=1.0,
                parameters={"color": color}
            )
        
        return None
    
    def _detect_color_translation(
        self, 
        in_positions: set, 
        out_positions: set
    ) -> Optional[Tuple[int, int]]:
        """
        Detect translation vector between two sets of positions.
        
        Returns:
            (dx, dy) tuple if consistent translation found, None otherwise
        """
        if len(in_positions) != len(out_positions):
            return None
        
        # Try to find consistent translation
        in_list = sorted(list(in_positions))
        out_list = sorted(list(out_positions))
        
        # Calculate translation for first point
        dy = out_list[0][0] - in_list[0][0]
        dx = out_list[0][1] - in_list[0][1]
        
        # Check if this translation works for all points
        translated = set((r + dy, c + dx) for r, c in in_positions)
        
        if translated == out_positions:
            return (dx, dy)
        
        return None
    
    def _get_object_by_color(self, grid: Grid, color: int) -> Optional[GeometricObject]:
        """Get the first object with the specified color from the grid."""
        for obj in grid.objects:
            if obj.color == color:
                return obj
        return None
    
    def describe_per_color_transformations(self, per_color: Dict[int, TransformationResult]) -> str:
        """
        Generate human-readable description of per-color transformations.
        """
        if not per_color:
            return "No per-color transformations detected."
        
        COLOR_NAMES = {
            0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
            5: "grey", 6: "magenta", 7: "orange", 8: "azure", 9: "brown"
        }
        
        descriptions = []
        for color, result in per_color.items():
            color_name = COLOR_NAMES.get(color, f"color_{color}")
            
            if result.transformation_type == "translation":
                dx = result.parameters.get("dx", 0)
                dy = result.parameters.get("dy", 0)
                desc = f"{color_name.upper()} ({color}): translate dx={dx}, dy={dy}"
            elif result.transformation_type == "rotation":
                angle = result.parameters.get("angle", 0)
                desc = f"{color_name.upper()} ({color}): rotate {angle}°"
            elif result.transformation_type == "color_change":
                to_color = result.parameters.get("to_color", 0)
                to_name = COLOR_NAMES.get(to_color, f"color_{to_color}")
                desc = f"{color_name.upper()} ({color}): change to {to_name} ({to_color})"
            elif result.transformation_type == "identity":
                desc = f"{color_name.upper()} ({color}): no change"
            else:
                desc = f"{color_name.upper()} ({color}): {result.transformation_type}"
            
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def detect_composite_transformation(
        self, 
        input_grid: Grid, 
        output_grid: Grid
    ) -> Optional[TransformationResult]:
        """
        Detect composite (combined) transformations.
        
        This tries common combinations when no single transformation matches:
        - Translation + Rotation
        - Translation + Reflection
        - Translation + Color change
        - Rotation + Color change
        - Translation + Rotation + Color change
        
        Returns the best matching composite transformation.
        """
        if input_grid.data.shape != output_grid.data.shape:
            return None
        
        in_data = input_grid.data
        out_data = output_grid.data
        
        # Find the main color (non-zero)
        in_colors = set(np.unique(in_data)) - {0}
        out_colors = set(np.unique(out_data)) - {0}
        
        if not in_colors:
            return None
        
        main_color = list(in_colors)[0]
        
        # Get object pixels in input
        in_pixels = set(zip(*np.where(in_data == main_color)))
        
        # Check if color changed
        color_changed = False
        new_color = main_color
        if in_colors != out_colors and len(out_colors) == 1:
            new_color = list(out_colors)[0]
            color_changed = True
        
        out_pixels = set(zip(*np.where(out_data == new_color)))
        
        if not in_pixels or not out_pixels:
            return None
        
        # Calculate bounding boxes
        in_rows, in_cols = zip(*in_pixels)
        in_bbox = (min(in_rows), min(in_cols), max(in_rows), max(in_cols))
        
        out_rows, out_cols = zip(*out_pixels)
        out_bbox = (min(out_rows), min(out_cols), max(out_rows), max(out_cols))
        
        # Extract shape from input
        in_h = in_bbox[2] - in_bbox[0] + 1
        in_w = in_bbox[3] - in_bbox[1] + 1
        in_shape = np.zeros((in_h, in_w), dtype=np.int8)
        for r, c in in_pixels:
            in_shape[r - in_bbox[0], c - in_bbox[1]] = 1
        
        # Try different composite transformations
        best_result = None
        best_confidence = 0
        
        # List of transformations to try
        rotations = [
            (90, lambda s: np.rot90(s, k=3)),   # 90° clockwise
            (180, lambda s: np.rot90(s, k=2)),  # 180°
            (270, lambda s: np.rot90(s, k=1))   # 270° clockwise
        ]
        
        reflections = [
            ("horizontal", lambda s: np.flipud(s)),
            ("vertical", lambda s: np.fliplr(s)),
            ("diagonal_main", lambda s: np.transpose(s))
        ]
        
        # Try TRANSLATE + ROTATE combinations
        for angle, rotate_fn in rotations:
            transformed = rotate_fn(in_shape)
            th, tw = transformed.shape
            
            # Find all possible translations
            for out_r in range(output_grid.height - th + 1):
                for out_c in range(output_grid.width - tw + 1):
                    # Generate expected pixels
                    expected_pixels = set()
                    for r in range(th):
                        for c in range(tw):
                            if transformed[r, c] == 1:
                                expected_pixels.add((out_r + r, out_c + c))
                    
                    if expected_pixels == out_pixels:
                        # Calculate translation
                        dx = out_c - in_bbox[1]
                        dy = out_r - in_bbox[0]
                        
                        # Build composite parameters
                        transformations = [
                            {"action": "rotate", "params": {"angle": angle}},
                            {"action": "translate", "params": {"dx": dx, "dy": dy}}
                        ]
                        
                        if color_changed:
                            transformations.append({
                                "action": "color_change", 
                                "params": {"from_color": main_color, "to_color": new_color}
                            })
                        
                        confidence = 1.0
                        params = {
                            "transformations": transformations,
                            "description": f"rotate {angle}° + translate ({dx}, {dy})" + 
                                          (f" + color {main_color}→{new_color}" if color_changed else "")
                        }
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_result = TransformationResult(
                                transformation_type="composite",
                                confidence=confidence,
                                parameters=params
                            )
        
        # Try TRANSLATE + REFLECT combinations
        for axis, reflect_fn in reflections:
            transformed = reflect_fn(in_shape)
            th, tw = transformed.shape
            
            for out_r in range(output_grid.height - th + 1):
                for out_c in range(output_grid.width - tw + 1):
                    expected_pixels = set()
                    for r in range(th):
                        for c in range(tw):
                            if transformed[r, c] == 1:
                                expected_pixels.add((out_r + r, out_c + c))
                    
                    if expected_pixels == out_pixels:
                        dx = out_c - in_bbox[1]
                        dy = out_r - in_bbox[0]
                        
                        transformations = [
                            {"action": "reflect", "params": {"axis": axis}},
                            {"action": "translate", "params": {"dx": dx, "dy": dy}}
                        ]
                        
                        if color_changed:
                            transformations.append({
                                "action": "color_change",
                                "params": {"from_color": main_color, "to_color": new_color}
                            })
                        
                        confidence = 1.0
                        params = {
                            "transformations": transformations,
                            "description": f"reflect {axis} + translate ({dx}, {dy})" +
                                          (f" + color {main_color}→{new_color}" if color_changed else "")
                        }
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_result = TransformationResult(
                                transformation_type="composite",
                                confidence=confidence,
                                parameters=params
                            )
        
        return best_result
