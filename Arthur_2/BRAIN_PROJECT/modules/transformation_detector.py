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
        translation = self.detect_translation(input_grid, output_grid)
        if translation:
            results.append(translation)
        
        rotation = self.detect_rotation(input_grid, output_grid)
        if rotation:
            results.append(rotation)
        
        reflection = self.detect_reflection(input_grid, output_grid)
        if reflection:
            results.append(reflection)
        
        color_change = self.detect_color_change(input_grid, output_grid)
        if color_change:
            results.append(color_change)
        
        scaling = self.detect_scaling(input_grid, output_grid)
        if scaling:
            results.append(scaling)
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
    
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
                            return TransformationResult(
                                transformation_type="rotation",
                                confidence=1.0,  # High confidence for object rotation
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
        Detect if the grid is reflected (mirrored).
        
        Returns:
            TransformationResult with axis if reflection detected
        """
        in_data = input_grid.data
        out_data = output_grid.data
        
        if in_data.shape != out_data.shape:
            return None
        
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
