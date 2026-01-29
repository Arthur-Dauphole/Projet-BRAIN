"""
executor.py - Action Executor (Execution Module)
=================================================
The "Hands" of the AI - Translates LLM reasoning into grid manipulation.

Executes symbolic actions (translate, rotate, fill, etc.) on grids
based on parsed instructions from the LLM.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np

from .types import Grid, GeometricObject


@dataclass
class ActionResult:
    """
    Result of an action execution.
    
    Attributes:
        success: Whether the action executed successfully
        output_grid: The resulting grid after action
        message: Description of what happened
        details: Additional execution details
    """
    success: bool
    output_grid: Optional[Grid] = None
    message: str = ""
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ActionExecutor:
    """
    Executes symbolic actions on ARC grids.
    
    Supported Actions:
        - translate: Move objects/pixels by (dx, dy)
        - fill: Fill regions with a color
        - copy: Copy objects to new positions
        - (more to be added)
    
    Usage:
        executor = ActionExecutor()
        result = executor.execute(grid, action_data)
    """
    
    # Registry of supported actions
    SUPPORTED_ACTIONS = [
        "translate", 
        "fill", 
        "copy", 
        "replace_color",
        "color_change",
        "rotate",
        "reflect",
        "scale",
        "draw_line",
        "tile",
        "composite"  # For combined transformations (e.g., translate + rotate)
    ]
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the executor.
        
        Args:
            verbose: Whether to print execution details
        """
        self.verbose = verbose
    
    def execute(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Execute an action on a grid based on parsed LLM instructions.
        
        Args:
            grid: The input Grid to transform
            action_data: Dictionary with action specification
                Required keys:
                    - "action": str - The action type (e.g., "translate")
                Optional keys:
                    - "params": dict - Action-specific parameters
                    - "color_filter": int - Apply only to specific color
                    - "object_filter": dict - Apply only to matching objects
        
        Returns:
            ActionResult with the transformed grid
        """
        if not action_data:
            return ActionResult(
                success=False,
                message="No action data provided"
            )
        
        action_type = action_data.get("action", "").lower()
        
        if not action_type:
            return ActionResult(
                success=False,
                message="No action type specified"
            )
        
        if action_type not in self.SUPPORTED_ACTIONS:
            return ActionResult(
                success=False,
                message=f"Unsupported action: {action_type}. Supported: {self.SUPPORTED_ACTIONS}"
            )
        
        # Dispatch to appropriate handler
        handler = getattr(self, f"_action_{action_type}", None)
        if handler:
            return handler(grid, action_data)
        
        return ActionResult(
            success=False,
            message=f"Handler not implemented for action: {action_type}"
        )
    
    def _action_translate(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Translate (shift) pixels/objects by (dx, dy).
        """
        params = action_data.get("params", {})
        dx = int(params.get("dx", 0))
        dy = int(params.get("dy", 0))
        color_filter = action_data.get("color_filter")
        if color_filter is not None:
            color_filter = int(color_filter)
        
        if self.verbose:
            print(f"  Executing TRANSLATE: dx={dx}, dy={dy}, color_filter={color_filter}")
        
        # Get dimensions
        height, width = grid.data.shape
        
        # Collect all pixel moves first, then apply them
        moves = []  # List of (row, col, value) to set
        keeps = []  # List of (row, col, value) to keep in place
        
        for r in range(height):
            for c in range(width):
                val = int(grid.data[r, c])
                
                if color_filter is not None:
                    if val == color_filter:
                        nr, nc = r + dy, c + dx
                        if 0 <= nr < height and 0 <= nc < width:
                            moves.append((nr, nc, val))
                    else:
                        keeps.append((r, c, val))
                else:
                    if val != 0:
                        nr, nc = r + dy, c + dx
                        if 0 <= nr < height and 0 <= nc < width:
                            moves.append((nr, nc, val))
        
        if self.verbose:
            print(f"  Moves to apply: {len(moves)}")
            print(f"  Keeps to apply: {len(keeps)}")
        
        # Create result as Python list first, then convert to numpy
        result_list = [[0 for _ in range(width)] for _ in range(height)]
        
        # Apply keeps
        for r, c, v in keeps:
            result_list[r][c] = v
        
        # Apply moves  
        for r, c, v in moves:
            result_list[r][c] = v
        
        # Convert to numpy array
        result = np.array(result_list, dtype=np.int64)
        
        if self.verbose:
            print(f"  Pixels moved: {len(moves)}")
            print(f"  Result non-zero: {np.count_nonzero(result)}")
        
        return ActionResult(
            success=True,
            output_grid=Grid(data=result),
            message=f"Translated dx={dx}, dy={dy}, moved {len(moves)} pixels",
            details={"dx": dx, "dy": dy, "color_filter": color_filter}
        )
    
    def _action_fill(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Fill regions or the entire grid with a color.
        
        Parameters:
            params.color: int - Color to fill with
            params.region: str (optional) - "all", "background", or bounding box
            color_filter: int (optional) - Only fill pixels of this color
        
        Returns:
            ActionResult with filled grid
        """
        params = action_data.get("params", {})
        fill_color = params.get("color", 0)
        region = params.get("region", "all")
        color_filter = action_data.get("color_filter", None)
        
        output_data = grid.data.copy()
        
        if region == "all":
            if color_filter is not None:
                # Fill only pixels matching the filter
                mask = output_data == color_filter
                output_data[mask] = fill_color
            else:
                # Fill entire grid
                output_data[:] = fill_color
        elif region == "background":
            # Fill only background (0) pixels
            mask = output_data == 0
            output_data[mask] = fill_color
        
        output_grid = Grid(data=output_data)
        
        return ActionResult(
            success=True,
            output_grid=output_grid,
            message=f"Filled with color {fill_color}",
            details={"fill_color": fill_color, "region": region}
        )
    
    def _action_replace_color(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Replace one color with another.
        
        Parameters:
            params.from_color: int - Color to replace
            params.to_color: int - New color
        
        Returns:
            ActionResult with color-replaced grid
        """
        params = action_data.get("params", {})
        from_color = params.get("from_color", 0)
        to_color = params.get("to_color", 0)
        
        output_data = grid.data.copy()
        mask = output_data == from_color
        output_data[mask] = to_color
        
        output_grid = Grid(data=output_data)
        
        return ActionResult(
            success=True,
            output_grid=output_grid,
            message=f"Replaced color {from_color} with {to_color}",
            details={"from_color": from_color, "to_color": to_color}
        )
    
    def _action_copy(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Copy objects/pixels to new positions.
        
        Parameters:
            params.dx: int - Horizontal offset for copy
            params.dy: int - Vertical offset for copy
            color_filter: int (optional) - Only copy pixels of this color
        
        Returns:
            ActionResult with copied elements
        """
        params = action_data.get("params", {})
        dx = params.get("dx", 0)
        dy = params.get("dy", 0)
        color_filter = action_data.get("color_filter", None)
        
        output_data = grid.data.copy()
        height, width = grid.shape
        
        # Find pixels to copy
        for row in range(height):
            for col in range(width):
                pixel_value = grid.data[row, col]
                
                should_copy = False
                if color_filter is not None:
                    should_copy = (pixel_value == color_filter)
                else:
                    should_copy = (pixel_value != 0)
                
                if should_copy:
                    new_row = row + dy
                    new_col = col + dx
                    
                    if 0 <= new_row < height and 0 <= new_col < width:
                        output_data[new_row, new_col] = pixel_value
        
        output_grid = Grid(data=output_data)
        
        return ActionResult(
            success=True,
            output_grid=output_grid,
            message=f"Copied with offset dx={dx}, dy={dy}",
            details={"dx": dx, "dy": dy, "color_filter": color_filter}
        )
    
    def _action_color_change(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Change colors according to a mapping.
        
        Parameters:
            params.color_map: dict - Mapping of old_color -> new_color
            OR
            params.from_color: int - Single color to change from
            params.to_color: int - Single color to change to
        
        Returns:
            ActionResult with color-changed grid
        """
        params = action_data.get("params", {})
        
        # Support both color_map and from_color/to_color formats
        color_map = params.get("color_map", {})
        
        if not color_map:
            # Try single color change format
            from_color = params.get("from_color")
            to_color = params.get("to_color")
            if from_color is not None and to_color is not None:
                color_map = {int(from_color): int(to_color)}
        
        # Also check for "changes" key from TransformationDetector
        if not color_map:
            color_map = params.get("changes", {})
        
        if not color_map:
            return ActionResult(
                success=False,
                message="No color mapping provided"
            )
        
        if self.verbose:
            print(f"  Executing COLOR_CHANGE: {color_map}")
        
        # Convert keys to int
        color_map = {int(k): int(v) for k, v in color_map.items()}
        
        height, width = grid.data.shape
        result_list = [[0 for _ in range(width)] for _ in range(height)]
        
        for r in range(height):
            for c in range(width):
                val = int(grid.data[r, c])
                if val in color_map:
                    result_list[r][c] = color_map[val]
                else:
                    result_list[r][c] = val
        
        result = np.array(result_list, dtype=np.int64)
        
        return ActionResult(
            success=True,
            output_grid=Grid(data=result),
            message=f"Changed colors: {color_map}",
            details={"color_map": color_map}
        )
    
    def _action_rotate(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Rotate the grid or specific objects around their centroid.
        
        Parameters:
            params.angle: int - Rotation angle (90, 180, 270)
            params.grid_level: bool - If True, rotate entire grid (no color_filter auto-detection)
            color_filter: int (optional) - Only rotate pixels of this color
                         If not provided and grid_level=False, will auto-detect the non-background color
        
        Returns:
            ActionResult with rotated grid
        """
        params = action_data.get("params", {})
        angle = int(params.get("angle", 90))
        color_filter = action_data.get("color_filter")
        grid_level = params.get("grid_level", False)
        
        if angle not in [90, 180, 270]:
            return ActionResult(
                success=False,
                message=f"Invalid rotation angle: {angle}. Must be 90, 180, or 270"
            )
        
        # Auto-detect color ONLY if:
        # - color_filter is not provided
        # - grid_level is False (not a grid-level rotation)
        # - there's only one non-background color
        if color_filter is None and not grid_level:
            unique_colors = [c for c in grid.unique_colors if c != 0]
            if len(unique_colors) == 1:
                color_filter = unique_colors[0]
                if self.verbose:
                    print(f"  Auto-detected color_filter: {color_filter}")
        
        if self.verbose:
            print(f"  Executing ROTATE: angle={angle}, color_filter={color_filter}")
        
        k = angle // 90  # Number of 90-degree rotations
        
        if color_filter is None:
            # Rotate entire grid
            rotated = np.rot90(grid.data, k=k)
            result = np.array(rotated, dtype=np.int64)
        else:
            # Rotate only specific color using centroid-based rotation
            color_filter = int(color_filter)
            height, width = grid.data.shape
            
            # Find pixels of this color
            rows, cols = np.where(grid.data == color_filter)
            if len(rows) == 0:
                return ActionResult(
                    success=False,
                    message=f"No pixels found with color {color_filter}"
                )
            
            # Calculate centroid (center of mass)
            centroid_r = np.mean(rows)
            centroid_c = np.mean(cols)
            
            # Get bounding box for reference
            min_r, max_r = rows.min(), rows.max()
            min_c, max_c = cols.min(), cols.max()
            
            if self.verbose:
                print(f"    Centroid: ({centroid_r:.1f}, {centroid_c:.1f})")
                print(f"    Bounding box: ({min_r}, {min_c}) to ({max_r}, {max_c})")
            
            # Create result grid
            result = grid.data.copy()
            result = np.array(result, dtype=np.int64)
            
            # Clear original position
            result[grid.data == color_filter] = 0
            
            # Rotate each pixel around the centroid
            rotated_pixels = []
            for r, c in zip(rows, cols):
                # Translate to centroid origin
                rel_r = r - centroid_r
                rel_c = c - centroid_c
                
                # Apply rotation
                if angle == 90:
                    # 90° clockwise: (r, c) -> (c, -r)
                    new_rel_r = rel_c
                    new_rel_c = -rel_r
                elif angle == 180:
                    # 180°: (r, c) -> (-r, -c)
                    new_rel_r = -rel_r
                    new_rel_c = -rel_c
                elif angle == 270:
                    # 270° clockwise (90° counter-clockwise): (r, c) -> (-c, r)
                    new_rel_r = -rel_c
                    new_rel_c = rel_r
                else:
                    new_rel_r, new_rel_c = rel_r, rel_c
                
                # Translate back from centroid
                new_r = round(new_rel_r + centroid_r)
                new_c = round(new_rel_c + centroid_c)
                
                rotated_pixels.append((new_r, new_c))
            
            # Place rotated pixels
            pixels_placed = 0
            for new_r, new_c in rotated_pixels:
                if 0 <= new_r < height and 0 <= new_c < width:
                    result[new_r, new_c] = color_filter
                    pixels_placed += 1
            
            if self.verbose:
                print(f"    Placed {pixels_placed}/{len(rotated_pixels)} rotated pixels")
        
        return ActionResult(
            success=True,
            output_grid=Grid(data=result),
            message=f"Rotated by {angle}° around centroid",
            details={"angle": angle, "color_filter": color_filter}
        )
    
    def _action_reflect(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Reflect (mirror) the grid or specific objects.
        
        Parameters:
            params.axis: str - "horizontal" (flip up-down), "vertical" (flip left-right),
                              "diagonal_main", "diagonal_anti"
            params.grid_level: bool - If True, reflect entire grid (no color_filter auto-detection)
            color_filter: int (optional) - Only reflect pixels of this color
                         If not provided and grid_level=False, will auto-detect
        
        Returns:
            ActionResult with reflected grid
        """
        params = action_data.get("params", {})
        axis = params.get("axis", "horizontal")
        color_filter = action_data.get("color_filter")
        grid_level = params.get("grid_level", False)
        
        # Auto-detect color ONLY if:
        # - color_filter is not provided
        # - grid_level is False (not a grid-level reflection)
        # - there's only one non-background color
        if color_filter is None and not grid_level:
            unique_colors = [c for c in grid.unique_colors if c != 0]
            if len(unique_colors) == 1:
                color_filter = unique_colors[0]
                if self.verbose:
                    print(f"  Auto-detected color_filter: {color_filter}")
        
        if self.verbose:
            print(f"  Executing REFLECT: axis={axis}, color_filter={color_filter}")
        
        if color_filter is None:
            # Reflect entire grid
            if axis == "horizontal":
                reflected = np.flipud(grid.data)
            elif axis == "vertical":
                reflected = np.fliplr(grid.data)
            elif axis == "diagonal_main":
                reflected = grid.data.T
            elif axis == "diagonal_anti":
                reflected = np.flip(np.flip(grid.data, 0), 1).T
            else:
                return ActionResult(
                    success=False,
                    message=f"Invalid axis: {axis}"
                )
            result = np.array(reflected, dtype=np.int64)
        else:
            # Reflect only specific color
            color_filter = int(color_filter)
            height, width = grid.data.shape
            
            # Find bounding box of colored pixels
            rows, cols = np.where(grid.data == color_filter)
            if len(rows) == 0:
                return ActionResult(
                    success=False,
                    message=f"No pixels found with color {color_filter}"
                )
            
            min_r, max_r = rows.min(), rows.max()
            min_c, max_c = cols.min(), cols.max()
            
            # Extract the object
            obj_height = max_r - min_r + 1
            obj_width = max_c - min_c + 1
            obj_data = np.zeros((obj_height, obj_width), dtype=np.int64)
            
            for r, c in zip(rows, cols):
                obj_data[r - min_r, c - min_c] = color_filter
            
            # Reflect the object
            if axis == "horizontal":
                reflected_obj = np.flipud(obj_data)
            elif axis == "vertical":
                reflected_obj = np.fliplr(obj_data)
            elif axis == "diagonal_main":
                reflected_obj = obj_data.T
            elif axis == "diagonal_anti":
                reflected_obj = np.flip(np.flip(obj_data, 0), 1).T
            else:
                return ActionResult(
                    success=False,
                    message=f"Invalid axis: {axis}"
                )
            
            new_height, new_width = reflected_obj.shape
            
            # Create result grid
            result = grid.data.copy()
            result = np.array(result, dtype=np.int64)
            
            # Clear original position
            result[grid.data == color_filter] = 0
            
            # Place reflected object at same top-left position
            for r in range(new_height):
                for c in range(new_width):
                    if reflected_obj[r, c] != 0:
                        nr, nc = min_r + r, min_c + c
                        if 0 <= nr < height and 0 <= nc < width:
                            result[nr, nc] = reflected_obj[r, c]
        
        return ActionResult(
            success=True,
            output_grid=Grid(data=result),
            message=f"Reflected along {axis} axis",
            details={"axis": axis, "color_filter": color_filter}
        )
    
    def _action_scale(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Scale (resize) objects or the entire grid.
        
        Parameters:
            params.factor: float - Scale factor (2 = double, 0.5 = half)
            color_filter: int (optional) - Only scale pixels of this color
        
        Returns:
            ActionResult with scaled grid
        """
        params = action_data.get("params", {})
        factor = float(params.get("factor", 2))
        color_filter = action_data.get("color_filter")
        
        if factor <= 0:
            return ActionResult(
                success=False,
                message=f"Invalid scale factor: {factor}"
            )
        
        if self.verbose:
            print(f"  Executing SCALE: factor={factor}, color_filter={color_filter}")
        
        height, width = grid.data.shape
        
        if color_filter is None:
            # Scale entire grid
            new_height = int(height * factor)
            new_width = int(width * factor)
            
            result = np.zeros((new_height, new_width), dtype=np.int64)
            
            for r in range(height):
                for c in range(width):
                    val = int(grid.data[r, c])
                    # Fill the scaled region
                    for dr in range(int(factor)):
                        for dc in range(int(factor)):
                            nr, nc = int(r * factor) + dr, int(c * factor) + dc
                            if 0 <= nr < new_height and 0 <= nc < new_width:
                                result[nr, nc] = val
        else:
            # Scale only specific color within same grid size
            color_filter = int(color_filter)
            
            # Find bounding box
            rows, cols = np.where(grid.data == color_filter)
            if len(rows) == 0:
                return ActionResult(
                    success=False,
                    message=f"No pixels found with color {color_filter}"
                )
            
            min_r, max_r = rows.min(), rows.max()
            min_c, max_c = cols.min(), cols.max()
            
            result = grid.data.copy()
            result = np.array(result, dtype=np.int64)
            
            # Clear original
            result[grid.data == color_filter] = 0
            
            # Draw scaled version
            for r, c in zip(rows, cols):
                rel_r, rel_c = r - min_r, c - min_c
                for dr in range(int(factor)):
                    for dc in range(int(factor)):
                        nr = min_r + int(rel_r * factor) + dr
                        nc = min_c + int(rel_c * factor) + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            result[nr, nc] = color_filter
        
        return ActionResult(
            success=True,
            output_grid=Grid(data=result),
            message=f"Scaled by factor {factor}",
            details={"factor": factor, "color_filter": color_filter}
        )
    
    def _action_draw_line(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Draw a line between two points of the same color.
        
        This action detects exactly 2 pixels of a specific color and draws
        a line connecting them using Bresenham's line algorithm.
        
        Parameters:
            color_filter: int - The color of the two points to connect
            OR
            params.point1: dict - {"row": int, "col": int} first point
            params.point2: dict - {"row": int, "col": int} second point
            params.color: int - Color to draw the line with
        
        Returns:
            ActionResult with the grid containing the drawn line
        """
        params = action_data.get("params", {})
        color_filter = action_data.get("color_filter")
        
        # Try to get explicit points from params
        point1 = params.get("point1")
        point2 = params.get("point2")
        line_color = params.get("color")
        
        if self.verbose:
            print(f"  Executing DRAW_LINE: color_filter={color_filter}, params={params}")
        
        height, width = grid.data.shape
        result = grid.data.copy()
        result = np.array(result, dtype=np.int64)
        
        # If explicit points are given, use them
        if point1 and point2:
            r1, c1 = point1.get("row", 0), point1.get("col", 0)
            r2, c2 = point2.get("row", 0), point2.get("col", 0)
            
            if line_color is None:
                # Try to infer color from the grid at those positions
                line_color = int(grid.data[r1, c1]) if grid.data[r1, c1] != 0 else 1
        else:
            # Auto-detect: find exactly 2 pixels of the specified color
            if color_filter is None:
                # Find any color with exactly 2 pixels
                for color in range(1, 10):  # Colors 1-9
                    positions = list(zip(*np.where(grid.data == color)))
                    if len(positions) == 2:
                        color_filter = color
                        break
            
            if color_filter is None:
                return ActionResult(
                    success=False,
                    message="Could not find two points to connect. Specify color_filter or params.point1/point2."
                )
            
            color_filter = int(color_filter)
            positions = list(zip(*np.where(grid.data == color_filter)))
            
            if len(positions) != 2:
                return ActionResult(
                    success=False,
                    message=f"Expected exactly 2 pixels of color {color_filter}, found {len(positions)}"
                )
            
            r1, c1 = positions[0]
            r2, c2 = positions[1]
            line_color = color_filter
        
        if self.verbose:
            print(f"    Drawing line from ({r1}, {c1}) to ({r2}, {c2}) with color {line_color}")
        
        # Draw the line using Bresenham's algorithm
        line_pixels = self._bresenham_line(r1, c1, r2, c2)
        
        for r, c in line_pixels:
            if 0 <= r < height and 0 <= c < width:
                result[r, c] = line_color
        
        return ActionResult(
            success=True,
            output_grid=Grid(data=result),
            message=f"Drew line from ({r1}, {c1}) to ({r2}, {c2}) with color {line_color}",
            details={
                "point1": {"row": int(r1), "col": int(c1)},
                "point2": {"row": int(r2), "col": int(c2)},
                "color": int(line_color),
                "pixels_drawn": len(line_pixels)
            }
        )
    
    def _bresenham_line(self, r1: int, c1: int, r2: int, c2: int) -> List[tuple]:
        """
        Generate all pixel coordinates for a line between two points.
        Uses Bresenham's line algorithm for clean integer-based line drawing.
        
        Args:
            r1, c1: Starting point (row, col)
            r2, c2: Ending point (row, col)
            
        Returns:
            List of (row, col) tuples forming the line
        """
        pixels = []
        
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        
        sr = 1 if r1 < r2 else -1
        sc = 1 if c1 < c2 else -1
        
        if dc > dr:
            # Line is more horizontal
            err = dc // 2
            r = r1
            c = c1
            while True:
                pixels.append((r, c))
                if c == c2:
                    break
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
                c += sc
        else:
            # Line is more vertical or diagonal
            err = dr // 2
            r = r1
            c = c1
            while True:
                pixels.append((r, c))
                if r == r2:
                    break
                err -= dc
                if err < 0:
                    c += sc
                    err += dr
                r += sr
        
        return pixels
    
    def apply_action(self, grid: Grid, action_data: dict) -> Grid:
        """
        Convenience method that returns just the output grid.
        
        Args:
            grid: Input grid
            action_data: Action specification
            
        Returns:
            Transformed Grid (or original if action fails)
        """
        result = self.execute(grid, action_data)
        
        if result.success and result.output_grid:
            return result.output_grid
        else:
            if self.verbose:
                print(f"  Action failed: {result.message}")
            return grid
    
    def _action_tile(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Tile (repeat) the input pattern to create a larger output.
        
        Parameters:
            params.repetitions_horizontal: int - Number of horizontal repetitions
            params.repetitions_vertical: int - Number of vertical repetitions
        
        Returns:
            ActionResult with tiled grid
        """
        params = action_data.get("params", {})
        reps_h = int(params.get("repetitions_horizontal", 2))
        reps_v = int(params.get("repetitions_vertical", 2))
        
        if reps_h < 1 or reps_v < 1:
            return ActionResult(
                success=False,
                message=f"Invalid repetitions: h={reps_h}, v={reps_v}"
            )
        
        if self.verbose:
            print(f"  Executing TILE: {reps_h}x horizontal, {reps_v}x vertical")
        
        in_h, in_w = grid.data.shape
        out_h = in_h * reps_v
        out_w = in_w * reps_h
        
        # Create output grid by tiling
        result = np.tile(grid.data, (reps_v, reps_h))
        
        return ActionResult(
            success=True,
            output_grid=Grid(data=result),
            message=f"Tiled pattern {reps_h}x{reps_v} to create {out_w}x{out_h} grid"
        )
    
    def _action_composite(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Execute a composite (combined) transformation.
        
        For rotate+translate combinations, we use an object-centric approach:
        1. Extract the object pixels
        2. Normalize (center at origin)
        3. Apply rotation to normalized shape
        4. Place at final position (original + translation offset)
        
        Parameters:
            params.transformations: List of transformation dictionaries
            color_filter: Color of the object to transform
        """
        params = action_data.get("params", {})
        transformations = params.get("transformations", [])
        color_filter = action_data.get("color_filter")
        
        if not transformations:
            return ActionResult(
                success=False,
                message="No transformations specified in composite action"
            )
        
        if self.verbose:
            trans_desc = " + ".join([t.get("action", "?") for t in transformations])
            print(f"  Executing COMPOSITE: {trans_desc} (color_filter={color_filter})")
        
        # Check if this is a rotate+translate combination (most common)
        action_types = [t.get("action") for t in transformations]
        
        if "rotate" in action_types and "translate" in action_types:
            # Use object-centric approach for rotate+translate
            return self._composite_rotate_translate(grid, transformations, color_filter)
        
        if "reflect" in action_types and "translate" in action_types:
            # Use object-centric approach for reflect+translate
            return self._composite_reflect_translate(grid, transformations, color_filter)
        
        # Fallback: sequential execution for other combinations
        current_grid = grid
        executed = []
        
        for i, transform in enumerate(transformations):
            sub_action = {
                "action": transform.get("action"),
                "params": transform.get("params", {}),
                "color_filter": color_filter
            }
            
            result = self.execute(current_grid, sub_action)
            
            if not result.success:
                return ActionResult(
                    success=False,
                    message=f"Composite failed at step {i+1} ({sub_action['action']}): {result.message}"
                )
            
            current_grid = result.output_grid
            executed.append(sub_action['action'])
        
        return ActionResult(
            success=True,
            output_grid=current_grid,
            message=f"Composite transformation: {' → '.join(executed)}"
        )
    
    def _composite_rotate_translate(
        self, 
        grid: Grid, 
        transformations: list, 
        color_filter: int
    ) -> ActionResult:
        """
        Handle rotate+translate as a single atomic operation.
        
        Instead of rotating then translating (which can cause precision issues),
        we:
        1. Extract object pixels
        2. Normalize to origin (0,0)
        3. Apply rotation to normalized shape
        4. Calculate final position and place there
        """
        # Find rotation and translation parameters
        angle = 0
        dx, dy = 0, 0
        
        for t in transformations:
            if t.get("action") == "rotate":
                angle = int(t.get("params", {}).get("angle", 0))
            elif t.get("action") == "translate":
                dx = int(t.get("params", {}).get("dx", 0))
                dy = int(t.get("params", {}).get("dy", 0))
        
        if self.verbose:
            print(f"  Rotate {angle}° then translate dx={dx}, dy={dy}")
        
        # Get object pixels
        if color_filter is None:
            return ActionResult(success=False, message="color_filter required for composite")
        
        pixels = []
        for r in range(grid.height):
            for c in range(grid.width):
                if int(grid.data[r, c]) == color_filter:
                    pixels.append((r, c))
        
        if not pixels:
            return ActionResult(success=False, message=f"No pixels of color {color_filter}")
        
        # Get bounding box
        rows, cols = zip(*pixels)
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        
        # Normalize pixels to origin
        normalized = [(r - min_r, c - min_c) for r, c in pixels]
        
        # Create shape array for rotation
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        shape = np.zeros((h, w), dtype=np.int8)
        for r, c in normalized:
            shape[r, c] = 1
        
        # Apply rotation
        if angle == 90:
            rotated = np.rot90(shape, k=3)  # 90° clockwise
        elif angle == 180:
            rotated = np.rot90(shape, k=2)
        elif angle == 270:
            rotated = np.rot90(shape, k=1)  # 270° clockwise = 90° counter
        else:
            rotated = shape
        
        # Get rotated pixels
        rotated_pixels = []
        for r in range(rotated.shape[0]):
            for c in range(rotated.shape[1]):
                if rotated[r, c] == 1:
                    rotated_pixels.append((r, c))
        
        # Calculate final position
        # Final top-left = original top-left + translation
        final_min_r = min_r + dy
        final_min_c = min_c + dx
        
        # Create output grid
        result = grid.data.copy()
        
        # Clear original pixels
        for r, c in pixels:
            result[r, c] = 0
        
        # Place rotated shape at final position
        placed = 0
        for r, c in rotated_pixels:
            new_r = final_min_r + r
            new_c = final_min_c + c
            if 0 <= new_r < grid.height and 0 <= new_c < grid.width:
                result[new_r, new_c] = color_filter
                placed += 1
        
        if self.verbose:
            print(f"  Placed {placed}/{len(rotated_pixels)} pixels at ({final_min_r}, {final_min_c})")
        
        return ActionResult(
            success=True,
            output_grid=Grid(data=result),
            message=f"Composite: rotate {angle}° + translate ({dx}, {dy})"
        )
    
    def _composite_reflect_translate(
        self, 
        grid: Grid, 
        transformations: list, 
        color_filter: int
    ) -> ActionResult:
        """Handle reflect+translate as a single atomic operation."""
        # Find reflection and translation parameters
        axis = "horizontal"
        dx, dy = 0, 0
        
        for t in transformations:
            if t.get("action") == "reflect":
                axis = t.get("params", {}).get("axis", "horizontal")
            elif t.get("action") == "translate":
                dx = int(t.get("params", {}).get("dx", 0))
                dy = int(t.get("params", {}).get("dy", 0))
        
        if self.verbose:
            print(f"  Reflect {axis} then translate dx={dx}, dy={dy}")
        
        # Get object pixels
        if color_filter is None:
            return ActionResult(success=False, message="color_filter required for composite")
        
        pixels = []
        for r in range(grid.height):
            for c in range(grid.width):
                if int(grid.data[r, c]) == color_filter:
                    pixels.append((r, c))
        
        if not pixels:
            return ActionResult(success=False, message=f"No pixels of color {color_filter}")
        
        # Get bounding box
        rows, cols = zip(*pixels)
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        
        # Normalize pixels to origin
        normalized = [(r - min_r, c - min_c) for r, c in pixels]
        
        # Create shape array
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        shape = np.zeros((h, w), dtype=np.int8)
        for r, c in normalized:
            shape[r, c] = 1
        
        # Apply reflection
        if axis == "horizontal":
            reflected = np.flipud(shape)
        elif axis == "vertical":
            reflected = np.fliplr(shape)
        elif axis == "diagonal_main":
            reflected = shape.T
        else:
            reflected = shape
        
        # Get reflected pixels
        reflected_pixels = []
        for r in range(reflected.shape[0]):
            for c in range(reflected.shape[1]):
                if reflected[r, c] == 1:
                    reflected_pixels.append((r, c))
        
        # Calculate final position
        final_min_r = min_r + dy
        final_min_c = min_c + dx
        
        # Create output grid
        result = grid.data.copy()
        
        # Clear original pixels
        for r, c in pixels:
            result[r, c] = 0
        
        # Place reflected shape at final position
        placed = 0
        for r, c in reflected_pixels:
            new_r = final_min_r + r
            new_c = final_min_c + c
            if 0 <= new_r < grid.height and 0 <= new_c < grid.width:
                result[new_r, new_c] = color_filter
                placed += 1
        
        if self.verbose:
            print(f"  Placed {placed}/{len(reflected_pixels)} pixels at ({final_min_r}, {final_min_c})")
        
        return ActionResult(
            success=True,
            output_grid=Grid(data=result),
            message=f"Composite: reflect {axis} + translate ({dx}, {dy})"
        )
    
    # ==================== MULTI-TRANSFORM SUPPORT ====================
    
    def execute_multi_actions(
        self, 
        grid: Grid, 
        multi_actions: List[Dict[str, Any]]
    ) -> ActionResult:
        """
        Execute multiple actions, each targeting a specific color.
        
        This applies different transformations to different colored objects
        in a single grid.
        
        Args:
            grid: The input Grid to transform
            multi_actions: List of action dictionaries, each with:
                - "color": int - The color to apply the action to
                - "action": str - The action type
                - "params": dict - Action-specific parameters
        
        Returns:
            ActionResult with the final transformed grid
        """
        if not multi_actions:
            return ActionResult(
                success=False,
                message="No actions provided"
            )
        
        if self.verbose:
            print(f"  Executing {len(multi_actions)} multi-actions...")
        
        height, width = grid.data.shape
        
        # Start with a blank grid (we'll rebuild it with transformed objects)
        result_data = np.zeros((height, width), dtype=np.int64)
        
        # Track which pixels have been processed
        processed_colors = set()
        action_messages = []
        
        for action_entry in multi_actions:
            color = action_entry.get("color")
            action_type = action_entry.get("action", "").lower()
            params = action_entry.get("params", {})
            
            if color is None:
                continue
            
            color = int(color)
            processed_colors.add(color)
            
            if self.verbose:
                print(f"    Processing color {color}: {action_type} with {params}")
            
            # Extract pixels of this color from original grid
            color_mask = grid.data == color
            if not np.any(color_mask):
                if self.verbose:
                    print(f"      No pixels found for color {color}")
                continue
            
            # Apply the transformation based on action type
            if action_type == "identity":
                # No change - just copy to result
                result_data[color_mask] = color
                action_messages.append(f"Color {color}: identity (no change)")
                
            elif action_type == "translate":
                dx = int(params.get("dx", 0))
                dy = int(params.get("dy", 0))
                
                # Apply translation
                for r in range(height):
                    for c in range(width):
                        if grid.data[r, c] == color:
                            nr, nc = r + dy, c + dx
                            if 0 <= nr < height and 0 <= nc < width:
                                result_data[nr, nc] = color
                
                action_messages.append(f"Color {color}: translate dx={dx}, dy={dy}")
                
            elif action_type == "rotate":
                angle = int(params.get("angle", 90))
                
                # Extract object pixels
                rows, cols = np.where(grid.data == color)
                if len(rows) == 0:
                    continue
                
                min_r, max_r = rows.min(), rows.max()
                min_c, max_c = cols.min(), cols.max()
                
                # Create local object
                obj_h = max_r - min_r + 1
                obj_w = max_c - min_c + 1
                obj_data = np.zeros((obj_h, obj_w), dtype=np.int64)
                
                for r, c in zip(rows, cols):
                    obj_data[r - min_r, c - min_c] = color
                
                # Rotate
                k = angle // 90
                rotated_obj = np.rot90(obj_data, k=k)
                new_h, new_w = rotated_obj.shape
                
                # Calculate new position (center on original center)
                center_r = (min_r + max_r) // 2
                center_c = (min_c + max_c) // 2
                new_min_r = center_r - new_h // 2
                new_min_c = center_c - new_w // 2
                
                # Place rotated object
                for r in range(new_h):
                    for c in range(new_w):
                        if rotated_obj[r, c] != 0:
                            nr, nc = new_min_r + r, new_min_c + c
                            if 0 <= nr < height and 0 <= nc < width:
                                result_data[nr, nc] = rotated_obj[r, c]
                
                action_messages.append(f"Color {color}: rotate {angle}°")
                
            elif action_type == "color_change":
                from_color = int(params.get("from_color", color))
                to_color = int(params.get("to_color", color))
                
                # Copy pixels with new color
                for r in range(height):
                    for c in range(width):
                        if grid.data[r, c] == from_color:
                            result_data[r, c] = to_color
                
                action_messages.append(f"Color {color}: change to {to_color}")
                
            elif action_type == "reflect":
                axis = params.get("axis", "horizontal")
                
                # Extract object pixels
                rows, cols = np.where(grid.data == color)
                if len(rows) == 0:
                    continue
                
                min_r, max_r = rows.min(), rows.max()
                min_c, max_c = cols.min(), cols.max()
                
                # Create local object
                obj_h = max_r - min_r + 1
                obj_w = max_c - min_c + 1
                obj_data = np.zeros((obj_h, obj_w), dtype=np.int64)
                
                for r, c in zip(rows, cols):
                    obj_data[r - min_r, c - min_c] = color
                
                # Reflect
                if axis == "horizontal":
                    reflected_obj = np.flipud(obj_data)
                elif axis == "vertical":
                    reflected_obj = np.fliplr(obj_data)
                else:
                    reflected_obj = obj_data
                
                # Place reflected object at same position
                for r in range(obj_h):
                    for c in range(obj_w):
                        if reflected_obj[r, c] != 0:
                            nr, nc = min_r + r, min_c + c
                            if 0 <= nr < height and 0 <= nc < width:
                                result_data[nr, nc] = reflected_obj[r, c]
                
                action_messages.append(f"Color {color}: reflect {axis}")
            
            elif action_type == "draw_line":
                # Find the two points of this color and draw a line between them
                rows, cols = np.where(grid.data == color)
                positions = list(zip(rows, cols))
                
                if len(positions) == 2:
                    r1, c1 = positions[0]
                    r2, c2 = positions[1]
                    
                    # Draw line using Bresenham
                    line_pixels = self._bresenham_line(r1, c1, r2, c2)
                    
                    for r, c in line_pixels:
                        if 0 <= r < height and 0 <= c < width:
                            result_data[r, c] = color
                    
                    action_messages.append(f"Color {color}: draw_line from ({r1},{c1}) to ({r2},{c2})")
                else:
                    # Just copy as-is if not exactly 2 points
                    result_data[grid.data == color] = color
                    action_messages.append(f"Color {color}: draw_line skipped (need exactly 2 points, found {len(positions)})")
            
            else:
                # Unknown action - just copy original
                result_data[color_mask] = color
                action_messages.append(f"Color {color}: unknown action '{action_type}', copied as-is")
        
        # Copy any colors that weren't processed (keep them as-is)
        for color in grid.unique_colors:
            if color not in processed_colors:
                color_mask = grid.data == color
                result_data[color_mask] = color
                if self.verbose:
                    print(f"    Color {color}: not in actions, keeping as-is")
        
        return ActionResult(
            success=True,
            output_grid=Grid(data=result_data),
            message=f"Executed {len(multi_actions)} actions: " + "; ".join(action_messages),
            details={"actions_executed": len(multi_actions), "colors_processed": list(processed_colors)}
        )
