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
        # Basic transformations
        "translate", 
        "rotate",
        "reflect",
        "scale",
        
        # Color operations
        "fill", 
        "replace_color",
        "color_change",
        "conditional_color",  # TIER 2: Color based on conditions
        
        # Object operations
        "copy", 
        "symmetry",       # TIER 2: Create symmetric copies
        "add_border",     # Color the border/contour of an object
        
        # Drawing operations
        "draw_line",
        "flood_fill",     # TIER 2: Fill connected regions
        
        # Pattern operations
        "tile",
        
        # Composite operations
        "composite",      # Combined transformations (e.g., translate + rotate)
    ]
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the executor.
        
        Args:
            verbose: Whether to print execution details
        """
        self.verbose = verbose
        self._error_context = {}  # Track error context for debugging
    
    # ==================== DEFENSIVE HELPERS (TIER 1) ====================
    
    def _safe_int(self, value: Any, default: int = 0, name: str = "param") -> int:
        """
        Safely convert value to int with logging.
        
        Args:
            value: Value to convert
            default: Default if conversion fails
            name: Parameter name for error messages
            
        Returns:
            Integer value or default
        """
        if value is None:
            return default
        
        if isinstance(value, (int, np.integer)):
            return int(value)
        
        if isinstance(value, float):
            if self.verbose:
                print(f"  ⚠ Warning: {name}={value} is float, converting to int")
            return int(value)
        
        if isinstance(value, str):
            # Try to parse string
            value_clean = value.strip().lower()
            
            # Handle word numbers
            word_numbers = {
                "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10
            }
            if value_clean in word_numbers:
                return word_numbers[value_clean]
            
            # Try numeric parse
            try:
                return int(float(value))  # Handle "3.0" strings
            except ValueError:
                if self.verbose:
                    print(f"  ⚠ Warning: Invalid {name}='{value}', using default={default}")
                return default
        
        if self.verbose:
            print(f"  ⚠ Warning: Unexpected type for {name}: {type(value)}, using default={default}")
        return default
    
    def _safe_float(self, value: Any, default: float = 1.0, name: str = "param") -> float:
        """
        Safely convert value to float with logging.
        
        Args:
            value: Value to convert
            default: Default if conversion fails
            name: Parameter name for error messages
            
        Returns:
            Float value or default
        """
        if value is None:
            return default
        
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                if self.verbose:
                    print(f"  ⚠ Warning: Invalid {name}='{value}', using default={default}")
                return default
        
        return default
    
    def _safe_color(self, value: Any, default: int = None, name: str = "color") -> Optional[int]:
        """
        Safely convert color value (handles names and numbers).
        
        Args:
            value: Color value (int, string name, or string number)
            default: Default if conversion fails
            name: Parameter name for error messages
            
        Returns:
            Color integer (0-9) or default
        """
        if value is None:
            return default
        
        if isinstance(value, (int, np.integer)):
            val = int(value)
            if 0 <= val <= 9:
                return val
            if self.verbose:
                print(f"  ⚠ Warning: {name}={val} out of range [0-9], using default={default}")
            return default
        
        if isinstance(value, str):
            value_clean = value.strip().lower()
            
            # Color name mapping
            color_names = {
                "black": 0, "blue": 1, "red": 2, "green": 3, "yellow": 4,
                "grey": 5, "gray": 5, "magenta": 6, "pink": 6, "orange": 7,
                "cyan": 8, "azure": 8, "teal": 8, "brown": 9, "maroon": 9
            }
            
            if value_clean in color_names:
                return color_names[value_clean]
            
            # Try numeric
            try:
                val = int(float(value_clean))
                if 0 <= val <= 9:
                    return val
            except ValueError:
                pass
            
            if self.verbose:
                print(f"  ⚠ Warning: Invalid {name}='{value}', using default={default}")
            return default
        
        return default
    
    def _validate_grid(self, grid: Grid, context: str = "input") -> bool:
        """
        Validate grid data is well-formed.
        
        Args:
            grid: Grid to validate
            context: Context string for error messages
            
        Returns:
            True if valid, False otherwise
        """
        if grid is None:
            if self.verbose:
                print(f"  ✗ Error: {context} grid is None")
            return False
        
        if grid.data is None:
            if self.verbose:
                print(f"  ✗ Error: {context} grid.data is None")
            return False
        
        if grid.data.size == 0:
            if self.verbose:
                print(f"  ✗ Error: {context} grid is empty")
            return False
        
        # Check for invalid values
        if not np.issubdtype(grid.data.dtype, np.integer):
            if self.verbose:
                print(f"  ⚠ Warning: {context} grid has non-integer dtype: {grid.data.dtype}")
            # Try to convert
            try:
                grid.data = grid.data.astype(np.int64)
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ Error: Could not convert grid to int: {e}")
                return False
        
        # Check for NaN/Inf (shouldn't happen with int, but be safe)
        if np.issubdtype(grid.data.dtype, np.floating):
            if np.any(~np.isfinite(grid.data)):
                if self.verbose:
                    print(f"  ✗ Error: {context} grid contains NaN or Inf values")
                return False
        
        return True
    
    def _get_params(self, action_data: dict) -> dict:
        """
        Safely extract params dict from action_data.
        
        Args:
            action_data: The action specification
            
        Returns:
            Params dictionary (never None)
        """
        params = action_data.get("params")
        if params is None:
            return {}
        if not isinstance(params, dict):
            if self.verbose:
                print(f"  ⚠ Warning: params is not a dict: {type(params)}")
            return {}
        return params
    
    # ==================== MAIN EXECUTE METHOD ====================
    
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
        # Clear error context
        self._error_context = {"action_data": action_data}
        
        # Validate inputs
        if not action_data:
            return ActionResult(
                success=False,
                message="No action data provided",
                details={"error_code": "NO_ACTION_DATA"}
            )
        
        if not isinstance(action_data, dict):
            return ActionResult(
                success=False,
                message=f"action_data must be dict, got {type(action_data).__name__}",
                details={"error_code": "INVALID_ACTION_TYPE"}
            )
        
        # Validate grid
        if not self._validate_grid(grid, "input"):
            return ActionResult(
                success=False,
                message="Invalid input grid",
                details={"error_code": "INVALID_GRID"}
            )
        
        action_type = action_data.get("action", "")
        
        if not action_type:
            return ActionResult(
                success=False,
                message="No action type specified",
                details={"error_code": "NO_ACTION_TYPE"}
            )
        
        # Normalize action type
        if isinstance(action_type, str):
            action_type = action_type.lower().strip()
            # Handle common aliases
            action_aliases = {
                "translation": "translate",
                "move": "translate",
                "shift": "translate",
                "rotation": "rotate",
                "turn": "rotate",
                "mirror": "reflect",
                "flip": "reflect",
                "reflection": "reflect",
                "recolor": "color_change",
                "change_color": "color_change",
                "line": "draw_line",
                "connect": "draw_line",
                "border": "add_border",
                "contour": "add_border",
                "resize": "scale",
                "repeat": "tile",
                "tiling": "tile",
            }
            action_type = action_aliases.get(action_type, action_type)
        else:
            return ActionResult(
                success=False,
                message=f"Action type must be string, got {type(action_type).__name__}",
                details={"error_code": "INVALID_ACTION_TYPE"}
            )
        
        if action_type not in self.SUPPORTED_ACTIONS:
            return ActionResult(
                success=False,
                message=f"Unsupported action: {action_type}. Supported: {self.SUPPORTED_ACTIONS}",
                details={"error_code": "UNSUPPORTED_ACTION", "action": action_type}
            )
        
        # Store context for error reporting
        self._error_context["action_type"] = action_type
        self._error_context["grid_shape"] = grid.shape
        
        # Dispatch to appropriate handler with error catching
        handler = getattr(self, f"_action_{action_type}", None)
        if handler:
            try:
                result = handler(grid, action_data)
                
                # Validate output grid
                if result.success and result.output_grid:
                    if not self._validate_grid(result.output_grid, "output"):
                        return ActionResult(
                            success=False,
                            message="Action produced invalid output grid",
                            details={"error_code": "INVALID_OUTPUT"}
                        )
                
                return result
                
            except Exception as e:
                # Catch any unexpected errors in handlers
                error_msg = f"Execution error in {action_type}: {type(e).__name__}: {str(e)}"
                if self.verbose:
                    print(f"  ✗ {error_msg}")
                    import traceback
                    traceback.print_exc()
                
                return ActionResult(
                    success=False,
                    message=error_msg,
                    details={
                        "error_code": "EXECUTION_ERROR",
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "context": self._error_context
                    }
                )
        
        return ActionResult(
            success=False,
            message=f"Handler not implemented for action: {action_type}",
            details={"error_code": "NO_HANDLER"}
        )
    
    def _action_translate(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Translate (shift) pixels/objects by (dx, dy).
        """
        params = self._get_params(action_data)
        dx = self._safe_int(params.get("dx", 0), default=0, name="dx")
        dy = self._safe_int(params.get("dy", 0), default=0, name="dy")
        color_filter = self._safe_color(action_data.get("color_filter"), name="color_filter")
        
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
        params = self._get_params(action_data)
        angle = self._safe_int(params.get("angle", 90), default=90, name="angle")
        color_filter = self._safe_color(action_data.get("color_filter"), name="color_filter")
        grid_level = bool(params.get("grid_level", False))
        
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
        params = self._get_params(action_data)
        factor = self._safe_float(params.get("factor", 2), default=2.0, name="factor")
        color_filter = self._safe_color(action_data.get("color_filter"), name="color_filter")
        
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
    
    # ==================== TIER 2: NEW DSL PRIMITIVES ====================
    
    def _action_symmetry(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Create symmetric copies of objects.
        
        Parameters:
            params.axis: str - "horizontal" | "vertical" | "both" | "diagonal"
            params.position: str - "adjacent" | "opposite" | {"offset_x": int, "offset_y": int}
            params.keep_original: bool - Whether to keep the original object (default True)
            color_filter: int (optional) - Only apply to specific color
        
        Example:
            Mirror a shape to create horizontal symmetry:
            {"action": "symmetry", "params": {"axis": "vertical"}, "color_filter": 2}
        
        Returns:
            ActionResult with symmetric grid
        """
        params = self._get_params(action_data)
        axis = params.get("axis", "vertical")
        position = params.get("position", "adjacent")
        keep_original = params.get("keep_original", True)
        color_filter = self._safe_color(action_data.get("color_filter"), name="color_filter")
        
        if self.verbose:
            print(f"  Executing SYMMETRY: axis={axis}, position={position}, color={color_filter}")
        
        height, width = grid.data.shape
        result = grid.data.copy() if keep_original else np.zeros_like(grid.data)
        
        # Determine which pixels to mirror
        if color_filter is not None:
            mask = grid.data == color_filter
        else:
            mask = grid.data != 0
        
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return ActionResult(
                success=False,
                message="No pixels found to create symmetry"
            )
        
        # Calculate bounding box of object
        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()
        obj_height = max_r - min_r + 1
        obj_width = max_c - min_c + 1
        center_r = (min_r + max_r) / 2
        center_c = (min_c + max_c) / 2
        
        # Determine offset for mirrored copy
        if isinstance(position, dict):
            offset_x = self._safe_int(position.get("offset_x", 0), name="offset_x")
            offset_y = self._safe_int(position.get("offset_y", 0), name="offset_y")
        elif position == "adjacent":
            if axis == "vertical":
                offset_x = obj_width + 1
                offset_y = 0
            elif axis == "horizontal":
                offset_x = 0
                offset_y = obj_height + 1
            else:
                offset_x = obj_width + 1
                offset_y = obj_height + 1
        elif position == "opposite":
            if axis == "vertical":
                offset_x = width - max_c - 1 - min_c
                offset_y = 0
            elif axis == "horizontal":
                offset_x = 0
                offset_y = height - max_r - 1 - min_r
            else:
                offset_x = width - max_c - 1 - min_c
                offset_y = height - max_r - 1 - min_r
        else:
            offset_x, offset_y = 0, 0
        
        # Create mirrored pixels
        for r, c in zip(rows, cols):
            pixel_val = int(grid.data[r, c])
            
            # Calculate mirrored position relative to object center
            rel_r = r - center_r
            rel_c = c - center_c
            
            if axis == "vertical":
                new_rel_c = -rel_c
                new_rel_r = rel_r
            elif axis == "horizontal":
                new_rel_c = rel_c
                new_rel_r = -rel_r
            elif axis == "both":
                new_rel_c = -rel_c
                new_rel_r = -rel_r
            elif axis == "diagonal":
                new_rel_c = rel_r
                new_rel_r = rel_c
            else:
                new_rel_c = rel_c
                new_rel_r = rel_r
            
            # Place mirrored pixel
            new_r = int(round(center_r + new_rel_r + offset_y))
            new_c = int(round(center_c + new_rel_c + offset_x))
            
            if 0 <= new_r < height and 0 <= new_c < width:
                result[new_r, new_c] = pixel_val
        
        return ActionResult(
            success=True,
            output_grid=Grid(data=result),
            message=f"Created {axis} symmetry with position={position}",
            details={"axis": axis, "position": position, "color_filter": color_filter}
        )
    
    def _action_flood_fill(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Fill connected regions with a color (paint bucket tool).
        
        Parameters:
            params.seed_point: dict {"row": int, "col": int} - Starting point
                OR "enclosed_regions" to auto-detect enclosed areas
                OR "background" to fill all background (0) pixels
            params.fill_color: int - Color to fill with
            params.boundary_colors: List[int] - Colors that stop the fill (optional)
            params.connectivity: int - 4 or 8 (default 4)
        
        Returns:
            ActionResult with flood-filled grid
        """
        params = self._get_params(action_data)
        seed_point = params.get("seed_point")
        fill_color = self._safe_color(params.get("fill_color", 1), default=1, name="fill_color")
        boundary_colors = params.get("boundary_colors", None)
        connectivity = self._safe_int(params.get("connectivity", 4), default=4, name="connectivity")
        
        if self.verbose:
            print(f"  Executing FLOOD_FILL: seed={seed_point}, fill_color={fill_color}, connectivity={connectivity}")
        
        height, width = grid.data.shape
        result = grid.data.copy()
        
        # Determine boundary colors
        if boundary_colors is None:
            # All non-zero colors are boundaries by default
            boundary_set = set(range(1, 10))
        else:
            boundary_set = set(self._safe_color(c, name="boundary") for c in boundary_colors if c is not None)
        
        # Determine neighbors based on connectivity
        if connectivity == 8:
            neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:  # 4-connectivity
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        def flood_fill_from(start_r: int, start_c: int, target_color: int):
            """BFS flood fill from a starting point."""
            if not (0 <= start_r < height and 0 <= start_c < width):
                return 0
            
            if int(result[start_r, start_c]) in boundary_set:
                return 0
            
            if result[start_r, start_c] != target_color:
                return 0
            
            filled = 0
            stack = [(start_r, start_c)]
            visited = set()
            
            while stack:
                r, c = stack.pop()
                
                if (r, c) in visited:
                    continue
                if not (0 <= r < height and 0 <= c < width):
                    continue
                if int(result[r, c]) in boundary_set:
                    continue
                if result[r, c] != target_color:
                    continue
                
                visited.add((r, c))
                result[r, c] = fill_color
                filled += 1
                
                for dr, dc in neighbors:
                    stack.append((r + dr, c + dc))
            
            return filled
        
        pixels_filled = 0
        
        if seed_point == "enclosed_regions":
            # Find enclosed regions (background pixels surrounded by non-background)
            # This is a simplified heuristic
            for r in range(1, height - 1):
                for c in range(1, width - 1):
                    if result[r, c] == 0:
                        # Check if surrounded by non-zero pixels in all 4 directions
                        has_boundary_up = any(result[rr, c] != 0 for rr in range(0, r))
                        has_boundary_down = any(result[rr, c] != 0 for rr in range(r + 1, height))
                        has_boundary_left = any(result[r, cc] != 0 for cc in range(0, c))
                        has_boundary_right = any(result[r, cc] != 0 for cc in range(c + 1, width))
                        
                        if has_boundary_up and has_boundary_down and has_boundary_left and has_boundary_right:
                            pixels_filled += flood_fill_from(r, c, 0)
        
        elif seed_point == "background":
            # Fill all background pixels
            for r in range(height):
                for c in range(width):
                    if result[r, c] == 0:
                        result[r, c] = fill_color
                        pixels_filled += 1
        
        elif isinstance(seed_point, dict):
            # Fill from specific point
            seed_r = self._safe_int(seed_point.get("row", 0), name="seed_row")
            seed_c = self._safe_int(seed_point.get("col", 0), name="seed_col")
            target_color = int(result[seed_r, seed_c]) if 0 <= seed_r < height and 0 <= seed_c < width else 0
            pixels_filled = flood_fill_from(seed_r, seed_c, target_color)
        
        else:
            return ActionResult(
                success=False,
                message=f"Invalid seed_point: {seed_point}. Use dict, 'enclosed_regions', or 'background'"
            )
        
        return ActionResult(
            success=True,
            output_grid=Grid(data=result),
            message=f"Flood filled {pixels_filled} pixels with color {fill_color}",
            details={"pixels_filled": pixels_filled, "fill_color": fill_color}
        )
    
    def _action_conditional_color(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Apply color changes based on spatial conditions.
        
        Parameters:
            params.rules: List of rule dictionaries:
                - "condition": str - The condition to check
                - "from_color": int - Color to change from (optional, applies to all if not specified)
                - "to_color": int - Color to change to
            
            Supported conditions:
                - "has_neighbor_color_X": Pixel has at least one neighbor of color X
                - "no_neighbor_color_X": Pixel has no neighbors of color X
                - "is_corner": Pixel is at grid corner
                - "is_edge": Pixel is on grid edge
                - "neighbor_count_ge_N": Pixel has >= N non-background neighbors
                - "neighbor_count_le_N": Pixel has <= N non-background neighbors
                - "is_isolated": Pixel has no non-background neighbors
        
        Example:
            {"action": "conditional_color", "params": {
                "rules": [
                    {"condition": "is_edge", "from_color": 2, "to_color": 1},
                    {"condition": "has_neighbor_color_0", "from_color": 2, "to_color": 3}
                ]
            }}
        
        Returns:
            ActionResult with conditionally colored grid
        """
        params = self._get_params(action_data)
        rules = params.get("rules", [])
        
        if not rules:
            return ActionResult(
                success=False,
                message="No rules provided for conditional_color"
            )
        
        if self.verbose:
            print(f"  Executing CONDITIONAL_COLOR with {len(rules)} rules")
        
        height, width = grid.data.shape
        result = grid.data.copy()
        
        # 4-connectivity neighbors
        neighbors_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        def get_neighbors(r: int, c: int) -> List[int]:
            """Get colors of all 4-connected neighbors."""
            colors = []
            for dr, dc in neighbors_4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    colors.append(int(grid.data[nr, nc]))
            return colors
        
        def check_condition(r: int, c: int, condition: str) -> bool:
            """Check if a condition is met for pixel at (r, c)."""
            neighbors = get_neighbors(r, c)
            
            # has_neighbor_color_X
            match = re.match(r"has_neighbor_color_(\d+)", condition)
            if match:
                target_color = int(match.group(1))
                return target_color in neighbors
            
            # no_neighbor_color_X
            match = re.match(r"no_neighbor_color_(\d+)", condition)
            if match:
                target_color = int(match.group(1))
                return target_color not in neighbors
            
            # is_corner
            if condition == "is_corner":
                return (r == 0 or r == height - 1) and (c == 0 or c == width - 1)
            
            # is_edge
            if condition == "is_edge":
                return r == 0 or r == height - 1 or c == 0 or c == width - 1
            
            # neighbor_count_ge_N
            match = re.match(r"neighbor_count_ge_(\d+)", condition)
            if match:
                n = int(match.group(1))
                non_bg_count = sum(1 for nc in neighbors if nc != 0)
                return non_bg_count >= n
            
            # neighbor_count_le_N
            match = re.match(r"neighbor_count_le_(\d+)", condition)
            if match:
                n = int(match.group(1))
                non_bg_count = sum(1 for nc in neighbors if nc != 0)
                return non_bg_count <= n
            
            # is_isolated
            if condition == "is_isolated":
                return all(nc == 0 for nc in neighbors)
            
            # all (always true)
            if condition == "all" or condition == "true":
                return True
            
            return False
        
        changes_made = 0
        
        for rule in rules:
            condition = rule.get("condition", "all")
            from_color = self._safe_color(rule.get("from_color"), name="from_color")
            to_color = self._safe_color(rule.get("to_color"), name="to_color")
            
            if to_color is None:
                continue
            
            for r in range(height):
                for c in range(width):
                    current_color = int(result[r, c])
                    
                    # Check if from_color matches (or if no from_color specified)
                    if from_color is not None and current_color != from_color:
                        continue
                    
                    # Check condition
                    if check_condition(r, c, condition):
                        result[r, c] = to_color
                        changes_made += 1
        
        return ActionResult(
            success=True,
            output_grid=Grid(data=result),
            message=f"Applied {len(rules)} conditional color rules, {changes_made} pixels changed",
            details={"rules_applied": len(rules), "pixels_changed": changes_made}
        )
    
    def _action_add_border(self, grid: Grid, action_data: dict) -> ActionResult:
        """
        Add a colored border/contour to an object.
        
        This colors the outer pixels (perimeter) of an object with a different color,
        keeping the interior pixels with the original color.
        
        Parameters:
            color_filter: The color of the object to add border to
            params.border_color: The color for the border
        
        Example:
            Input: 3x3 red square (color 2)
            Output: 3x3 square with red interior (2) and blue border (1)
            
            Before:     After:
            2 2 2       1 1 1
            2 2 2  -->  1 2 1
            2 2 2       1 1 1
        """
        params = action_data.get("params", {})
        color_filter = action_data.get("color_filter")
        border_color = params.get("border_color")
        
        if color_filter is None:
            return ActionResult(
                success=False,
                message="color_filter required for add_border action"
            )
        
        if border_color is None:
            return ActionResult(
                success=False,
                message="border_color parameter required for add_border action"
            )
        
        color_filter = int(color_filter)
        border_color = int(border_color)
        
        if self.verbose:
            print(f"  Executing ADD_BORDER: object color={color_filter}, border color={border_color}")
        
        # Get object pixels
        pixels = set()
        for r in range(grid.height):
            for c in range(grid.width):
                if int(grid.data[r, c]) == color_filter:
                    pixels.add((r, c))
        
        if not pixels:
            return ActionResult(
                success=False,
                message=f"No pixels of color {color_filter} found"
            )
        
        # Find border pixels (pixels with at least one neighbor not in the object)
        border_pixels = set()
        for r, c in pixels:
            is_border = False
            # Check 4-connectivity (up, down, left, right)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                # If neighbor is outside grid or not part of object, this is a border pixel
                if (nr, nc) not in pixels:
                    is_border = True
                    break
            if is_border:
                border_pixels.add((r, c))
        
        # Interior pixels = all pixels - border pixels
        interior_pixels = pixels - border_pixels
        
        # Create output grid
        result = grid.data.copy()
        
        # Color border pixels with border_color
        for r, c in border_pixels:
            result[r, c] = border_color
        
        # Keep interior pixels with original color (already there, but explicit)
        for r, c in interior_pixels:
            result[r, c] = color_filter
        
        if self.verbose:
            print(f"  Border pixels: {len(border_pixels)}, Interior pixels: {len(interior_pixels)}")
        
        return ActionResult(
            success=True,
            output_grid=Grid(data=result),
            message=f"Added border (color {border_color}) to object (color {color_filter})"
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
        
        # Check for color_change in transformations (to apply after geometric transforms)
        color_change_from = None
        color_change_to = None
        for t in transformations:
            if t.get("action") == "color_change":
                color_change_from = t.get("params", {}).get("from_color")
                color_change_to = t.get("params", {}).get("to_color")
                break
        
        if "rotate" in action_types and "translate" in action_types:
            # Use object-centric approach for rotate+translate
            result = self._composite_rotate_translate(grid, transformations, color_filter)
            # Apply color change if needed
            if result.success and color_change_to is not None:
                result = self._apply_color_change_to_result(result, color_filter, color_change_to)
            return result
        
        if "reflect" in action_types and "translate" in action_types:
            # Use object-centric approach for reflect+translate
            result = self._composite_reflect_translate(grid, transformations, color_filter)
            # Apply color change if needed
            if result.success and color_change_to is not None:
                result = self._apply_color_change_to_result(result, color_filter, color_change_to)
            return result
        
        # Handle rotate only (without translate)
        if "rotate" in action_types and "translate" not in action_types:
            result = self._composite_rotate_only(grid, transformations, color_filter)
            if result.success and color_change_to is not None:
                result = self._apply_color_change_to_result(result, color_filter, color_change_to)
            return result
        
        # Handle reflect only (without translate)
        if "reflect" in action_types and "translate" not in action_types:
            result = self._composite_reflect_only(grid, transformations, color_filter)
            if result.success and color_change_to is not None:
                result = self._apply_color_change_to_result(result, color_filter, color_change_to)
            return result
        
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
    
    def _apply_color_change_to_result(
        self, 
        result: ActionResult, 
        from_color: int, 
        to_color: int
    ) -> ActionResult:
        """Apply color change to a result grid."""
        if not result.success or result.output_grid is None:
            return result
        
        data = result.output_grid.data.copy()
        data[data == from_color] = to_color
        
        return ActionResult(
            success=True,
            output_grid=Grid(data=data),
            message=f"{result.message} + color {from_color}→{to_color}"
        )
    
    def _composite_rotate_only(
        self, 
        grid: Grid, 
        transformations: list, 
        color_filter: int
    ) -> ActionResult:
        """Handle rotation without translation using object-centric approach."""
        # Find rotation angle
        angle = 0
        for t in transformations:
            if t.get("action") == "rotate":
                angle = int(t.get("params", {}).get("angle", 0))
                break
        
        if self.verbose:
            print(f"  Composite rotate only: {angle}° (color={color_filter})")
        
        # Create single-action request
        action_data = {
            "action": "rotate",
            "params": {"angle": angle},
            "color_filter": color_filter
        }
        
        return self._action_rotate(grid, action_data)
    
    def _composite_reflect_only(
        self, 
        grid: Grid, 
        transformations: list, 
        color_filter: int
    ) -> ActionResult:
        """Handle reflection without translation using object-centric approach."""
        # Find reflection axis
        axis = "horizontal"
        for t in transformations:
            if t.get("action") == "reflect":
                axis = t.get("params", {}).get("axis", "horizontal")
                break
        
        if self.verbose:
            print(f"  Composite reflect only: {axis} (color={color_filter})")
        
        # Create single-action request
        action_data = {
            "action": "reflect",
            "params": {"axis": axis},
            "color_filter": color_filter
        }
        
        return self._action_reflect(grid, action_data)
    
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
