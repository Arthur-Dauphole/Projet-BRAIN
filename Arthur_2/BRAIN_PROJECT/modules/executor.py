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
    SUPPORTED_ACTIONS = ["translate", "fill", "copy", "replace_color"]
    
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
        
        Parameters:
            params.dx: int - Horizontal shift (positive = right)
            params.dy: int - Vertical shift (positive = down)
            color_filter: int (optional) - Only translate pixels of this color
        
        Returns:
            ActionResult with translated grid
        """
        params = action_data.get("params", {})
        dx = params.get("dx", 0)  # Horizontal shift (columns)
        dy = params.get("dy", 0)  # Vertical shift (rows)
        color_filter = action_data.get("color_filter", None)
        
        if self.verbose:
            print(f"  Executing TRANSLATE: dx={dx}, dy={dy}, color_filter={color_filter}")
        
        # Create output grid (start with background)
        output_data = np.zeros_like(grid.data)
        
        height, width = grid.shape
        
        # Process each pixel
        for row in range(height):
            for col in range(width):
                pixel_value = grid.data[row, col]
                
                # Check if this pixel should be translated
                if color_filter is not None:
                    if pixel_value == color_filter:
                        # Translate this pixel
                        new_row = row + dy
                        new_col = col + dx
                        
                        # Boundary check (clip - don't wrap)
                        if 0 <= new_row < height and 0 <= new_col < width:
                            output_data[new_row, new_col] = pixel_value
                    else:
                        # Keep non-filtered pixels in place
                        output_data[row, col] = pixel_value
                else:
                    # No filter - translate all non-background pixels
                    if pixel_value != 0:
                        new_row = row + dy
                        new_col = col + dx
                        
                        if 0 <= new_row < height and 0 <= new_col < width:
                            output_data[new_row, new_col] = pixel_value
        
        output_grid = Grid(data=output_data)
        
        return ActionResult(
            success=True,
            output_grid=output_grid,
            message=f"Translated by dx={dx}, dy={dy}",
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
