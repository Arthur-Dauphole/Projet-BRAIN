"""
prompt_maker.py - Prompt Creator (Bridge Module)
================================================
Step 2 of the pipeline: Prompting

Transforms the detected symbols and patterns into structured prompts
for the LLM reasoning engine.

Supports:
    - Single transformation (all objects same transform)
    - Multi-transform mode (different transform per color)
"""

from typing import List, Optional, Dict, Any
from .types import Grid, ARCTask, GeometricObject


class PromptMaker:
    """
    Bridge module that creates prompts for LLM reasoning.
    
    Responsibilities:
        - Convert grid data to textual representation
        - Format detected objects into natural language
        - Structure examples for few-shot learning
        - Create task-specific prompts
    """
    
    # System prompt template - MUST output structured JSON
    SYSTEM_PROMPT = """You are an ARC-AGI puzzle solver. Analyze input-output pairs and output a JSON action.

COORDINATE SYSTEM:
- row = vertical position (0 = top, increases downward)
- col = horizontal position (0 = left, increases rightward)
- dx = column shift (dx > 0 means move RIGHT)
- dy = row shift (dy > 0 means move DOWN)

CRITICAL RULES:
1. The "**DETECTED TRANSFORMATION**" line tells you EXACTLY which transformation to use.
2. Apply the SAME transformation to the test input, regardless of its color.
3. The transformation pattern is INDEPENDENT of object colors - different examples may use different colors.

AVAILABLE ACTIONS:

1. TRANSLATION (move objects by dx, dy):
```json
{"action": "translate", "params": {"dx": NUMBER, "dy": NUMBER}}
```
Use when DETECTED TRANSFORMATION shows "TRANSLATION".

2. COLOR CHANGE (replace one color with another):
```json
{"action": "color_change", "params": {"from_color": OLD_COLOR, "to_color": NEW_COLOR}}
```
Use when DETECTED TRANSFORMATION shows "COLOR CHANGE".

3. ROTATION (rotate objects around their centroid):
```json
{"action": "rotate", "params": {"angle": 90_OR_180_OR_270}, "color_filter": COLOR_OF_OBJECT}
```
Use when DETECTED TRANSFORMATION shows "ROTATION".
- color_filter = the color of the object to rotate (find it from the test input)
- The object rotates around its center of mass (centroid)

4. REFLECTION (mirror/flip an object):
```json
{"action": "reflect", "params": {"axis": "horizontal_OR_vertical"}, "color_filter": COLOR_OF_OBJECT}
```
Use when DETECTED TRANSFORMATION shows "REFLECTION".
- "horizontal" = flip up-down (mirror vertically)
- "vertical" = flip left-right (mirror horizontally)
- color_filter = the color of the object to reflect (find it from the test input)

5. DRAW LINE (connect two points with a line):
```json
{"action": "draw_line", "color_filter": COLOR_OF_THE_TWO_POINTS}
```
Use when DETECTED TRANSFORMATION shows "DRAW LINE".
The system will automatically find the 2 pixels of that color and draw a line between them.

OUTPUT FORMAT (MANDATORY):
End with exactly ONE JSON block matching the DETECTED TRANSFORMATION type."""

    def __init__(self, include_grid_ascii: bool = True, include_objects: bool = True):
        """
        Initialize the prompt maker.
        
        Args:
            include_grid_ascii: Whether to include ASCII representation of grids
            include_objects: Whether to include detected object descriptions
        """
        self.include_grid_ascii = include_grid_ascii
        self.include_objects = include_objects
    
    def create_task_prompt(self, task: ARCTask) -> str:
        """
        Create a complete prompt for solving an ARC task.
        
        Args:
            task: The ARCTask to create a prompt for
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add task header
        prompt_parts.append(f"# ARC Task: {task.task_id}\n")
        
        # Add training examples
        prompt_parts.append("## Training Examples\n")
        for i, pair in enumerate(task.train_pairs, 1):
            prompt_parts.append(f"### Example {i}\n")
            prompt_parts.append(self._format_pair(pair.input_grid, pair.output_grid))
            prompt_parts.append("")
        
        # Add test input
        prompt_parts.append("## Test Input\n")
        if task.test_pairs:
            test_input = task.test_pairs[0].input_grid
            prompt_parts.append(self._format_grid(test_input, "Input"))
        
        # Add instruction
        prompt_parts.append("\n## Your Task\n")
        prompt_parts.append("Based on the training examples, determine the transformation rule ")
        prompt_parts.append("and apply it to the test input to produce the output grid.\n")
        prompt_parts.append("\nProvide your answer in the following format:\n")
        prompt_parts.append("1. **Reasoning**: Explain the transformation rule you discovered\n")
        prompt_parts.append("2. **Output Grid**: Provide the resulting grid as a 2D array\n")
        
        return "".join(prompt_parts)
    
    def create_single_grid_prompt(self, grid: Grid, instruction: str = "") -> str:
        """
        Create a prompt for analyzing a single grid.
        
        Args:
            grid: The grid to analyze
            instruction: Custom instruction for the analysis
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        prompt_parts.append("# Grid Analysis\n\n")
        prompt_parts.append(self._format_grid(grid, "Grid"))
        
        if instruction:
            prompt_parts.append(f"\n## Instruction\n{instruction}\n")
        
        return "".join(prompt_parts)
    
    def _format_pair(self, input_grid: Grid, output_grid: Optional[Grid]) -> str:
        """
        Format an input-output pair with transformation analysis.
        
        Args:
            input_grid: The input grid
            output_grid: The output grid (can be None)
            
        Returns:
            Formatted string representation
        """
        parts = []
        parts.append(self._format_grid(input_grid, "Input"))
        
        if output_grid:
            parts.append(self._format_grid(output_grid, "Output"))
            
            # Add transformation analysis
            analysis = self._analyze_transformation(input_grid, output_grid)
            if analysis:
                parts.append(f"**Transformation Analysis:**\n{analysis}\n")
        
        return "\n".join(parts)
    
    def _analyze_transformation(self, input_grid: Grid, output_grid: Grid) -> str:
        """
        Analyze and describe the transformation between input and output.
        
        Args:
            input_grid: The input grid
            output_grid: The output grid
            
        Returns:
            Analysis string describing the transformation with clear DETECTED TRANSFORMATION line
        """
        import numpy as np
        
        in_data = input_grid.data
        out_data = output_grid.data
        
        # Priority 1: Check for TRANSLATION (objects moved)
        if input_grid.objects and output_grid.objects:
            for in_obj in input_grid.objects:
                for out_obj in output_grid.objects:
                    # Match objects by color and area
                    if in_obj.color == out_obj.color and in_obj.area == out_obj.area:
                        if in_obj.bounding_box and out_obj.bounding_box:
                            in_row, in_col = in_obj.bounding_box[0], in_obj.bounding_box[1]
                            out_row, out_col = out_obj.bounding_box[0], out_obj.bounding_box[1]
                            
                            dx = out_col - in_col
                            dy = out_row - in_row
                            
                            # Only report if there's actual movement
                            if dx != 0 or dy != 0:
                                return f"**DETECTED TRANSFORMATION: TRANSLATION with dx={dx}, dy={dy}**"
                        break
        
        # Priority 2: Check for OBJECT REFLECTION (object level - BEFORE rotation!)
        # Reflection is prioritized over rotation because for same-dimension shapes,
        # a reflection is more likely than a 180° rotation
        if input_grid.objects and output_grid.objects:
            for in_obj in input_grid.objects:
                for out_obj in output_grid.objects:
                    # Match by color and area, AND same dimensions (reflection keeps dimensions)
                    if (in_obj.color == out_obj.color and 
                        in_obj.area == out_obj.area and
                        in_obj.width == out_obj.width and 
                        in_obj.height == out_obj.height):
                        reflection = self._detect_object_reflection(in_obj, out_obj)
                        if reflection:
                            return f"**DETECTED TRANSFORMATION: REFLECTION with axis={reflection} (use color_filter with the color from TEST input)**"
        
        # Priority 3: Check for OBJECT ROTATION (shape rotates in place)
        # Only check rotation if reflection wasn't detected
        if input_grid.objects and output_grid.objects:
            for in_obj in input_grid.objects:
                for out_obj in output_grid.objects:
                    if in_obj.color == out_obj.color and in_obj.area == out_obj.area:
                        # Check if dimensions are swapped (90° or 270° rotation)
                        if in_obj.width == out_obj.height and in_obj.height == out_obj.width:
                            # Verify it's actually a rotation by checking pixel patterns
                            rotation_angle = self._detect_object_rotation_angle(in_obj, out_obj)
                            if rotation_angle:
                                return f"**DETECTED TRANSFORMATION: ROTATION with angle={rotation_angle} (use color_filter with the color from TEST input)**"
                        # Check for 180° rotation (same dimensions) - only if not a reflection
                        elif in_obj.width == out_obj.width and in_obj.height == out_obj.height:
                            # Double-check it's not a reflection first
                            reflection = self._detect_object_reflection(in_obj, out_obj)
                            if not reflection:
                                rotation_angle = self._detect_object_rotation_angle(in_obj, out_obj)
                                if rotation_angle:
                                    return f"**DETECTED TRANSFORMATION: ROTATION with angle={rotation_angle} (use color_filter with the color from TEST input)**"
        
        # Priority 4: Check for GRID-LEVEL ROTATION
        for angle in [90, 180, 270]:
            rotated = np.rot90(in_data, k=angle // 90)
            if rotated.shape == out_data.shape and np.array_equal(rotated, out_data):
                return f"**DETECTED TRANSFORMATION: ROTATION with angle={angle}**"
        
        # Priority 5: Check for REFLECTION (grid level)
        if in_data.shape == out_data.shape:
            if np.array_equal(np.flipud(in_data), out_data):
                return "**DETECTED TRANSFORMATION: REFLECTION with axis=horizontal**"
            if np.array_equal(np.fliplr(in_data), out_data):
                return "**DETECTED TRANSFORMATION: REFLECTION with axis=vertical**"
        
        # Priority 5: Check for DRAW LINE (2 points become a connected line)
        if in_data.shape == out_data.shape:
            for color in range(1, 10):  # Check colors 1-9
                in_positions = list(zip(*np.where(in_data == color)))
                out_positions = set(zip(*np.where(out_data == color)))
                
                # We need exactly 2 input points and more output points (line drawn)
                if len(in_positions) == 2 and len(out_positions) > 2:
                    # Check if input points are in output
                    if all((r, c) in out_positions for r, c in in_positions):
                        p1, p2 = in_positions[0], in_positions[1]
                        
                        # Classify line type
                        if p1[0] == p2[0]:
                            line_type = "horizontal"
                        elif p1[1] == p2[1]:
                            line_type = "vertical"
                        else:
                            line_type = "diagonal"
                        
                        return f"**DETECTED TRANSFORMATION: DRAW LINE between two points of color {color} ({line_type} line)**"
        
        # Priority 6: Check for COLOR CHANGE (pixels stay in place, only color changes)
        if in_data.shape == out_data.shape:
            non_zero_in = set(zip(*np.where(in_data != 0)))
            non_zero_out = set(zip(*np.where(out_data != 0)))
            
            # Same positions must have non-zero pixels for pure color change
            if non_zero_in == non_zero_out and len(non_zero_in) > 0:
                color_changes = {}
                is_pure_color_change = True
                
                for r, c in non_zero_in:
                    in_val = int(in_data[r, c])
                    out_val = int(out_data[r, c])
                    if in_val != out_val:
                        if in_val in color_changes:
                            if color_changes[in_val] != out_val:
                                is_pure_color_change = False
                                break
                        else:
                            color_changes[in_val] = out_val
                
                if is_pure_color_change and color_changes:
                    from_color = list(color_changes.keys())[0]
                    to_color = color_changes[from_color]
                    return f"**DETECTED TRANSFORMATION: COLOR CHANGE from_color={from_color}, to_color={to_color}**"
        
        return "**DETECTED TRANSFORMATION: UNKNOWN**"
    
    def _detect_object_reflection(self, in_obj: GeometricObject, out_obj: GeometricObject) -> Optional[str]:
        """
        Detect if an object has been reflected (mirrored).
        
        Returns:
            "horizontal" or "vertical" if reflection detected, None otherwise
        """
        import numpy as np
        
        if not in_obj.pixels or not out_obj.pixels:
            return None
        
        # Normalize both objects to origin
        in_min_r = min(p[0] for p in in_obj.pixels)
        in_min_c = min(p[1] for p in in_obj.pixels)
        in_normalized = frozenset((p[0] - in_min_r, p[1] - in_min_c) for p in in_obj.pixels)
        
        out_min_r = min(p[0] for p in out_obj.pixels)
        out_min_c = min(p[1] for p in out_obj.pixels)
        out_normalized = frozenset((p[0] - out_min_r, p[1] - out_min_c) for p in out_obj.pixels)
        
        # Get dimensions of normalized shape
        in_h = max(p[0] for p in in_normalized) + 1
        in_w = max(p[1] for p in in_normalized) + 1
        
        # Check vertical reflection (flip left-right)
        vertical_reflected = frozenset((r, in_w - 1 - c) for r, c in in_normalized)
        # Re-normalize
        if vertical_reflected:
            vr_min_r = min(p[0] for p in vertical_reflected)
            vr_min_c = min(p[1] for p in vertical_reflected)
            vertical_reflected = frozenset((p[0] - vr_min_r, p[1] - vr_min_c) for p in vertical_reflected)
        
        if vertical_reflected == out_normalized:
            return "vertical"
        
        # Check horizontal reflection (flip up-down)
        horizontal_reflected = frozenset((in_h - 1 - r, c) for r, c in in_normalized)
        # Re-normalize
        if horizontal_reflected:
            hr_min_r = min(p[0] for p in horizontal_reflected)
            hr_min_c = min(p[1] for p in horizontal_reflected)
            horizontal_reflected = frozenset((p[0] - hr_min_r, p[1] - hr_min_c) for p in horizontal_reflected)
        
        if horizontal_reflected == out_normalized:
            return "horizontal"
        
        return None
    
    def _detect_object_rotation_angle(self, in_obj: GeometricObject, out_obj: GeometricObject) -> int:
        """
        Detect the rotation angle between two objects by comparing their pixel patterns.
        
        Returns:
            Rotation angle (90, 180, 270) or 0 if no rotation detected
        """
        if not in_obj.pixels or not out_obj.pixels:
            return 0
        
        # Normalize pixels to origin (0,0)
        in_min_r = min(p[0] for p in in_obj.pixels)
        in_min_c = min(p[1] for p in in_obj.pixels)
        in_normalized = set((p[0] - in_min_r, p[1] - in_min_c) for p in in_obj.pixels)
        
        out_min_r = min(p[0] for p in out_obj.pixels)
        out_min_c = min(p[1] for p in out_obj.pixels)
        out_normalized = set((p[0] - out_min_r, p[1] - out_min_c) for p in out_obj.pixels)
        
        h = max(p[0] for p in in_normalized) + 1
        w = max(p[1] for p in in_normalized) + 1
        
        # Check each rotation angle
        for angle in [90, 180, 270]:
            rotated = set()
            for r, c in in_normalized:
                if angle == 90:
                    new_r, new_c = c, h - 1 - r
                elif angle == 180:
                    new_r, new_c = h - 1 - r, w - 1 - c
                elif angle == 270:
                    new_r, new_c = w - 1 - c, r
                else:
                    new_r, new_c = r, c
                rotated.add((new_r, new_c))
            
            # Normalize rotated pixels
            if rotated:
                min_r = min(p[0] for p in rotated)
                min_c = min(p[1] for p in rotated)
                rotated_normalized = set((p[0] - min_r, p[1] - min_c) for p in rotated)
                
                if rotated_normalized == out_normalized:
                    return angle
        
        return 0
    
    def _format_grid(self, grid: Grid, label: str) -> str:
        """
        Format a single grid with optional ASCII art and object descriptions.
        
        Args:
            grid: The grid to format
            label: Label for the grid (e.g., "Input", "Output")
            
        Returns:
            Formatted string representation
        """
        parts = []
        parts.append(f"**{label} ({grid.height}x{grid.width})**\n")
        
        # Add ASCII representation
        if self.include_grid_ascii:
            parts.append("```")
            parts.append(self._grid_to_ascii(grid))
            parts.append("```\n")
        
        # Add detected objects
        if self.include_objects and grid.objects:
            parts.append("**Detected Objects:**\n")
            for obj in grid.objects:
                parts.append(f"- {self._format_object(obj)}\n")
        
        return "".join(parts)
    
    def _grid_to_ascii(self, grid: Grid) -> str:
        """
        Convert a grid to ASCII representation.
        
        Args:
            grid: The grid to convert
            
        Returns:
            ASCII art string
        """
        # Use single digits for colors 0-9
        lines = []
        for row in grid.data:
            line = " ".join(str(cell) for cell in row)
            lines.append(line)
        return "\n".join(lines)
    
    def _format_object(self, obj: GeometricObject) -> str:
        """
        Format a single object description with explicit coordinates.
        
        Args:
            obj: The object to describe
            
        Returns:
            Description string
        """
        color_name = obj.properties.get("color_name", f"color_{obj.color}")
        if obj.bounding_box:
            min_row, min_col, max_row, max_col = obj.bounding_box
            return (f"{color_name} {obj.object_type} "
                    f"at TOP-LEFT corner (row={min_row}, col={min_col}), "
                    f"size {obj.width}x{obj.height}")
        return f"{color_name} {obj.object_type} (pixels: {obj.area})"
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the LLM.
        
        Returns:
            System prompt string
        """
        return self.SYSTEM_PROMPT
    
    def create_reasoning_chain_prompt(self, task: ARCTask) -> str:
        """
        Create a prompt that encourages step-by-step reasoning with JSON output.
        
        Args:
            task: The ARCTask to reason about
            
        Returns:
            Chain-of-thought prompt string
        """
        base_prompt = self.create_task_prompt(task)
        
        # Extract the detected transformations to give a clear hint
        transformations_summary = []
        for i, pair in enumerate(task.train_pairs, 1):
            analysis = self._analyze_transformation(pair.input_grid, pair.output_grid)
            if "DETECTED TRANSFORMATION:" in analysis:
                transformations_summary.append(f"Example {i}: {analysis}")
        
        cot_instruction = """
## TRANSFORMATION SUMMARY

"""
        cot_instruction += "\n".join(transformations_summary) if transformations_summary else "No clear pattern detected."
        
        cot_instruction += """

## YOUR TASK

Based on the DETECTED TRANSFORMATION above, output the corresponding JSON action.

RULES:
1. Use the EXACT transformation type shown (TRANSLATION, ROTATION, COLOR CHANGE, REFLECTION)
2. Use the EXACT parameters shown (dx, dy, angle, from_color, to_color, axis)
3. The transformation applies to ALL objects regardless of their specific color in the test input

## OUTPUT (MANDATORY)

Match the JSON format to the detected transformation type:

For TRANSLATION with dx=X, dy=Y:
```json
{"action": "translate", "params": {"dx": X, "dy": Y}}
```

For ROTATION with angle=A:
```json
{"action": "rotate", "params": {"angle": A}}
```

For COLOR CHANGE from_color=F, to_color=T:
```json
{"action": "color_change", "params": {"from_color": F, "to_color": T}}
```

For REFLECTION with axis=A:
```json
{"action": "reflect", "params": {"axis": "A"}}
```
"""
        
        return base_prompt + cot_instruction
    
    # ==================== MULTI-TRANSFORM SUPPORT ====================
    
    MULTI_TRANSFORM_SYSTEM_PROMPT = """You are an ARC-AGI puzzle solver analyzing MULTI-OBJECT transformations.

Each COLOR in the grid may have a DIFFERENT transformation.

COORDINATE SYSTEM:
- dx = column shift (dx > 0 means move RIGHT)
- dy = row shift (dy > 0 means move DOWN)

CRITICAL: Look at the "PER-COLOR TRANSFORMATIONS" section. Each color has its own action.

OUTPUT FORMAT (MANDATORY):
Return a JSON ARRAY of actions, one per color:

```json
[
  {"color": 2, "action": "translate", "params": {"dx": 2, "dy": 0}},
  {"color": 1, "action": "rotate", "params": {"angle": 90}},
  {"color": 3, "action": "color_change", "params": {"from_color": 3, "to_color": 4}}
]
```

AVAILABLE ACTIONS:
- translate: {"dx": NUMBER, "dy": NUMBER}
- rotate: {"angle": 90 | 180 | 270}
- color_change: {"from_color": NUMBER, "to_color": NUMBER}
- reflect: {"axis": "horizontal" | "vertical"}
- identity: {} (no change)

IMPORTANT: Output ONE action per color found in the input."""
    
    def get_multi_transform_system_prompt(self) -> str:
        """Get the system prompt for multi-transform mode."""
        return self.MULTI_TRANSFORM_SYSTEM_PROMPT
    
    def create_multi_transform_prompt(
        self, 
        task: ARCTask, 
        per_color_transforms: Dict[int, Any]
    ) -> str:
        """
        Create a prompt for multi-object transformations.
        
        Args:
            task: The ARCTask to solve
            per_color_transforms: Dictionary of detected per-color transformations
            
        Returns:
            Prompt string for multi-transform mode
        """
        prompt_parts = []
        
        # Header
        prompt_parts.append(f"# ARC Task: {task.task_id} (MULTI-TRANSFORM MODE)\n")
        prompt_parts.append("This task has DIFFERENT transformations for DIFFERENT colors.\n\n")
        
        # Training examples
        prompt_parts.append("## Training Examples\n")
        for i, pair in enumerate(task.train_pairs, 1):
            prompt_parts.append(f"### Example {i}\n")
            prompt_parts.append(self._format_grid(pair.input_grid, "Input"))
            prompt_parts.append(self._format_grid(pair.output_grid, "Output"))
            prompt_parts.append("")
        
        # Per-color transformation analysis
        prompt_parts.append("## PER-COLOR TRANSFORMATIONS DETECTED\n")
        prompt_parts.append(self._format_per_color_transforms(per_color_transforms))
        prompt_parts.append("\n")
        
        # Test input
        prompt_parts.append("## Test Input\n")
        if task.test_pairs:
            test_input = task.test_pairs[0].input_grid
            prompt_parts.append(self._format_grid(test_input, "Input"))
            
            # List colors in test input
            test_colors = test_input.unique_colors
            prompt_parts.append(f"\n**Colors in test input:** {test_colors}\n")
        
        # Instructions
        prompt_parts.append("""
## YOUR TASK

Apply the DETECTED transformations to each color in the test input.

OUTPUT (MANDATORY):
Return a JSON array with one action per color:

```json
[
""")
        
        # Generate expected format based on detected transforms
        for color, transform in per_color_transforms.items():
            action = self._transform_to_action_hint(color, transform)
            prompt_parts.append(f"  {action},\n")
        
        prompt_parts.append("""]
```

Replace the parameters with the values from the DETECTED TRANSFORMATIONS above.
""")
        
        return "".join(prompt_parts)
    
    def _format_per_color_transforms(self, per_color_transforms: Dict[int, Any]) -> str:
        """Format per-color transformations as a readable string."""
        COLOR_NAMES = {
            0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
            5: "grey", 6: "magenta", 7: "orange", 8: "azure", 9: "brown"
        }
        
        lines = []
        for color, transform in per_color_transforms.items():
            color_name = COLOR_NAMES.get(color, f"color_{color}")
            
            if hasattr(transform, 'transformation_type'):
                t_type = transform.transformation_type
                params = transform.parameters
            else:
                t_type = transform.get('transformation_type', 'unknown')
                params = transform.get('parameters', {})
            
            if t_type == "translation":
                dx = params.get("dx", 0)
                dy = params.get("dy", 0)
                lines.append(f"- **{color_name.upper()} ({color})**: TRANSLATE dx={dx}, dy={dy}")
            elif t_type == "rotation":
                angle = params.get("angle", 0)
                lines.append(f"- **{color_name.upper()} ({color})**: ROTATE angle={angle}°")
            elif t_type == "color_change":
                to_color = params.get("to_color", 0)
                to_name = COLOR_NAMES.get(to_color, f"color_{to_color}")
                lines.append(f"- **{color_name.upper()} ({color})**: COLOR CHANGE to {to_name} ({to_color})")
            elif t_type == "reflection":
                axis = params.get("axis", "horizontal")
                lines.append(f"- **{color_name.upper()} ({color})**: REFLECT axis={axis}")
            elif t_type == "identity":
                lines.append(f"- **{color_name.upper()} ({color})**: NO CHANGE (identity)")
            else:
                lines.append(f"- **{color_name.upper()} ({color})**: {t_type} {params}")
        
        return "\n".join(lines) if lines else "No per-color transformations detected."
    
    def _transform_to_action_hint(self, color: int, transform: Any) -> str:
        """Convert a transformation to an action JSON hint."""
        if hasattr(transform, 'transformation_type'):
            t_type = transform.transformation_type
            params = transform.parameters
        else:
            t_type = transform.get('transformation_type', 'unknown')
            params = transform.get('parameters', {})
        
        if t_type == "translation":
            return f'{{"color": {color}, "action": "translate", "params": {{"dx": DX, "dy": DY}}}}'
        elif t_type == "rotation":
            return f'{{"color": {color}, "action": "rotate", "params": {{"angle": ANGLE}}}}'
        elif t_type == "color_change":
            return f'{{"color": {color}, "action": "color_change", "params": {{"from_color": {color}, "to_color": TO_COLOR}}}}'
        elif t_type == "reflection":
            return f'{{"color": {color}, "action": "reflect", "params": {{"axis": "AXIS"}}}}'
        else:
            return f'{{"color": {color}, "action": "identity", "params": {{}}}}'