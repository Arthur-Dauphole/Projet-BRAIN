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
    
    # System prompt template - SIMPLIFIED with few-shot examples
    SYSTEM_PROMPT = """You are an ARC puzzle solver. Your ONLY job: output a JSON action.

## FEW-SHOT EXAMPLES

Example 1 - TRANSLATION:
- Detected: "TRANSLATION dx=3, dy=2"
- Output:
```json
{"action": "translate", "params": {"dx": 3, "dy": 2}}
```

Example 2 - COLOR CHANGE:
- Detected: "COLOR CHANGE from_color=2, to_color=4"
- Output:
```json
{"action": "color_change", "params": {"from_color": 2, "to_color": 4}}
```

Example 3 - ROTATION:
- Detected: "ROTATION angle=90"
- Test input has color 3
- Output:
```json
{"action": "rotate", "params": {"angle": 90}, "color_filter": 3}
```

Example 4 - REFLECTION:
- Detected: "REFLECTION axis=vertical"
- Test input has color 1
- Output:
```json
{"action": "reflect", "params": {"axis": "vertical"}, "color_filter": 1}
```

Example 5 - DRAW LINE:
- Detected: "DRAW LINE color=2"
- Output:
```json
{"action": "draw_line", "color_filter": 2}
```

## RULES

1. Look at "DETECTED TRANSFORMATION" - it tells you EXACTLY what to output
2. Copy the parameters from the detection (dx, dy, angle, from_color, to_color, axis)
3. For ROTATION/REFLECTION: use "color_filter" with the color from TEST INPUT
4. ALWAYS end with a ```json block

## COORDINATE SYSTEM
- dx > 0 = move RIGHT
- dy > 0 = move DOWN"""

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
        
        # Priority 4: Check for GRID-LEVEL REFLECTION (entire grid flipped)
        # This should be checked BEFORE grid-level rotation to avoid confusion
        if in_data.shape == out_data.shape:
            if np.array_equal(np.flipud(in_data), out_data):
                return "**DETECTED TRANSFORMATION: REFLECTION with axis=horizontal (GRID-LEVEL)**"
            if np.array_equal(np.fliplr(in_data), out_data):
                return "**DETECTED TRANSFORMATION: REFLECTION with axis=vertical (GRID-LEVEL)**"
        
        # Priority 5: Check for GRID-LEVEL ROTATION
        for angle in [90, 180, 270]:
            rotated = np.rot90(in_data, k=angle // 90)
            if rotated.shape == out_data.shape and np.array_equal(rotated, out_data):
                return f"**DETECTED TRANSFORMATION: ROTATION with angle={angle} (GRID-LEVEL)**"
        
        # Priority 6: Check for DRAW LINE (2 points become a connected line)
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
                        
                        return f"**DETECTED TRANSFORMATION: DRAW LINE color={color} ({line_type})**"
        
        # Priority 7: Check for COLOR CHANGE (pixels stay in place, only color changes)
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
            Chain-of-thought prompt string with direct JSON instruction
        """
        prompt_parts = []
        
        # Header
        prompt_parts.append(f"# ARC Task: {task.task_id}\n\n")
        
        # Training examples (simplified)
        prompt_parts.append("## Training Examples\n\n")
        for i, pair in enumerate(task.train_pairs, 1):
            prompt_parts.append(f"### Example {i}\n")
            prompt_parts.append(self._format_grid(pair.input_grid, "Input"))
            prompt_parts.append(self._format_grid(pair.output_grid, "Output"))
            prompt_parts.append("\n")
        
        # CRITICAL: Detect transformation and give EXPLICIT instruction
        transformations = []
        for i, pair in enumerate(task.train_pairs, 1):
            analysis = self._analyze_transformation(pair.input_grid, pair.output_grid)
            if "DETECTED TRANSFORMATION:" in analysis:
                transformations.append(analysis)
        
        # Get test input info
        test_colors = []
        if task.test_pairs:
            test_input = task.test_pairs[0].input_grid
            test_colors = [c for c in test_input.unique_colors if c != 0]
            prompt_parts.append("## Test Input\n")
            prompt_parts.append(self._format_grid(test_input, "Input"))
            prompt_parts.append(f"\n**Colors in test:** {test_colors}\n")
        
        # Build the EXPLICIT instruction
        prompt_parts.append("\n## DETECTED TRANSFORMATION\n\n")
        
        if transformations:
            # Use the first consistent transformation
            main_transform = transformations[0]
            prompt_parts.append(f"{main_transform}\n\n")
            
            # Generate the EXACT JSON to output
            json_instruction = self._generate_json_instruction(main_transform, test_colors)
            prompt_parts.append("## YOUR OUTPUT\n\n")
            prompt_parts.append("Copy this JSON (fill in the values from above):\n\n")
            prompt_parts.append(f"```json\n{json_instruction}\n```\n")
        else:
            prompt_parts.append("**DETECTED TRANSFORMATION: UNKNOWN**\n\n")
            prompt_parts.append("Analyze the examples and output a JSON action.\n")
        
        return "".join(prompt_parts)
    
    def _generate_json_instruction(self, transformation: str, test_colors: List[int]) -> str:
        """
        Generate the exact JSON instruction based on detected transformation.
        
        Args:
            transformation: The detected transformation string
            test_colors: Colors present in test input
            
        Returns:
            JSON string template
        """
        import re
        
        # Get the main color from test (first non-zero color)
        main_color = test_colors[0] if test_colors else 1
        
        # Parse TRANSLATION
        match = re.search(r'TRANSLATION.*?dx=(-?\d+).*?dy=(-?\d+)', transformation)
        if match:
            dx, dy = match.groups()
            return f'{{"action": "translate", "params": {{"dx": {dx}, "dy": {dy}}}}}'
        
        # Parse ROTATION
        match = re.search(r'ROTATION.*?angle=(\d+)', transformation)
        if match:
            angle = match.group(1)
            return f'{{"action": "rotate", "params": {{"angle": {angle}}}, "color_filter": {main_color}}}'
        
        # Parse REFLECTION - check if it's grid-level or object-level
        match = re.search(r'REFLECTION.*?axis=(\w+)', transformation)
        if match:
            axis = match.group(1)
            # Check if this is a grid-level reflection
            if "GRID-LEVEL" in transformation or "color_filter" not in transformation.lower():
                # Grid-level reflection - use grid_level flag to prevent auto-detection
                return f'{{"action": "reflect", "params": {{"axis": "{axis}", "grid_level": true}}}}'
            else:
                # Object-level reflection
                return f'{{"action": "reflect", "params": {{"axis": "{axis}"}}, "color_filter": {main_color}}}'
        
        # Parse COLOR CHANGE
        match = re.search(r'COLOR CHANGE.*?from_color=(\d+).*?to_color=(\d+)', transformation)
        if match:
            from_c, to_c = match.groups()
            return f'{{"action": "color_change", "params": {{"from_color": {from_c}, "to_color": {to_c}}}}}'
        
        # Parse DRAW LINE - multiple patterns to catch
        match = re.search(r'DRAW LINE.*?color[= ]+(\d+)', transformation)
        if not match:
            match = re.search(r'DRAW LINE.*?(\d+)', transformation)
        if match:
            color = match.group(1)
            return f'{{"action": "draw_line", "color_filter": {color}}}'
        
        # Fallback
        return '{"action": "translate", "params": {"dx": 0, "dy": 0}}'
    
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