"""
prompt_maker.py - Prompt Creator (Bridge Module)
================================================
Step 2 of the pipeline: Prompting

Transforms the detected symbols and patterns into structured prompts
for the LLM reasoning engine.
"""

from typing import List, Optional
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

CRITICAL: Look at the "Transformation Analysis" section in each example.
It shows exactly how objects moved: the dx and dy values are already calculated for you.
Use these values directly in your JSON output.

OUTPUT FORMAT (MANDATORY):
You MUST end your response with a JSON block like this:
```json
{"action": "translate", "params": {"dx": NUMBER, "dy": NUMBER}, "color_filter": COLOR}
```

Example: If analysis shows "dx=3, dy=0", output:
```json
{"action": "translate", "params": {"dx": 3, "dy": 0}, "color_filter": 2}
```"""

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
            Analysis string describing the transformation
        """
        if not input_grid.objects or not output_grid.objects:
            return ""
        
        analysis_lines = []
        
        # Compare objects by color
        for in_obj in input_grid.objects:
            # Find matching object in output (same color)
            for out_obj in output_grid.objects:
                if in_obj.color == out_obj.color:
                    if in_obj.bounding_box and out_obj.bounding_box:
                        in_row, in_col = in_obj.bounding_box[0], in_obj.bounding_box[1]
                        out_row, out_col = out_obj.bounding_box[0], out_obj.bounding_box[1]
                        
                        dx = out_col - in_col  # Horizontal shift (column difference)
                        dy = out_row - in_row  # Vertical shift (row difference)
                        
                        color_name = in_obj.properties.get("color_name", f"color_{in_obj.color}")
                        analysis_lines.append(
                            f"- {color_name} object: moved from (row={in_row}, col={in_col}) "
                            f"to (row={out_row}, col={out_col}) → "
                            f"**dx={dx}** (horizontal), **dy={dy}** (vertical)"
                        )
                    break
        
        return "\n".join(analysis_lines)
    
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
        
        cot_instruction = """
## YOUR TASK

1. Look at the **Transformation Analysis** in each training example above.
2. Find the consistent dx and dy values across ALL examples.
3. Output the JSON with those exact values.

REMEMBER:
- dx = horizontal movement (positive = RIGHT)
- dy = vertical movement (positive = DOWN)
- Use the dx and dy values shown in the Transformation Analysis!

## OUTPUT (MANDATORY)

End your response with:
```json
{"action": "translate", "params": {"dx": VALUE, "dy": VALUE}, "color_filter": COLOR}
```
"""
        
        return base_prompt + cot_instruction
