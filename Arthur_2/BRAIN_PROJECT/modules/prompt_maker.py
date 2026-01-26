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
    
    # System prompt template
    SYSTEM_PROMPT = """You are an expert at solving ARC-AGI (Abstraction and Reasoning Corpus) puzzles.
Your task is to analyze input-output pairs and discover the transformation rule.

Key principles:
1. Look for geometric patterns (shapes, lines, symmetry)
2. Identify color relationships and transformations
3. Consider spatial operations (rotation, reflection, translation)
4. Think about object relationships (containment, adjacency, grouping)
5. Reason step by step before providing your answer

Always explain your reasoning clearly before giving the final transformation rule."""

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
        Format an input-output pair.
        
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
        
        return "\n".join(parts)
    
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
        Format a single object description.
        
        Args:
            obj: The object to describe
            
        Returns:
            Description string
        """
        color_name = obj.properties.get("color_name", f"color_{obj.color}")
        return (f"{color_name} {obj.object_type} "
                f"(pos: {obj.bounding_box}, size: {obj.width}x{obj.height}, "
                f"pixels: {obj.area})")
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the LLM.
        
        Returns:
            System prompt string
        """
        return self.SYSTEM_PROMPT
    
    def create_reasoning_chain_prompt(self, task: ARCTask) -> str:
        """
        Create a prompt that encourages step-by-step reasoning.
        
        Args:
            task: The ARCTask to reason about
            
        Returns:
            Chain-of-thought prompt string
        """
        base_prompt = self.create_task_prompt(task)
        
        cot_instruction = """
## Reasoning Process

Please follow these steps:

1. **Observe**: What patterns do you see in the input grids?
2. **Compare**: How do the outputs differ from the inputs?
3. **Hypothesize**: What transformation rule could explain all examples?
4. **Verify**: Does your rule work for all training examples?
5. **Apply**: Apply the rule to the test input.

Show your work for each step.
"""
        
        return base_prompt + cot_instruction
