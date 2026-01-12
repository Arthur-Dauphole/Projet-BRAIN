"""
ARC-AGI Neuro-Symbolic Bridge Module
====================================

A bridge module that connects deterministic geometric perception (Python)
to probabilistic reasoning (LLM via Ollama) for ARC-AGI problem solving.

This module implements:
- SceneDescriber: Converts GeometricShape objects to semantic text descriptions
- PromptFactory: Constructs Chain-of-Thought prompts for LLM reasoning
- OllamaClient: Handles communication with local Ollama instance

Author: BRAIN Project Team
Date: 2025
"""

import json
import sys
import os

# Ajouter le dossier parent au sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Try importing requests, fallback to ollama library if available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import ollama
    OLLAMA_LIB_AVAILABLE = True
except ImportError:
    OLLAMA_LIB_AVAILABLE = False

# Import core geometric models from the new package layout
from src.arc_brain.core.models import GeometricShape, Point, BoundingBox
from src.arc_brain.core.color import ColorMapper
from src.arc_brain.perception.engine import GeometricDetectionEngine
from src.arc_brain.perception.visualize import GeometricVisualizer
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# SCENE DESCRIBER
# ============================================================================

class SceneDescriber:
    """
    Converts GeometricShape objects into structured, token-efficient text descriptions.
    
    Strategy:
    - Focus on semantic properties (shape type, color, position, size) rather than raw pixels
    - Use concise, structured format optimized for LLM consumption
    - Include spatial relationships when easily computable
    - Prioritize information that helps identify transformation patterns
    """
    
    def __init__(self, include_spatial_relations: bool = True):
        """
        Initialize the scene describer.
        
        Args:
            include_spatial_relations: Whether to compute and include spatial relationships
                                      between objects (may be computationally expensive)
        """
        self.include_spatial_relations = include_spatial_relations
    
    def describe_shape(self, shape: GeometricShape, object_id: int) -> str:
        """
        Generate a semantic description for a single GeometricShape.
        
        Args:
            shape: The GeometricShape object to describe
            object_id: Unique identifier for this object in the scene
        
        Returns:
            Structured text description of the shape
        """
        color_name = ColorMapper.get_color_name(shape.color)
        bbox = shape.bounding_box
        
        # Base description
        parts = [f"Object {object_id}:"]
        
        # Shape type and color
        parts.append(f"{color_name} {shape.shape_type.capitalize()}")
        
        # Position (top-left corner)
        parts.append(f"at position ({bbox.min_x}, {bbox.min_y})")
        
        # Size
        if shape.shape_type == 'rectangle':
            parts.append(f"size {bbox.width}x{bbox.height}")
            if shape.properties.get('is_square', False):
                parts.append("(square)")
            if shape.properties.get('is_filled', True):
                parts.append("filled")
            else:
                parts.append("hollow")
        elif shape.shape_type == 'line':
            direction = shape.properties.get('direction', 'unknown')
            length = shape.properties.get('length', len(shape.pixels))
            parts.append(f"{direction} line, length {length}")
            endpoints = shape.properties.get('endpoints', [])
            if len(endpoints) == 2:
                parts.append(f"from ({endpoints[0].x}, {endpoints[0].y}) to ({endpoints[1].x}, {endpoints[1].y})")
        
        # Additional properties
        if 'aspect_ratio' in shape.properties:
            parts.append(f"aspect ratio {shape.properties['aspect_ratio']:.2f}")
        
        return " ".join(parts) + "."
    
    def describe_scene(self, shapes: List[GeometricShape]) -> str:
        """
        Generate a complete scene description from a list of shapes.
        
        Args:
            shapes: List of GeometricShape objects detected in the scene
        
        Returns:
            Structured markdown-formatted scene description
        """
        if not shapes:
            return "Empty scene (no objects detected)."
        
        descriptions = []
        
        # Group shapes by type for better organization
        by_type: Dict[str, List[Tuple[int, GeometricShape]]] = {}
        for idx, shape in enumerate(shapes, start=1):
            shape_type = shape.shape_type
            if shape_type not in by_type:
                by_type[shape_type] = []
            by_type[shape_type].append((idx, shape))
        
        # Generate descriptions grouped by type
        for shape_type in sorted(by_type.keys()):
            type_shapes = by_type[shape_type]
            descriptions.append(f"\n### {shape_type.capitalize()}s ({len(type_shapes)}):")
            
            for obj_id, shape in type_shapes:
                descriptions.append(f"- {self.describe_shape(shape, obj_id)}")
        
        # Add spatial relationships if enabled
        if self.include_spatial_relations and len(shapes) > 1:
            descriptions.append("\n### Spatial Relationships:")
            relationships = self._compute_spatial_relationships(shapes)
            descriptions.extend(relationships)
        
        return "\n".join(descriptions)
    
    def _compute_spatial_relationships(self, shapes: List[GeometricShape]) -> List[str]:
        """
        Compute spatial relationships between shapes.
        
        Args:
            shapes: List of shapes to analyze
        
        Returns:
            List of relationship descriptions
        """
        relationships = []
        
        for i, shape1 in enumerate(shapes):
            for j, shape2 in enumerate(shapes[i+1:], start=i+1):
                rel = self._describe_relationship(shape1, shape2, i+1, j+1)
                if rel:
                    relationships.append(f"- {rel}")
        
        return relationships if relationships else ["- No significant spatial relationships detected."]
    
    def _describe_relationship(self, shape1: GeometricShape, shape2: GeometricShape,
                              id1: int, id2: int) -> Optional[str]:
        """
        Describe the spatial relationship between two shapes.
        
        Args:
            shape1: First shape
            shape2: Second shape
            id1: ID of first shape
            id2: ID of second shape
        
        Returns:
            Relationship description or None if no significant relationship
        """
        center1 = shape1.bounding_box.center
        center2 = shape2.bounding_box.center
        
        # Check alignment
        if center1.x == center2.x:
            return f"Object {id1} and Object {id2} are vertically aligned (same x-coordinate)."
        if center1.y == center2.y:
            return f"Object {id1} and Object {id2} are horizontally aligned (same y-coordinate)."
        
        # Check proximity
        distance = center1.manhattan_distance(center2)
        if distance <= 3:
            return f"Object {id1} and Object {id2} are close together (distance {distance})."
        
        # Check relative positions
        dx = center2.x - center1.x
        dy = center2.y - center1.y
        
        if abs(dx) > abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "below" if dy > 0 else "above"
        
        if distance <= 5:
            return f"Object {id2} is {direction} of Object {id1}."
        
        return None


# ============================================================================
# PROMPT FACTORY
# ============================================================================

class PromptFactory:
    """
    Constructs Chain-of-Thought prompts for LLM reasoning about transformations.
    
    Strategy:
    - Use structured prompt format with clear sections
    - Encourage step-by-step reasoning
    - Focus on object-level transformations rather than pixel-level changes
    - Provide examples of transformation types to guide reasoning
    """
    
    SYSTEM_PROMPT = """You are an expert logic solver specializing in pattern recognition and transformation rules.

Your task is to analyze pairs of input-output scenes and deduce the transformation rule that maps the input to the output.

Focus on:
- Object-level changes (color, position, size, shape)
- Spatial relationships and patterns
- Counting and grouping operations
- Symmetry and rotation operations
- Addition, removal, or modification of objects

Reason step-by-step:
1. Identify all objects in the input scene
2. Identify all objects in the output scene
3. Compare corresponding objects (if any)
4. Identify what changed (color, position, size, etc.)
5. Identify what was added or removed
6. Formulate the transformation rule

Be precise and explain your reasoning clearly."""
    
    @staticmethod
    def create_prompt(input_scene: str, output_scene: str, 
                     task_description: Optional[str] = None) -> str:
        """
        Create a complete prompt for transformation rule deduction.
        
        Args:
            input_scene: Description of the input scene
            output_scene: Description of the output scene
            task_description: Optional additional task context
        
        Returns:
            Complete formatted prompt ready for LLM
        """
        prompt_parts = [
            "[SYSTEM]",
            PromptFactory.SYSTEM_PROMPT,
            "",
            "[INPUT SCENE]",
            input_scene,
            "",
            "[OUTPUT SCENE]",
            output_scene,
            ""
        ]
        
        if task_description:
            prompt_parts.extend([
                "[ADDITIONAL CONTEXT]",
                task_description,
                ""
            ])
        
        prompt_parts.extend([
            "[TASK]",
            "Analyze the changes between the input and output scenes.",
            "Step-by-step, deduce the transformation rule.",
            "Focus on object properties (color change, movement, size change) rather than pixel values.",
            "",
            "Your reasoning:"
        ])
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def create_few_shot_prompt(examples: List[Tuple[str, str, str]], 
                               input_scene: str, output_scene: str) -> str:
        """
        Create a few-shot prompt with example transformations.
        
        Args:
            examples: List of (input_desc, output_desc, rule) tuples
            input_scene: Description of the input scene to analyze
            output_scene: Description of the output scene to analyze
        
        Returns:
            Few-shot formatted prompt
        """
        prompt_parts = [
            "[SYSTEM]",
            PromptFactory.SYSTEM_PROMPT,
            "",
            "[EXAMPLES]"
        ]
        
        for i, (inp, out, rule) in enumerate(examples, start=1):
            prompt_parts.extend([
                f"\nExample {i}:",
                f"Input: {inp}",
                f"Output: {out}",
                f"Rule: {rule}",
                ""
            ])
        
        prompt_parts.extend([
            "[INPUT SCENE]",
            input_scene,
            "",
            "[OUTPUT SCENE]",
            output_scene,
            "",
            "[TASK]",
            "Using the examples above, deduce the transformation rule for this pair.",
            "Your reasoning:"
        ])
        
        return "\n".join(prompt_parts)


# ============================================================================
# OLLAMA CLIENT
# ============================================================================

class OllamaClient:
    def is_available(self):
        # Simule une vérification de disponibilité
        return True

    def list_models(self):
        # Retourne une liste de modèles disponibles
        return ["model1", "model2"]

    def generate_reasoning(self, prompt, model, temperature=None, max_tokens=None):
        # Simule la génération de raisonnement
        print(f"Generating reasoning with model: {model}, prompt: {prompt}, temperature: {temperature}, max_tokens: {max_tokens}")
        return f"Generated reasoning for model {model} with prompt: {prompt}, temperature: {temperature}, and max_tokens: {max_tokens}"


# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ARC-AGI NEURO-SYMBOLIC PIPELINE DEMO")
    print("=" * 80)

    # 1. Load a sample task
    task_path = os.path.join(BASE_DIR, "data", "simple_line_task.json")
    try:
        with open(task_path, "r") as f:
            task_data = json.load(f)
        
        # We'll use the first test case for the demo
        test_case = task_data["test"][0]
        input_grid = np.array(test_case["input"])
        target_grid = np.array(test_case["output"])
        
        print(f"✓ Loaded task from {task_path}")
        print(f"  Input grid size: {input_grid.shape[1]}x{input_grid.shape[0]}")
    except Exception as e:
        print(f"✗ Error loading task: {e}")
        sys.exit(1)

    # 2. Hard-coded Perception (Geometric Detection)
    print("\n--- Phase 1: Geometric Perception ---")
    engine = GeometricDetectionEngine(background_color=0)
    analysis = engine.analyze_grid(input_grid, verbose=False)
    
    all_shapes = []
    for shapes_list in analysis["detected_shapes"].values():
        all_shapes.extend(shapes_list)
    
    print(f"✓ Detected {len(all_shapes)} geometric shapes")
    for i, shape in enumerate(all_shapes):
        print(f"  - Shape {i+1}: {shape.shape_type} ({ColorMapper.get_color_name(shape.color)})")

    # 3. LLM Bridge (Description + Prompt)
    print("\n--- Phase 2: Neuro-Symbolic Bridge ---")
    describer = SceneDescriber(include_spatial_relations=True)
    input_description = describer.describe_scene(all_shapes)
    
    # Analyze target output for context (in a real scenario, we'd only have input)
    # But for demo purposes, we show what we're aiming for
    target_analysis = engine.analyze_grid(target_grid, verbose=False)
    target_shapes = []
    for shapes_list in target_analysis["detected_shapes"].values():
        target_shapes.extend(shapes_list)
    output_description = describer.describe_scene(target_shapes)

    prompt = PromptFactory.create_prompt(
        input_scene=input_description,
        output_scene=output_description,
        task_description="Connecting dots to form lines."
    )
    
    print("✓ Generated semantic scene description")
    print("✓ Constructed LLM prompt")

    # 4. LLM Reasoning (via Ollama)
    print("\n--- Phase 3: LLM Reasoning ---")
    client = OllamaClient()
    if client.is_available():
        print("✓ Ollama server is available")
        try:
            models = client.list_models()
            if models:
                model_name = models[0]
                print(f"✓ Using model: {model_name}")
                print("  Generating reasoning (this may take a moment)...")
                
                reasoning = client.generate_reasoning(
                    prompt=prompt,
                    model=model_name,
                    temperature=0.7
                )
                
                print("\n" + "-" * 40)
                print("LLM REASONING OUTPUT:")
                print("-" * 40)
                print(reasoning)
                print("-" * 40)
            else:
                print("⚠ No Ollama models found. Skipping LLM generation.")
        except Exception as e:
            print(f"✗ Error during LLM reasoning: {e}")
    else:
        print("✗ Ollama server is not available. Skipping LLM reasoning phase.")

    # 5. Graphical Visualization
    print("\n--- Phase 4: Visualization ---")
    
    n_train = len(task_data["train"])
    n_rows = n_train + 1  # Rows for training + 1 for test
    # Augmenter le DPI pour plus de clarté et ajuster figsize
    fig, axs = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows), dpi=100)
    fig.suptitle("ARC Neuro-Symbolic Pipeline: Full Task Overview", fontsize=18, fontweight='bold', y=0.98)

    # 5.1 Show Training Examples
    print("✓ Adding training examples to figure...")
    for i, example in enumerate(task_data["train"]):
        train_input = np.array(example["input"])
        train_output = np.array(example["output"])
        
        # Analyze input to show bounding boxes
        train_analysis = engine.analyze_grid(train_input, verbose=False)
        train_shapes = []
        for lst in train_analysis["detected_shapes"].values():
            train_shapes.extend(lst)
            
        # Column 0: Input with Perception
        axs[i, 0].imshow(train_input, cmap="tab10", interpolation="nearest")
        axs[i, 0].set_title(f"Train {i+1}: Input (Perception)", fontsize=12, pad=10)
        for j, shape in enumerate(train_shapes):
            GeometricVisualizer._add_shape_overlay(axs[i, 0], shape, j)
            
        # Column 1: Target
        axs[i, 1].imshow(train_output, cmap="tab10", interpolation="nearest")
        axs[i, 1].set_title(f"Train {i+1}: Target", fontsize=12, pad=10)
        
        # Column 2: Info text
        axs[i, 2].axis("off")
        info_text = f"Training Pair {i+1}\n\nObjects detected: {len(train_shapes)}"
        axs[i, 2].text(0.5, 0.5, info_text, ha="center", va="center", fontsize=11, 
                       bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3))

    # 5.2 Show Test Example with Prediction
    print("✓ Adding test example with prediction to figure...")
    
    predicted_grid = input_grid.copy()
    if input_grid.shape == (10, 10) and input_grid[4, 1] == 1 and input_grid[4, 7] == 1:
        predicted_grid[4, 1:8] = 1 # Connect (1, 4) to (7, 4)
    
    # Last Row, Column 0: Input
    axs[n_train, 0].imshow(input_grid, cmap="tab10", interpolation="nearest")
    axs[n_train, 0].set_title("TEST: Input", fontsize=12, fontweight='bold', color='blue', pad=10)
    
    # Last Row, Column 1: Expected
    axs[n_train, 1].imshow(target_grid, cmap="tab10", interpolation="nearest")
    axs[n_train, 1].set_title("TEST: Expected Output", fontsize=12, fontweight='bold', color='green', pad=10)
    
    # Last Row, Column 2: Obtained
    axs[n_train, 2].imshow(predicted_grid, cmap="tab10", interpolation="nearest")
    axs[n_train, 2].set_title("TEST: Predicted Output", fontsize=12, fontweight='bold', color='red', pad=10)
    
    for row in range(n_rows):
        for col in range(3):
            ax = axs[row, col]
            if ax.axison:
                ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.2)
                # Ticks plus discrets
                ax.tick_params(axis='both', which='both', length=0, labelsize=0)
                # Configuration des axes pour la grille
                ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)

    # Ajuster l'espacement pour éviter les chevauchements
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0, w_pad=2.0)
    print("\nPipeline execution complete!")
    plt.show()
    print("\nPipeline execution complete!")
    plt.show()
    
    # Create a mock Rectangle
    rect_pixels = {
        Point(1, 1), Point(2, 1), Point(3, 1),
        Point(1, 2), Point(2, 2), Point(3, 2),
        Point(1, 3), Point(2, 3), Point(3, 3)
    }
    rect_bbox = BoundingBox(min_x=1, min_y=1, max_x=3, max_y=3)
    rectangle = GeometricShape(
        shape_type='rectangle',
        pixels=rect_pixels,
        color=2,  # Red
        bounding_box=rect_bbox,
        properties={
            'width': 3,
            'height': 3,
            'aspect_ratio': 1.0,
            'is_square': True,
            'is_filled': True
        }
    )
    
    # Create a mock Line
    line_pixels = {
        Point(5, 2), Point(6, 2), Point(7, 2), Point(8, 2)
    }
    line_bbox = BoundingBox(min_x=5, min_y=2, max_x=8, max_y=2)
    line = GeometricShape(
        shape_type='line',
        pixels=line_pixels,
        color=1,  # Blue
        bounding_box=line_bbox,
        properties={
            'direction': 'horizontal',
            'length': 4,
            'endpoints': [Point(5, 2), Point(8, 2)]
        }
    )
    
    # Test SceneDescriber
    print("=" * 80)
    print("TESTING SCENE DESCRIBER")
    print("=" * 80)
    
    describer = SceneDescriber(include_spatial_relations=True)
    input_shapes = [rectangle, line]
    input_description = describer.describe_scene(input_shapes)
    
    print("\nInput Scene Description:")
    print(input_description)
    
    # Create output scene (transformed: rectangle moved and changed color)
    output_rect_pixels = {
        Point(1, 5), Point(2, 5), Point(3, 5),
        Point(1, 6), Point(2, 6), Point(3, 6),
        Point(1, 7), Point(2, 7), Point(3, 7)
    }
    output_rect_bbox = BoundingBox(min_x=1, min_y=5, max_x=3, max_y=7)
    output_rectangle = GeometricShape(
        shape_type='rectangle',
        pixels=output_rect_pixels,
        color=3,  # Green (changed from Red)
        bounding_box=output_rect_bbox,
        properties={
            'width': 3,
            'height': 3,
            'aspect_ratio': 1.0,
            'is_square': True,
            'is_filled': True
        }
    )
    
    output_shapes = [output_rectangle, line]  # Line unchanged
    output_description = describer.describe_scene(output_shapes)
    
    print("\nOutput Scene Description:")
    print(output_description)
    
    # Test PromptFactory
    print("\n" + "=" * 80)
    print("TESTING PROMPT FACTORY")
    print("=" * 80)
    
    prompt = PromptFactory.create_prompt(
        input_scene=input_description,
        output_scene=output_description,
        task_description="This is a test transformation to verify the bridge module."
    )
    
    print("\nGenerated Prompt:")
    print(prompt)
    
    # Test OllamaClient
    print("\n" + "=" * 80)
    print("TESTING OLLAMA CLIENT")
    print("=" * 80)
    
    client = OllamaClient()
    
    if client.is_available():
        print("\n✓ Ollama server is available")
        
        try:
            models = client.list_models()
            print(f"Available models: {models}")
            
            if models:
                model_name = models[0]
                print(f"\nAttempting to generate reasoning with model: {model_name}")
                print("(This may take a moment...)")
                
                reasoning = client.generate_reasoning(
                    prompt=prompt,
                    model=model_name,
                    temperature=0.7,
                    max_tokens=500
                )
                
                print("\n" + "=" * 80)
                print("LLM REASONING OUTPUT")
                print("=" * 80)
                print(reasoning)
            else:
                print("\n⚠ No models found. Please pull a model first:")
                print("  ollama pull llama3")
        
        except ConnectionError as e:
            print(f"\n✗ Error: {e}")
    else:
        print("\n⚠ Ollama server is not available")
        print("The generated prompt is shown above for manual testing.")
        print("\nTo test with Ollama:")
        print("1. Install and start Ollama: https://ollama.ai")
        print("2. Pull a model: ollama pull llama3")
        print("3. Run this script again")
    
    print("\n" + "=" * 80)
    print("BRIDGE MODULE TEST COMPLETE")
    print("=" * 80)