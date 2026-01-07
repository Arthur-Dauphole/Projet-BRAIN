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
    """
    Client for communicating with a local Ollama instance.
    
    Handles API calls to the Ollama REST API endpoint (default: http://localhost:11434).
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 60):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL of the Ollama API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.api_url = f"{self.base_url}/api/generate"
    
    def is_available(self) -> bool:
        """
        Check if Ollama server is available.
        
        Returns:
            True if server is reachable, False otherwise
        """
        if OLLAMA_LIB_AVAILABLE:
            try:
                ollama.list()  # Simple check using ollama library
                return True
            except Exception:
                return False
        elif REQUESTS_AVAILABLE:
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                return response.status_code == 200
            except (requests.exceptions.RequestException, requests.exceptions.Timeout):
                return False
        else:
            return False
    
    def generate_reasoning(self, prompt: str, model: str = "llama3",
                          temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """
        Generate reasoning text from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            model: Model name to use (default: "llama3")
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate (None for no limit)
        
        Returns:
            Generated reasoning text
        
        Raises:
            ConnectionError: If Ollama server is not available
            ImportError: If neither requests nor ollama library is available
        """
        if not self.is_available():
            raise ConnectionError(
                f"Ollama server is not available at {self.base_url}. "
                "Please ensure Ollama is running and accessible."
            )
        
        # Use ollama library if available (preferred method)
        if OLLAMA_LIB_AVAILABLE:
            try:
                options = {"temperature": temperature}
                if max_tokens is not None:
                    options["num_predict"] = max_tokens
                
                response = ollama.generate(
                    model=model,
                    prompt=prompt,
                    options=options
                )
                return response.get("response", "").strip()
            except Exception as e:
                raise ConnectionError(f"Error using ollama library: {str(e)}")
        
        # Fallback to requests library
        elif REQUESTS_AVAILABLE:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            if max_tokens is not None:
                payload["options"]["num_predict"] = max_tokens
            
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "").strip()
            
            except requests.exceptions.Timeout:
                raise ConnectionError(
                    f"Request to Ollama timed out after {self.timeout} seconds."
                )
            except requests.exceptions.RequestException as e:
                raise ConnectionError(
                    f"Error communicating with Ollama: {str(e)}"
                )
        else:
            raise ImportError(
                "Neither 'requests' nor 'ollama' library is available. "
                "Please install one: pip install requests OR pip install ollama"
            )
    
    def list_models(self) -> List[str]:
        """
        List available models in Ollama.
        
        Returns:
            List of model names
        
        Raises:
            ConnectionError: If Ollama server is not available
            ImportError: If neither requests nor ollama library is available
        """
        if not self.is_available():
            raise ConnectionError(
                f"Ollama server is not available at {self.base_url}."
            )
        
        # Use ollama library if available (preferred method)
        if OLLAMA_LIB_AVAILABLE:
            try:
                models_data = ollama.list()
                models = [model["name"] for model in models_data.get("models", [])]
                return models
            except Exception as e:
                raise ConnectionError(f"Error listing models: {str(e)}")
        
        # Fallback to requests library
        elif REQUESTS_AVAILABLE:
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                response.raise_for_status()
                
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                return models
            
            except requests.exceptions.RequestException as e:
                raise ConnectionError(f"Error listing models: {str(e)}")
        else:
            raise ImportError(
                "Neither 'requests' nor 'ollama' library is available. "
                "Please install one: pip install requests OR pip install ollama"
            )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Mock data: Create sample GeometricShape objects
    from arc_geometric_detection import BoundingBox, Point
    
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
