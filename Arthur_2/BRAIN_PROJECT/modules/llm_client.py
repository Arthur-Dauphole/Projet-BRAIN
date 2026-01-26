"""
llm_client.py - LLM Client (Reasoning Module)
==============================================
Step 3 of the pipeline: LLM Reasoning

Handles communication with the LLM (via Ollama) for reasoning
about grid transformations.
"""

import json
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """
    Structured response from the LLM.
    
    Attributes:
        raw_text: The complete raw response from the LLM
        reasoning: Extracted reasoning/explanation
        predicted_grid: Extracted output grid (if parseable)
        action_data: Extracted action JSON for the executor (single action)
        multi_actions: List of actions for multi-transform mode
        confidence: Self-reported confidence (if available)
        metadata: Additional response metadata
    """
    raw_text: str
    reasoning: Optional[str] = None
    predicted_grid: Optional[List[List[int]]] = None
    action_data: Optional[Dict[str, Any]] = None
    multi_actions: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMClient:
    """
    Client for interacting with LLM via Ollama.
    
    Responsibilities:
        - Send prompts to Ollama
        - Parse and structure LLM responses
        - Extract predicted grids from text
        - Handle errors and retries
    """
    
    DEFAULT_MODEL = "llama3"
    
    def __init__(
        self, 
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 120
    ):
        """
        Initialize the LLM client.
        
        Args:
            model: Ollama model name (default: llama3.2)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            timeout: Request timeout in seconds
        """
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._ollama = None
    
    def _get_client(self):
        """Lazy initialization of Ollama client."""
        if self._ollama is None:
            try:
                import ollama
                self._ollama = ollama
            except ImportError:
                raise ImportError(
                    "Ollama package not installed. "
                    "Please run: pip install ollama"
                )
        return self._ollama
    
    def query(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Send a query to the LLM and get a structured response.
        
        Args:
            prompt: The main prompt/question
            system_prompt: Optional system prompt for context
            
        Returns:
            Structured LLMResponse object
        """
        ollama = self._get_client()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Call Ollama
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            )
            
            raw_text = response["message"]["content"]
            
            # Parse the response
            return self._parse_response(raw_text)
            
        except Exception as e:
            # Return error response
            return LLMResponse(
                raw_text=f"Error: {str(e)}",
                metadata={"error": True, "error_message": str(e)}
            )
    
    def _parse_response(self, raw_text: str) -> LLMResponse:
        """
        Parse raw LLM response into structured format.
        
        Args:
            raw_text: Raw response text from LLM
            
        Returns:
            Structured LLMResponse
        """
        response = LLMResponse(raw_text=raw_text)
        
        # Try to extract reasoning
        response.reasoning = self._extract_reasoning(raw_text)
        
        # Try to extract multi-actions (array of actions for multi-transform)
        response.multi_actions = self._extract_multi_actions(raw_text)
        
        # Try to extract single action JSON (PRIORITY - for executor)
        response.action_data = self._extract_action_json(raw_text)
        
        # Try to extract grid (fallback if LLM provides one directly)
        response.predicted_grid = self._extract_grid(raw_text)
        
        return response
    
    def _extract_multi_actions(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract an array of actions from the LLM response (for multi-transform mode).
        
        Looks for JSON arrays in the format:
        ```json
        [
          {"color": 2, "action": "translate", "params": {"dx": 2, "dy": 0}},
          {"color": 1, "action": "rotate", "params": {"angle": 90}}
        ]
        ```
        
        Args:
            text: Raw response text
            
        Returns:
            List of parsed action dictionaries or None
        """
        # Pattern 1: JSON array in code block
        array_pattern = r"```json\s*(\[\s*\{[^`]+\}\s*\])\s*```"
        matches = re.findall(array_pattern, text, re.DOTALL)
        
        if matches:
            for json_str in reversed(matches):
                try:
                    actions = json.loads(json_str.strip())
                    if isinstance(actions, list) and len(actions) > 0:
                        # Validate and normalize each action
                        valid_actions = []
                        for action in actions:
                            if self._validate_multi_action(action):
                                valid_actions.append(self._normalize_multi_action(action))
                        
                        if valid_actions:
                            return valid_actions
                except json.JSONDecodeError:
                    continue
        
        # Pattern 2: Generic code block with array
        generic_pattern = r"```\s*(\[\s*\{[^`]+\}\s*\])\s*```"
        matches = re.findall(generic_pattern, text, re.DOTALL)
        
        if matches:
            for json_str in reversed(matches):
                try:
                    actions = json.loads(json_str.strip())
                    if isinstance(actions, list) and len(actions) > 0:
                        valid_actions = []
                        for action in actions:
                            if self._validate_multi_action(action):
                                valid_actions.append(self._normalize_multi_action(action))
                        
                        if valid_actions:
                            return valid_actions
                except json.JSONDecodeError:
                    continue
        
        # Pattern 3: Standalone JSON array
        standalone_pattern = r'\[\s*(\{[^\]]+\})\s*\]'
        matches = re.findall(standalone_pattern, text, re.DOTALL)
        
        if matches:
            for match in reversed(matches):
                try:
                    # Reconstruct the array
                    json_str = "[" + match + "]"
                    actions = json.loads(json_str)
                    if isinstance(actions, list) and len(actions) > 0:
                        valid_actions = []
                        for action in actions:
                            if self._validate_multi_action(action):
                                valid_actions.append(self._normalize_multi_action(action))
                        
                        if valid_actions:
                            return valid_actions
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _validate_multi_action(self, data: dict) -> bool:
        """
        Validate that a multi-action entry has required fields.
        
        Args:
            data: Parsed JSON data for one action
            
        Returns:
            True if valid
        """
        if not isinstance(data, dict):
            return False
        
        # Must have "color" and "action" fields
        if "color" not in data:
            return False
        
        if "action" not in data:
            return False
        
        return True
    
    def _normalize_multi_action(self, data: dict) -> dict:
        """
        Normalize a multi-action entry.
        
        Args:
            data: Parsed action data
            
        Returns:
            Normalized action data
        """
        # Use the existing normalize function for the action part
        result = dict(data)
        
        # Ensure color is int
        if "color" in result:
            try:
                result["color"] = int(result["color"])
            except (ValueError, TypeError):
                pass
        
        # Normalize params
        if "params" in result and isinstance(result["params"], dict):
            params = dict(result["params"])
            
            # Color-related params
            for key in ["from_color", "to_color", "color"]:
                if key in params:
                    try:
                        params[key] = int(params[key])
                    except (ValueError, TypeError):
                        pass
            
            # Numeric params
            for key in ["dx", "dy", "angle", "factor"]:
                if key in params:
                    val = params[key]
                    if isinstance(val, str):
                        try:
                            if key == "factor":
                                params[key] = float(val)
                            else:
                                params[key] = int(val)
                        except ValueError:
                            pass
            
            result["params"] = params
        
        return result
    
    def _extract_action_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract the action JSON block from the LLM response.
        
        Looks for JSON blocks in the format:
        ```json
        {"action": "translate", "params": {"dx": 3, "dy": 0}, "color_filter": 2}
        ```
        
        Args:
            text: Raw response text
            
        Returns:
            Parsed action dictionary or None
        """
        # Pattern 1: JSON in code block with json marker
        json_block_pattern = r"```json\s*(\{[^`]+\})\s*```"
        matches = re.findall(json_block_pattern, text, re.DOTALL)
        
        if matches:
            # Take the last JSON block (most likely the final answer)
            for json_str in reversed(matches):
                try:
                    action_data = json.loads(json_str.strip())
                    if self._validate_action_data(action_data):
                        return self._normalize_action_data(action_data)
                except json.JSONDecodeError:
                    continue
        
        # Pattern 2: JSON in generic code block
        generic_block_pattern = r"```\s*(\{[^`]+\})\s*```"
        matches = re.findall(generic_block_pattern, text, re.DOTALL)
        
        if matches:
            for json_str in reversed(matches):
                try:
                    action_data = json.loads(json_str.strip())
                    if self._validate_action_data(action_data):
                        return self._normalize_action_data(action_data)
                except json.JSONDecodeError:
                    continue
        
        # Pattern 3: Standalone JSON object (no code block)
        # Look for {"action": ...} pattern
        standalone_pattern = r'(\{"action"\s*:\s*"[^"]+"\s*,\s*"params"\s*:\s*\{[^}]+\}[^}]*\})'
        matches = re.findall(standalone_pattern, text)
        
        if matches:
            for json_str in reversed(matches):
                try:
                    action_data = json.loads(json_str)
                    if self._validate_action_data(action_data):
                        return self._normalize_action_data(action_data)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _validate_action_data(self, data: dict) -> bool:
        """
        Validate that action data has required fields.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            True if valid action data
        """
        if not isinstance(data, dict):
            return False
        
        # Must have "action" field
        if "action" not in data:
            return False
        
        # Action must be a string
        if not isinstance(data.get("action"), str):
            return False
        
        return True
    
    def _normalize_action_data(self, data: dict) -> dict:
        """
        Normalize action data by converting color names to numbers.
        
        Args:
            data: Parsed action data
            
        Returns:
            Normalized action data with color names converted to integers
        """
        # Color name to number mapping
        COLOR_MAP = {
            "black": 0, "blue": 1, "red": 2, "green": 3, "yellow": 4,
            "grey": 5, "gray": 5, "magenta": 6, "pink": 6, "orange": 7,
            "cyan": 8, "teal": 8, "brown": 9, "maroon": 9,
            # Also handle string numbers
            "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
            "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        }
        
        def convert_color(value):
            """Convert a color value to integer."""
            if value is None:
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                value_lower = value.lower().strip()
                if value_lower in COLOR_MAP:
                    return COLOR_MAP[value_lower]
                # Try to parse as integer
                try:
                    return int(value)
                except ValueError:
                    return None
            return None
        
        # Deep copy to avoid modifying original
        result = dict(data)
        
        # Convert color_filter
        if "color_filter" in result:
            result["color_filter"] = convert_color(result["color_filter"])
        
        # Convert params
        if "params" in result and isinstance(result["params"], dict):
            params = dict(result["params"])
            
            # Color-related params
            for key in ["from_color", "to_color", "color"]:
                if key in params:
                    params[key] = convert_color(params[key])
            
            # Numeric params
            for key in ["dx", "dy", "angle", "factor"]:
                if key in params:
                    val = params[key]
                    if isinstance(val, str):
                        try:
                            if key == "factor":
                                params[key] = float(val)
                            else:
                                params[key] = int(val)
                        except ValueError:
                            pass
            
            result["params"] = params
        
        return result
    
    def _extract_reasoning(self, text: str) -> Optional[str]:
        """
        Extract the reasoning section from the response.
        
        Args:
            text: Raw response text
            
        Returns:
            Extracted reasoning or None
        """
        # Look for common reasoning markers
        patterns = [
            r"\*\*Reasoning\*\*:?\s*(.+?)(?=\*\*Output|```|\Z)",
            r"Reasoning:?\s*(.+?)(?=Output|```|\Z)",
            r"(?:Let me|I will|First,|The pattern)(.+?)(?=```|\Z)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern found, return everything before any grid
        grid_start = text.find("```")
        if grid_start > 0:
            return text[:grid_start].strip()
        
        return None
    
    def _extract_grid(self, text: str) -> Optional[List[List[int]]]:
        """
        Extract a grid from the response text.
        
        Args:
            text: Raw response text
            
        Returns:
            2D list representing the grid, or None if not found
        """
        # Try to find JSON array
        json_patterns = [
            r"\[\s*\[[\d,\s\[\]]+\]\s*\]",  # [[0,1],[2,3]]
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    # Take the last match (usually the output)
                    grid = json.loads(matches[-1])
                    if self._validate_grid(grid):
                        return grid
                except json.JSONDecodeError:
                    continue
        
        # Try to parse from code block
        code_block_pattern = r"```(?:\w+)?\s*([\d\s]+)\s*```"
        matches = re.findall(code_block_pattern, text)
        
        for match in matches:
            grid = self._parse_ascii_grid(match)
            if grid:
                return grid
        
        return None
    
    def _parse_ascii_grid(self, text: str) -> Optional[List[List[int]]]:
        """
        Parse an ASCII representation of a grid.
        
        Args:
            text: ASCII grid text
            
        Returns:
            2D list or None
        """
        lines = text.strip().split("\n")
        grid = []
        
        for line in lines:
            row = []
            for char in line.split():
                try:
                    row.append(int(char))
                except ValueError:
                    continue
            if row:
                grid.append(row)
        
        if self._validate_grid(grid):
            return grid
        return None
    
    def _validate_grid(self, grid: List[List[int]]) -> bool:
        """
        Validate that a grid is well-formed.
        
        Args:
            grid: The grid to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not grid or not isinstance(grid, list):
            return False
        
        if not all(isinstance(row, list) for row in grid):
            return False
        
        # Check all rows have same length
        if len(set(len(row) for row in grid)) != 1:
            return False
        
        # Check all values are valid (0-9)
        for row in grid:
            for val in row:
                if not isinstance(val, int) or val < 0 or val > 9:
                    return False
        
        return True
    
    def check_connection(self) -> bool:
        """
        Check if Ollama is running and the model is available.
        
        Returns:
            True if connection is successful
        """
        try:
            ollama = self._get_client()
            response = ollama.list()
            
            # Handle both old and new API formats
            # New API: response.models is a list of Model objects with .model attribute
            # Old API: response is a dict with "models" key containing dicts with "name" key
            model_names = []
            
            if hasattr(response, 'models'):
                # New API format (ollama >= 0.4)
                for m in response.models:
                    if hasattr(m, 'model'):
                        model_names.append(m.model)
                    elif hasattr(m, 'name'):
                        model_names.append(m.name)
            elif isinstance(response, dict) and "models" in response:
                # Old API format
                model_names = [m.get("name", "") for m in response["models"]]
            
            # Check if our model is available (with or without tag)
            for name in model_names:
                if name.startswith(self.model):
                    return True
            
            print(f"Warning: Model '{self.model}' not found. Available: {model_names}")
            return False
            
        except Exception as e:
            print(f"Connection error: {e}")
            return False
