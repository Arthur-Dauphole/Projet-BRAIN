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
        confidence: Self-reported confidence (if available)
        metadata: Additional response metadata
    """
    raw_text: str
    reasoning: Optional[str] = None
    predicted_grid: Optional[List[List[int]]] = None
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
    
    DEFAULT_MODEL = "llama3.2"
    
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
        
        # Try to extract grid
        response.predicted_grid = self._extract_grid(raw_text)
        
        return response
    
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
            models = ollama.list()
            model_names = [m["name"] for m in models.get("models", [])]
            
            # Check if our model is available (with or without tag)
            for name in model_names:
                if name.startswith(self.model):
                    return True
            
            print(f"Warning: Model '{self.model}' not found. Available: {model_names}")
            return False
            
        except Exception as e:
            print(f"Connection error: {e}")
            return False
