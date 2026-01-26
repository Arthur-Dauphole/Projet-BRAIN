"""
BRAIN Project - Modules Package
===============================
Neuro-Symbolic ARC-AGI Solver Pipeline

Pipeline Flow:
    Input Grid -> Perception -> Prompting -> LLM Reasoning -> Execution -> Analysis -> Visualization
"""

from .types import Grid, GeometricObject, ARCTask, TaskPair
from .detector import SymbolDetector
from .prompt_maker import PromptMaker
from .llm_client import LLMClient, LLMResponse
from .executor import ActionExecutor, ActionResult
from .analyzer import ResultAnalyzer
from .visualizer import Visualizer

__all__ = [
    # Data Types
    "Grid",
    "GeometricObject",
    "ARCTask",
    "TaskPair",
    # Pipeline Components
    "SymbolDetector",      # Step 1: Perception
    "PromptMaker",         # Step 2: Prompting
    "LLMClient",           # Step 3: Reasoning
    "LLMResponse",         # Step 3: Response structure
    "ActionExecutor",      # Step 4: Execution (THE HANDS)
    "ActionResult",        # Step 4: Execution result
    "ResultAnalyzer",      # Step 5: Analysis
    "Visualizer",          # Step 6: Visualization
]
