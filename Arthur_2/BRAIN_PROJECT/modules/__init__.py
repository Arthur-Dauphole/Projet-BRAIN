"""
BRAIN Project - Modules Package
===============================
Neuro-Symbolic ARC-AGI Solver Pipeline

Pipeline Flow:
    Input Grid -> Perception -> Transformation Detection -> Prompting -> LLM Reasoning -> Execution -> Analysis -> Visualization
"""

from .types import Grid, GeometricObject, ARCTask, TaskPair
from .detector import SymbolDetector
from .transformation_detector import TransformationDetector, TransformationResult
from .prompt_maker import PromptMaker
from .llm_client import LLMClient, LLMResponse
from .executor import ActionExecutor, ActionResult
from .analyzer import ResultAnalyzer
from .visualizer import Visualizer
from .batch_runner import BatchRunner, BatchResult, TaskResult, run_batch_evaluation

__all__ = [
    # Data Types
    "Grid",
    "GeometricObject",
    "ARCTask",
    "TaskPair",
    # Pipeline Components
    "SymbolDetector",           # Step 1: Perception (shapes)
    "TransformationDetector",   # Step 1b: Perception (transformations)
    "TransformationResult",     # Step 1b: Result structure
    "PromptMaker",              # Step 2: Prompting
    "LLMClient",                # Step 3: Reasoning
    "LLMResponse",              # Step 3: Response structure
    "ActionExecutor",           # Step 4: Execution (THE HANDS)
    "ActionResult",             # Step 4: Execution result
    "ResultAnalyzer",           # Step 5: Analysis
    "Visualizer",               # Step 6: Visualization
    # Batch Evaluation
    "BatchRunner",              # Batch task evaluation
    "BatchResult",              # Batch results structure
    "TaskResult",               # Single task result
    "run_batch_evaluation",     # Convenience function
]
