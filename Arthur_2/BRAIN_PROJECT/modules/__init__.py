"""
BRAIN Project - Modules Package
===============================
Neuro-Symbolic ARC-AGI Solver Pipeline

Pipeline Flow:
    Input Grid -> Perception -> Transformation Detection -> Prompting -> LLM Reasoning -> Execution -> Analysis -> Visualization

Features (TIER 1-3):
    - Robust error handling and logging (TIER 1)
    - Extended DSL: symmetry, flood_fill, conditional_color (TIER 2)
    - Rule Memory (RAG) for few-shot learning (TIER 3)
    - Self-correction loop with error feedback (TIER 3)
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

# TIER 1: Structured Logging
from .logger import BRAINLogger, LogLevel, LogEntry, PerformanceMetrics, get_logger, set_logger

# TIER 3: Rule Memory (RAG)
from .rule_memory import RuleMemory, StoredRule, TaskSignature

# Model Comparison
from .model_comparator import ModelComparator, ModelComparisonResult, ModelResult, RECOMMENDED_MODELS

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
    
    # TIER 1: Logging
    "BRAINLogger",              # Structured logging
    "LogLevel",                 # Log component identifiers
    "LogEntry",                 # Structured log entry
    "PerformanceMetrics",       # Performance metrics collection
    "get_logger",               # Get global logger
    "set_logger",               # Set global logger
    
    # TIER 3: Rule Memory (RAG)
    "RuleMemory",               # Rule storage and retrieval
    "StoredRule",               # Stored rule structure
    "TaskSignature",            # Task feature signature
    
    # Model Comparison
    "ModelComparator",          # Multi-model comparison
    "ModelComparisonResult",    # Comparison results
    "ModelResult",              # Single model result
    "RECOMMENDED_MODELS",       # Suggested models dict
]
