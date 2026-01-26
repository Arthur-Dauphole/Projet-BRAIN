"""
BRAIN Project - Modules Package
===============================
Neuro-Symbolic ARC-AGI Solver Pipeline

Pipeline Flow:
    Input Grid -> Perception -> Prompting -> LLM Reasoning -> Analysis -> Visualization
"""

from .types import Grid, GeometricObject
from .detector import SymbolDetector
from .prompt_maker import PromptMaker
from .llm_client import LLMClient
from .analyzer import ResultAnalyzer
from .visualizer import Visualizer

__all__ = [
    "Grid",
    "GeometricObject",
    "SymbolDetector",
    "PromptMaker",
    "LLMClient",
    "ResultAnalyzer",
    "Visualizer",
]
