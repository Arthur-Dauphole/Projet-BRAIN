"""
analyzer.py - Result Analyzer (Evaluation Module)
=================================================
Step 4 of the pipeline: Analysis

Evaluates LLM predictions against expected outputs and
provides detailed analysis of results.
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import numpy as np

from .types import Grid
from .llm_client import LLMResponse


@dataclass
class AnalysisResult:
    """
    Detailed analysis of a single prediction.
    
    Attributes:
        is_correct: Whether the prediction exactly matches expected
        accuracy: Pixel-level accuracy (0.0 to 1.0)
        predicted_grid: The predicted grid
        expected_grid: The expected/target grid
        diff_mask: Boolean mask showing differences
        error_analysis: Detailed breakdown of errors
        metrics: Additional computed metrics
    """
    is_correct: bool
    accuracy: float
    predicted_grid: Optional[Grid] = None
    expected_grid: Optional[Grid] = None
    diff_mask: Optional[np.ndarray] = None
    error_analysis: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass 
class BatchAnalysisResult:
    """
    Analysis results for a batch of predictions.
    
    Attributes:
        total_tasks: Number of tasks evaluated
        correct_count: Number of exactly correct predictions
        overall_accuracy: Overall pixel-level accuracy
        individual_results: List of individual AnalysisResult
        summary: Summary statistics
    """
    total_tasks: int
    correct_count: int
    overall_accuracy: float
    individual_results: List[AnalysisResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class ResultAnalyzer:
    """
    Evaluation module for analyzing LLM predictions.
    
    Responsibilities:
        - Compare predicted vs expected grids
        - Calculate accuracy metrics
        - Identify error patterns
        - Generate analysis reports
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        pass
    
    def analyze(
        self, 
        llm_response: LLMResponse, 
        expected_grid: Grid
    ) -> AnalysisResult:
        """
        Analyze a single LLM response against expected output.
        
        Args:
            llm_response: The LLM's response
            expected_grid: The expected output grid
            
        Returns:
            Detailed AnalysisResult
        """
        # Handle case where no grid was extracted
        if llm_response.predicted_grid is None:
            return AnalysisResult(
                is_correct=False,
                accuracy=0.0,
                expected_grid=expected_grid,
                error_analysis={"error": "No grid could be extracted from response"}
            )
        
        predicted_grid = Grid.from_list(llm_response.predicted_grid)
        
        return self.compare_grids(predicted_grid, expected_grid)
    
    def compare_grids(
        self, 
        predicted: Grid, 
        expected: Grid
    ) -> AnalysisResult:
        """
        Compare two grids and compute detailed analysis.
        
        Args:
            predicted: The predicted grid
            expected: The expected grid
            
        Returns:
            Detailed AnalysisResult
        """
        # Check shape match
        if predicted.shape != expected.shape:
            return AnalysisResult(
                is_correct=False,
                accuracy=0.0,
                predicted_grid=predicted,
                expected_grid=expected,
                error_analysis={
                    "error": "Shape mismatch",
                    "predicted_shape": predicted.shape,
                    "expected_shape": expected.shape
                }
            )
        
        # Compute difference mask
        diff_mask = predicted.data != expected.data
        
        # Calculate metrics
        total_pixels = expected.data.size
        correct_pixels = np.sum(~diff_mask)
        accuracy = correct_pixels / total_pixels
        is_correct = np.array_equal(predicted.data, expected.data)
        
        # Detailed error analysis
        error_analysis = self._analyze_errors(predicted, expected, diff_mask)
        
        # Compute additional metrics
        metrics = self._compute_metrics(predicted, expected, diff_mask)
        
        return AnalysisResult(
            is_correct=is_correct,
            accuracy=accuracy,
            predicted_grid=predicted,
            expected_grid=expected,
            diff_mask=diff_mask,
            error_analysis=error_analysis,
            metrics=metrics
        )
    
    def _analyze_errors(
        self, 
        predicted: Grid, 
        expected: Grid, 
        diff_mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze the nature of prediction errors.
        
        Args:
            predicted: Predicted grid
            expected: Expected grid
            diff_mask: Boolean mask of differences
            
        Returns:
            Dictionary with error analysis
        """
        if not np.any(diff_mask):
            return {"no_errors": True}
        
        analysis = {}
        
        # Count errors by position
        error_positions = list(zip(*np.where(diff_mask)))
        analysis["error_count"] = len(error_positions)
        analysis["error_positions"] = error_positions[:10]  # Limit to first 10
        
        # Analyze color confusions
        pred_errors = predicted.data[diff_mask]
        exp_errors = expected.data[diff_mask]
        
        color_confusion = {}
        for p, e in zip(pred_errors.flat, exp_errors.flat):
            key = f"{e}->{p}"  # expected -> predicted
            color_confusion[key] = color_confusion.get(key, 0) + 1
        
        analysis["color_confusions"] = color_confusion
        
        # Identify error patterns
        analysis["error_pattern"] = self._identify_error_pattern(diff_mask)
        
        return analysis
    
    def _identify_error_pattern(self, diff_mask: np.ndarray) -> str:
        """
        Identify common error patterns.
        
        Args:
            diff_mask: Boolean mask of differences
            
        Returns:
            Description of error pattern
        """
        if not np.any(diff_mask):
            return "none"
        
        # Check if errors are in a contiguous region
        # (Simplified check - could be more sophisticated)
        error_rows = np.any(diff_mask, axis=1)
        error_cols = np.any(diff_mask, axis=0)
        
        if np.sum(error_rows) == 1:
            return "single_row_error"
        elif np.sum(error_cols) == 1:
            return "single_column_error"
        elif np.sum(diff_mask) <= 3:
            return "sparse_errors"
        elif np.all(diff_mask):
            return "complete_mismatch"
        else:
            return "scattered_errors"
    
    def _compute_metrics(
        self, 
        predicted: Grid, 
        expected: Grid, 
        diff_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute additional evaluation metrics.
        
        Args:
            predicted: Predicted grid
            expected: Expected grid
            diff_mask: Boolean mask of differences
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Pixel accuracy
        metrics["pixel_accuracy"] = 1.0 - (np.sum(diff_mask) / diff_mask.size)
        
        # Per-color accuracy
        for color in range(10):
            expected_mask = expected.data == color
            if np.any(expected_mask):
                predicted_mask = predicted.data == color
                intersection = np.logical_and(expected_mask, predicted_mask)
                union = np.logical_or(expected_mask, predicted_mask)
                if np.any(union):
                    metrics[f"iou_color_{color}"] = np.sum(intersection) / np.sum(union)
        
        # Shape preservation (did we get dimensions right?)
        metrics["shape_match"] = 1.0 if predicted.shape == expected.shape else 0.0
        
        return metrics
    
    def analyze_batch(
        self, 
        predictions: List[Tuple[LLMResponse, Grid]]
    ) -> BatchAnalysisResult:
        """
        Analyze a batch of predictions.
        
        Args:
            predictions: List of (LLMResponse, expected_grid) tuples
            
        Returns:
            BatchAnalysisResult with overall statistics
        """
        results = []
        correct_count = 0
        total_accuracy = 0.0
        
        for response, expected in predictions:
            result = self.analyze(response, expected)
            results.append(result)
            
            if result.is_correct:
                correct_count += 1
            total_accuracy += result.accuracy
        
        overall_accuracy = total_accuracy / len(predictions) if predictions else 0.0
        
        return BatchAnalysisResult(
            total_tasks=len(predictions),
            correct_count=correct_count,
            overall_accuracy=overall_accuracy,
            individual_results=results,
            summary={
                "exact_match_rate": correct_count / len(predictions) if predictions else 0.0,
                "average_pixel_accuracy": overall_accuracy,
            }
        )
    
    def generate_report(self, result: AnalysisResult) -> str:
        """
        Generate a human-readable report of the analysis.
        
        Args:
            result: The analysis result
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 50)
        lines.append("ANALYSIS REPORT")
        lines.append("=" * 50)
        
        lines.append(f"\n✓ Correct: {result.is_correct}")
        lines.append(f"📊 Pixel Accuracy: {result.accuracy:.2%}")
        
        if result.error_analysis:
            lines.append("\n📋 Error Analysis:")
            for key, value in result.error_analysis.items():
                if key != "error_positions":  # Skip verbose data
                    lines.append(f"  - {key}: {value}")
        
        if result.metrics:
            lines.append("\n📈 Metrics:")
            for key, value in result.metrics.items():
                lines.append(f"  - {key}: {value:.4f}")
        
        lines.append("\n" + "=" * 50)
        
        return "\n".join(lines)
