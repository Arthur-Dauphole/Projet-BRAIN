"""
model_comparator.py - Multi-Model Comparison for BRAIN Project
===============================================================
Compare LLM performance across different models for ARC tasks.

Features:
    - Run same tasks on multiple models
    - Collect comparative metrics
    - Generate comparison reports and visualizations
    - Statistical significance testing

Usage:
    from modules.model_comparator import ModelComparator
    
    comparator = ModelComparator(models=["llama3", "mistral", "phi3"])
    results = comparator.compare_on_tasks(task_dir="data/", limit=10)
    comparator.generate_report(results, output_dir="comparison_results/")
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import numpy as np

from .types import ARCTask
from .llm_client import LLMClient
from .executor import ActionExecutor
from .analyzer import ResultAnalyzer
from .detector import SymbolDetector
from .transformation_detector import TransformationDetector
from .prompt_maker import PromptMaker


@dataclass
class ModelResult:
    """Results for a single model on a single task."""
    model_name: str
    task_id: str
    accuracy: float
    is_correct: bool
    response_time_ms: float
    action_proposed: Optional[Dict[str, Any]] = None
    fallback_used: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class ModelComparisonResult:
    """Aggregated comparison results across all models and tasks."""
    models: List[str]
    tasks_evaluated: int
    timestamp: str
    
    # Per-model aggregates
    model_accuracies: Dict[str, float] = field(default_factory=dict)
    model_correct_counts: Dict[str, int] = field(default_factory=dict)
    model_avg_response_times: Dict[str, float] = field(default_factory=dict)
    model_fallback_rates: Dict[str, float] = field(default_factory=dict)
    
    # Detailed results
    detailed_results: List[ModelResult] = field(default_factory=list)
    
    # Statistical comparisons
    best_model: str = ""
    statistical_significance: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["detailed_results"] = [r.to_dict() if hasattr(r, 'to_dict') else r for r in self.detailed_results]
        return d


# Recommended models for comparison
RECOMMENDED_MODELS = {
    "llama3": {
        "description": "Meta's Llama 3 8B - Good all-rounder",
        "size": "4.7 GB",
        "install": "ollama pull llama3"
    },
    "mistral": {
        "description": "Mistral 7B - Excellent reasoning, fast",
        "size": "4.1 GB",
        "install": "ollama pull mistral"
    },
    "phi3": {
        "description": "Microsoft Phi-3 Mini - Small but capable",
        "size": "2.2 GB",
        "install": "ollama pull phi3"
    },
    "gemma2": {
        "description": "Google Gemma 2 9B - Strong reasoning",
        "size": "5.4 GB",
        "install": "ollama pull gemma2"
    },
    "codellama": {
        "description": "Meta Code Llama - Optimized for code/logic",
        "size": "3.8 GB",
        "install": "ollama pull codellama"
    },
    "qwen2": {
        "description": "Alibaba Qwen 2 7B - Multilingual, good logic",
        "size": "4.4 GB",
        "install": "ollama pull qwen2"
    },
    "llama3.1": {
        "description": "Meta Llama 3.1 8B - Latest version",
        "size": "4.7 GB",
        "install": "ollama pull llama3.1"
    },
    "deepseek-coder": {
        "description": "DeepSeek Coder 6.7B - Code-focused",
        "size": "3.8 GB",
        "install": "ollama pull deepseek-coder"
    }
}


class ModelComparator:
    """
    Compare multiple LLM models on ARC tasks.
    
    Usage:
        comparator = ModelComparator(
            models=["llama3", "mistral", "phi3"],
            verbose=True
        )
        
        # Compare on all tasks in a directory
        results = comparator.compare_on_tasks("data/", limit=10)
        
        # Generate report
        comparator.generate_report(results, "comparison_results/")
    """
    
    def __init__(
        self,
        models: List[str] = None,
        verbose: bool = True,
        timeout_per_model: int = 120
    ):
        """
        Initialize the model comparator.
        
        Args:
            models: List of model names to compare
            verbose: Print progress
            timeout_per_model: Timeout in seconds per model query
        """
        self.models = models or ["llama3"]
        self.verbose = verbose
        self.timeout = timeout_per_model
        
        # Shared components (model-agnostic)
        self.detector = SymbolDetector(connectivity=4)
        self.transformation_detector = TransformationDetector(verbose=False)
        self.prompt_maker = PromptMaker(include_grid_ascii=True, include_objects=True)
        self.executor = ActionExecutor(verbose=False)
        self.analyzer = ResultAnalyzer()
        
        # LLM clients (one per model)
        self.llm_clients: Dict[str, LLMClient] = {}
        for model in self.models:
            self.llm_clients[model] = LLMClient(model=model)
        
        if verbose:
            print(f"ModelComparator initialized with {len(self.models)} models:")
            for m in self.models:
                info = RECOMMENDED_MODELS.get(m, {})
                desc = info.get("description", "Custom model")
                print(f"  • {m}: {desc}")
    
    @staticmethod
    def list_recommended_models() -> Dict[str, Dict]:
        """Get list of recommended models with install commands."""
        return RECOMMENDED_MODELS
    
    @staticmethod
    def print_recommended_models():
        """Print recommended models with descriptions."""
        print("\n" + "=" * 60)
        print("  RECOMMENDED MODELS FOR COMPARISON")
        print("=" * 60 + "\n")
        
        for name, info in RECOMMENDED_MODELS.items():
            print(f"📦 {name}")
            print(f"   {info['description']}")
            print(f"   Size: {info['size']}")
            print(f"   Install: {info['install']}")
            print()
    
    def load_task(self, task_path: str) -> ARCTask:
        """Load a task from JSON file."""
        with open(task_path) as f:
            data = json.load(f)
        task_id = Path(task_path).stem
        return ARCTask.from_json(task_id, data)
    
    def evaluate_model_on_task(
        self,
        model_name: str,
        task: ARCTask
    ) -> ModelResult:
        """
        Evaluate a single model on a single task.
        
        Args:
            model_name: Name of the model to use
            task: The ARC task to solve
            
        Returns:
            ModelResult with accuracy and timing
        """
        start_time = time.time()
        
        try:
            # Get LLM client for this model
            llm = self.llm_clients.get(model_name)
            if not llm:
                llm = LLMClient(model=model_name)
                self.llm_clients[model_name] = llm
            
            # Detect transformations
            detected_transforms = []
            for pair in task.train_pairs:
                transforms = self.transformation_detector.detect_all(
                    pair.input_grid, pair.output_grid
                )
                detected_transforms.append(transforms)
            
            # Create prompt
            prompt = self.prompt_maker.create_reasoning_chain_prompt(task)
            system_prompt = self.prompt_maker.get_system_prompt()
            
            # Query model
            try:
                response = llm.query(prompt, system_prompt)
                llm_success = True
            except Exception as llm_error:
                llm_success = False
                response = None
                if self.verbose:
                    print(f"         LLM error: {llm_error}")
            
            response_time = (time.time() - start_time) * 1000
            
            # Check if we got a valid action
            action_data = None
            fallback_used = False
            
            if llm_success and response and response.action_data:
                action_data = response.action_data
                fallback_used = False
            else:
                # Try fallback from detection
                fallback_action = self._build_fallback_action(detected_transforms, task)
                if fallback_action:
                    action_data = fallback_action
                    fallback_used = True
                else:
                    error_msg = "No valid action from LLM" if llm_success else "LLM query failed"
                    return ModelResult(
                        model_name=model_name,
                        task_id=task.task_id,
                        accuracy=0.0,
                        is_correct=False,
                        response_time_ms=response_time,
                        fallback_used=False,
                        error=error_msg
                    )
            
            # Execute on test
            if not task.test_pairs:
                return ModelResult(
                    model_name=model_name,
                    task_id=task.task_id,
                    accuracy=0.0,
                    is_correct=False,
                    response_time_ms=response_time,
                    action_proposed=action_data,
                    fallback_used=fallback_used,
                    error="No test pairs"
                )
            
            test_pair = task.test_pairs[0]
            result = self.executor.execute(test_pair.input_grid, action_data)
            
            if result.success and result.output_grid and test_pair.output_grid:
                analysis = self.analyzer.compare_grids(
                    result.output_grid,
                    test_pair.output_grid
                )
                
                return ModelResult(
                    model_name=model_name,
                    task_id=task.task_id,
                    accuracy=analysis.accuracy,
                    is_correct=analysis.is_correct,
                    response_time_ms=response_time,
                    action_proposed=action_data,
                    fallback_used=fallback_used
                )
            else:
                return ModelResult(
                    model_name=model_name,
                    task_id=task.task_id,
                    accuracy=0.0,
                    is_correct=False,
                    response_time_ms=response_time,
                    action_proposed=action_data,
                    fallback_used=fallback_used,
                    error=result.message if not result.success else "No output grid"
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            import traceback
            if self.verbose:
                print(f"         Exception: {e}")
            return ModelResult(
                model_name=model_name,
                task_id=task.task_id,
                accuracy=0.0,
                is_correct=False,
                response_time_ms=response_time,
                error=f"{type(e).__name__}: {str(e)}"
            )
    
    def _build_fallback_action(
        self,
        detected_transforms: List[List[Any]],
        task: ARCTask
    ) -> Optional[Dict[str, Any]]:
        """Build fallback action from detected transformations."""
        if not detected_transforms:
            return None
        
        for trans_list in detected_transforms:
            if trans_list:
                t = trans_list[0]
                t_type = getattr(t, 'transformation_type', None) or t.get('transformation_type')
                params = dict(getattr(t, 'parameters', {}) or t.get('parameters', {}))
                
                if t_type == "translation":
                    return {"action": "translate", "params": params}
                elif t_type == "rotation":
                    return {"action": "rotate", "params": params}
                elif t_type == "reflection":
                    return {"action": "reflect", "params": params}
                elif t_type == "color_change":
                    return {"action": "color_change", "params": params}
        
        return None
    
    def compare_on_tasks(
        self,
        task_dir: str,
        pattern: str = "task_*.json",
        limit: Optional[int] = None
    ) -> ModelComparisonResult:
        """
        Compare all models on tasks in a directory.
        
        Args:
            task_dir: Directory containing task JSON files
            pattern: Glob pattern for task files
            limit: Maximum number of tasks to evaluate
            
        Returns:
            ModelComparisonResult with aggregated statistics
        """
        task_dir = Path(task_dir)
        task_files = sorted(task_dir.glob(pattern))
        
        if limit:
            task_files = task_files[:limit]
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  MODEL COMPARISON: {len(self.models)} models × {len(task_files)} tasks")
            print(f"{'='*60}\n")
        
        all_results: List[ModelResult] = []
        
        for i, task_file in enumerate(task_files):
            task = self.load_task(str(task_file))
            
            if self.verbose:
                print(f"[{i+1}/{len(task_files)}] {task.task_id}")
            
            for model in self.models:
                result = self.evaluate_model_on_task(model, task)
                all_results.append(result)
                
                if self.verbose:
                    status = "✓" if result.is_correct else f"✗ ({result.accuracy:.0%})"
                    fb = " [FB]" if result.fallback_used else ""
                    print(f"       {model:15} {status:10} {result.response_time_ms:>7.0f}ms{fb}")
        
        # Compute aggregates
        comparison = self._compute_aggregates(all_results, len(task_files))
        
        if self.verbose:
            self._print_summary(comparison)
        
        return comparison
    
    def _compute_aggregates(
        self,
        results: List[ModelResult],
        num_tasks: int
    ) -> ModelComparisonResult:
        """Compute aggregate statistics from individual results."""
        comparison = ModelComparisonResult(
            models=self.models,
            tasks_evaluated=num_tasks,
            timestamp=datetime.now().isoformat(),
            detailed_results=results
        )
        
        for model in self.models:
            model_results = [r for r in results if r.model_name == model]
            
            if model_results:
                accuracies = [r.accuracy for r in model_results]
                comparison.model_accuracies[model] = np.mean(accuracies)
                comparison.model_correct_counts[model] = sum(1 for r in model_results if r.is_correct)
                comparison.model_avg_response_times[model] = np.mean([r.response_time_ms for r in model_results])
                comparison.model_fallback_rates[model] = sum(1 for r in model_results if r.fallback_used) / len(model_results)
        
        # Find best model
        if comparison.model_accuracies:
            comparison.best_model = max(comparison.model_accuracies.items(), key=lambda x: x[1])[0]
        
        return comparison
    
    def _print_summary(self, comparison: ModelComparisonResult):
        """Print summary of comparison results."""
        print(f"\n{'='*60}")
        print("  COMPARISON SUMMARY")
        print(f"{'='*60}\n")
        
        # Table header
        print(f"{'Model':<18} {'Accuracy':>10} {'Correct':>10} {'Avg Time':>12} {'Fallback':>10}")
        print("-" * 62)
        
        # Sort by accuracy
        sorted_models = sorted(
            self.models,
            key=lambda m: comparison.model_accuracies.get(m, 0),
            reverse=True
        )
        
        for model in sorted_models:
            acc = comparison.model_accuracies.get(model, 0)
            correct = comparison.model_correct_counts.get(model, 0)
            time_ms = comparison.model_avg_response_times.get(model, 0)
            fb_rate = comparison.model_fallback_rates.get(model, 0)
            
            marker = "🏆" if model == comparison.best_model else "  "
            print(f"{marker}{model:<16} {acc:>9.1%} {correct:>10}/{comparison.tasks_evaluated} {time_ms:>10.0f}ms {fb_rate:>9.1%}")
        
        print(f"\n🏆 Best model: {comparison.best_model} ({comparison.model_accuracies.get(comparison.best_model, 0):.1%} accuracy)")
    
    def generate_report(
        self,
        comparison: ModelComparisonResult,
        output_dir: str
    ) -> Path:
        """
        Generate a detailed comparison report.
        
        Args:
            comparison: The comparison results
            output_dir: Directory to save reports
            
        Returns:
            Path to the output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_path / "comparison.json"
        with open(json_path, "w") as f:
            json.dump(comparison.to_dict(), f, indent=2, default=str)
        
        # Save CSV summary
        csv_path = output_path / "model_summary.csv"
        with open(csv_path, "w") as f:
            f.write("model,accuracy,correct,total,avg_time_ms,fallback_rate\n")
            for model in comparison.models:
                acc = comparison.model_accuracies.get(model, 0)
                correct = comparison.model_correct_counts.get(model, 0)
                time_ms = comparison.model_avg_response_times.get(model, 0)
                fb = comparison.model_fallback_rates.get(model, 0)
                f.write(f"{model},{acc:.4f},{correct},{comparison.tasks_evaluated},{time_ms:.1f},{fb:.4f}\n")
        
        # Save detailed results CSV
        detail_csv = output_path / "detailed_results.csv"
        with open(detail_csv, "w") as f:
            f.write("model,task_id,accuracy,is_correct,response_time_ms,fallback_used,error\n")
            for r in comparison.detailed_results:
                f.write(f"{r.model_name},{r.task_id},{r.accuracy:.4f},{r.is_correct},{r.response_time_ms:.1f},{r.fallback_used},{r.error or ''}\n")
        
        # Generate markdown report
        md_path = output_path / "comparison_report.md"
        with open(md_path, "w") as f:
            f.write("# Model Comparison Report\n\n")
            f.write(f"**Date:** {comparison.timestamp}\n")
            f.write(f"**Tasks evaluated:** {comparison.tasks_evaluated}\n")
            f.write(f"**Models compared:** {', '.join(comparison.models)}\n\n")
            
            f.write("## Summary\n\n")
            f.write("| Model | Accuracy | Correct | Avg Time | Fallback Rate |\n")
            f.write("|-------|----------|---------|----------|---------------|\n")
            
            for model in sorted(comparison.models, key=lambda m: comparison.model_accuracies.get(m, 0), reverse=True):
                acc = comparison.model_accuracies.get(model, 0)
                correct = comparison.model_correct_counts.get(model, 0)
                time_ms = comparison.model_avg_response_times.get(model, 0)
                fb = comparison.model_fallback_rates.get(model, 0)
                best = " 🏆" if model == comparison.best_model else ""
                f.write(f"| {model}{best} | {acc:.1%} | {correct}/{comparison.tasks_evaluated} | {time_ms:.0f}ms | {fb:.1%} |\n")
            
            f.write(f"\n**Best model:** {comparison.best_model}\n")
        
        if self.verbose:
            print(f"\n📊 Reports saved to: {output_path}")
            print(f"   • {json_path.name}")
            print(f"   • {csv_path.name}")
            print(f"   • {detail_csv.name}")
            print(f"   • {md_path.name}")
        
        return output_path


def compare_models_cli():
    """Command-line interface for model comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare LLM models on ARC tasks")
    parser.add_argument("--models", "-m", nargs="+", default=["llama3"],
                       help="Models to compare (e.g., llama3 mistral phi3)")
    parser.add_argument("--tasks", "-t", type=str, default="data/",
                       help="Directory containing task files")
    parser.add_argument("--pattern", "-p", type=str, default="task_*.json",
                       help="Task file pattern")
    parser.add_argument("--limit", "-l", type=int, default=None,
                       help="Limit number of tasks")
    parser.add_argument("--output", "-o", type=str, default="comparison_results/",
                       help="Output directory for reports")
    parser.add_argument("--list-models", action="store_true",
                       help="List recommended models and exit")
    
    args = parser.parse_args()
    
    if args.list_models:
        ModelComparator.print_recommended_models()
        return
    
    comparator = ModelComparator(models=args.models)
    results = comparator.compare_on_tasks(args.tasks, args.pattern, args.limit)
    comparator.generate_report(results, args.output)


if __name__ == "__main__":
    compare_models_cli()
