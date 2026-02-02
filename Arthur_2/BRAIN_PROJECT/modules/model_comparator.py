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
        
        ALIGNED WITH main.py solve_task() for consistent results:
        - Detects objects before prompting
        - Uses direct fallback for composite/add_border transformations
        - Same fallback logic as main.py
        
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
            
            # === STEP 1a: DETECT OBJECTS (same as main.py) ===
            for pair in task.train_pairs:
                self.detector.detect(pair.input_grid)
                self.detector.detect(pair.output_grid)
            for pair in task.test_pairs:
                self.detector.detect(pair.input_grid)
            
            # === STEP 1b: DETECT TRANSFORMATIONS ===
            detected_transforms = []
            for pair in task.train_pairs:
                transforms = self.transformation_detector.detect_all(
                    pair.input_grid, pair.output_grid
                )
                detected_transforms.append(transforms)
            
            # === CHECK FOR DIRECT FALLBACK (composite, add_border) ===
            # Same logic as main.py - bypass LLM for these transformations
            use_direct_fallback = False
            direct_fallback_action = None
            
            for trans_list in detected_transforms:
                for t in trans_list:
                    t_type = t.transformation_type if hasattr(t, 'transformation_type') else ''
                    confidence = t.confidence if hasattr(t, 'confidence') else 0
                    
                    if t_type in ("composite", "add_border") and confidence >= 0.95:
                        use_direct_fallback = True
                        direct_fallback_action = self._build_fallback_action(detected_transforms, task)
                        break
                if use_direct_fallback:
                    break
            
            # === STEP 2-3: PROMPTING + LLM (skip if direct fallback) ===
            action_data = None
            fallback_used = False
            llm_success = False
            response = None
            
            if use_direct_fallback and direct_fallback_action:
                # Use direct fallback - same as main.py
                action_data = dict(direct_fallback_action)
                # Adjust color_filter for test input
                if task.test_pairs:
                    test_colors = [c for c in task.test_pairs[0].input_grid.unique_colors if c != 0]
                    if test_colors:
                        action_data["color_filter"] = test_colors[0]
                fallback_used = True
            else:
                # Normal LLM path
                prompt = self.prompt_maker.create_reasoning_chain_prompt(task)
                system_prompt = self.prompt_maker.get_system_prompt()
                
                try:
                    response = llm.query(prompt, system_prompt)
                    llm_success = True
                except Exception as llm_error:
                    llm_success = False
                    response = None
                    if self.verbose:
                        print(f"         LLM error: {llm_error}")
                
                # Check if we got a valid action from LLM
                if llm_success and response and response.action_data:
                    action_data = response.action_data
                    fallback_used = False
                else:
                    # Try fallback from detection (same as main.py)
                    fallback_action = self._build_fallback_action(detected_transforms, task)
                    if fallback_action:
                        action_data = fallback_action
                        fallback_used = True
            
            response_time = (time.time() - start_time) * 1000
            
            # No action available
            if not action_data:
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
            
            # === STEP 4: EXECUTE ===
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
        """
        Build fallback action from detected transformations.
        
        ALIGNED WITH main.py _build_fallback_action() for consistent results.
        Supports all transformation types.
        """
        if not detected_transforms:
            return None
        
        # Get main color from test input (for color_filter)
        color_filter = None
        if task.test_pairs:
            test_colors = [c for c in task.test_pairs[0].input_grid.unique_colors if c != 0]
            if test_colors:
                color_filter = test_colors[0]
        
        for trans_list in detected_transforms:
            if trans_list:
                for t in trans_list:
                    t_type = getattr(t, 'transformation_type', None)
                    if t_type is None and isinstance(t, dict):
                        t_type = t.get('transformation_type')
                    
                    params = {}
                    if hasattr(t, 'parameters'):
                        params = dict(t.parameters) if t.parameters else {}
                    elif isinstance(t, dict):
                        params = dict(t.get('parameters', {}))
                    
                    if not t_type:
                        continue
                    
                    # Build action based on transformation type
                    if t_type == "translation":
                        return {
                            "action": "translate",
                            "params": {
                                "dx": params.get("dx", 0),
                                "dy": params.get("dy", 0)
                            }
                        }
                    
                    elif t_type == "rotation":
                        angle = params.get("angle", 90)
                        is_grid_level = params.get("grid_level", False)
                        
                        if is_grid_level:
                            return {
                                "action": "rotate",
                                "params": {"angle": angle, "grid_level": True}
                            }
                        else:
                            action = {"action": "rotate", "params": {"angle": angle}}
                            if color_filter:
                                action["color_filter"] = color_filter
                            return action
                    
                    elif t_type == "reflection":
                        axis = params.get("axis", "horizontal")
                        is_grid_level = params.get("grid_level", False)
                        
                        if is_grid_level or color_filter is None:
                            return {
                                "action": "reflect",
                                "params": {"axis": axis, "grid_level": True}
                            }
                        else:
                            return {
                                "action": "reflect",
                                "params": {"axis": axis},
                                "color_filter": color_filter
                            }
                    
                    elif t_type == "color_change":
                        return {
                            "action": "color_change",
                            "params": {
                                "from_color": params.get("from_color", 1),
                                "to_color": params.get("to_color", 2)
                            }
                        }
                    
                    elif t_type == "draw_line":
                        # Auto-detect color from test input
                        draw_color = None
                        if task.test_pairs:
                            test_data = task.test_pairs[0].input_grid.data
                            for c in range(1, 10):
                                count = int(np.sum(test_data == c))
                                if count == 2:
                                    draw_color = c
                                    break
                        
                        if draw_color is None:
                            draw_color = params.get("color") or color_filter or 1
                        
                        return {
                            "action": "draw_line",
                            "color_filter": draw_color
                        }
                    
                    elif t_type == "tiling":
                        return {
                            "action": "tile",
                            "params": {
                                "repetitions_horizontal": params.get("repetitions_horizontal", 2),
                                "repetitions_vertical": params.get("repetitions_vertical", 2)
                            }
                        }
                    
                    elif t_type == "scaling":
                        return {
                            "action": "scale",
                            "params": {"factor": params.get("factor", 2)}
                        }
                    
                    elif t_type == "add_border":
                        return {
                            "action": "add_border",
                            "params": {
                                "border_color": params.get("border_color", 1),
                                "interior_color": params.get("interior_color")
                            },
                            "color_filter": color_filter
                        }
                    
                    elif t_type == "composite":
                        # Build composite action from sub-transformations
                        sub_transforms = params.get("transformations", [])
                        return {
                            "action": "composite",
                            "params": {"transformations": sub_transforms},
                            "color_filter": color_filter
                        }
        
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


class ModelComparisonVisualizer:
    """
    Visualize model comparison results.
    
    Generates publication-quality figures comparing LLM performance.
    """
    
    def __init__(self, comparison: ModelComparisonResult = None, results_path: str = None):
        """
        Initialize visualizer with comparison results.
        
        Args:
            comparison: ModelComparisonResult object
            results_path: Path to comparison.json file
        """
        if comparison:
            self.comparison = comparison
        elif results_path:
            self.comparison = self._load_results(results_path)
        else:
            raise ValueError("Provide either comparison or results_path")
        
        self._setup_style()
    
    def _load_results(self, path: str) -> ModelComparisonResult:
        """Load results from JSON file."""
        import json
        with open(path) as f:
            data = json.load(f)
        
        # Reconstruct ModelComparisonResult
        detailed = [ModelResult(**r) for r in data.get("detailed_results", [])]
        
        return ModelComparisonResult(
            models=data["models"],
            tasks_evaluated=data["tasks_evaluated"],
            timestamp=data["timestamp"],
            model_accuracies=data.get("model_accuracies", {}),
            model_correct_counts=data.get("model_correct_counts", {}),
            model_avg_response_times=data.get("model_avg_response_times", {}),
            model_fallback_rates=data.get("model_fallback_rates", {}),
            detailed_results=detailed,
            best_model=data.get("best_model", ""),
        )
    
    def _setup_style(self):
        """Setup matplotlib style for publication quality."""
        import matplotlib.pyplot as plt
        import shutil
        
        latex_available = shutil.which('latex') is not None
        
        plt.rcParams.update({
            "text.usetex": latex_available,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"] if latex_available else ["DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.figsize": (6, 4),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 1.5,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        })
        
        # Color palette (colorblind-friendly)
        self.colors = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#33BBEE', '#EE3377']
    
    def plot_accuracy_comparison(self, save_path: str = None, show: bool = True):
        """
        Bar plot comparing accuracy across models.
        
        Args:
            save_path: Path to save figure
            show: Whether to display the figure
        """
        import matplotlib.pyplot as plt
        
        models = self.comparison.models
        accuracies = [self.comparison.model_accuracies.get(m, 0) * 100 for m in models]
        
        fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
        
        bars = ax.bar(models, accuracies, color=self.colors[:len(models)], edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.annotate(f'{acc:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        # Highlight best model
        best_idx = models.index(self.comparison.best_model) if self.comparison.best_model in models else 0
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(2)
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Model')
        ax.set_title(f'Model Accuracy Comparison (n={self.comparison.tasks_evaluated} tasks)')
        ax.set_ylim(0, 105)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_time_comparison(self, save_path: str = None, show: bool = True):
        """
        Bar plot comparing response times across models.
        """
        import matplotlib.pyplot as plt
        
        models = self.comparison.models
        times = [self.comparison.model_avg_response_times.get(m, 0) / 1000 for m in models]  # Convert to seconds
        
        fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
        
        bars = ax.bar(models, times, color=self.colors[:len(models)], edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, t in zip(bars, times):
            height = bar.get_height()
            ax.annotate(f'{t:.1f}s',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Average Response Time (s)')
        ax.set_xlabel('Model')
        ax.set_title('Model Response Time Comparison')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_accuracy_vs_time(self, save_path: str = None, show: bool = True):
        """
        Scatter plot: accuracy vs response time for each model.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
        
        for i, model in enumerate(self.comparison.models):
            acc = self.comparison.model_accuracies.get(model, 0) * 100
            time_s = self.comparison.model_avg_response_times.get(model, 0) / 1000
            
            marker = '*' if model == self.comparison.best_model else 'o'
            size = 200 if model == self.comparison.best_model else 100
            
            ax.scatter(time_s, acc, c=self.colors[i], s=size, marker=marker, 
                      label=model, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Average Response Time (s)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy vs Response Time Trade-off')
        ax.legend(loc='best', framealpha=0.9)
        ax.set_ylim(0, 105)
        
        # Add quadrant annotations
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_accuracy_boxplot(self, save_path: str = None, show: bool = True):
        """
        Box plot showing accuracy distribution per model.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
        
        # Group accuracies by model
        data = []
        labels = []
        for model in self.comparison.models:
            model_results = [r.accuracy * 100 for r in self.comparison.detailed_results 
                           if r.model_name == model]
            if model_results:
                data.append(model_results)
                labels.append(model)
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Color the boxes
        for i, (box, median) in enumerate(zip(bp['boxes'], bp['medians'])):
            box.set_facecolor(self.colors[i % len(self.colors)])
            box.set_alpha(0.7)
            median.set_color('black')
            median.set_linewidth(2)
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Model')
        ax.set_title('Accuracy Distribution by Model')
        ax.set_ylim(-5, 105)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_per_task_comparison(self, save_path: str = None, show: bool = True):
        """
        Grouped bar chart showing accuracy per task for each model.
        """
        import matplotlib.pyplot as plt
        
        # Get unique tasks
        tasks = list(set(r.task_id for r in self.comparison.detailed_results))
        tasks.sort()
        
        # Limit to 15 tasks max for readability
        if len(tasks) > 15:
            tasks = tasks[:15]
        
        fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
        
        x = np.arange(len(tasks))
        width = 0.8 / len(self.comparison.models)
        
        for i, model in enumerate(self.comparison.models):
            accuracies = []
            for task in tasks:
                result = next((r for r in self.comparison.detailed_results 
                              if r.task_id == task and r.model_name == model), None)
                accuracies.append(result.accuracy * 100 if result else 0)
            
            offset = (i - len(self.comparison.models) / 2 + 0.5) * width
            ax.bar(x + offset, accuracies, width, label=model, 
                  color=self.colors[i % len(self.colors)], edgecolor='black', linewidth=0.3)
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Task')
        ax.set_title('Per-Task Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('task_', '') for t in tasks], rotation=45, ha='right')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_ylim(0, 105)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_fallback_comparison(self, save_path: str = None, show: bool = True):
        """
        Bar chart comparing fallback usage rates.
        """
        import matplotlib.pyplot as plt
        
        models = self.comparison.models
        fallback_rates = [self.comparison.model_fallback_rates.get(m, 0) * 100 for m in models]
        
        fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
        
        bars = ax.bar(models, fallback_rates, color=self.colors[:len(models)], 
                     edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, rate in zip(bars, fallback_rates):
            height = bar.get_height()
            ax.annotate(f'{rate:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Fallback Usage Rate (%)')
        ax.set_xlabel('Model')
        ax.set_title('LLM Fallback Rate (lower is better)')
        ax.set_ylim(0, max(fallback_rates) * 1.2 + 5 if fallback_rates else 10)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_summary_dashboard(self, save_path: str = None, show: bool = True):
        """
        Create a 2x2 dashboard with all key metrics.
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
        
        models = self.comparison.models
        
        # 1. Accuracy comparison
        ax = axes[0, 0]
        accuracies = [self.comparison.model_accuracies.get(m, 0) * 100 for m in models]
        bars = ax.bar(models, accuracies, color=self.colors[:len(models)], edgecolor='black')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Overall Accuracy')
        ax.set_ylim(0, 105)
        for bar, acc in zip(bars, accuracies):
            ax.annotate(f'{acc:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
        
        # 2. Response time
        ax = axes[0, 1]
        times = [self.comparison.model_avg_response_times.get(m, 0) / 1000 for m in models]
        bars = ax.bar(models, times, color=self.colors[:len(models)], edgecolor='black')
        ax.set_ylabel('Avg Response Time (s)')
        ax.set_title('Response Time')
        for bar, t in zip(bars, times):
            ax.annotate(f'{t:.1f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
        
        # 3. Accuracy vs Time scatter
        ax = axes[1, 0]
        for i, model in enumerate(models):
            acc = self.comparison.model_accuracies.get(model, 0) * 100
            time_s = self.comparison.model_avg_response_times.get(model, 0) / 1000
            marker = '*' if model == self.comparison.best_model else 'o'
            ax.scatter(time_s, acc, c=self.colors[i], s=150, marker=marker, 
                      label=model, edgecolors='black')
        ax.set_xlabel('Response Time (s)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy vs Time Trade-off')
        ax.legend(loc='best', fontsize=7)
        ax.set_ylim(0, 105)
        
        # 4. Correct count
        ax = axes[1, 1]
        correct = [self.comparison.model_correct_counts.get(m, 0) for m in models]
        total = self.comparison.tasks_evaluated
        bars = ax.bar(models, correct, color=self.colors[:len(models)], edgecolor='black')
        ax.axhline(y=total, color='red', linestyle='--', label=f'Total tasks ({total})')
        ax.set_ylabel('Tasks Solved')
        ax.set_title('Tasks Correctly Solved')
        ax.legend(loc='best', fontsize=7)
        for bar, c in zip(bars, correct):
            ax.annotate(f'{c}/{total}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
        
        fig.suptitle(f'Model Comparison Dashboard\n{self.comparison.timestamp[:10]}', fontsize=12, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def save_all_plots(self, output_dir: str, formats: List[str] = None):
        """
        Generate and save all comparison plots.
        
        Args:
            output_dir: Directory to save figures
            formats: List of formats (default: ['png', 'pdf'])
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        formats = formats or ['png', 'pdf']
        
        print(f"\n📊 Generating comparison visualizations...")
        
        plots = [
            ('accuracy_comparison', self.plot_accuracy_comparison),
            ('time_comparison', self.plot_time_comparison),
            ('accuracy_vs_time', self.plot_accuracy_vs_time),
            ('accuracy_boxplot', self.plot_accuracy_boxplot),
            ('per_task_comparison', self.plot_per_task_comparison),
            ('fallback_comparison', self.plot_fallback_comparison),
            ('summary_dashboard', self.plot_summary_dashboard),
        ]
        
        for name, plot_func in plots:
            for fmt in formats:
                save_path = output_path / f"{name}.{fmt}"
                try:
                    plot_func(save_path=str(save_path), show=False)
                except Exception as e:
                    print(f"  ⚠ Warning: Could not generate {name}: {e}")
        
        print(f"\n✅ All plots saved to: {output_path}")
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
    parser.add_argument("--visualize", "-v", action="store_true",
                       help="Generate visualization plots")
    
    args = parser.parse_args()
    
    if args.list_models:
        ModelComparator.print_recommended_models()
        return
    
    comparator = ModelComparator(models=args.models)
    results = comparator.compare_on_tasks(args.tasks, args.pattern, args.limit)
    comparator.generate_report(results, args.output)
    
    # Generate visualizations if requested
    if args.visualize:
        viz = ModelComparisonVisualizer(comparison=results)
        viz.save_all_plots(f"{args.output}/figures")


if __name__ == "__main__":
    compare_models_cli()
