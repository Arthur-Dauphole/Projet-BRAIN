#!/usr/bin/env python3
"""
compare_models.py - Multi-Model Comparison Tool
================================================
Compare multiple LLM models on ARC tasks by running main.py batch for each model.

This approach ensures 100% consistency with main.py results by using it directly
instead of reimplementing the pipeline.

Usage:
    # List recommended models
    python compare_models.py --list-models
    
    # Compare 2 models on 5 tasks
    python compare_models.py --models llama3 mistral --limit 5
    
    # Full comparison with visualizations
    python compare_models.py --models llama3 mistral phi3 --visualize
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import shutil

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.batch_runner import BatchRunner, BatchResult, run_batch_evaluation
from modules.model_comparator import (
    ModelComparisonResult,
    ModelResult,
    ModelComparisonVisualizer,
    RECOMMENDED_MODELS,
)


def print_install_instructions():
    """Print instructions for installing models."""
    print("\n" + "=" * 70)
    print("  HOW TO INSTALL NEW MODELS FOR COMPARISON")
    print("=" * 70)
    print("""
Ollama makes it easy to download and run different models.
To install a model, use: ollama pull <model_name>

Recommended models for ARC-AGI comparison:
""")
    
    for name, info in RECOMMENDED_MODELS.items():
        print(f"  📦 {name}")
        print(f"     {info['description']}")
        print(f"     Size: {info['size']}")
        print(f"     Install: {info['install']}")
        print()
    
    print("Quick start (install 3 models for comparison):")
    print("  ollama pull llama3")
    print("  ollama pull mistral")
    print("  ollama pull phi3")
    print()


def run_batch_for_model(
    model: str,
    task_dir: str,
    pattern: str,
    limit: Optional[int],
    output_base: str,
    verbose: bool = True
) -> Optional[BatchResult]:
    """
    Run batch evaluation for a single model using BatchRunner.
    
    This uses the exact same code path as `main.py --batch`.
    
    Args:
        model: Model name
        task_dir: Directory with task files
        pattern: Task file pattern
        limit: Max tasks
        output_base: Base output directory
        verbose: Print progress
        
    Returns:
        BatchResult or None if failed
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Running batch for model: {model}")
        print(f"{'='*60}")
    
    try:
        # Create BatchRunner with this model
        runner = BatchRunner(
            model=model,
            verbose=verbose,
            visualize=False,  # No popups during batch
            multi_mode=False
        )
        
        # Run batch (correct signature: directory, pattern, limit)
        result = runner.run_batch(
            directory=task_dir,
            pattern=pattern,
            limit=limit
        )
        
        # Save results to model-specific folder
        runner.save_results(result, f"{output_base}/{model}")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error running batch for {model}: {e}")
        import traceback
        traceback.print_exc()
        return None


def aggregate_results(
    model_results: Dict[str, BatchResult],
    models: List[str]
) -> ModelComparisonResult:
    """
    Aggregate batch results from multiple models into a comparison result.
    
    Args:
        model_results: Dict mapping model name to BatchResult
        models: List of model names
        
    Returns:
        ModelComparisonResult for visualization
    """
    detailed_results: List[ModelResult] = []
    model_accuracies = {}
    model_correct_counts = {}
    model_avg_response_times = {}
    model_fallback_rates = {}
    
    tasks_evaluated = 0
    
    for model in models:
        batch_result = model_results.get(model)
        if not batch_result:
            continue
        
        tasks_evaluated = max(tasks_evaluated, batch_result.total_tasks)
        
        # Aggregate metrics (use correct BatchResult attributes)
        model_accuracies[model] = batch_result.overall_accuracy
        model_correct_counts[model] = batch_result.correct_tasks
        model_avg_response_times[model] = batch_result.avg_time_per_task * 1000  # Convert to ms
        model_fallback_rates[model] = batch_result.fallback_usage_rate
        
        # Convert to ModelResult for detailed analysis
        for task_result in batch_result.task_results:
            detailed_results.append(ModelResult(
                model_name=model,
                task_id=task_result.task_id,
                accuracy=task_result.accuracy,
                is_correct=task_result.is_correct,
                response_time_ms=task_result.execution_time * 1000,
                action_proposed={"action": task_result.action_used} if task_result.action_used else None,
                fallback_used=task_result.was_fallback_used,
                error=task_result.error_message
            ))
    
    # Find best model
    best_model = max(model_accuracies.items(), key=lambda x: x[1])[0] if model_accuracies else ""
    
    return ModelComparisonResult(
        models=models,
        tasks_evaluated=tasks_evaluated,
        timestamp=datetime.now().isoformat(),
        model_accuracies=model_accuracies,
        model_correct_counts=model_correct_counts,
        model_avg_response_times=model_avg_response_times,
        model_fallback_rates=model_fallback_rates,
        detailed_results=detailed_results,
        best_model=best_model
    )


def print_comparison_summary(comparison: ModelComparisonResult):
    """Print a summary of comparison results."""
    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}\n")
    
    # Table header
    print(f"{'Model':<18} {'Accuracy':>10} {'Correct':>10} {'Avg Time':>12} {'Fallback':>10}")
    print("-" * 62)
    
    # Sort by accuracy
    sorted_models = sorted(
        comparison.models,
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


def save_comparison_report(
    comparison: ModelComparisonResult,
    model_results: Dict[str, BatchResult],
    output_dir: Path
):
    """Save comparison reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / "comparison.json"
    with open(json_path, "w") as f:
        json.dump(comparison.to_dict(), f, indent=2, default=str)
    
    # Save CSV summary
    csv_path = output_dir / "model_summary.csv"
    with open(csv_path, "w") as f:
        f.write("model,accuracy,correct,total,avg_time_ms,fallback_rate\n")
        for model in comparison.models:
            acc = comparison.model_accuracies.get(model, 0)
            correct = comparison.model_correct_counts.get(model, 0)
            time_ms = comparison.model_avg_response_times.get(model, 0)
            fb = comparison.model_fallback_rates.get(model, 0)
            f.write(f"{model},{acc:.4f},{correct},{comparison.tasks_evaluated},{time_ms:.1f},{fb:.4f}\n")
    
    # Save markdown report
    md_path = output_dir / "comparison_report.md"
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
    
    print(f"\n📊 Reports saved to: {output_dir}")
    print(f"   • {json_path.name}")
    print(f"   • {csv_path.name}")
    print(f"   • {md_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM models on ARC-AGI tasks (uses main.py batch internally)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_models.py --list-models              # Show available models
  python compare_models.py -m llama3 mistral          # Compare 2 models
  python compare_models.py -m llama3 mistral -l 5     # Limit to 5 tasks
  python compare_models.py -m llama3 mistral -v       # With visualizations
        """
    )
    
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=["llama3"],
        help="Models to compare (e.g., llama3 mistral phi3)"
    )
    
    parser.add_argument(
        "--tasks", "-t",
        type=str,
        default="data/",
        help="Directory containing task files (default: data/)"
    )
    
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default="task_*.json",
        help="Task file pattern (default: task_*.json)"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of tasks to evaluate"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="comparison_results/",
        help="Output directory for reports (default: comparison_results/)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List recommended models and installation instructions"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Less verbose output"
    )
    
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Generate visualization plots after comparison"
    )
    
    parser.add_argument(
        "--viz-only",
        type=str,
        metavar="PATH",
        help="Only generate visualizations from existing comparison.json"
    )
    
    args = parser.parse_args()
    
    # Just list models
    if args.list_models:
        print_install_instructions()
        return
    
    # Visualization only mode
    if args.viz_only:
        viz_path = Path(args.viz_only)
        if viz_path.is_dir():
            json_path = viz_path / "comparison.json"
        else:
            json_path = viz_path
        
        if not json_path.exists():
            print(f"❌ Error: comparison.json not found at {json_path}")
            sys.exit(1)
        
        print(f"\n📊 Generating visualizations from: {json_path}")
        viz = ModelComparisonVisualizer(results_path=str(json_path))
        output_dir = json_path.parent / "figures"
        viz.save_all_plots(str(output_dir))
        print(f"\n✅ Visualizations saved to: {output_dir}")
        return
    
    # Check if tasks directory exists
    task_dir = Path(args.tasks)
    if not task_dir.exists():
        print(f"❌ Error: Task directory not found: {args.tasks}")
        sys.exit(1)
    
    # Count available tasks
    task_files = list(task_dir.glob(args.pattern))
    if not task_files:
        print(f"❌ Error: No tasks found matching '{args.pattern}' in {args.tasks}")
        sys.exit(1)
    
    num_tasks = min(len(task_files), args.limit) if args.limit else len(task_files)
    
    print(f"\n🧠 BRAIN Model Comparison Tool (via main.py batch)")
    print(f"{'='*60}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Tasks: {num_tasks} to evaluate")
    print(f"Output: {args.output}")
    print()
    
    # Run batch for each model
    output_base = Path(args.output)
    model_results: Dict[str, BatchResult] = {}
    
    for model in args.models:
        result = run_batch_for_model(
            model=model,
            task_dir=str(task_dir),
            pattern=args.pattern,
            limit=args.limit,
            output_base=str(output_base),
            verbose=not args.quiet
        )
        
        if result:
            model_results[model] = result
    
    # Aggregate results
    comparison = aggregate_results(model_results, args.models)
    
    # Print summary
    print_comparison_summary(comparison)
    
    # Save reports
    save_comparison_report(comparison, model_results, output_base)
    
    # Generate visualizations if requested
    if args.visualize:
        viz = ModelComparisonVisualizer(comparison=comparison)
        fig_dir = output_base / "figures"
        viz.save_all_plots(str(fig_dir))
    
    print(f"\n✅ Comparison complete!")
    print(f"   Best model: {comparison.best_model} ({comparison.model_accuracies.get(comparison.best_model, 0):.1%})")
    print(f"   Reports saved to: {output_base}")
    
    if args.visualize:
        print(f"   Visualizations: {output_base}/figures/")


if __name__ == "__main__":
    main()
