#!/usr/bin/env python3
"""
compare_models.py - Multi-Model Comparison Tool
================================================
Compare multiple LLM models on ARC tasks.

Usage:
    # List recommended models
    python compare_models.py --list-models
    
    # Compare 2 models on 5 tasks
    python compare_models.py --models llama3 mistral --limit 5
    
    # Full comparison on all tasks
    python compare_models.py --models llama3 mistral phi3 --output results/comparison/
"""

import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.model_comparator import (
    ModelComparator, 
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
    print("Then run:")
    print("  python compare_models.py --models llama3 mistral phi3 --limit 10")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM models on ARC-AGI tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_models.py --list-models              # Show available models
  python compare_models.py -m llama3 mistral          # Compare 2 models
  python compare_models.py -m llama3 mistral -l 5     # Limit to 5 tasks
  python compare_models.py -m llama3 mistral phi3 -o comparison_results/
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
    
    args = parser.parse_args()
    
    # Just list models
    if args.list_models:
        print_install_instructions()
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
    
    print(f"\n🧠 BRAIN Model Comparison Tool")
    print(f"{'='*50}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Tasks: {len(task_files)} available" + (f" (limited to {args.limit})" if args.limit else ""))
    print(f"Output: {args.output}")
    print()
    
    # Verify models are available (optional check)
    print("Verifying models...")
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        installed_models = result.stdout.lower()
        
        missing = []
        for model in args.models:
            base_model = model.split(":")[0]  # Handle llama3:latest
            if base_model not in installed_models:
                missing.append(model)
        
        if missing:
            print(f"\n⚠️  Warning: These models may not be installed: {', '.join(missing)}")
            print("   Install with: ollama pull <model_name>")
            response = input("\nContinue anyway? [y/N] ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)
    except Exception:
        pass  # Ollama check failed, continue anyway
    
    # Run comparison
    comparator = ModelComparator(
        models=args.models,
        verbose=not args.quiet
    )
    
    results = comparator.compare_on_tasks(
        task_dir=args.tasks,
        pattern=args.pattern,
        limit=args.limit
    )
    
    # Generate reports
    output_dir = comparator.generate_report(results, args.output)
    
    print(f"\n✅ Comparison complete!")
    print(f"   Best model: {results.best_model} ({results.model_accuracies.get(results.best_model, 0):.1%})")
    print(f"   Reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
