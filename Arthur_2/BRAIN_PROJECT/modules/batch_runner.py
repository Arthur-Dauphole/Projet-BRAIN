"""
batch_runner.py - Batch Task Evaluation Module
===============================================
Run multiple ARC tasks and collect statistics.

Features:
    - Execute all task_*.json files in a directory
    - Collect accuracy, timing, and transformation statistics
    - Generate JSON/CSV reports in timestamped folders
    - Support for filtering by task name pattern
    - Non-blocking execution (no visualization during batch)

Usage:
    from modules.batch_runner import BatchRunner
    
    runner = BatchRunner(model="llama3")
    results = runner.run_batch("data/", pattern="task_*.json")
    runner.print_summary(results)
    runner.save_results(results, "results/")  # Creates timestamped folder
"""

import json
import time
import glob
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class TaskResult:
    """Result of a single task evaluation."""
    task_id: str
    task_file: str
    success: bool
    is_correct: bool
    accuracy: float
    execution_time: float  # seconds
    detected_transformations: List[str] = field(default_factory=list)
    action_used: Optional[str] = None
    error_message: Optional[str] = None
    num_train_examples: int = 0
    num_test_examples: int = 0
    grid_size: Optional[str] = None  # e.g., "10x10"
    
    # === NEW: Enhanced data for analysis ===
    # Primary transformation info
    primary_transformation: Optional[str] = None  # Main transformation type detected
    transformation_confidence: float = 0.0  # Confidence level (0-1)
    transformation_params: Dict[str, Any] = field(default_factory=dict)  # Parameters used
    
    # LLM vs Fallback tracking
    was_fallback_used: bool = False  # True if fallback was used instead of LLM
    llm_proposed_action: Optional[str] = None  # What the LLM originally proposed
    fallback_reason: Optional[str] = None  # Why fallback was used
    
    # Timing breakdown
    llm_response_time: float = 0.0  # Time spent waiting for LLM
    detection_time: float = 0.0  # Time spent on transformation detection
    execution_action_time: float = 0.0  # Time spent executing action
    
    # Mode and complexity
    mode_used: str = "single"  # single, multi_transform
    num_colors_in_input: int = 0  # Complexity metric
    num_objects_in_input: int = 0  # Complexity metric
    
    # Grid data for visualization (not serialized to JSON)
    input_grid: Any = field(default=None, repr=False)
    predicted_grid: Any = field(default=None, repr=False)
    expected_grid: Any = field(default=None, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding grid data."""
        
        def convert_numpy(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        return {
            'task_id': self.task_id,
            'task_file': self.task_file,
            'success': self.success,
            'is_correct': self.is_correct,
            'accuracy': float(self.accuracy),
            'execution_time': float(self.execution_time),
            'detected_transformations': self.detected_transformations,
            'action_used': self.action_used,
            'error_message': self.error_message,
            'num_train_examples': int(self.num_train_examples),
            'num_test_examples': int(self.num_test_examples),
            'grid_size': self.grid_size,
            # Enhanced data - convert numpy types
            'primary_transformation': self.primary_transformation,
            'transformation_confidence': float(self.transformation_confidence),
            'transformation_params': convert_numpy(self.transformation_params),
            'was_fallback_used': bool(self.was_fallback_used),
            'llm_proposed_action': self.llm_proposed_action,
            'fallback_reason': self.fallback_reason,
            'timing': {
                'total': round(float(self.execution_time), 3),
                'llm_response': round(float(self.llm_response_time), 3),
                'detection': round(float(self.detection_time), 3),
                'action_execution': round(float(self.execution_action_time), 3),
            },
            'complexity': {
                'mode': self.mode_used,
                'num_colors': int(self.num_colors_in_input),
                'num_objects': int(self.num_objects_in_input),
            }
        }


@dataclass 
class BatchResult:
    """Aggregated results from a batch run."""
    total_tasks: int = 0
    successful_tasks: int = 0
    correct_tasks: int = 0
    failed_tasks: int = 0
    
    total_time: float = 0.0
    avg_time_per_task: float = 0.0
    
    overall_accuracy: float = 0.0
    accuracy_when_successful: float = 0.0
    
    transformation_counts: Dict[str, int] = field(default_factory=dict)
    action_counts: Dict[str, int] = field(default_factory=dict)
    
    task_results: List[TaskResult] = field(default_factory=list)
    
    # Metadata
    run_timestamp: str = ""
    run_folder: str = ""  # Timestamped folder name
    model_used: str = ""
    pattern_used: str = ""
    directory: str = ""
    
    # === NEW: Enhanced statistics for analysis ===
    program_version: str = "1.10.0"  # Track version
    
    # Per-transformation accuracy breakdown
    accuracy_by_transformation: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Format: {"rotation": {"accuracy": 0.85, "count": 10, "correct": 8}, ...}
    
    # LLM vs Fallback statistics
    llm_success_rate: float = 0.0  # % of times LLM gave correct action
    fallback_usage_rate: float = 0.0  # % of times fallback was used
    
    # Timing statistics
    avg_llm_time: float = 0.0
    avg_detection_time: float = 0.0
    avg_execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "summary": {
                "total_tasks": self.total_tasks,
                "successful_tasks": self.successful_tasks,
                "correct_tasks": self.correct_tasks,
                "failed_tasks": self.failed_tasks,
                "success_rate": f"{self.successful_tasks/max(1,self.total_tasks):.1%}",
                "correctness_rate": f"{self.correct_tasks/max(1,self.total_tasks):.1%}",
                "overall_accuracy": f"{self.overall_accuracy:.1%}",
                "accuracy_when_successful": f"{self.accuracy_when_successful:.1%}",
            },
            "timing": {
                "total_time_seconds": round(self.total_time, 2),
                "avg_time_per_task_seconds": round(self.avg_time_per_task, 2),
                "avg_llm_time_seconds": round(self.avg_llm_time, 3),
                "avg_detection_time_seconds": round(self.avg_detection_time, 3),
                "avg_action_execution_seconds": round(self.avg_execution_time, 3),
            },
            "transformation_stats": self.transformation_counts,
            "action_stats": self.action_counts,
            "accuracy_by_transformation": self.accuracy_by_transformation,
            "llm_vs_fallback": {
                "llm_success_rate": f"{self.llm_success_rate:.1%}",
                "fallback_usage_rate": f"{self.fallback_usage_rate:.1%}",
            },
            "metadata": {
                "run_timestamp": self.run_timestamp,
                "run_folder": self.run_folder,
                "model_used": self.model_used,
                "pattern_used": self.pattern_used,
                "directory": self.directory,
                "program_version": self.program_version,
            },
            "task_results": [r.to_dict() for r in self.task_results]
        }
        return result


class BatchRunner:
    """
    Run multiple ARC tasks and collect statistics.
    
    Example:
        runner = BatchRunner(model="llama3", verbose=True)
        results = runner.run_batch("data/", pattern="task_*.json")
        runner.print_summary(results)
        runner.save_report(results, "batch_results.json")
    """
    
    def __init__(
        self,
        model: str = "llama3",
        verbose: bool = True,
        visualize: bool = False,  # Default off for batch
        multi_mode: bool = False
    ):
        """
        Initialize the batch runner.
        
        Args:
            model: LLM model name for Ollama
            verbose: Print progress messages
            visualize: Show visualizations (usually False for batch)
            multi_mode: Use multi-transform mode
        """
        self.model = model
        self.verbose = verbose
        self.visualize = visualize
        self.multi_mode = multi_mode
        
        # Lazy import to avoid circular imports
        self._orchestrator = None
    
    def _get_orchestrator(self, fresh: bool = False):
        """
        Lazy initialization of orchestrator.
        
        Args:
            fresh: If True, create a new orchestrator (useful for debugging)
        """
        if self._orchestrator is None or fresh:
            # Import here to avoid circular imports
            from main import BRAINOrchestrator
            self._orchestrator = BRAINOrchestrator(
                model=self.model,
                verbose=False,  # Keep quiet during batch - only show progress
                visualize=self.visualize
            )
        return self._orchestrator
    
    def _log(self, message: str):
        """Print message if verbose mode."""
        if self.verbose:
            print(message)
    
    def find_tasks(self, directory: str, pattern: str = "task_*.json") -> List[Path]:
        """
        Find all task files matching the pattern.
        
        Args:
            directory: Directory to search
            pattern: Glob pattern for task files
            
        Returns:
            List of Path objects for matching files
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find matching files
        files = list(dir_path.glob(pattern))
        
        # Sort alphabetically for consistent ordering
        files.sort(key=lambda p: p.name)
        
        return files
    
    def run_single_task(self, task_file: Path) -> TaskResult:
        """
        Run a single task and return the result.
        
        Args:
            task_file: Path to the task JSON file
            
        Returns:
            TaskResult with all metrics and grid data for visualization
        """
        orchestrator = self._get_orchestrator()
        
        result = TaskResult(
            task_id=task_file.stem,
            task_file=str(task_file),
            success=False,
            is_correct=False,
            accuracy=0.0,
            execution_time=0.0
        )
        
        start_time = time.time()
        
        try:
            # Load task
            task = orchestrator.load_task(str(task_file))
            
            result.num_train_examples = len(task.train_pairs)
            result.num_test_examples = len(task.test_pairs)
            
            # Store input and expected grids for visualization
            if task.test_pairs:
                grid = task.test_pairs[0].input_grid
                result.grid_size = f"{grid.height}x{grid.width}"
                result.input_grid = grid
                # Complexity metrics
                result.num_colors_in_input = len([c for c in grid.unique_colors if c != 0])
                result.num_objects_in_input = len(grid.objects) if hasattr(grid, 'objects') and grid.objects else 0
                if task.test_pairs[0].output_grid:
                    result.expected_grid = task.test_pairs[0].output_grid
            
            # Solve task
            result.mode_used = "multi_transform" if self.multi_mode else "single"
            if self.multi_mode:
                solve_result = orchestrator.solve_task_multi_transform(task)
            else:
                solve_result = orchestrator.solve_task(task)
            
            # Extract results
            result.success = True
            
            # Get detected transformations (with enhanced details)
            if "detected_transformations" in solve_result:
                for trans_list in solve_result["detected_transformations"]:
                    if trans_list:
                        for t in trans_list:
                            if hasattr(t, 'transformation_type'):
                                result.detected_transformations.append(t.transformation_type)
                                # Primary transformation is the first one with highest confidence
                                if result.primary_transformation is None:
                                    result.primary_transformation = t.transformation_type
                                    result.transformation_confidence = getattr(t, 'confidence', 0.0)
                                    result.transformation_params = dict(getattr(t, 'parameters', {}))
                            elif isinstance(t, dict) and 'transformation_type' in t:
                                result.detected_transformations.append(t['transformation_type'])
                                if result.primary_transformation is None:
                                    result.primary_transformation = t['transformation_type']
                                    result.transformation_confidence = t.get('confidence', 0.0)
                                    result.transformation_params = dict(t.get('parameters', {}))
            
            # Get action used and LLM vs fallback tracking
            if "action_data" in solve_result and solve_result["action_data"]:
                action = solve_result["action_data"]
                if isinstance(action, dict):
                    result.action_used = action.get("action", "unknown")
            
            # NEW: Extract enhanced metadata from solve_result
            if "metadata" in solve_result:
                meta = solve_result["metadata"]
                result.was_fallback_used = meta.get("was_fallback_used", False)
                result.llm_proposed_action = meta.get("llm_proposed_action")
                result.fallback_reason = meta.get("fallback_reason")
                result.llm_response_time = meta.get("llm_response_time", 0.0)
                result.detection_time = meta.get("detection_time", 0.0)
                result.execution_action_time = meta.get("execution_time", 0.0)
            
            # Get accuracy and predicted grid
            if solve_result.get("analyses"):
                analyses = solve_result["analyses"]
                if analyses:
                    result.accuracy = analyses[0].accuracy
                    result.is_correct = analyses[0].is_correct
                    # Store predicted grid for visualization
                    if hasattr(analyses[0], 'predicted_grid'):
                        result.predicted_grid = analyses[0].predicted_grid
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        
        return result
    
    def run_batch(
        self,
        directory: str,
        pattern: str = "task_*.json",
        limit: Optional[int] = None
    ) -> BatchResult:
        """
        Run all tasks in a directory and collect statistics.
        
        Args:
            directory: Directory containing task files
            pattern: Glob pattern for task files
            limit: Maximum number of tasks to run (None for all)
            
        Returns:
            BatchResult with aggregated statistics
        """
        # Find tasks
        task_files = self.find_tasks(directory, pattern)
        
        if limit:
            task_files = task_files[:limit]
        
        self._log(f"\n{'='*60}")
        self._log(f"  BRAIN Batch Evaluation")
        self._log(f"{'='*60}")
        self._log(f"  Directory: {directory}")
        self._log(f"  Pattern: {pattern}")
        self._log(f"  Tasks found: {len(task_files)}")
        self._log(f"  Model: {self.model}")
        self._log(f"  Mode: {'Multi-transform' if self.multi_mode else 'Single-transform'}")
        self._log(f"{'='*60}\n")
        
        if not task_files:
            self._log("No tasks found!")
            return BatchResult()
        
        # Create timestamped folder name
        timestamp = datetime.now()
        run_folder = f"batch_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize result
        batch_result = BatchResult(
            total_tasks=len(task_files),
            run_timestamp=timestamp.isoformat(),
            run_folder=run_folder,
            model_used=self.model,
            pattern_used=pattern,
            directory=str(directory)
        )
        
        start_time = time.time()
        
        # Run each task
        for i, task_file in enumerate(task_files):
            self._log(f"[{i+1}/{len(task_files)}] Running: {task_file.name}")
            
            result = self.run_single_task(task_file)
            batch_result.task_results.append(result)
            
            # Update counters
            if result.success:
                batch_result.successful_tasks += 1
                if result.is_correct:
                    batch_result.correct_tasks += 1
            else:
                batch_result.failed_tasks += 1
            
            # Count transformations
            for trans in result.detected_transformations:
                batch_result.transformation_counts[trans] = \
                    batch_result.transformation_counts.get(trans, 0) + 1
            
            # Count actions
            if result.action_used:
                batch_result.action_counts[result.action_used] = \
                    batch_result.action_counts.get(result.action_used, 0) + 1
            
            # Progress indicator
            status = "✓" if result.is_correct else ("⚠" if result.success else "✗")
            acc_str = f"{result.accuracy:.0%}" if result.success else "N/A"
            time_str = f"{result.execution_time:.1f}s"
            self._log(f"       {status} Accuracy: {acc_str}, Time: {time_str}")
        
        # Calculate aggregates
        batch_result.total_time = time.time() - start_time
        batch_result.avg_time_per_task = batch_result.total_time / max(1, len(task_files))
        
        # Calculate accuracy metrics
        successful_results = [r for r in batch_result.task_results if r.success]
        if successful_results:
            batch_result.overall_accuracy = sum(r.accuracy for r in batch_result.task_results) / len(batch_result.task_results)
            batch_result.accuracy_when_successful = sum(r.accuracy for r in successful_results) / len(successful_results)
        
        # === NEW: Calculate enhanced statistics ===
        
        # 1. Accuracy by transformation type
        trans_accuracy = {}  # {trans_type: {"total": N, "correct": M, "accuracy_sum": X}}
        for r in batch_result.task_results:
            if r.primary_transformation and r.success:
                trans = r.primary_transformation
                if trans not in trans_accuracy:
                    trans_accuracy[trans] = {"total": 0, "correct": 0, "accuracy_sum": 0.0}
                trans_accuracy[trans]["total"] += 1
                trans_accuracy[trans]["accuracy_sum"] += r.accuracy
                if r.is_correct:
                    trans_accuracy[trans]["correct"] += 1
        
        # Convert to final format
        for trans, stats in trans_accuracy.items():
            batch_result.accuracy_by_transformation[trans] = {
                "count": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["accuracy_sum"] / max(1, stats["total"]),
                "success_rate": stats["correct"] / max(1, stats["total"])
            }
        
        # 2. LLM vs Fallback statistics
        fallback_count = sum(1 for r in batch_result.task_results if r.was_fallback_used)
        llm_only_results = [r for r in batch_result.task_results if not r.was_fallback_used and r.success]
        llm_correct = sum(1 for r in llm_only_results if r.is_correct)
        
        batch_result.fallback_usage_rate = fallback_count / max(1, len(batch_result.task_results))
        batch_result.llm_success_rate = llm_correct / max(1, len(llm_only_results)) if llm_only_results else 0.0
        
        # 3. Average timing breakdown
        llm_times = [r.llm_response_time for r in batch_result.task_results if r.llm_response_time > 0]
        detection_times = [r.detection_time for r in batch_result.task_results if r.detection_time > 0]
        exec_times = [r.execution_action_time for r in batch_result.task_results if r.execution_action_time > 0]
        
        batch_result.avg_llm_time = sum(llm_times) / max(1, len(llm_times)) if llm_times else 0.0
        batch_result.avg_detection_time = sum(detection_times) / max(1, len(detection_times)) if detection_times else 0.0
        batch_result.avg_execution_time = sum(exec_times) / max(1, len(exec_times)) if exec_times else 0.0
        
        return batch_result
    
    def print_summary(self, result: BatchResult):
        """
        Print a formatted summary of batch results.
        
        Args:
            result: BatchResult to summarize
        """
        print("\n" + "=" * 60)
        print("  BATCH EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"\n📊 RESULTS:")
        print(f"   Total tasks:      {result.total_tasks}")
        print(f"   Successful:       {result.successful_tasks} ({result.successful_tasks/max(1,result.total_tasks):.1%})")
        print(f"   Correct (100%):   {result.correct_tasks} ({result.correct_tasks/max(1,result.total_tasks):.1%})")
        print(f"   Failed:           {result.failed_tasks}")
        
        print(f"\n📈 ACCURACY:")
        print(f"   Overall:          {result.overall_accuracy:.1%}")
        print(f"   When successful:  {result.accuracy_when_successful:.1%}")
        
        print(f"\n⏱️  TIMING:")
        print(f"   Total time:       {result.total_time:.1f}s")
        print(f"   Avg per task:     {result.avg_time_per_task:.1f}s")
        
        if result.transformation_counts:
            print(f"\n🔄 TRANSFORMATIONS DETECTED:")
            for trans, count in sorted(result.transformation_counts.items(), key=lambda x: -x[1]):
                print(f"   {trans}: {count}")
        
        if result.action_counts:
            print(f"\n🎯 ACTIONS EXECUTED:")
            for action, count in sorted(result.action_counts.items(), key=lambda x: -x[1]):
                print(f"   {action}: {count}")
        
        # === NEW: Show accuracy by transformation ===
        if result.accuracy_by_transformation:
            print(f"\n📊 ACCURACY BY TRANSFORMATION TYPE:")
            for trans, stats in sorted(result.accuracy_by_transformation.items(), key=lambda x: -x[1]['accuracy']):
                acc = stats['accuracy']
                correct = stats['correct']
                total = stats['count']
                print(f"   {trans}: {acc:.1%} ({correct}/{total} correct)")
        
        # === NEW: Show LLM vs Fallback stats ===
        print(f"\n🤖 LLM vs FALLBACK:")
        print(f"   Fallback usage:   {result.fallback_usage_rate:.1%}")
        print(f"   LLM success rate: {result.llm_success_rate:.1%}")
        
        # === NEW: Show timing breakdown ===
        if result.avg_llm_time > 0 or result.avg_detection_time > 0:
            print(f"\n⏱️  TIMING BREAKDOWN (avg per task):")
            print(f"   Detection:       {result.avg_detection_time:.3f}s")
            print(f"   LLM response:    {result.avg_llm_time:.3f}s")
            print(f"   Action exec:     {result.avg_execution_time:.3f}s")
        
        # Show failed tasks
        failed = [r for r in result.task_results if not r.success]
        if failed:
            print(f"\n❌ FAILED TASKS ({len(failed)}):")
            for r in failed[:5]:  # Show first 5
                print(f"   - {r.task_id}: {r.error_message}")
            if len(failed) > 5:
                print(f"   ... and {len(failed)-5} more")
        
        # Show incorrect tasks (success but not 100%)
        incorrect = [r for r in result.task_results if r.success and not r.is_correct]
        if incorrect:
            print(f"\n⚠️  INCORRECT TASKS ({len(incorrect)}):")
            for r in sorted(incorrect, key=lambda x: x.accuracy)[:5]:
                print(f"   - {r.task_id}: {r.accuracy:.1%}")
            if len(incorrect) > 5:
                print(f"   ... and {len(incorrect)-5} more")
        
        print("\n" + "=" * 60)
    
    def save_report(self, result: BatchResult, filepath: str):
        """
        Save batch results to a JSON file.
        
        Args:
            result: BatchResult to save
            filepath: Path for the JSON file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        self._log(f"\n📄 Report saved to: {filepath}")
    
    def generate_csv_report(self, result: BatchResult, filepath: str):
        """
        Save task-level results to a CSV file.
        
        Args:
            result: BatchResult to save
            filepath: Path for the CSV file
        """
        import csv
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header - Enhanced for data analysis
            writer.writerow([
                'task_id', 'success', 'is_correct', 'accuracy', 
                'execution_time', 'transformations', 'action_used',
                'grid_size', 'train_examples', 'test_examples',
                # NEW columns for analysis
                'primary_transformation', 'transformation_confidence',
                'was_fallback_used', 'llm_proposed_action', 'fallback_reason',
                'llm_time', 'detection_time', 'action_time',
                'num_colors', 'num_objects', 'mode',
                'error'
            ])
            
            # Data rows
            for r in result.task_results:
                writer.writerow([
                    r.task_id,
                    r.success,
                    r.is_correct,
                    f"{r.accuracy:.4f}",
                    f"{r.execution_time:.2f}",
                    "|".join(r.detected_transformations),
                    r.action_used or "",
                    r.grid_size or "",
                    r.num_train_examples,
                    r.num_test_examples,
                    # NEW data
                    r.primary_transformation or "",
                    f"{r.transformation_confidence:.4f}",
                    r.was_fallback_used,
                    r.llm_proposed_action or "",
                    r.fallback_reason or "",
                    f"{r.llm_response_time:.3f}",
                    f"{r.detection_time:.3f}",
                    f"{r.execution_action_time:.3f}",
                    r.num_colors_in_input,
                    r.num_objects_in_input,
                    r.mode_used,
                    r.error_message or ""
                ])
        
        self._log(f"📄 CSV saved to: {filepath}")
    
    def save_results(
        self, 
        result: BatchResult, 
        output_dir: str = "results/",
        save_images: bool = True,
        show_summary: bool = True
    ) -> str:
        """
        Save all batch results to a timestamped folder.
        
        Creates a folder structure like:
            results/
                batch_20260127_143545/
                    summary.json      # Full report
                    tasks.csv         # Task-level results
                    README.txt        # Quick summary
                    images/           # Visual results
                        batch_summary.png
                        task_xxx.png (individual tasks)
        
        Args:
            result: BatchResult to save
            output_dir: Base directory for results
            save_images: Whether to save visualization images
            show_summary: Whether to display the summary visualization
            
        Returns:
            Path to the created folder
        """
        # Create timestamped folder
        folder_path = Path(output_dir) / result.run_folder
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_path = folder_path / "summary.json"
        self.save_report(result, str(json_path))
        
        # Save CSV report
        csv_path = folder_path / "tasks.csv"
        self.generate_csv_report(result, str(csv_path))
        
        # Generate quick summary README
        readme_path = folder_path / "README.txt"
        self._generate_readme(result, str(readme_path))
        
        # Generate visualizations
        if save_images:
            images_path = folder_path / "images"
            images_path.mkdir(exist_ok=True)
            
            self._generate_visualizations(
                result, 
                str(images_path),
                show_summary=show_summary
            )
        
        self._log(f"\n📁 Results saved to: {folder_path}/")
        
        return str(folder_path)
    
    def _generate_visualizations(
        self,
        result: BatchResult,
        images_dir: str,
        show_summary: bool = True,
        interactive: bool = True
    ):
        """
        Generate all visualization images for the batch.
        
        Args:
            result: BatchResult with task data
            images_dir: Directory to save images
            show_summary: Whether to display the visualization
            interactive: Use interactive browser (True) or static summary (False)
        """
        from modules.visualizer import Visualizer
        
        visualizer = Visualizer()
        images_path = Path(images_dir)
        
        # Prepare task visuals data
        task_visuals = []
        
        for task_result in result.task_results:
            visual_data = {
                'task_id': task_result.task_id,
                'input_grid': task_result.input_grid,
                'predicted_grid': task_result.predicted_grid,
                'expected_grid': task_result.expected_grid,
                'is_correct': task_result.is_correct,
                'accuracy': task_result.accuracy,
            }
            task_visuals.append(visual_data)
            
            # Save individual task image if we have the grids
            if task_result.input_grid is not None and task_result.expected_grid is not None:
                task_image_path = images_path / f"{task_result.task_id}.png"
                visualizer.create_task_detail_image(
                    task_id=task_result.task_id,
                    input_grid=task_result.input_grid,
                    predicted_grid=task_result.predicted_grid,
                    expected_grid=task_result.expected_grid,
                    is_correct=task_result.is_correct,
                    accuracy=task_result.accuracy,
                    save_path=str(task_image_path)
                )
        
        # Create and save batch summary image
        if task_visuals:
            summary_path = images_path / "batch_summary.png"
            visualizer.create_batch_summary(
                task_visuals=task_visuals,
                title=f"Batch Results: {result.run_folder}",
                save_path=str(summary_path),
                show=False  # Don't show static summary if using interactive
            )
            self._log(f"📊 Images saved to: {images_path}/")
            
            # Show interactive browser or static summary
            if show_summary and task_visuals:
                if interactive:
                    self._log(f"🖥️  Opening interactive browser (use ◀/▶ or arrow keys to navigate, Q to quit)")
                    visualizer.create_interactive_browser(
                        task_visuals=task_visuals,
                        title=f"Batch Results: {result.run_folder}"
                    )
                else:
                    # Show static summary
                    visualizer.create_batch_summary(
                        task_visuals=task_visuals,
                        title=f"Batch Results: {result.run_folder}",
                        save_path=None,
                        show=True
                    )
    
    def _generate_readme(self, result: BatchResult, filepath: str):
        """Generate a quick summary README file."""
        lines = [
            "=" * 50,
            "BRAIN Batch Evaluation Results",
            "=" * 50,
            "",
            f"Program version: {result.program_version}",
            f"Run timestamp: {result.run_timestamp}",
            f"Model: {result.model_used}",
            f"Directory: {result.directory}",
            f"Pattern: {result.pattern_used}",
            "",
            "RESULTS:",
            f"  Total tasks:       {result.total_tasks}",
            f"  Successful:        {result.successful_tasks} ({result.successful_tasks/max(1,result.total_tasks):.1%})",
            f"  Correct (100%):    {result.correct_tasks} ({result.correct_tasks/max(1,result.total_tasks):.1%})",
            f"  Failed:            {result.failed_tasks}",
            "",
            "ACCURACY:",
            f"  Overall:           {result.overall_accuracy:.1%}",
            f"  When successful:   {result.accuracy_when_successful:.1%}",
            "",
            "TIMING:",
            f"  Total time:        {result.total_time:.1f}s",
            f"  Avg per task:      {result.avg_time_per_task:.1f}s",
            f"  Avg LLM time:      {result.avg_llm_time:.3f}s",
            f"  Avg detection:     {result.avg_detection_time:.3f}s",
            f"  Avg execution:     {result.avg_execution_time:.3f}s",
            "",
            "LLM vs FALLBACK:",
            f"  Fallback usage:    {result.fallback_usage_rate:.1%}",
            f"  LLM success rate:  {result.llm_success_rate:.1%}",
            "",
        ]
        
        # Add accuracy by transformation if available
        if result.accuracy_by_transformation:
            lines.append("ACCURACY BY TRANSFORMATION:")
            for trans, stats in sorted(result.accuracy_by_transformation.items(), key=lambda x: -x[1]['accuracy']):
                acc = stats['accuracy']
                correct = stats['correct']
                total = stats['count']
                lines.append(f"  {trans}: {acc:.1%} ({correct}/{total})")
            lines.append("")
        
        lines.extend([
            "FILES:",
            "  summary.json  - Full detailed report (JSON)",
            "  tasks.csv     - Task-level results for data analysis",
            "  images/       - Visual results",
            "",
            "=" * 50,
        ])
        
        with open(filepath, 'w') as f:
            f.write("\n".join(lines))


def run_batch_evaluation(
    directory: str = "data/",
    pattern: str = "task_*.json",
    model: str = "llama3",
    output_dir: str = "results/",
    multi_mode: bool = False,
    limit: Optional[int] = None
) -> BatchResult:
    """
    Convenience function to run a batch evaluation.
    
    Creates a timestamped folder with all results:
        results/batch_YYYYMMDD_HHMMSS/
            summary.json
            tasks.csv
            README.txt
    
    Args:
        directory: Directory containing task files
        pattern: Glob pattern for task files
        model: LLM model name
        output_dir: Base directory for output (timestamped folder created inside)
        multi_mode: Use multi-transform mode
        limit: Maximum number of tasks
        
    Returns:
        BatchResult with all statistics
    """
    runner = BatchRunner(
        model=model,
        verbose=True,
        visualize=False,  # IMPORTANT: No visualization during batch
        multi_mode=multi_mode
    )
    
    result = runner.run_batch(directory, pattern, limit)
    runner.print_summary(result)
    
    # Save all results to timestamped folder
    runner.save_results(result, output_dir)
    
    return result


if __name__ == "__main__":
    # Quick test
    import argparse
    
    parser = argparse.ArgumentParser(description="Run batch task evaluation")
    parser.add_argument("--dir", "-d", default="data/", help="Task directory")
    parser.add_argument("--pattern", "-p", default="task_*.json", help="File pattern")
    parser.add_argument("--model", "-m", default="llama3", help="LLM model")
    parser.add_argument("--output", "-o", default="results/", help="Output directory")
    parser.add_argument("--limit", "-l", type=int, help="Max tasks to run")
    parser.add_argument("--multi", action="store_true", help="Multi-transform mode")
    
    args = parser.parse_args()
    
    run_batch_evaluation(
        directory=args.dir,
        pattern=args.pattern,
        model=args.model,
        output_dir=args.output,
        multi_mode=args.multi,
        limit=args.limit
    )
