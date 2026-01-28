"""
main.py - THE ORCHESTRATOR
==========================
Main entry point for the BRAIN Project.

Pipeline Flow:
    Input Grid -> Perception -> Prompting -> LLM Reasoning -> Analysis -> Visualization

Supports:
    - Single transformation mode (default)
    - Multi-transform mode (--multi) for different transformations per color
    - Batch evaluation mode (--batch) for running multiple tasks

Usage:
    python main.py                                    # Interactive mode
    python main.py --task data/task.json              # Solve a specific task
    python main.py --task data/task.json --multi      # Multi-transform mode
    python main.py --demo                             # Run demo with sample data
    python main.py --batch data/                      # Run all tasks in directory
    python main.py --batch data/ --limit 10           # Run first 10 tasks
    python main.py --batch data/ --output results/    # Save reports to directory
"""

import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

# Import all modules
from modules import (
    Grid,
    ARCTask,
    SymbolDetector,
    TransformationDetector,
    PromptMaker,
    LLMClient,
    ResultAnalyzer,
    Visualizer,
    ActionExecutor,
)


class BRAINOrchestrator:
    """
    Main orchestrator for the BRAIN pipeline.
    
    Coordinates the flow:
        Input -> Perception -> Prompting -> LLM -> Execution -> Analysis -> Visualization
    """
    
    def __init__(
        self,
        model: str = "llama3",
        verbose: bool = True,
        visualize: bool = True
    ):
        """
        Initialize all pipeline components.
        
        Args:
            model: LLM model name for Ollama
            verbose: Whether to print progress messages
            visualize: Whether to show visualizations
        """
        self.verbose = verbose
        self.visualize = visualize
        
        # Initialize pipeline components
        self._log("Initializing BRAIN Pipeline...")
        
        # Step 1a: Shape Perception
        self.detector = SymbolDetector(connectivity=4)
        self._log("  ✓ Symbol Detector (Shapes)")
        
        # Step 1b: Transformation Detection
        self.transformation_detector = TransformationDetector(verbose=verbose)
        self._log("  ✓ Transformation Detector (Patterns)")
        
        # Step 2: Prompting
        self.prompt_maker = PromptMaker(
            include_grid_ascii=True,
            include_objects=True
        )
        self._log("  ✓ Prompt Maker (Bridge)")
        
        # Step 3: LLM Reasoning
        self.llm_client = LLMClient(model=model)
        self._log(f"  ✓ LLM Client (Reasoning) - Model: {model}")
        
        # Step 4: Action Execution (THE HANDS)
        self.executor = ActionExecutor(verbose=verbose)
        self._log("  ✓ Action Executor (Hands)")
        
        # Step 5: Analysis
        self.analyzer = ResultAnalyzer()
        self._log("  ✓ Result Analyzer (Evaluation)")
        
        # Step 6: Visualization
        self.visualizer = Visualizer()
        self._log("  ✓ Visualizer (Dashboard)")
        
        self._log("Pipeline ready!\n")
    
    def _log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)
    
    def load_task(self, filepath: str) -> ARCTask:
        """
        Load an ARC task from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Loaded ARCTask object
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Task file not found: {filepath}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        task_id = path.stem
        task = ARCTask.from_json(task_id, data)
        
        self._log(f"Loaded task: {task}")
        return task
    
    def solve_task(self, task: ARCTask) -> dict:
        """
        Run the complete pipeline to solve an ARC task.
        
        Args:
            task: The ARCTask to solve
            
        Returns:
            Dictionary with results
        """
        results = {
            "task_id": task.task_id,
            "predictions": [],
            "analyses": [],
            "action_data": None
        }
        
        # === STEP 1a: PERCEPTION (Shapes) ===
        self._log("=" * 50)
        self._log("STEP 1a: PERCEPTION (Shape Detection)")
        self._log("=" * 50)
        
        # Detect objects in all grids
        for pair in task.train_pairs:
            self.detector.detect(pair.input_grid)
            self.detector.detect(pair.output_grid)
            self._log(f"  Train input: {pair.input_grid}")
            self._log(f"  Train output: {pair.output_grid}")
        
        for pair in task.test_pairs:
            self.detector.detect(pair.input_grid)
            self._log(f"  Test input: {pair.input_grid}")
        
        # === STEP 1b: TRANSFORMATION DETECTION ===
        self._log("\n" + "=" * 50)
        self._log("STEP 1b: TRANSFORMATION DETECTION")
        self._log("=" * 50)
        
        detected_transformations = []
        for i, pair in enumerate(task.train_pairs):
            transformations = self.transformation_detector.detect_all(
                pair.input_grid, pair.output_grid
            )
            if transformations:
                self._log(f"  Example {i+1}: {self.transformation_detector.describe_transformations(transformations)}")
                detected_transformations.append(transformations)
            else:
                self._log(f"  Example {i+1}: No clear transformation detected")
        
        results["detected_transformations"] = detected_transformations
        
        # === AUTO-DETECT MULTI-TRANSFORM ===
        # Check if this task has multiple colors with potentially different transformations
        if task.train_pairs:
            first_pair = task.train_pairs[0]
            input_colors = [c for c in first_pair.input_grid.unique_colors if c != 0]
            
            # If there are multiple colors, check for per-color transformations
            if len(input_colors) > 1:
                per_color = self.transformation_detector.detect_per_color_transformations(
                    first_pair.input_grid, first_pair.output_grid
                )
                
                # Check if different colors have different transformations OR different parameters
                if per_color and len(per_color) > 1:
                    transforms_signatures = set()
                    for color, t in per_color.items():
                        if hasattr(t, 'transformation_type'):
                            t_type = t.transformation_type
                            params = dict(t.parameters) if t.parameters else {}
                        elif isinstance(t, dict):
                            t_type = t.get('transformation_type', '')
                            params = dict(t.get('parameters', {}))
                        else:
                            continue
                        
                        # Create signature including type AND params
                        sig = f"{t_type}:{sorted(params.items())}"
                        transforms_signatures.add(sig)
                    
                    # If we have different signatures, use multi-transform mode
                    if len(transforms_signatures) > 1:
                        self._log("\n  🔄 AUTO-SWITCH: Multiple different transformations detected, using multi-transform mode")
                        return self.solve_task_multi_transform(task)
        
        # === STEP 2: PROMPTING ===
        self._log("\n" + "=" * 50)
        self._log("STEP 2: PROMPTING (Prompt Creation)")
        self._log("=" * 50)
        
        prompt = self.prompt_maker.create_reasoning_chain_prompt(task)
        system_prompt = self.prompt_maker.get_system_prompt()
        
        self._log(f"  Created prompt ({len(prompt)} chars)")
        
        # === STEP 3: LLM REASONING ===
        self._log("\n" + "=" * 50)
        self._log("STEP 3: LLM REASONING")
        self._log("=" * 50)
        
        # Check LLM connection
        if not self.llm_client.check_connection():
            self._log("  ⚠ Warning: LLM connection issue")
        
        self._log("  Querying LLM...")
        response = self.llm_client.query(prompt, system_prompt)
        
        self._log(f"  Got response ({len(response.raw_text)} chars)")
        if response.reasoning:
            self._log(f"  Reasoning: {response.reasoning[:200]}...")
        
        # Check for action data (JSON instruction)
        if response.action_data:
            self._log(f"  ✓ Action JSON extracted: {response.action_data}")
            results["action_data"] = response.action_data
        else:
            self._log("  ⚠ No action JSON found in LLM response")
            # FALLBACK: Build action from detected transformations
            if detected_transformations:
                self._log("  🔧 FALLBACK: Using detected transformations directly")
                fallback_action = self._build_fallback_action(detected_transformations, task)
                if fallback_action:
                    results["action_data"] = fallback_action
                    self._log(f"  ✓ Fallback action: {fallback_action}")
        
        results["predictions"].append(response)
        
        # === STEP 4: ACTION EXECUTION ===
        self._log("\n" + "=" * 50)
        self._log("STEP 4: ACTION EXECUTION (The Hands)")
        self._log("=" * 50)
        
        for i, test_pair in enumerate(task.test_pairs):
            test_input = test_pair.input_grid
            predicted_grid = None
            
            # Try LLM action first
            action_to_use = response.action_data or results.get("action_data")
            
            if action_to_use:
                # CRITICAL FIX: For draw_line, always detect color from TEST input
                # The LLM might return the training color instead of test color
                if action_to_use.get("action") == "draw_line":
                    import numpy as np
                    test_data = test_input.data
                    for c in range(1, 10):
                        if np.sum(test_data == c) == 2:
                            old_color = action_to_use.get("color_filter")
                            if old_color != c:
                                self._log(f"  🔧 Fixing draw_line color: {old_color} -> {c}")
                            action_to_use = dict(action_to_use)  # Make a copy
                            action_to_use["color_filter"] = c
                            break
                
                # Execute the action on the test input
                self._log(f"  Executing action on test input {i+1}...")
                action_result = self.executor.execute(test_input, action_to_use)
                
                if action_result.success:
                    predicted_grid = action_result.output_grid
                    self._log(f"  ✓ Action executed: {action_result.message}")
                else:
                    self._log(f"  ✗ Action failed: {action_result.message}")
            
            # Fallback to LLM's direct grid prediction if no action
            if predicted_grid is None and response.predicted_grid:
                self._log("  Using LLM's direct grid prediction (fallback)")
                predicted_grid = Grid.from_list(response.predicted_grid)
            
            if predicted_grid is None:
                self._log("  ✗ No prediction available!")
                continue
            
            # === STEP 5: ANALYSIS ===
            self._log("\n" + "=" * 50)
            self._log("STEP 5: ANALYSIS (Evaluation)")
            self._log("=" * 50)
            
            if test_pair.output_grid:
                analysis = self.analyzer.compare_grids(predicted_grid, test_pair.output_grid)
                results["analyses"].append(analysis)
                
                self._log(f"  Test {i+1} Results:")
                self._log(f"    ✓ Correct: {analysis.is_correct}")
                self._log(f"    📊 Accuracy: {analysis.accuracy:.2%}")
                
                if not analysis.is_correct and analysis.error_analysis:
                    self._log(f"    Error details: {analysis.error_analysis.get('error_count', 'N/A')} errors")
                
                # === STEP 6: VISUALIZATION ===
                if self.visualize:
                    self._log("\n" + "=" * 50)
                    self._log("STEP 6: VISUALIZATION")
                    self._log("=" * 50)
                    
                    self.visualizer.show_comparison(
                        test_input,
                        predicted_grid,
                        test_pair.output_grid,
                        title=f"Task {task.task_id} - Test {i+1}"
                    )
        
        # Final summary
        self._log("\n" + "=" * 50)
        self._log("PIPELINE COMPLETE")
        self._log("=" * 50)
        
        if results["analyses"]:
            correct = sum(1 for a in results["analyses"] if a.is_correct)
            total = len(results["analyses"])
            self._log(f"  Results: {correct}/{total} correct")
        
        return results
    
    def solve_task_multi_transform(self, task: ARCTask) -> dict:
        """
        Solve a task using MULTI-TRANSFORM mode.
        
        This mode detects and applies DIFFERENT transformations to DIFFERENT colors.
        
        Args:
            task: The ARCTask to solve
            
        Returns:
            Dictionary with results
        """
        results = {
            "task_id": task.task_id,
            "mode": "multi_transform",
            "predictions": [],
            "analyses": [],
            "per_color_transforms": {}
        }
        
        # === STEP 1a: PERCEPTION (Shapes) ===
        self._log("=" * 50)
        self._log("STEP 1a: PERCEPTION (Shape Detection)")
        self._log("=" * 50)
        
        # Detect objects in all grids
        for pair in task.train_pairs:
            self.detector.detect(pair.input_grid)
            self.detector.detect(pair.output_grid)
            self._log(f"  Train input: {pair.input_grid}")
            self._log(f"  Train output: {pair.output_grid}")
        
        for pair in task.test_pairs:
            self.detector.detect(pair.input_grid)
            self._log(f"  Test input: {pair.input_grid}")
        
        # === STEP 1b: PER-COLOR TRANSFORMATION DETECTION ===
        self._log("\n" + "=" * 50)
        self._log("STEP 1b: PER-COLOR TRANSFORMATION DETECTION")
        self._log("=" * 50)
        
        # Detect per-color transformations from first training example
        # (we assume the pattern is consistent across examples)
        if task.train_pairs:
            first_pair = task.train_pairs[0]
            per_color_transforms = self.transformation_detector.detect_per_color_transformations(
                first_pair.input_grid, 
                first_pair.output_grid
            )
            
            if per_color_transforms:
                self._log("  Detected per-color transformations:")
                desc = self.transformation_detector.describe_per_color_transformations(per_color_transforms)
                for line in desc.split("\n"):
                    self._log(f"    {line}")
                results["per_color_transforms"] = per_color_transforms
            else:
                self._log("  ⚠ No per-color transformations detected")
                self._log("  Falling back to single-transform mode...")
                return self.solve_task(task)
        
        # === STEP 2: PROMPTING (Multi-Transform Mode) ===
        self._log("\n" + "=" * 50)
        self._log("STEP 2: PROMPTING (Multi-Transform Mode)")
        self._log("=" * 50)
        
        prompt = self.prompt_maker.create_multi_transform_prompt(task, per_color_transforms)
        system_prompt = self.prompt_maker.get_multi_transform_system_prompt()
        
        self._log(f"  Created multi-transform prompt ({len(prompt)} chars)")
        
        # === STEP 3: LLM REASONING ===
        self._log("\n" + "=" * 50)
        self._log("STEP 3: LLM REASONING (Multi-Actions)")
        self._log("=" * 50)
        
        # Check LLM connection
        if not self.llm_client.check_connection():
            self._log("  ⚠ Warning: LLM connection issue")
        
        self._log("  Querying LLM for multiple actions...")
        response = self.llm_client.query(prompt, system_prompt)
        
        self._log(f"  Got response ({len(response.raw_text)} chars)")
        if response.reasoning:
            self._log(f"  Reasoning: {response.reasoning[:200]}...")
        
        # Check for multi-actions
        multi_actions = response.multi_actions
        
        if multi_actions:
            self._log(f"  ✓ Multi-actions extracted: {len(multi_actions)} actions")
            for action in multi_actions:
                self._log(f"    - Color {action.get('color')}: {action.get('action')} {action.get('params', {})}")
        else:
            self._log("  ⚠ No multi-actions found in LLM response")
            # Try to build actions from detected transformations
            self._log("  Building actions from detected transformations...")
            multi_actions = self._build_actions_from_transforms(per_color_transforms)
            if multi_actions:
                self._log(f"  ✓ Built {len(multi_actions)} actions from detected transformations")
                for action in multi_actions:
                    self._log(f"    - Color {action.get('color')}: {action.get('action')} {action.get('params', {})}")
        
        results["multi_actions"] = multi_actions
        results["predictions"].append(response)
        
        # === STEP 4: ACTION EXECUTION (Multi-Actions) ===
        self._log("\n" + "=" * 50)
        self._log("STEP 4: ACTION EXECUTION (Multi-Transform)")
        self._log("=" * 50)
        
        for i, test_pair in enumerate(task.test_pairs):
            test_input = test_pair.input_grid
            predicted_grid = None
            
            if multi_actions:
                self._log(f"  Executing {len(multi_actions)} actions on test input {i+1}...")
                action_result = self.executor.execute_multi_actions(test_input, multi_actions)
                
                if action_result.success:
                    predicted_grid = action_result.output_grid
                    self._log(f"  ✓ Multi-actions executed: {action_result.message}")
                else:
                    self._log(f"  ✗ Multi-actions failed: {action_result.message}")
            
            if predicted_grid is None:
                self._log("  ✗ No prediction available!")
                continue
            
            # === STEP 5: ANALYSIS ===
            self._log("\n" + "=" * 50)
            self._log("STEP 5: ANALYSIS (Evaluation)")
            self._log("=" * 50)
            
            if test_pair.output_grid:
                analysis = self.analyzer.compare_grids(predicted_grid, test_pair.output_grid)
                results["analyses"].append(analysis)
                
                self._log(f"  Test {i+1} Results:")
                self._log(f"    ✓ Correct: {analysis.is_correct}")
                self._log(f"    📊 Accuracy: {analysis.accuracy:.2%}")
                
                if not analysis.is_correct and analysis.error_analysis:
                    self._log(f"    Error details: {analysis.error_analysis.get('error_count', 'N/A')} errors")
                
                # === STEP 6: VISUALIZATION ===
                if self.visualize:
                    self._log("\n" + "=" * 50)
                    self._log("STEP 6: VISUALIZATION")
                    self._log("=" * 50)
                    
                    self.visualizer.show_comparison(
                        test_input,
                        predicted_grid,
                        test_pair.output_grid,
                        title=f"Task {task.task_id} - Test {i+1} (Multi-Transform)"
                    )
        
        # Final summary
        self._log("\n" + "=" * 50)
        self._log("PIPELINE COMPLETE (Multi-Transform Mode)")
        self._log("=" * 50)
        
        if results["analyses"]:
            correct = sum(1 for a in results["analyses"] if a.is_correct)
            total = len(results["analyses"])
            self._log(f"  Results: {correct}/{total} correct")
        
        return results
    
    def _build_fallback_action(
        self,
        detected_transformations: List[List[Any]],
        task: ARCTask
    ) -> Optional[Dict[str, Any]]:
        """
        Build an action dictionary from detected transformations (fallback when LLM fails).
        
        Args:
            detected_transformations: List of transformation lists from each training example
            task: The ARCTask (to get test input colors)
            
        Returns:
            Action dictionary or None if unable to build
        """
        if not detected_transformations:
            return None
        
        # Get the first valid transformation
        for trans_list in detected_transformations:
            if trans_list:
                for t in trans_list:
                    t_type = None
                    params = {}
                    
                    if hasattr(t, 'transformation_type'):
                        t_type = t.transformation_type
                        params = dict(t.parameters)
                    elif isinstance(t, dict):
                        t_type = t.get('transformation_type')
                        params = dict(t.get('parameters', {}))
                    
                    if t_type:
                        # Get main color from test input
                        color_filter = None
                        if task.test_pairs:
                            test_colors = [c for c in task.test_pairs[0].input_grid.unique_colors if c != 0]
                            if test_colors:
                                color_filter = test_colors[0]
                        
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
                                # Grid-level rotation
                                return {
                                    "action": "rotate",
                                    "params": {"angle": angle, "grid_level": True}
                                }
                            else:
                                # Object-level rotation
                                action = {
                                    "action": "rotate",
                                    "params": {"angle": angle}
                                }
                                if color_filter:
                                    action["color_filter"] = color_filter
                                return action
                        elif t_type == "reflection":
                            axis = params.get("axis", "horizontal")
                            # Check if this is grid-level or object-level reflection
                            is_grid_level = params.get("grid_level", False)
                            
                            if is_grid_level or color_filter is None:
                                # Grid-level reflection
                                return {
                                    "action": "reflect",
                                    "params": {"axis": axis, "grid_level": True}
                                }
                            else:
                                # Object-level reflection
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
                            # draw_line needs color_filter to find the 2 points
                            # ALWAYS detect color from TEST INPUT (not training examples)
                            # because the test may use a different color
                            draw_color = None
                            if task.test_pairs:
                                import numpy as np
                                test_data = task.test_pairs[0].input_grid.data
                                for c in range(1, 10):
                                    count = int(np.sum(test_data == c))
                                    if count == 2:
                                        draw_color = c
                                        self._log(f"  Auto-detected draw_line color from test: {draw_color}")
                                        break
                            
                            # Fallback to training color if test detection failed
                            if draw_color is None:
                                draw_color = params.get("color") or color_filter or 1
                            
                            return {
                                "action": "draw_line",
                                "color_filter": draw_color
                            }
        
        return None
    
    def _build_actions_from_transforms(
        self, 
        per_color_transforms: Dict[int, Any]
    ) -> List[Dict[str, Any]]:
        """
        Build action list from detected per-color transformations.
        
        This is a fallback when the LLM doesn't return proper multi-actions.
        
        Args:
            per_color_transforms: Dictionary of color -> TransformationResult
            
        Returns:
            List of action dictionaries ready for the executor
        """
        actions = []
        
        for color, transform in per_color_transforms.items():
            if hasattr(transform, 'transformation_type'):
                t_type = transform.transformation_type
                params = dict(transform.parameters)
            else:
                t_type = transform.get('transformation_type', 'identity')
                params = dict(transform.get('parameters', {}))
            
            # Remove 'color' from params if present (it's at top level)
            params.pop('color', None)
            
            action = {
                "color": color,
                "action": t_type,
                "params": params
            }
            actions.append(action)
        
        return actions
    
    def run_demo(self):
        """Run a demo with a simple task."""
        self._log("Running BRAIN Demo...")
        self._log("-" * 50)
        
        # Create a simple demo task
        demo_task_data = {
            "train": [
                {
                    "input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    "output": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
                },
                {
                    "input": [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                    "output": [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
                }
            ],
            "test": [
                {
                    "input": [[0, 0, 0], [0, 3, 0], [0, 0, 0]],
                    "output": [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
                }
            ]
        }
        
        task = ARCTask.from_json("demo_fill", demo_task_data)
        
        # Show the task
        if self.visualize:
            self.visualizer.show_task(task)
        
        # Solve it
        return self.solve_task(task)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BRAIN Project - Neuro-Symbolic ARC-AGI Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                          # Run demo
  python main.py --task data/task.json           # Solve a single task
  python main.py --batch data/                   # Run all tasks in data/
  python main.py --batch data/ --limit 5         # Run first 5 tasks
  python main.py --batch data/ --output results/ # Save reports
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    
    mode_group.add_argument(
        "--task", "-t",
        type=str,
        help="Path to task JSON file (single task mode)"
    )
    
    mode_group.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run demo mode"
    )
    
    mode_group.add_argument(
        "--batch", "-b",
        type=str,
        metavar="DIR",
        help="Run batch evaluation on all tasks in directory"
    )
    
    # General options
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="llama3",
        help="Ollama model name (default: llama3)"
    )
    
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualizations"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (less output)"
    )
    
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Use multi-transform mode (different transformation per color)"
    )
    
    # Batch-specific options
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default="task_*.json",
        help="File pattern for batch mode (default: task_*.json)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/",
        help="Output directory for batch reports (default: results/)"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Maximum number of tasks to run in batch mode"
    )
    
    args = parser.parse_args()
    
    # === BATCH MODE ===
    if args.batch:
        from modules.batch_runner import BatchRunner
        
        # IMPORTANT: Always disable visualization DURING batch execution
        # This allows the batch to run without user interaction
        # Visualization is shown AT THE END with all results
        runner = BatchRunner(
            model=args.model,
            verbose=not args.quiet,
            visualize=False,  # ALWAYS False during batch - no popups!
            multi_mode=args.multi
        )
        
        result = runner.run_batch(
            directory=args.batch,
            pattern=args.pattern,
            limit=args.limit
        )
        
        runner.print_summary(result)
        
        # Save all results to timestamped folder
        # show_summary=True displays the visual recap at the end
        result_folder = runner.save_results(
            result, 
            args.output,
            save_images=True,
            show_summary=not args.no_viz  # Show visual summary unless --no-viz
        )
        
        print(f"\n✅ Batch complete! Results in: {result_folder}")
        
        return
    
    # === SINGLE TASK / DEMO MODE ===
    # Create orchestrator
    brain = BRAINOrchestrator(
        model=args.model,
        verbose=not args.quiet,
        visualize=not args.no_viz
    )
    
    if args.demo:
        # Run demo
        brain.run_demo()
    elif args.task:
        # Solve specific task
        task = brain.load_task(args.task)
        
        if args.multi:
            # Use multi-transform mode
            brain.solve_task_multi_transform(task)
        else:
            # Use standard single-transform mode
            brain.solve_task(task)
    else:
        # Interactive mode - show help
        print("\n" + "=" * 60)
        print("  BRAIN Project - Neuro-Symbolic ARC-AGI Solver")
        print("=" * 60)
        print("\nUsage:")
        print("  python main.py --demo                    # Run demo")
        print("  python main.py --task FILE.json          # Solve a task")
        print("  python main.py --batch data/             # Batch evaluation")
        print("  python main.py --batch data/ --limit 10  # Run first 10 tasks")
        print("\nFor help: python main.py --help")


if __name__ == "__main__":
    main()
