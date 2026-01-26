"""
main.py - THE ORCHESTRATOR
==========================
Main entry point for the BRAIN Project.

Pipeline Flow:
    Input Grid -> Perception -> Prompting -> LLM Reasoning -> Analysis -> Visualization

Usage:
    python main.py                          # Interactive mode
    python main.py --task data/task.json    # Solve a specific task
    python main.py --demo                   # Run demo with sample data
"""

import json
import argparse
from pathlib import Path
from typing import Optional

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
            self._log("  ⚠ No action JSON found in response")
            if response.predicted_grid:
                self._log(f"  Fallback: Grid extracted directly ({len(response.predicted_grid)}x{len(response.predicted_grid[0])})")
        
        results["predictions"].append(response)
        
        # === STEP 4: ACTION EXECUTION ===
        self._log("\n" + "=" * 50)
        self._log("STEP 4: ACTION EXECUTION (The Hands)")
        self._log("=" * 50)
        
        for i, test_pair in enumerate(task.test_pairs):
            test_input = test_pair.input_grid
            predicted_grid = None
            
            if response.action_data:
                # Execute the action on the test input
                self._log(f"  Executing action on test input {i+1}...")
                action_result = self.executor.execute(test_input, response.action_data)
                
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
        description="BRAIN Project - Neuro-Symbolic ARC-AGI Solver"
    )
    
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="Path to task JSON file"
    )
    
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run demo mode"
    )
    
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
    
    args = parser.parse_args()
    
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
        brain.solve_task(task)
    else:
        # Interactive mode
        print("\n" + "=" * 60)
        print("  BRAIN Project - Neuro-Symbolic ARC-AGI Solver")
        print("=" * 60)
        print("\nUsage:")
        print("  python main.py --demo           # Run demo")
        print("  python main.py --task FILE.json # Solve a task")
        print("\nFor help: python main.py --help")


if __name__ == "__main__":
    main()
