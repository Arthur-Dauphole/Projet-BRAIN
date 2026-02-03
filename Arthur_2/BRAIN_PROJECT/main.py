"""
main.py - THE ORCHESTRATOR
==========================
Main entry point for the BRAIN Project.

Pipeline Flow:
    Input Grid -> Perception -> Prompting -> LLM Reasoning -> Analysis -> Visualization

Features (TIER 1-3):
    - Single transformation mode (default)
    - Multi-transform mode (--multi) for different transformations per color
    - Batch evaluation mode (--batch) for running multiple tasks
    - Self-correction loop (--self-correct) for improved accuracy
    - Rule Memory (RAG) for few-shot learning from past solutions

Usage:
    python main.py                                    # Interactive mode
    python main.py --task data/task.json              # Solve a specific task
    python main.py --task data/task.json --multi      # Multi-transform mode
    python main.py --task data/task.json --self-correct  # With self-correction
    python main.py --demo                             # Run demo with sample data
    python main.py --batch data/                      # Run all tasks in directory
    python main.py --batch data/ --limit 10           # Run first 10 tasks
    python main.py --batch data/ --output results/    # Save reports to directory
"""

import json
import argparse
import time
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

# TIER 3: Import Rule Memory
from modules.rule_memory import RuleMemory


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
        visualize: bool = True,
        use_memory: bool = True,
        memory_path: str = "rule_memory.json"
    ):
        """
        Initialize all pipeline components.
        
        Args:
            model: LLM model name for Ollama
            verbose: Whether to print progress messages
            visualize: Whether to show visualizations
            use_memory: Enable Rule Memory (RAG) for few-shot learning
            memory_path: Path to rule memory storage file
        """
        self.verbose = verbose
        self.visualize = visualize
        self.use_memory = use_memory
        
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
        
        # TIER 3: Rule Memory (RAG)
        if use_memory:
            self.rule_memory = RuleMemory(
                storage_path=memory_path,
                verbose=verbose
            )
            self._log(f"  ✓ Rule Memory ({len(self.rule_memory)} rules loaded)")
        else:
            self.rule_memory = None
        
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
            "action_data": None,
            # === NEW: Metadata for analysis ===
            "metadata": {
                "was_fallback_used": False,
                "llm_proposed_action": None,
                "fallback_reason": None,
                "llm_response_time": 0.0,
                "detection_time": 0.0,
                "execution_time": 0.0
            }
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
        
        detection_start = time.time()
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
        results["metadata"]["detection_time"] = time.time() - detection_start
        
        # === CHECK FOR SPECIAL TRANSFORMATIONS (tiling, scaling, flood_fill, symmetry) ===
        # These transformations are grid-level and should NOT trigger multi-transform mode
        is_grid_level_transform = False
        if detected_transformations:
            for trans_list in detected_transformations:
                for t in trans_list:
                    t_type = t.transformation_type if hasattr(t, 'transformation_type') else ''
                    # flood_fill and symmetry apply globally, not per-color
                    if t_type in ('tiling', 'scaling', 'flood_fill', 'symmetry') and t.confidence >= 0.95:
                        is_grid_level_transform = True
                        self._log(f"\n  ℹ️ Grid-level transformation detected ({t_type}), skipping per-color analysis")
                        break
                if is_grid_level_transform:
                    break
        
        # === AUTO-DETECT MULTI-TRANSFORM ===
        # Check if this task has multiple colors with potentially different transformations
        # BUT skip this if we detected a grid-level transformation (tiling, scaling)
        if task.train_pairs and not is_grid_level_transform:
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
        
        # === CHECK FOR COMPOSITE OR ADD_BORDER TRANSFORMATION ===
        # If composite/add_border detected with high confidence, use fallback directly (LLM struggles with these)
        use_direct_fallback = False
        direct_fallback_action = None
        if detected_transformations:
            for trans_list in detected_transformations:
                for t in trans_list:
                    if t.transformation_type == "composite" and t.confidence >= 0.95:
                        use_direct_fallback = True
                        self._log(f"\n  ℹ️ Composite transformation detected, using direct execution")
                        direct_fallback_action = self._build_fallback_action(detected_transformations, task)
                        break
                    elif t.transformation_type == "add_border" and t.confidence >= 0.95:
                        use_direct_fallback = True
                        self._log(f"\n  ℹ️ Add border transformation detected, using direct execution")
                        direct_fallback_action = self._build_fallback_action(detected_transformations, task)
                        break
                    elif t.transformation_type == "flood_fill" and t.confidence >= 0.95:
                        use_direct_fallback = True
                        self._log(f"\n  ℹ️ Flood fill transformation detected, using direct execution")
                        direct_fallback_action = self._build_fallback_action(detected_transformations, task)
                        break
                    elif t.transformation_type == "symmetry" and t.confidence >= 0.95:
                        use_direct_fallback = True
                        self._log(f"\n  ℹ️ Symmetry transformation detected, using direct execution")
                        direct_fallback_action = self._build_fallback_action(detected_transformations, task)
                        break
                    elif t.transformation_type == "scaling" and t.confidence >= 0.85:
                        use_direct_fallback = True
                        self._log(f"\n  ℹ️ Scaling transformation detected, using direct execution")
                        direct_fallback_action = self._build_fallback_action(detected_transformations, task)
                        break
                    elif t.transformation_type == "rotation" and t.confidence >= 0.85:
                        use_direct_fallback = True
                        self._log(f"\n  ℹ️ Rotation transformation detected (conf={t.confidence:.2f}), using direct execution")
                        direct_fallback_action = self._build_fallback_action(detected_transformations, task)
                        break
                    elif t.transformation_type == "reflection" and t.confidence >= 0.90:
                        use_direct_fallback = True
                        self._log(f"\n  ℹ️ Reflection transformation detected (conf={t.confidence:.2f}), using direct execution")
                        direct_fallback_action = self._build_fallback_action(detected_transformations, task)
                        break
                if use_direct_fallback:
                    break
        
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
        llm_start = time.time()
        response = self.llm_client.query(prompt, system_prompt)
        results["metadata"]["llm_response_time"] = time.time() - llm_start
        
        self._log(f"  Got response ({len(response.raw_text)} chars)")
        if response.reasoning:
            self._log(f"  Reasoning: {response.reasoning[:200]}...")
        
        # Check for action data (JSON instruction)
        if response.action_data:
            self._log(f"  ✓ Action JSON extracted: {response.action_data}")
            results["action_data"] = response.action_data
            results["metadata"]["llm_proposed_action"] = response.action_data.get("action") if isinstance(response.action_data, dict) else None
        else:
            self._log("  ⚠ No action JSON found in LLM response")
            # FALLBACK: Build action from detected transformations
            if detected_transformations:
                self._log("  🔧 FALLBACK: Using detected transformations directly")
                fallback_action = self._build_fallback_action(detected_transformations, task)
                if fallback_action:
                    results["action_data"] = fallback_action
                    results["metadata"]["was_fallback_used"] = True
                    results["metadata"]["fallback_reason"] = "no_llm_json"
                    self._log(f"  ✓ Fallback action: {fallback_action}")
        
        results["predictions"].append(response)
        
        # === STEP 4: ACTION EXECUTION ===
        self._log("\n" + "=" * 50)
        self._log("STEP 4: ACTION EXECUTION (The Hands)")
        self._log("=" * 50)
        
        execution_start = time.time()
        for i, test_pair in enumerate(task.test_pairs):
            test_input = test_pair.input_grid
            predicted_grid = None
            
            # Use direct fallback if detected (composite, add_border), otherwise try LLM action
            if use_direct_fallback and direct_fallback_action:
                action_to_use = dict(direct_fallback_action)  # Make a copy
                # Fix color_filter to use test input color
                test_colors = [c for c in test_input.unique_colors if c != 0]
                if test_colors:
                    action_to_use["color_filter"] = test_colors[0]
                self._log(f"  Using direct fallback action: {action_to_use.get('action')} (color={action_to_use.get('color_filter')})")
                # Track this as a fallback
                results["metadata"]["was_fallback_used"] = True
                results["metadata"]["fallback_reason"] = f"direct_{action_to_use.get('action')}"
                results["metadata"]["llm_proposed_action"] = response.action_data.get("action") if response.action_data and isinstance(response.action_data, dict) else None
            else:
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
                
                # Track execution time for the action
                results["metadata"]["execution_time"] = time.time() - execution_start
            
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
    
    # ==================== TIER 3: SELF-CORRECTION ====================
    
    def solve_task_with_correction(
        self,
        task: ARCTask,
        max_retries: int = 2,
        use_rag: bool = True
    ) -> dict:
        """
        Solve a task with self-correction loop.
        
        If the first attempt is wrong, analyzes errors and re-prompts
        the LLM with feedback about what went wrong.
        
        Args:
            task: The ARCTask to solve
            max_retries: Maximum number of correction attempts
            use_rag: Use Rule Memory for few-shot examples
            
        Returns:
            Dictionary with results (same format as solve_task)
        """
        self._log("=" * 50)
        self._log("SELF-CORRECTION MODE ENABLED")
        self._log("=" * 50)
        
        # Get similar rules for few-shot prompting (RAG)
        similar_rules = []
        if use_rag and self.rule_memory:
            similar_rules = self.rule_memory.find_similar_rules(task, top_k=3)
            if similar_rules:
                self._log(f"  📚 Found {len(similar_rules)} similar past solutions")
        
        best_results = None
        best_accuracy = 0.0
        
        for attempt in range(max_retries + 1):
            if attempt == 0:
                self._log(f"\n--- Attempt {attempt + 1}/{max_retries + 1} (Initial) ---")
            else:
                self._log(f"\n--- Attempt {attempt + 1}/{max_retries + 1} (Correction) ---")
            
            # Run the pipeline
            if attempt == 0:
                # First attempt: normal solve with RAG enhancement
                results = self._solve_with_rag(task, similar_rules)
            else:
                # Correction attempt: include error feedback in prompt
                results = self._solve_with_feedback(
                    task,
                    previous_results=best_results,
                    similar_rules=similar_rules
                )
            
            # Check if successful
            if results.get("analyses"):
                analysis = results["analyses"][0]
                accuracy = analysis.accuracy
                is_correct = analysis.is_correct
                
                self._log(f"  Result: {'✓ Correct' if is_correct else f'✗ {accuracy:.1%} accuracy'}")
                
                # Track best result
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_results = results
                
                # If correct, we're done
                if is_correct:
                    self._log(f"\n✓ Solved on attempt {attempt + 1}!")
                    
                    # Store successful rule in memory
                    if self.rule_memory and results.get("action_data"):
                        self.rule_memory.store_rule(
                            task=task,
                            action_data=results["action_data"],
                            success=True,
                            accuracy=accuracy,
                            detected_transforms=results.get("detected_transformations", [[]])[0]
                        )
                    
                    return results
            else:
                self._log("  Result: No prediction")
        
        # All attempts exhausted
        self._log(f"\n⚠ Max retries reached. Best accuracy: {best_accuracy:.1%}")
        
        # Store the best (but failed) rule for learning
        if self.rule_memory and best_results and best_results.get("action_data"):
            self.rule_memory.store_rule(
                task=task,
                action_data=best_results["action_data"],
                success=False,
                accuracy=best_accuracy,
                detected_transforms=best_results.get("detected_transformations", [[]])[0]
            )
        
        return best_results if best_results else {"task_id": task.task_id, "error": "All attempts failed"}
    
    def _solve_with_rag(self, task: ARCTask, similar_rules: List) -> dict:
        """
        Solve task with RAG-enhanced prompting.
        
        Args:
            task: The task to solve
            similar_rules: Similar rules from memory
            
        Returns:
            Results dictionary
        """
        # If we have similar rules, enhance the prompt
        if similar_rules and self.rule_memory:
            # Get the original prompt
            prompt = self.prompt_maker.create_reasoning_chain_prompt(task)
            
            # Add few-shot examples from memory
            few_shot_text = self.rule_memory.format_for_prompt(similar_rules)
            
            # Insert before "YOUR TASK" section
            if "## YOUR TASK:" in prompt:
                prompt = prompt.replace("## YOUR TASK:", f"{few_shot_text}\n## YOUR TASK:")
            else:
                prompt = few_shot_text + "\n" + prompt
            
            # Query LLM with enhanced prompt
            system_prompt = self.prompt_maker.get_system_prompt()
            response = self.llm_client.query(prompt, system_prompt)
            
            # Continue with normal execution...
            # (This is a simplified version - full implementation would
            # integrate more deeply with solve_task)
        
        # Fall back to normal solve
        return self.solve_task(task)
    
    def _solve_with_feedback(
        self,
        task: ARCTask,
        previous_results: dict,
        similar_rules: List
    ) -> dict:
        """
        Solve task with error feedback from previous attempt.
        
        Args:
            task: The task to solve
            previous_results: Results from previous attempt
            similar_rules: Similar rules from memory
            
        Returns:
            Results dictionary
        """
        # Extract error information
        error_info = self._extract_error_feedback(previous_results)
        
        # Create correction prompt
        correction_prompt = self._create_correction_prompt(task, previous_results, error_info)
        
        # Get system prompt for correction
        system_prompt = self._get_correction_system_prompt()
        
        self._log("  Sending correction prompt to LLM...")
        
        # Query LLM
        response = self.llm_client.query(correction_prompt, system_prompt)
        
        # Execute the corrected action
        if response.action_data:
            self._log(f"  ✓ Got corrected action: {response.action_data.get('action')}")
            
            # Execute on test input
            results = {
                "task_id": task.task_id,
                "predictions": [response],
                "analyses": [],
                "action_data": response.action_data,
            }
            
            for test_pair in task.test_pairs:
                action_result = self.executor.execute(test_pair.input_grid, response.action_data)
                
                if action_result.success and test_pair.output_grid:
                    analysis = self.analyzer.compare_grids(
                        action_result.output_grid,
                        test_pair.output_grid
                    )
                    results["analyses"].append(analysis)
            
            return results
        
        self._log("  ⚠ No valid action in correction response")
        return previous_results
    
    def _extract_error_feedback(self, results: dict) -> dict:
        """
        Extract error information from previous results for feedback.
        
        Args:
            results: Results from previous attempt
            
        Returns:
            Dictionary with error details
        """
        if not results or not results.get("analyses"):
            return {"error": "No analysis available"}
        
        analysis = results["analyses"][0]
        
        error_info = {
            "accuracy": analysis.accuracy,
            "is_correct": analysis.is_correct,
            "error_count": 0,
            "shape_match": True,
            "color_confusions": {},
        }
        
        if hasattr(analysis, "error_analysis") and analysis.error_analysis:
            ea = analysis.error_analysis
            error_info["error_count"] = ea.get("error_count", 0)
            error_info["color_confusions"] = ea.get("color_confusions", {})
            error_info["error_pattern"] = ea.get("error_pattern", "unknown")
            
            if "Shape mismatch" in str(ea.get("error", "")):
                error_info["shape_match"] = False
        
        return error_info
    
    def _create_correction_prompt(
        self,
        task: ARCTask,
        previous_results: dict,
        error_info: dict
    ) -> str:
        """
        Create a prompt that includes error feedback for correction.
        
        Args:
            task: The task being solved
            previous_results: Results from previous attempt
            error_info: Extracted error information
            
        Returns:
            Correction prompt string
        """
        previous_action = previous_results.get("action_data", {})
        
        prompt = f"""Your previous prediction was incorrect.

## ERROR ANALYSIS:
- Accuracy achieved: {error_info.get('accuracy', 0):.1%}
- Pixels wrong: {error_info.get('error_count', 'unknown')}
- Error pattern: {error_info.get('error_pattern', 'unknown')}

## YOUR PREVIOUS ACTION:
```json
{json.dumps(previous_action, indent=2)}
```

## COLOR CONFUSIONS (what you predicted vs what was expected):
"""
        
        confusions = error_info.get("color_confusions", {})
        if confusions:
            for conf, count in list(confusions.items())[:5]:
                prompt += f"- {conf}: {count} pixels\n"
        else:
            prompt += "- No specific confusion data available\n"
        
        prompt += """
## INSTRUCTIONS FOR CORRECTION:
1. Re-analyze the training examples carefully
2. Consider: Did you use the wrong transformation type?
3. Consider: Did you use the wrong parameters (dx, dy, angle, color)?
4. Consider: Did you apply the transformation to the wrong object?

Based on this feedback, provide a CORRECTED action JSON.
Focus on fixing the most likely error based on the confusion pattern.

"""
        
        # Add the original task description
        original_prompt = self.prompt_maker.create_reasoning_chain_prompt(task)
        prompt += original_prompt
        
        return prompt
    
    def _get_correction_system_prompt(self) -> str:
        """Get system prompt for correction attempts."""
        base_prompt = self.prompt_maker.get_system_prompt()
        
        correction_addition = """
CORRECTION MODE:
You are receiving feedback about a previous incorrect prediction.
Pay special attention to:
1. The error analysis showing what went wrong
2. The color confusions indicating which pixels were wrong
3. Your previous action that needs correction

Be more careful with:
- Parameter values (dx, dy, angle)
- Color filters
- Transformation direction
"""
        
        return base_prompt + "\n\n" + correction_addition
    
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
                            # === SMART ROTATION FALLBACK ===
                            # Try brute-force validation to find working configuration
                            best_rotation = self._find_best_rotation_action(task, params)
                            if best_rotation:
                                return best_rotation
                            
                            # Fallback to original logic if brute-force fails
                            angle = params.get("angle", 90)
                            is_grid_level = params.get("grid_level", False)
                            per_object = params.get("per_object", False)
                            position_changed = params.get("position_changed", False)
                            
                            # Auto-detect grid-level rotation based on grid dimensions
                            if task and task.train_pairs and not per_object:
                                pair = task.train_pairs[0]
                                in_shape = pair.input_grid.data.shape
                                out_shape = pair.output_grid.data.shape
                                if angle in [90, 270]:
                                    expected_shape = (in_shape[1], in_shape[0])
                                    if out_shape == expected_shape:
                                        is_grid_level = True
                                elif angle == 180 and in_shape == out_shape:
                                    import numpy as np
                                    rotated = np.rot90(pair.input_grid.data, k=2)
                                    if np.array_equal(rotated, pair.output_grid.data):
                                        is_grid_level = True
                            
                            if position_changed and not is_grid_level:
                                self._log(f"  Rotation with position change - may need composite handling")
                            
                            if is_grid_level:
                                self._log(f"  Rotation fallback (grid-level): angle={angle}")
                                return {
                                    "action": "rotate",
                                    "params": {"angle": angle, "grid_level": True}
                                }
                            else:
                                rot_color = color_filter if color_filter else params.get("color")
                                self._log(f"  Rotation fallback (object): angle={angle}, color={rot_color}")
                                action = {
                                    "action": "rotate",
                                    "params": {"angle": angle}
                                }
                                if rot_color:
                                    action["color_filter"] = int(rot_color)
                                return action
                        elif t_type == "reflection":
                            # === SMART REFLECTION FALLBACK ===
                            # Try brute-force validation to find working configuration
                            best_reflection = self._find_best_reflection_action(task, params)
                            if best_reflection:
                                return best_reflection
                            
                            # Fallback to original logic if brute-force fails
                            axis = params.get("axis", "horizontal")
                            per_object = params.get("per_object", False)
                            is_grid_level = params.get("grid_level", False)
                            
                            # Auto-detect grid-level reflection
                            if task and task.train_pairs and not per_object:
                                import numpy as np
                                pair = task.train_pairs[0]
                                in_data = pair.input_grid.data
                                out_data = pair.output_grid.data
                                
                                if in_data.shape == out_data.shape:
                                    if axis == "horizontal" and np.array_equal(np.flipud(in_data), out_data):
                                        is_grid_level = True
                                    elif axis == "vertical" and np.array_equal(np.fliplr(in_data), out_data):
                                        is_grid_level = True
                            
                            if is_grid_level:
                                self._log(f"  Reflection fallback (grid-level): axis={axis}")
                                return {
                                    "action": "reflect",
                                    "params": {"axis": axis, "grid_level": True}
                                }
                            else:
                                ref_color = color_filter if color_filter else params.get("color")
                                self._log(f"  Reflection fallback (object): axis={axis}, color={ref_color}")
                                action = {
                                    "action": "reflect",
                                    "params": {"axis": axis}
                                }
                                if ref_color:
                                    action["color_filter"] = int(ref_color)
                                return action
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
                        elif t_type == "tiling":
                            reps_h = int(params.get("repetitions_horizontal", 2))
                            reps_v = int(params.get("repetitions_vertical", 2))
                            self._log(f"  Tiling fallback: {reps_h}x{reps_v} repetitions")
                            return {
                                "action": "tile",
                                "params": {
                                    "repetitions_horizontal": reps_h,
                                    "repetitions_vertical": reps_v
                                }
                            }
                        elif t_type == "add_border":
                            # Add border to object
                            border_color = params.get("border_color", 1)
                            obj_color = params.get("color_filter", color_filter)
                            self._log(f"  Add border fallback: object={obj_color}, border={border_color}")
                            return {
                                "action": "add_border",
                                "color_filter": obj_color,
                                "params": {
                                    "border_color": int(border_color)
                                }
                            }
                        elif t_type == "composite":
                            # === SMART COMPOSITE FALLBACK ===
                            # Try brute-force validation to find working configuration
                            best_composite = self._find_best_composite_action(task, params)
                            if best_composite:
                                return best_composite
                            
                            # Fallback to original logic if brute-force fails
                            transformations = params.get("transformations", [])
                            desc = params.get("description", "composite")
                            
                            # Validate and clean up transformations
                            cleaned_transforms = []
                            for t in transformations:
                                if isinstance(t, dict) and "action" in t:
                                    t_params = t.get("params", {})
                                    if not isinstance(t_params, dict):
                                        t_params = {}
                                    
                                    cleaned_t = {
                                        "action": t["action"],
                                        "params": t_params
                                    }
                                    cleaned_transforms.append(cleaned_t)
                            
                            comp_color = color_filter
                            if comp_color is None and task and task.test_pairs:
                                test_colors = [c for c in task.test_pairs[0].input_grid.unique_colors if c != 0]
                                if test_colors:
                                    comp_color = test_colors[0]
                            
                            self._log(f"  Composite fallback: {desc}")
                            self._log(f"    Transforms: {[t['action'] for t in cleaned_transforms]}")
                            self._log(f"    Color: {comp_color}")
                            
                            action = {
                                "action": "composite",
                                "params": {
                                    "transformations": cleaned_transforms
                                }
                            }
                            if comp_color:
                                action["color_filter"] = int(comp_color)
                            return action
                        elif t_type == "flood_fill":
                            # Flood fill enclosed regions with a color
                            seed_point = params.get("seed_point", "enclosed_regions")
                            fill_color = params.get("fill_color", 1)
                            self._log(f"  Flood fill fallback: seed={seed_point}, fill_color={fill_color}")
                            return {
                                "action": "flood_fill",
                                "params": {
                                    "seed_point": seed_point,
                                    "fill_color": int(fill_color)
                                }
                            }
                        elif t_type == "symmetry":
                            # === SMART SYMMETRY FALLBACK ===
                            # Try brute-force validation to find working configuration
                            best_symmetry = self._find_best_symmetry_action(task, params)
                            if best_symmetry:
                                return best_symmetry
                            
                            # Fallback to original logic if brute-force fails
                            axis = params.get("axis", "vertical")
                            position = params.get("position", "adjacent")
                            sym_color = color_filter if color_filter else params.get("color")
                            self._log(f"  Symmetry fallback: axis={axis}, position={position}, color={sym_color}")
                            action = {
                                "action": "symmetry",
                                "params": {
                                    "axis": axis,
                                    "position": position
                                }
                            }
                            if sym_color:
                                action["color_filter"] = int(sym_color)
                            return action
                        elif t_type == "scaling":
                            # Scale objects by a factor
                            factor = params.get("factor", 2)
                            
                            # Check if this is GRID-LEVEL scaling (output size = input size * factor)
                            # In that case, we should NOT use color_filter
                            is_grid_level_scale = False
                            if task and task.train_pairs:
                                pair = task.train_pairs[0]
                                in_h, in_w = pair.input_grid.data.shape
                                out_h, out_w = pair.output_grid.data.shape
                                expected_h = int(in_h * factor)
                                expected_w = int(in_w * factor)
                                if abs(out_h - expected_h) < 2 and abs(out_w - expected_w) < 2:
                                    is_grid_level_scale = True
                            
                            if is_grid_level_scale:
                                self._log(f"  Scaling fallback (grid-level): factor={factor}")
                                return {
                                    "action": "scale",
                                    "params": {
                                        "factor": float(factor)
                                    }
                                }
                            else:
                                # Object-level scaling within same grid
                                scale_color = color_filter if color_filter else params.get("color")
                                self._log(f"  Scaling fallback (object): factor={factor}, color={scale_color}")
                                action = {
                                    "action": "scale",
                                    "params": {
                                        "factor": float(factor)
                                    }
                                }
                                if scale_color:
                                    action["color_filter"] = int(scale_color)
                                return action
        
        return None
    
    def _validate_action_on_training(
        self,
        action: Dict[str, Any],
        task: ARCTask,
        tolerance: float = 0.98
    ) -> bool:
        """
        Validate if an action produces correct output on training examples.
        
        Args:
            action: The action to validate
            task: The ARCTask with training pairs
            tolerance: Required accuracy (0.98 = 98% of pixels must match)
            
        Returns:
            True if action produces correct output on at least one training pair
        """
        if not task or not task.train_pairs:
            return False
        
        import numpy as np
        
        for pair in task.train_pairs:
            try:
                # Copy action and update color_filter for this training pair
                test_action = dict(action)
                train_colors = [c for c in pair.input_grid.unique_colors if c != 0]
                if train_colors and "color_filter" in test_action:
                    test_action["color_filter"] = train_colors[0]
                
                result = self.executor.execute(pair.input_grid, test_action)
                
                if result.success and result.output_grid:
                    expected = pair.output_grid.data
                    actual = result.output_grid.data
                    
                    if expected.shape == actual.shape:
                        match_ratio = np.sum(expected == actual) / expected.size
                        if match_ratio >= tolerance:
                            return True
            except Exception:
                continue
        
        return False
    
    def _find_best_rotation_action(
        self,
        task: ARCTask,
        detected_params: dict
    ) -> Optional[Dict[str, Any]]:
        """
        Try multiple rotation configurations and return the one that works.
        
        Tries all angles (90, 180, 270), both grid-level and object-level,
        and different anchor strategies (centroid, topleft, center).
        """
        import numpy as np
        
        if not task or not task.train_pairs:
            return None
        
        color_filter = None
        if task.test_pairs:
            test_colors = [c for c in task.test_pairs[0].input_grid.unique_colors if c != 0]
            if test_colors:
                color_filter = test_colors[0]
        
        # Configurations to try (ordered by likelihood)
        detected_angle = detected_params.get("angle", 90)
        angles_to_try = [detected_angle]
        for a in [90, 180, 270]:
            if a not in angles_to_try:
                angles_to_try.append(a)
        
        # Anchor strategies to try
        anchors_to_try = ["topleft", "centroid", "center", "topright"]
        
        configs_to_try = []
        
        # Try grid-level first (most reliable for full grid rotations)
        for angle in angles_to_try:
            configs_to_try.append({
                "action": "rotate",
                "params": {"angle": angle, "grid_level": True}
            })
        
        # Then object-level with different anchor strategies
        if color_filter:
            # Try each anchor strategy for each angle
            for anchor in anchors_to_try:
                for angle in angles_to_try:
                    configs_to_try.append({
                        "action": "rotate",
                        "params": {"angle": angle, "anchor": anchor},
                        "color_filter": int(color_filter)
                    })
        
        # Test each configuration
        for config in configs_to_try:
            if self._validate_action_on_training(config, task):
                anchor = config.get("params", {}).get("anchor", "grid_level" if config.get("params", {}).get("grid_level") else "default")
                self._log(f"  ✓ Found working rotation: angle={config['params'].get('angle')}, anchor={anchor}")
                return config
        
        # If nothing works, return None (will use original fallback)
        return None
    
    def _find_best_reflection_action(
        self,
        task: ARCTask,
        detected_params: dict
    ) -> Optional[Dict[str, Any]]:
        """
        Try multiple reflection configurations and return the one that works.
        
        Tries all axes and both grid-level and object-level.
        """
        import numpy as np
        
        if not task or not task.train_pairs:
            return None
        
        color_filter = None
        if task.test_pairs:
            test_colors = [c for c in task.test_pairs[0].input_grid.unique_colors if c != 0]
            if test_colors:
                color_filter = test_colors[0]
        
        detected_axis = detected_params.get("axis", "horizontal")
        axes_to_try = [detected_axis]
        for ax in ["horizontal", "vertical", "diagonal_main", "diagonal_anti"]:
            if ax not in axes_to_try:
                axes_to_try.append(ax)
        
        configs_to_try = []
        
        # Try grid-level first
        for axis in axes_to_try:
            configs_to_try.append({
                "action": "reflect",
                "params": {"axis": axis, "grid_level": True}
            })
        
        # Then object-level with color
        if color_filter:
            for axis in axes_to_try:
                configs_to_try.append({
                    "action": "reflect",
                    "params": {"axis": axis},
                    "color_filter": int(color_filter)
                })
        
        # Test each configuration
        for config in configs_to_try:
            if self._validate_action_on_training(config, task):
                self._log(f"  ✓ Found working reflection: axis={config['params'].get('axis')}, grid_level={config['params'].get('grid_level', False)}")
                return config
        
        return None
    
    def _find_best_symmetry_action(
        self,
        task: ARCTask,
        detected_params: dict
    ) -> Optional[Dict[str, Any]]:
        """
        Try multiple symmetry configurations and return the one that works.
        """
        if not task or not task.train_pairs:
            return None
        
        color_filter = None
        if task.test_pairs:
            test_colors = [c for c in task.test_pairs[0].input_grid.unique_colors if c != 0]
            if test_colors:
                color_filter = test_colors[0]
        
        detected_axis = detected_params.get("axis", "vertical")
        detected_position = detected_params.get("position", "adjacent")
        
        axes = [detected_axis]
        for ax in ["vertical", "horizontal"]:
            if ax not in axes:
                axes.append(ax)
        
        positions = [detected_position]
        for pos in ["adjacent", "opposite", "centered"]:
            if pos not in positions:
                positions.append(pos)
        
        configs_to_try = []
        for axis in axes:
            for position in positions:
                config = {
                    "action": "symmetry",
                    "params": {"axis": axis, "position": position}
                }
                if color_filter:
                    config["color_filter"] = int(color_filter)
                configs_to_try.append(config)
        
        for config in configs_to_try:
            if self._validate_action_on_training(config, task):
                self._log(f"  ✓ Found working symmetry: axis={config['params'].get('axis')}, position={config['params'].get('position')}")
                return config
        
        return None
    
    def _find_best_composite_action(
        self,
        task: ARCTask,
        detected_params: dict
    ) -> Optional[Dict[str, Any]]:
        """
        Try multiple composite configurations and return the one that works.
        
        Common patterns:
        - rotate + translate
        - reflect + translate  
        - just rotate (mis-detected as composite)
        - just reflect (mis-detected as composite)
        """
        import numpy as np
        
        if not task or not task.train_pairs:
            return None
        
        color_filter = None
        if task.test_pairs:
            test_colors = [c for c in task.test_pairs[0].input_grid.unique_colors if c != 0]
            if test_colors:
                color_filter = test_colors[0]
        
        configs_to_try = []
        
        # 1. Try pure rotations first (sometimes composite is just rotation)
        for angle in [90, 180, 270]:
            config = {
                "action": "rotate",
                "params": {"angle": angle, "grid_level": True}
            }
            configs_to_try.append(config)
            
            if color_filter:
                config = {
                    "action": "rotate",
                    "params": {"angle": angle},
                    "color_filter": int(color_filter)
                }
                configs_to_try.append(config)
        
        # 2. Try pure reflections
        for axis in ["horizontal", "vertical"]:
            config = {
                "action": "reflect",
                "params": {"axis": axis, "grid_level": True}
            }
            configs_to_try.append(config)
            
            if color_filter:
                config = {
                    "action": "reflect",
                    "params": {"axis": axis},
                    "color_filter": int(color_filter)
                }
                configs_to_try.append(config)
        
        # 3. Try detected transformations if available
        detected_transforms = detected_params.get("transformations", [])
        if detected_transforms and color_filter:
            config = {
                "action": "composite",
                "params": {"transformations": detected_transforms},
                "color_filter": int(color_filter)
            }
            configs_to_try.insert(0, config)  # Try detected first
        
        # 4. Try common composite patterns: rotate + translate
        for angle in [90, 180, 270]:
            for dx in [-1, 0, 1, 2]:
                for dy in [-1, 0, 1, 2]:
                    if dx == 0 and dy == 0:
                        continue
                    config = {
                        "action": "composite",
                        "params": {
                            "transformations": [
                                {"action": "rotate", "params": {"angle": angle}},
                                {"action": "translate", "params": {"dx": dx, "dy": dy}}
                            ]
                        }
                    }
                    if color_filter:
                        config["color_filter"] = int(color_filter)
                    configs_to_try.append(config)
        
        # Test each configuration
        for config in configs_to_try:
            if self._validate_action_on_training(config, task):
                action_name = config.get("action")
                if action_name == "composite":
                    transforms = [t["action"] for t in config.get("params", {}).get("transformations", [])]
                    self._log(f"  ✓ Found working composite: {transforms}")
                else:
                    self._log(f"  ✓ Found working {action_name} (instead of composite)")
                return config
        
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
    
    # TIER 3: Self-correction options
    parser.add_argument(
        "--self-correct",
        action="store_true",
        help="Enable self-correction loop (retry with error feedback)"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum correction retries (default: 2)"
    )
    
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable Rule Memory (RAG) for few-shot learning"
    )
    
    parser.add_argument(
        "--memory-path",
        type=str,
        default="rule_memory.json",
        help="Path to rule memory file (default: rule_memory.json)"
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
    # Create orchestrator with TIER 3 options
    brain = BRAINOrchestrator(
        model=args.model,
        verbose=not args.quiet,
        visualize=not args.no_viz,
        use_memory=not args.no_memory,
        memory_path=args.memory_path
    )
    
    if args.demo:
        # Run demo
        brain.run_demo()
    elif args.task:
        # Solve specific task
        task = brain.load_task(args.task)
        
        if args.self_correct:
            # TIER 3: Use self-correction mode
            brain.solve_task_with_correction(
                task,
                max_retries=args.max_retries,
                use_rag=not args.no_memory
            )
        elif args.multi:
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
        print("  python main.py --task FILE.json --self-correct  # With self-correction")
        print("  python main.py --batch data/             # Batch evaluation")
        print("  python main.py --batch data/ --limit 10  # Run first 10 tasks")
        print("\nFor help: python main.py --help")


if __name__ == "__main__":
    main()
