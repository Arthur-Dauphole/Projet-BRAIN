"""
rule_memory.py - Rule Memory System for BRAIN Project
======================================================
Stores and retrieves successful transformation rules for few-shot learning.

Features:
    - Task signature extraction for similarity matching
    - Rule storage with success/failure tracking
    - Similarity-based rule retrieval (simple and embedding-based)
    - JSON persistence for long-term memory
    
This implements a simple RAG (Retrieval-Augmented Generation) system
to help the LLM learn from past successful solutions.
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import numpy as np

from .types import Grid, ARCTask


@dataclass
class TaskSignature:
    """
    Extractable features from a task for similarity matching.
    
    These features capture the "shape" of a task without storing
    the actual grid data, enabling efficient similarity search.
    """
    # Grid properties
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    size_change: bool
    
    # Color properties
    input_colors: List[int]
    output_colors: List[int]
    colors_added: List[int]
    colors_removed: List[int]
    num_colors_input: int
    num_colors_output: int
    
    # Object properties (from detection)
    num_objects_input: int = 0
    num_objects_output: int = 0
    object_types_input: List[str] = field(default_factory=list)
    object_types_output: List[str] = field(default_factory=list)
    
    # Transformation hints (from TransformationDetector)
    detected_transforms: List[str] = field(default_factory=list)
    transform_confidence: float = 0.0
    
    # Hash for quick comparison
    signature_hash: str = ""
    
    def __post_init__(self):
        """Compute signature hash after initialization."""
        if not self.signature_hash:
            self.signature_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute a hash of the signature for quick comparison."""
        key_features = (
            self.input_shape,
            self.output_shape,
            tuple(sorted(self.input_colors)),
            tuple(sorted(self.output_colors)),
            self.size_change,
        )
        return hashlib.md5(str(key_features).encode()).hexdigest()[:12]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TaskSignature":
        """Create from dictionary."""
        # Convert lists that might be tuples
        data = dict(data)
        if isinstance(data.get("input_shape"), list):
            data["input_shape"] = tuple(data["input_shape"])
        if isinstance(data.get("output_shape"), list):
            data["output_shape"] = tuple(data["output_shape"])
        return cls(**data)


@dataclass
class StoredRule:
    """
    A stored rule from a successfully (or unsuccessfully) solved task.
    
    Attributes:
        task_id: Original task identifier
        signature: Task signature for similarity matching
        action_data: The action that was executed
        success: Whether the action produced correct output
        accuracy: Pixel-level accuracy achieved
        timestamp: When this rule was stored
        metadata: Additional information
    """
    task_id: str
    signature: TaskSignature
    action_data: Dict[str, Any]
    success: bool
    accuracy: float
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "signature": self.signature.to_dict(),
            "action_data": self.action_data,
            "success": self.success,
            "accuracy": self.accuracy,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "StoredRule":
        """Create from dictionary."""
        data = dict(data)
        data["signature"] = TaskSignature.from_dict(data["signature"])
        return cls(**data)


class RuleMemory:
    """
    Memory system for storing and retrieving transformation rules.
    
    This implements a simple RAG-like system where:
    1. Task signatures are extracted and stored with their solutions
    2. New tasks are matched against stored signatures
    3. Similar past solutions are retrieved as few-shot examples
    
    Usage:
        memory = RuleMemory("rule_memory.json")
        
        # Store a successful rule
        memory.store_rule(task, action_data, success=True, accuracy=1.0)
        
        # Find similar rules for a new task
        similar = memory.find_similar_rules(new_task, top_k=3)
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        auto_save: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the rule memory.
        
        Args:
            storage_path: Path to JSON file for persistence (None = in-memory only)
            auto_save: Automatically save after each store operation
            verbose: Print debug information
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.auto_save = auto_save
        self.verbose = verbose
        
        self.rules: List[StoredRule] = []
        self._signature_cache: Dict[str, TaskSignature] = {}
        
        # Load existing rules if storage exists
        if self.storage_path and self.storage_path.exists():
            self._load()
    
    def compute_task_signature(self, task: ARCTask) -> TaskSignature:
        """
        Extract a searchable signature from a task.
        
        Args:
            task: The ARC task to analyze
            
        Returns:
            TaskSignature with extracted features
        """
        if not task.train_pairs:
            raise ValueError("Task has no training pairs")
        
        # Use first training pair as representative
        first_pair = task.train_pairs[0]
        input_grid = first_pair.input_grid
        output_grid = first_pair.output_grid
        
        # Extract grid properties
        input_shape = input_grid.shape
        output_shape = output_grid.shape
        size_change = input_shape != output_shape
        
        # Extract color properties
        input_colors = sorted(list(input_grid.unique_colors))
        output_colors = sorted(list(output_grid.unique_colors))
        
        colors_added = [c for c in output_colors if c not in input_colors]
        colors_removed = [c for c in input_colors if c not in output_colors]
        
        # Extract object counts (if available)
        num_objects_input = len(input_grid.objects) if hasattr(input_grid, 'objects') and input_grid.objects else 0
        num_objects_output = len(output_grid.objects) if hasattr(output_grid, 'objects') and output_grid.objects else 0
        
        # Extract object types
        object_types_input = []
        object_types_output = []
        
        if hasattr(input_grid, 'objects') and input_grid.objects:
            object_types_input = [obj.object_type for obj in input_grid.objects if hasattr(obj, 'object_type')]
        if hasattr(output_grid, 'objects') and output_grid.objects:
            object_types_output = [obj.object_type for obj in output_grid.objects if hasattr(obj, 'object_type')]
        
        return TaskSignature(
            input_shape=input_shape,
            output_shape=output_shape,
            size_change=size_change,
            input_colors=input_colors,
            output_colors=output_colors,
            colors_added=colors_added,
            colors_removed=colors_removed,
            num_colors_input=len([c for c in input_colors if c != 0]),
            num_colors_output=len([c for c in output_colors if c != 0]),
            num_objects_input=num_objects_input,
            num_objects_output=num_objects_output,
            object_types_input=object_types_input,
            object_types_output=object_types_output,
        )
    
    def compute_signature_with_transforms(
        self,
        task: ARCTask,
        detected_transforms: List[Any]
    ) -> TaskSignature:
        """
        Compute signature including detected transformations.
        
        Args:
            task: The ARC task
            detected_transforms: List of TransformationResult from detector
            
        Returns:
            Enhanced TaskSignature
        """
        sig = self.compute_task_signature(task)
        
        if detected_transforms:
            transform_types = []
            max_confidence = 0.0
            
            for t in detected_transforms:
                if hasattr(t, 'transformation_type'):
                    transform_types.append(t.transformation_type)
                    if hasattr(t, 'confidence'):
                        max_confidence = max(max_confidence, t.confidence)
                elif isinstance(t, dict):
                    transform_types.append(t.get('transformation_type', 'unknown'))
                    max_confidence = max(max_confidence, t.get('confidence', 0))
            
            sig.detected_transforms = transform_types
            sig.transform_confidence = max_confidence
            sig.signature_hash = sig._compute_hash()  # Recompute hash
        
        return sig
    
    def store_rule(
        self,
        task: ARCTask,
        action_data: Dict[str, Any],
        success: bool,
        accuracy: float,
        detected_transforms: List[Any] = None,
        metadata: Dict[str, Any] = None
    ) -> StoredRule:
        """
        Store a rule from a solved (or attempted) task.
        
        Args:
            task: The ARC task that was solved
            action_data: The action that was executed
            success: Whether the action produced correct output
            accuracy: Pixel-level accuracy achieved
            detected_transforms: Optional transformation detection results
            metadata: Additional metadata to store
            
        Returns:
            The stored rule
        """
        # Check for duplicate - don't store if same task with same or lower accuracy
        existing = self._find_existing_rule(task.task_id)
        if existing:
            if accuracy <= existing.accuracy:
                if self.verbose:
                    print(f"  ⊘ Skipping duplicate: {task.task_id} (existing: {existing.accuracy:.1%})")
                return existing
            else:
                # Replace with better version
                self.rules.remove(existing)
                if self.verbose:
                    print(f"  ↑ Updating rule: {task.task_id} ({existing.accuracy:.1%} -> {accuracy:.1%})")
        
        # Compute signature
        if detected_transforms:
            signature = self.compute_signature_with_transforms(task, detected_transforms)
        else:
            signature = self.compute_task_signature(task)
        
        # Create rule
        rule = StoredRule(
            task_id=task.task_id,
            signature=signature,
            action_data=action_data,
            success=success,
            accuracy=accuracy,
            metadata=metadata or {}
        )
        
        # Add to memory
        self.rules.append(rule)
        
        if self.verbose:
            status = "✓" if success else "✗"
            print(f"  {status} Stored rule: {task.task_id} ({accuracy:.1%}) -> {action_data.get('action', '?')}")
        
        # Auto-save if enabled
        if self.auto_save and self.storage_path:
            self._save()
        
        return rule
    
    def _find_existing_rule(self, task_id: str) -> Optional[StoredRule]:
        """Find an existing rule for a task."""
        for rule in self.rules:
            if rule.task_id == task_id:
                return rule
        return None
    
    def find_similar_rules(
        self,
        task: ARCTask,
        top_k: int = 3,
        min_accuracy: float = 0.5,
        success_only: bool = True
    ) -> List[StoredRule]:
        """
        Find rules from similar past tasks.
        
        Args:
            task: The new task to find matches for
            top_k: Maximum number of rules to return
            min_accuracy: Minimum accuracy threshold for returned rules
            success_only: Only return successful rules
            
        Returns:
            List of similar StoredRules, sorted by similarity
        """
        if not self.rules:
            return []
        
        # Compute signature for new task
        new_sig = self.compute_task_signature(task)
        
        # Score all rules
        scored_rules = []
        
        for rule in self.rules:
            # Filter by success and accuracy
            if success_only and not rule.success:
                continue
            if rule.accuracy < min_accuracy:
                continue
            
            # Compute similarity score
            score = self._compute_similarity(new_sig, rule.signature)
            
            if score > 0:
                scored_rules.append((score, rule))
        
        # Sort by score (descending) and return top_k
        scored_rules.sort(key=lambda x: x[0], reverse=True)
        
        return [rule for _, rule in scored_rules[:top_k]]
    
    def _compute_similarity(self, sig1: TaskSignature, sig2: TaskSignature) -> float:
        """
        Compute similarity score between two task signatures.
        
        Returns a score between 0 and 1, where 1 is identical.
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            Similarity score (0-1)
        """
        score = 0.0
        max_score = 0.0
        
        # Shape match (high weight)
        max_score += 3.0
        if sig1.input_shape == sig2.input_shape:
            score += 1.5
        elif self._shapes_similar(sig1.input_shape, sig2.input_shape):
            score += 0.75  # Partial credit for similar shapes
        if sig1.output_shape == sig2.output_shape:
            score += 1.5
        elif self._shapes_similar(sig1.output_shape, sig2.output_shape):
            score += 0.75
        
        # Size change match (important)
        max_score += 2.0
        if sig1.size_change == sig2.size_change:
            score += 2.0
        
        # Color count match
        max_score += 1.0
        if sig1.num_colors_input == sig2.num_colors_input:
            score += 0.5
        if sig1.num_colors_output == sig2.num_colors_output:
            score += 0.5
        
        # Color pattern match (colors added/removed)
        max_score += 1.0
        if sig1.colors_added == sig2.colors_added:
            score += 0.5
        if sig1.colors_removed == sig2.colors_removed:
            score += 0.5
        
        # Object count similarity
        max_score += 1.0
        if sig1.num_objects_input > 0 and sig2.num_objects_input > 0:
            obj_diff = abs(sig1.num_objects_input - sig2.num_objects_input)
            score += max(0, 1.0 - obj_diff * 0.2)
        
        # Object type match
        max_score += 1.0
        if sig1.object_types_input and sig2.object_types_input:
            common_types = set(sig1.object_types_input) & set(sig2.object_types_input)
            all_types = set(sig1.object_types_input) | set(sig2.object_types_input)
            if all_types:
                score += len(common_types) / len(all_types)
        
        # Transformation type match (highest weight - most important for RAG)
        if sig1.detected_transforms and sig2.detected_transforms:
            max_score += 3.0
            common_transforms = set(sig1.detected_transforms) & set(sig2.detected_transforms)
            all_transforms = set(sig1.detected_transforms) | set(sig2.detected_transforms)
            if all_transforms:
                # Exact match bonus
                if sig1.detected_transforms == sig2.detected_transforms:
                    score += 3.0
                else:
                    score += 2.0 * len(common_transforms) / len(all_transforms)
        
        # Normalize to 0-1
        return score / max_score if max_score > 0 else 0.0
    
    def _shapes_similar(self, shape1: Tuple[int, int], shape2: Tuple[int, int], tolerance: int = 2) -> bool:
        """Check if two shapes are similar within tolerance."""
        return (abs(shape1[0] - shape2[0]) <= tolerance and 
                abs(shape1[1] - shape2[1]) <= tolerance)
    
    def get_rules_by_transform(self, transform_type: str) -> List[StoredRule]:
        """
        Get all rules for a specific transformation type.
        
        Args:
            transform_type: The transformation type to filter by
            
        Returns:
            List of matching rules
        """
        return [
            rule for rule in self.rules
            if transform_type in rule.signature.detected_transforms
        ]
    
    def get_successful_rules(self) -> List[StoredRule]:
        """Get all successful rules."""
        return [rule for rule in self.rules if rule.success]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored rules.
        
        Returns:
            Dictionary with statistics
        """
        if not self.rules:
            return {"total_rules": 0}
        
        successful = [r for r in self.rules if r.success]
        
        # Count by transformation type
        transform_counts = {}
        for rule in self.rules:
            for t in rule.signature.detected_transforms:
                transform_counts[t] = transform_counts.get(t, 0) + 1
        
        return {
            "total_rules": len(self.rules),
            "successful_rules": len(successful),
            "success_rate": len(successful) / len(self.rules),
            "avg_accuracy": sum(r.accuracy for r in self.rules) / len(self.rules),
            "transform_counts": transform_counts,
            "unique_tasks": len(set(r.task_id for r in self.rules)),
        }
    
    def format_for_prompt(self, rules: List[StoredRule], max_examples: int = 3) -> str:
        """
        Format rules as few-shot examples for LLM prompt.
        
        Args:
            rules: Rules to format
            max_examples: Maximum number of examples
            
        Returns:
            Formatted string for inclusion in prompt
        """
        if not rules:
            return ""
        
        lines = ["\n## SIMILAR PAST SOLUTIONS:\n"]
        
        for i, rule in enumerate(rules[:max_examples]):
            sig = rule.signature
            lines.append(f"### Example {i+1} (accuracy: {rule.accuracy:.0%}):")
            lines.append(f"- Input shape: {sig.input_shape}")
            lines.append(f"- Output shape: {sig.output_shape}")
            lines.append(f"- Colors: {sig.input_colors} -> {sig.output_colors}")
            
            if sig.detected_transforms:
                lines.append(f"- Detected transformation: {', '.join(sig.detected_transforms)}")
            
            lines.append(f"- Successful action:")
            lines.append(f"  ```json")
            lines.append(f"  {json.dumps(rule.action_data, indent=2)}")
            lines.append(f"  ```")
            lines.append("")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all stored rules."""
        self.rules.clear()
        self._signature_cache.clear()
        
        if self.storage_path and self.storage_path.exists():
            self.storage_path.unlink()
    
    def _save(self) -> None:
        """Save rules to storage file."""
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": "1.0",
            "rules": [rule.to_dict() for rule in self.rules],
            "statistics": self.get_statistics()
        }
        
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        if self.verbose:
            print(f"  💾 Saved {len(self.rules)} rules to {self.storage_path}")
    
    def _load(self) -> None:
        """Load rules from storage file."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            
            self.rules = [StoredRule.from_dict(r) for r in data.get("rules", [])]
            
            if self.verbose:
                print(f"  📂 Loaded {len(self.rules)} rules from {self.storage_path}")
        
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Warning: Could not load rules: {e}")
            self.rules = []
    
    def __len__(self) -> int:
        return len(self.rules)
    
    def __repr__(self) -> str:
        return f"RuleMemory({len(self.rules)} rules)"
