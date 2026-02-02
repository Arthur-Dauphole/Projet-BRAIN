"""
logger.py - Structured Logging System for BRAIN Project
========================================================
Provides consistent, structured logging across all pipeline components.

Features:
    - Pipeline step logging with timing
    - Structured data attachment
    - Multiple output formats (console, file, JSON)
    - Log level filtering
    - Performance metrics collection
"""

import logging
import json
import time
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager


class LogLevel(Enum):
    """Pipeline component identifiers for structured logging."""
    PIPELINE = "PIPELINE"
    PERCEPTION = "PERCEPTION"
    DETECTION = "DETECTION"
    PROMPTING = "PROMPTING"
    LLM = "LLM"
    EXECUTION = "EXECUTION"
    ANALYSIS = "ANALYSIS"
    VISUALIZATION = "VISUALIZATION"
    BATCH = "BATCH"
    MEMORY = "MEMORY"
    ERROR = "ERROR"


@dataclass
class LogEntry:
    """Structured log entry with metadata."""
    timestamp: str
    level: str
    component: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class PerformanceMetrics:
    """Collected performance metrics for a pipeline run."""
    total_duration_ms: float = 0.0
    step_durations: Dict[str, float] = field(default_factory=dict)
    llm_calls: int = 0
    llm_total_time_ms: float = 0.0
    detection_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class BRAINLogger:
    """
    Centralized logging system for the BRAIN pipeline.
    
    Usage:
        logger = BRAINLogger(verbose=True)
        logger.step(LogLevel.DETECTION, "Detected 3 objects", objects=3)
        
        with logger.timed_step(LogLevel.LLM, "Querying model"):
            # ... LLM call ...
    """
    
    # ANSI color codes for terminal output
    COLORS = {
        LogLevel.PIPELINE: "\033[1;36m",      # Bold Cyan
        LogLevel.PERCEPTION: "\033[0;34m",    # Blue
        LogLevel.DETECTION: "\033[0;35m",     # Magenta
        LogLevel.PROMPTING: "\033[0;33m",     # Yellow
        LogLevel.LLM: "\033[0;32m",           # Green
        LogLevel.EXECUTION: "\033[0;36m",     # Cyan
        LogLevel.ANALYSIS: "\033[0;34m",      # Blue
        LogLevel.VISUALIZATION: "\033[0;37m", # White
        LogLevel.BATCH: "\033[1;33m",         # Bold Yellow
        LogLevel.MEMORY: "\033[0;35m",        # Magenta
        LogLevel.ERROR: "\033[1;31m",         # Bold Red
    }
    RESET = "\033[0m"
    
    # Icons for different components
    ICONS = {
        LogLevel.PIPELINE: "🔄",
        LogLevel.PERCEPTION: "👁️",
        LogLevel.DETECTION: "🔍",
        LogLevel.PROMPTING: "📝",
        LogLevel.LLM: "🧠",
        LogLevel.EXECUTION: "⚙️",
        LogLevel.ANALYSIS: "📊",
        LogLevel.VISUALIZATION: "📈",
        LogLevel.BATCH: "📦",
        LogLevel.MEMORY: "💾",
        LogLevel.ERROR: "❌",
    }
    
    def __init__(
        self,
        verbose: bool = True,
        log_file: Optional[str] = None,
        json_log: bool = False,
        use_colors: bool = True,
        collect_metrics: bool = True
    ):
        """
        Initialize the logger.
        
        Args:
            verbose: Print to console
            log_file: Path to log file (optional)
            json_log: Output logs as JSON lines
            use_colors: Use ANSI colors in console output
            collect_metrics: Collect performance metrics
        """
        self.verbose = verbose
        self.log_file = Path(log_file) if log_file else None
        self.json_log = json_log
        self.use_colors = use_colors
        self.collect_metrics = collect_metrics
        
        self.entries: List[LogEntry] = []
        self.metrics = PerformanceMetrics()
        self._step_start_times: Dict[str, float] = {}
        
        # Setup file logging if requested
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def step(
        self,
        component: LogLevel,
        message: str,
        **data
    ) -> None:
        """
        Log a pipeline step.
        
        Args:
            component: The pipeline component logging this message
            message: Human-readable message
            **data: Additional structured data to attach
        """
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level="INFO",
            component=component.value,
            message=message,
            data=data
        )
        
        self.entries.append(entry)
        
        if self.verbose:
            self._print_entry(entry, component)
        
        if self.log_file:
            self._write_to_file(entry)
    
    def success(self, component: LogLevel, message: str, **data) -> None:
        """Log a success message with checkmark."""
        self.step(component, f"✓ {message}", **data)
    
    def warning(self, component: LogLevel, message: str, **data) -> None:
        """Log a warning message."""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level="WARNING",
            component=component.value,
            message=f"⚠ {message}",
            data=data
        )
        
        self.entries.append(entry)
        self.metrics.warnings.append(message)
        
        if self.verbose:
            self._print_entry(entry, component, is_warning=True)
        
        if self.log_file:
            self._write_to_file(entry)
    
    def error(self, component: LogLevel, message: str, exception: Exception = None, **data) -> None:
        """Log an error message."""
        if exception:
            data["exception_type"] = type(exception).__name__
            data["exception_message"] = str(exception)
        
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level="ERROR",
            component=component.value,
            message=f"✗ {message}",
            data=data
        )
        
        self.entries.append(entry)
        self.metrics.errors.append(message)
        
        if self.verbose:
            self._print_entry(entry, LogLevel.ERROR)
        
        if self.log_file:
            self._write_to_file(entry)
    
    @contextmanager
    def timed_step(self, component: LogLevel, message: str, **data):
        """
        Context manager for timing a step.
        
        Usage:
            with logger.timed_step(LogLevel.LLM, "Querying model"):
                response = llm.query(prompt)
        """
        start_time = time.time()
        
        self.step(component, f"{message}...")
        
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            
            entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                level="INFO",
                component=component.value,
                message=f"  ↳ Completed in {duration_ms:.1f}ms",
                data=data,
                duration_ms=duration_ms
            )
            
            self.entries.append(entry)
            
            # Update metrics
            if self.collect_metrics:
                key = component.value.lower()
                if key not in self.metrics.step_durations:
                    self.metrics.step_durations[key] = 0.0
                self.metrics.step_durations[key] += duration_ms
                
                if component == LogLevel.LLM:
                    self.metrics.llm_calls += 1
                    self.metrics.llm_total_time_ms += duration_ms
                elif component == LogLevel.DETECTION:
                    self.metrics.detection_time_ms += duration_ms
                elif component == LogLevel.EXECUTION:
                    self.metrics.execution_time_ms += duration_ms
            
            if self.verbose:
                self._print_entry(entry, component, indent=True)
            
            if self.log_file:
                self._write_to_file(entry)
    
    def section(self, title: str, width: int = 50) -> None:
        """Print a section header."""
        if self.verbose:
            print("=" * width)
            print(f"  {title}")
            print("=" * width)
    
    def subsection(self, title: str, width: int = 40) -> None:
        """Print a subsection header."""
        if self.verbose:
            print("-" * width)
            print(f"  {title}")
            print("-" * width)
    
    def _print_entry(
        self,
        entry: LogEntry,
        component: LogLevel,
        is_warning: bool = False,
        indent: bool = False
    ) -> None:
        """Print a log entry to console."""
        prefix = "  " if indent else ""
        icon = self.ICONS.get(component, "•")
        
        if self.use_colors:
            color = self.COLORS.get(component, "")
            reset = self.RESET
        else:
            color = ""
            reset = ""
        
        if self.json_log:
            print(entry.to_json())
        else:
            component_tag = f"[{component.value}]"
            print(f"{prefix}{color}{icon} {component_tag:14} {entry.message}{reset}")
            
            # Print attached data if present
            if entry.data and not indent:
                for key, value in entry.data.items():
                    if not key.startswith("_"):  # Skip private keys
                        print(f"{prefix}  └─ {key}: {value}")
    
    def _write_to_file(self, entry: LogEntry) -> None:
        """Write entry to log file."""
        if self.log_file:
            with open(self.log_file, "a") as f:
                if self.json_log:
                    f.write(entry.to_json() + "\n")
                else:
                    f.write(f"[{entry.timestamp}] [{entry.level}] [{entry.component}] {entry.message}\n")
                    if entry.data:
                        f.write(f"  Data: {json.dumps(entry.data, default=str)}\n")
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get collected performance metrics."""
        return self.metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary dictionary of metrics."""
        return {
            "total_entries": len(self.entries),
            "llm_calls": self.metrics.llm_calls,
            "llm_total_time_ms": self.metrics.llm_total_time_ms,
            "detection_time_ms": self.metrics.detection_time_ms,
            "execution_time_ms": self.metrics.execution_time_ms,
            "step_durations": self.metrics.step_durations,
            "error_count": len(self.metrics.errors),
            "warning_count": len(self.metrics.warnings),
        }
    
    def print_metrics_summary(self) -> None:
        """Print a summary of collected metrics."""
        if not self.verbose:
            return
        
        print("\n" + "=" * 50)
        print("  PERFORMANCE METRICS")
        print("=" * 50)
        
        summary = self.get_metrics_summary()
        
        print(f"\n📊 Pipeline Statistics:")
        print(f"   • Total log entries: {summary['total_entries']}")
        print(f"   • LLM calls: {summary['llm_calls']}")
        print(f"   • LLM total time: {summary['llm_total_time_ms']:.1f}ms")
        print(f"   • Detection time: {summary['detection_time_ms']:.1f}ms")
        print(f"   • Execution time: {summary['execution_time_ms']:.1f}ms")
        
        if summary['error_count'] > 0:
            print(f"\n❌ Errors: {summary['error_count']}")
            for err in self.metrics.errors[:5]:  # Show first 5
                print(f"   • {err}")
        
        if summary['warning_count'] > 0:
            print(f"\n⚠ Warnings: {summary['warning_count']}")
            for warn in self.metrics.warnings[:5]:  # Show first 5
                print(f"   • {warn}")
        
        print()
    
    def clear(self) -> None:
        """Clear all log entries and reset metrics."""
        self.entries.clear()
        self.metrics = PerformanceMetrics()
        self._step_start_times.clear()


# Global logger instance (optional convenience)
_global_logger: Optional[BRAINLogger] = None


def get_logger() -> BRAINLogger:
    """Get or create the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = BRAINLogger()
    return _global_logger


def set_logger(logger: BRAINLogger) -> None:
    """Set the global logger instance."""
    global _global_logger
    _global_logger = logger
