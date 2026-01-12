---
description: Repository Information Overview
alwaysApply: true
---

# arc-brain Information

## Summary
`arc-brain` (ARTHUR) is a neuro-symbolic framework for solving ARC-AGI (Abstraction and Reasoning Corpus) tasks. It combines deterministic geometric perception and symbolic reasoning with probabilistic LLM-based reasoning via a dedicated bridge.

## Structure
- `src/arc_brain/`: Core Python package.
  - `cli/`: Command-line interface logic.
  - `core/`: Basic models (Point, BoundingBox) and color utilities.
  - `perception/`: Geometric detection engine and specialized detectors (Rectangle, Line, Symmetry).
  - `reasoning/`: Symbolic solver, action generators, and search algorithms.
- `src/arc_neuro_bridge.py`: Bridge module converting geometric data into semantic descriptions for LLMs (Ollama integration).
- `data/`: Contains ARC-like task grids in JSON format (`arc_test_grids.json`, `simple_line_task.json`).
- `tests/`: Unit tests and test fixtures.
- `arc_geometric_detection.py`: Legacy wrapper for perception components.
- `arc-agi_reasoning_engine.py`: Legacy wrapper for reasoning components.
- `test_runner.py`: Specialized automated test suite for geometric detection validation.

## Language & Runtime
**Language**: Python  
**Version**: >= 3.10  
**Build System**: setuptools  
**Package Manager**: pip (configured via `pyproject.toml`)

## Dependencies
**Main Dependencies**:
- `numpy`: Grid and array manipulations.
- `matplotlib`: Visualization of grids and detected shapes.
- `scipy`: Advanced image processing and labeling.
- `requests`: Communication with Ollama LLM service.

**Development Dependencies**:
- `pytest`: Primary testing framework.
- `ruff`: Linting and formatting.
- `mypy`: Static type checking.

## Build & Installation
```bash
# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Main Files & Resources
- **CLI Entry Point**: `arc-brain` (mapped to `src/arc_brain/cli/main:main`)
- **Perception Engine**: `src/arc_brain/perception/engine.py`
- **Reasoning Solver**: `src/arc_brain/reasoning/solver.py`
- **LLM Bridge**: `src/arc_neuro_bridge.py`
- **Test Grids**: `data/arc_test_grids.json`

## Testing
**Framework**: pytest
**Test Location**: `tests/unit/`
**Naming Convention**: `test_*.py`
**Specialized Tests**: `test_runner.py` for geometric detection accuracy against JSON ground truth.

**Run Command**:
```bash
# Run unit tests
pytest

# Run specialized geometric detection tests
python test_runner.py --verbose
```
