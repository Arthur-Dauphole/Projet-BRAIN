import numpy as np

from src.arc_brain.perception.engine import GeometricDetectionEngine


def test_engine_detects_simple_rectangle() -> None:
    grid = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    engine = GeometricDetectionEngine(background_color=0)
    analysis = engine.analyze_grid(grid, verbose=False)

    rects = analysis["detected_shapes"]["rectangles"]
    lines = analysis["detected_shapes"]["lines"]

    assert len(rects) == 1
    assert len(lines) == 0


def test_engine_detects_horizontal_line() -> None:
    grid = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    engine = GeometricDetectionEngine(background_color=0)
    analysis = engine.analyze_grid(grid, verbose=False)

    rects = analysis["detected_shapes"]["rectangles"]
    lines = analysis["detected_shapes"]["lines"]

    assert len(rects) == 0
    assert len(lines) == 1
    assert lines[0].properties.get("direction") == "horizontal"

