from src.arc_brain.core.models import BoundingBox, GeometricShape, Point
from arc_neuro_bridge import SceneDescriber, PromptFactory


def _make_dummy_rectangle() -> GeometricShape:
    pixels = {Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)}
    bbox = BoundingBox(min_x=0, min_y=0, max_x=1, max_y=1)
    return GeometricShape(
        shape_type="rectangle",
        pixels=pixels,
        color=2,
        bounding_box=bbox,
        properties={"is_square": True, "is_filled": True, "aspect_ratio": 1.0},
    )


def test_scene_describer_returns_non_empty_string() -> None:
    shape = _make_dummy_rectangle()
    describer = SceneDescriber(include_spatial_relations=False)

    description = describer.describe_scene([shape])

    assert isinstance(description, str)
    assert "Rectangle" in description or "rectangle" in description
    assert "Object 1" in description


def test_prompt_factory_structure() -> None:
    input_scene = "### Rectangles (1):\n- Object 1: Red rectangle."
    output_scene = "### Rectangles (1):\n- Object 1: Green rectangle."

    prompt = PromptFactory.create_prompt(input_scene=input_scene, output_scene=output_scene)

    assert "[SYSTEM]" in prompt
    assert "[INPUT SCENE]" in prompt
    assert "[OUTPUT SCENE]" in prompt
    assert "[TASK]" in prompt
    assert "Step-by-step" in prompt or "Reason step-by-step" in prompt

