import numpy as np

from src.arc_brain.core.models import BoundingBox, GeometricShape, Point


def test_point_distance_and_equality() -> None:
    p1 = Point(1, 2)
    p2 = Point(4, 6)

    assert p1 == Point(1, 2)
    assert p1 != p2
    assert p1.manhattan_distance(p2) == 7
    assert np.isclose(p1.euclidean_distance(p2), 5.0)


def test_bounding_box_properties() -> None:
    bbox = BoundingBox(min_x=1, min_y=2, max_x=3, max_y=5)

    assert bbox.width == 3
    assert bbox.height == 4
    assert bbox.area == 12
    assert bbox.contains(Point(2, 3))
    assert not bbox.contains(Point(0, 0))

    corners = bbox.corners()
    assert Point(1, 2) in corners
    assert Point(3, 5) in corners


def test_geometric_shape_density_and_flags() -> None:
    pixels = {Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)}
    bbox = BoundingBox(min_x=0, min_y=0, max_x=1, max_y=1)

    shape = GeometricShape(
        shape_type="rectangle",
        pixels=pixels,
        color=1,
        bounding_box=bbox,
        properties={},
    )

    # Aire de la bbox = 4, 4 pixels => densité 1.0
    assert shape.properties["area"] == 4
    assert np.isclose(shape.properties["density"], 1.0)
    assert shape.is_filled()
    assert not shape.is_hollow()

