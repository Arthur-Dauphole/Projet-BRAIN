"""
Geometric object representation and shape classification.
"""

class GeometryObject:
    """
    Represents a connected visual object extracted from the grid.
    Stores its cells, original color and derived geometric properties.
    """

    def __init__(self, cells, color_code):
        self.cells = cells
        self.color_code = color_code
        self.shape = "unknown"

    def width(self):
        """Compute bounding box width."""
        xs = [x for x, y in self.cells]
        return max(xs) - min(xs) + 1

    def height(self):
        """Compute bounding box height."""
        ys = [y for x, y in self.cells]
        return max(ys) - min(ys) + 1

    def detect_shape(self):
        """
        Classify the object geometry using simple geometric rules:
        - point : single cell
        - line  : width=1 or height=1 but area>1
        - square: width=height and area matches bounding box
        - rectangle: area matches bounding box but not square
        - other : anything else
        """
        w = self.width()
        h = self.height()
        area = len(self.cells)

        if area == 1:
            self.shape = "point"
        elif w == 1 or h == 1:
            self.shape = "line"
        elif area == w * h and w == h:
            self.shape = "square"
        elif area == w * h:
            self.shape = "rectangle"
        else:
            self.shape = "other"