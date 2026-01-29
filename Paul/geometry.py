"""
Geometric object representation and shape classification.
"""

class GeometryObject:
    def __init__(self, cells, color_code):
        self.cells = cells
        self.color_code = color_code
        self.shape = "unknown"

    def width(self):
        xs = [x for x, y in self.cells]
        return max(xs) - min(xs) + 1 if xs else 0

    def height(self):
        ys = [y for x, y in self.cells]
        return max(ys) - min(ys) + 1 if ys else 0

    def detect_shape(self):
        w = self.width()
        h = self.height()
        area = len(self.cells)
        bounding_box_area = w * h

        if area == 1:
            self.shape = "point"
        elif w == 1 or h == 1:
            self.shape = "line"
        elif area == bounding_box_area:
            self.shape = "square" if w == h else "rectangle"
        else:
            self.shape = "other"

    def get_internal_holes(self, grid_w, grid_h, offset=(0, 0)):
        """Identifie les cases vides situées à l'intérieur de la boîte englobante décalée."""
        if not self.cells:
            return []

        dx, dy = offset
        cells_x = [c[0] for c in self.cells]
        cells_y = [c[1] for c in self.cells]
        
        min_x, max_x = min(cells_x) + dx, max(cells_x) + dx
        min_y, max_y = min(cells_y) + dy, max(cells_y) + dy
        
        shifted_cells = set((x + dx, y + dy) for x, y in self.cells)

        holes = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if 0 <= x < grid_w and 0 <= y < grid_h:
                    if (x, y) not in shifted_cells:
                        holes.append((x, y))
        return holes

    @property
    def top_left(self):
        xs = [x for x, y in self.cells]
        ys = [y for x, y in self.cells]
        return (min(xs), min(ys)) if xs else (0, 0)

    @property
    def shape_signature(self):
        min_x, min_y = self.top_left
        return frozenset((x - min_x, y - min_y) for x, y in self.cells)
    
    def get_all_variants(self):
        variants = []
        def rotate(pts): return frozenset((-y, x) for x, y in pts)
        def flip(pts): return frozenset((-x, y) for x, y in pts)
        def normalize(pts):
            mx = min(x for x, y in pts)
            my = min(y for x, y in pts)
            return frozenset((x - mx, y - my) for x, y in pts)

        current = normalize(list(self.shape_signature))
        for _ in range(4):
            variants.append(current)
            variants.append(normalize(flip(current)))
            current = normalize(rotate(current))
        return variants