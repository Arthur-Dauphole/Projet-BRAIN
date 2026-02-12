"""
Geometric object representation and shape classification.
"""

class GeometryObject:
    def __init__(self, cells, color_code):
        self.cells = set(cells) # Set pour recherche rapide O(1)
        self.color_code = color_code
        self.shape = "unknown"
        # On calcule les dims tout de suite
        xs = [x for x, y in self.cells]
        ys = [y for x, y in self.cells]
        self.min_x, self.max_x = (min(xs), max(xs)) if xs else (0,0)
        self.min_y, self.max_y = (min(ys), max(ys)) if ys else (0,0)

    @property
    def top_left(self):
        return (self.min_x, self.min_y)

    @property
    def center(self):
        """Calcule le centre de gravité (barycentre) pour distinguer rotation et translation."""
        if not self.cells: return (0,0)
        mean_x = sum(x for x, y in self.cells) / len(self.cells)
        mean_y = sum(y for x, y in self.cells) / len(self.cells)
        return (mean_x, mean_y)

    def width(self):
        return self.max_x - self.min_x + 1

    def height(self):
        return self.max_y - self.min_y + 1

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
        """
        Identifie les VRAIS trous fermés à l'intérieur de l'objet.
        Utilise un algorithme de propagation (Flood Fill) depuis les bords 
        de la Bounding Box pour éliminer les concavités ouvertes.
        """
        if not self.cells:
            return []

        dx, dy = offset
        
        # 1. Définir la zone de travail (Bounding Box décalée)
        # On ajoute une marge de 1 pixel autour pour être sûr de contourner les formes bizarres
        bx_min, bx_max = self.min_x + dx - 1, self.max_x + dx + 1
        by_min, by_max = self.min_y + dy - 1, self.max_y + dy + 1
        
        shifted_cells = set((x + dx, y + dy) for x, y in self.cells)

        # 2. Flood Fill depuis les bords pour marquer l'"extérieur"
        outside = set()
        stack = []

        # On initialise la pile avec tous les pixels du bord de notre zone de travail
        for x in range(bx_min, bx_max + 1):
            stack.append((x, by_min))
            stack.append((x, by_max))
        for y in range(by_min, by_max + 1):
            stack.append((bx_min, y))
            stack.append((bx_max, y))
            
        seen = set(stack)

        while stack:
            cx, cy = stack.pop()
            outside.add((cx, cy))

            for ndx, ndy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = cx + ndx, cy + ndy
                # Si on est dans la zone de travail, pas encore vu, et pas un obstacle (l'objet)
                if (bx_min <= nx <= bx_max and by_min <= ny <= by_max):
                    if (nx, ny) not in seen and (nx, ny) not in shifted_cells:
                        seen.add((nx, ny))
                        stack.append((nx, ny))

        # 3. Tout ce qui est dans la BB, qui n'est pas l'objet, et pas "outside", est un trou
        holes = []
        # On scanne uniquement la bounding box stricte de l'objet cible
        search_min_x = max(0, self.min_x + dx)
        search_max_x = min(grid_w - 1, self.max_x + dx)
        search_min_y = max(0, self.min_y + dy)
        search_max_y = min(grid_h - 1, self.max_y + dy)

        for x in range(search_min_x, search_max_x + 1):
            for y in range(search_min_y, search_max_y + 1):
                if (x, y) not in shifted_cells and (x, y) not in outside:
                    holes.append((x, y))
                    
        return holes

    @property
    def shape_signature(self):
        return frozenset((x - self.min_x, y - self.min_y) for x, y in self.cells)
    
    def get_all_variants(self):
        variants = []
        def rotate(pts): return frozenset((-y, x) for x, y in pts)
        def flip(pts): return frozenset((-x, y) for x, y in pts)
        def normalize(pts):
            if not pts: return frozenset()
            mx = min(x for x, y in pts)
            my = min(y for x, y in pts)
            return frozenset((x - mx, y - my) for x, y in pts)

        current = normalize(list(self.shape_signature))
        # On génère les 8 symétries du groupe diédral D4
        for _ in range(4):
            variants.append(current)
            variants.append(normalize(flip(current)))
            current = normalize(rotate(current))
        return variants