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

    def analyze_topology(self):
        """
        Sépare les cellules de l'objet en deux catégories : 
        le contour et l'intérieur.
        """
        contour_cells = []
        interior_cells = []
        
        # On transforme la liste de cellules en ensemble pour une recherche rapide
        cell_set = set(self.cells)
        
        for (x, y) in self.cells:
            # Un pixel fait partie du contour s'il a au moins un voisin vide 
            # (ou s'il touche le bord de l'objet dans la grille)
            is_contour = False
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                if (x + dx, y + dy) not in cell_set:
                    is_contour = True
                    break
            
            if is_contour:
                contour_cells.append((x, y))
            else:
                interior_cells.append((x, y))
                
        return contour_cells, interior_cells
    
    @property
    def top_left(self):
        """Retourne les coordonnées (x, y) du coin haut-gauche."""
        xs = [x for x, y in self.cells]
        ys = [y for x, y in self.cells]
        return (min(xs), min(ys))

    @property
    def shape_signature(self):
        """
        Crée une empreinte unique de la forme, indépendante de sa position.
        On ramène tous les points par rapport à (0,0).
        """
        min_x, min_y = self.top_left
        # On crée un set de points normalisés (relatifs)
        normalized_cells = frozenset((x - min_x, y - min_y) for x, y in self.cells)
        return normalized_cells