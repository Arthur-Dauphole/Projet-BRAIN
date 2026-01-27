import numpy as np
from geometry import GeometryObject

ARC_COLORS = {
    0: "Noir", 1: "Bleu", 2: "Rouge", 3: "Vert", 4: "Jaune",
    5: "Gris", 6: "Magenta", 7: "Orange", 8: "Azur", 9: "Marron"
}

class Transformation:
    def __init__(self, obj_in, obj_out):
        self.obj_in = obj_in
        self.obj_out = obj_out
        self.actions = []
        self.analyze()

    def analyze(self):
        x1, y1 = self.obj_in.top_left
        x2, y2 = self.obj_out.top_left
        dx, dy = x2 - x1, y2 - y1
        if dx != 0 or dy != 0: 
            self.actions.append(f"Translation ({dx:+d}, {dy:+d})")
        if self.obj_in.color_code != self.obj_out.color_code:
            self.actions.append(f"Couleur {self.obj_in.color_code}->{self.obj_out.color_code}")
        if not self.actions:
            self.actions.append("Identique")

class ReasoningEngine:
    def compare_grids(self, objects_input, objects_output, grid_out):
        transformations = []
        available_outputs = list(objects_output)
        grid_h = len(grid_out)
        grid_w = len(grid_out[0]) if grid_h > 0 else 0
        
        # 1. Matching et détection de remplissage
        for obj_in in objects_input:
            for obj_out in available_outputs[:]:
                if obj_out.shape_signature in obj_in.get_all_variants():
                    # Calculer le mouvement
                    x1, y1 = obj_in.top_left
                    x2, y2 = obj_out.top_left
                    dx, dy = x2 - x1, y2 - y1
                    
                    trans = Transformation(obj_in, obj_out)
                    transformations.append(trans)
                    
                    # Vérifier les trous à la position de sortie (avec offset)
                    holes = obj_in.get_internal_holes(grid_w, grid_h, offset=(dx, dy))
                    if holes:
                        hole_colors = [grid_out[y][x] for x, y in holes if grid_out[y][x] != 0]
                        if hole_colors:
                            fill_color = ARC_COLORS.get(hole_colors[0], hole_colors[0])
                            transformations.append(f"Règle : L'intérieur de {ARC_COLORS.get(obj_in.color_code)} est rempli en {fill_color}.")
                            
                            # Nettoyage : On supprime les objets "remplissage" de la liste des nouveaux objets
                            for out_item in available_outputs[:]:
                                if any(cell in holes for cell in out_item.cells):
                                    if out_item in available_outputs:
                                        available_outputs.remove(out_item)

                    if obj_out in available_outputs:
                        available_outputs.remove(obj_out)
                    break

        # 2. Objets vraiment orphelins
        for obj_out in available_outputs:
            transformations.append(f"Nouvel objet {obj_out.shape} ({ARC_COLORS.get(obj_out.color_code)}) apparu.")

        return transformations