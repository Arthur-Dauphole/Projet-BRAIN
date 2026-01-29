import numpy as np
from geometry import GeometryObject

ARC_COLORS = {
    0: "Noir", 1: "Bleu", 2: "Rouge", 3: "Vert", 4: "Jaune",
    5: "Gris", 6: "Magenta", 7: "Orange", 8: "Azur", 9: "Marron"
}

def normalize_shape(cells):
    """
    Ramène une liste de pixels (r, c) vers (0,0) et la trie pour comparaison.
    """
    if not cells: return []
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    # On soustrait le min pour coller la forme en haut à gauche (0,0)
    return sorted([(r - min_r, c - min_c) for r, c in cells])

class Transformation:
    def __init__(self, obj_in, obj_out):
        self.obj_in = obj_in
        self.obj_out = obj_out
        self.actions = []
        self.analyze()

    def analyze(self):
        # 1. TRANSLATION
        x1, y1 = self.obj_in.top_left
        x2, y2 = self.obj_out.top_left
        dx, dy = x2 - x1, y2 - y1
        if dx != 0 or dy != 0: 
            self.actions.append(f"Translation ({dx:+d}, {dy:+d})")

        # 2. ROTATIONS & SYMÉTRIES
        # On compare la forme pure (normalisée) de l'entrée et de la sortie
        shape_in_norm = normalize_shape(self.obj_in.cells)
        shape_out_norm = normalize_shape(self.obj_out.cells)

        # Si les formes sont différentes, on cherche pourquoi (Rotation ?)
        if shape_in_norm != shape_out_norm:
            found_geo = False
            # Liste des transformations à tester : (Nom, Fonction lambda sur (r,c))
            geo_tests = [
                ("Rotation 90°", lambda r, c: (c, -r)),
                ("Rotation 180°", lambda r, c: (-r, -c)),
                ("Rotation 270°", lambda r, c: (-c, r)),
                ("Symétrie Horizontale", lambda r, c: (-r, c)), # Flip haut/bas
                ("Symétrie Verticale", lambda r, c: (r, -c))    # Flip gauche/droite
            ]

            for name, func in geo_tests:
                # On applique la transformation sur les pixels d'origine
                transformed_cells = [func(r, c) for r, c in shape_in_norm]
                # On re-normalise le résultat (car une rotation peut donner des coords négatives)
                if normalize_shape(transformed_cells) == shape_out_norm:
                    self.actions.append(name)
                    found_geo = True
                    break # On s'arrête à la première transfo trouvée
            
            # Si on a tout testé et rien trouvé (cas rare de déformation non rigide)
            if not found_geo:
                self.actions.append("Déformation inconnue")

        # 3. COULEUR
        if self.obj_in.color_code != self.obj_out.color_code:
            self.actions.append(f"Couleur {self.obj_in.color_code}->{self.obj_out.color_code}")

        # 4. IDENTIQUE (Seulement si rien d'autre n'a changé)
        if not self.actions:
            self.actions.append("Identique")

class ReasoningEngine:
    def compare_grids(self, objects_input, objects_output, grid_out):
        transformations = []
        available_outputs = list(objects_output)
        grid_h = len(grid_out)
        grid_w = len(grid_out[0]) if grid_h > 0 else 0
        
        # 1. Matching et détection de transformations
        for obj_in in objects_input:
            for obj_out in available_outputs[:]:
                # On utilise shape_signature (qui est invariant par translation) pour le matching
                # Note: get_all_variants() doit renvoyer les signatures des rotations possibles
                if obj_out.shape_signature in obj_in.get_all_variants():
                    
                    # C'est ici que la classe Transformation fait le travail d'analyse
                    trans = Transformation(obj_in, obj_out)
                    transformations.append(trans)
                    
                    # --- Détection de Remplissage (Holes) ---
                    # On calcule le delta pour savoir où regarder dans la grille de sortie
                    x1, y1 = obj_in.top_left
                    x2, y2 = obj_out.top_left
                    dx, dy = x2 - x1, y2 - y1
                    
                    holes = obj_in.get_internal_holes(grid_w, grid_h, offset=(dx, dy))
                    if holes:
                        hole_colors = [grid_out[y][x] for x, y in holes if 0 <= y < grid_h and 0 <= x < grid_w and grid_out[y][x] != 0]
                        if hole_colors:
                            # On prend la couleur dominante ou la première trouvée
                            fill_id = hole_colors[0] 
                            fill_name = ARC_COLORS.get(fill_id, str(fill_id))
                            obj_color_name = ARC_COLORS.get(obj_in.color_code, str(obj_in.color_code))
                            
                            transformations.append(f"Règle : L'intérieur de {obj_color_name} est rempli en {fill_name}.")
                            
                            # Nettoyage : Si le remplissage a créé des "objets" parasites dans available_outputs, on les vire
                            for out_item in available_outputs[:]:
                                # Si un objet de sortie est composé entièrement de pixels qui sont des trous remplis
                                if all((c, r) in holes for r, c in out_item.cells): # Attention à l'ordre r,c selon ta classe
                                    if out_item in available_outputs:
                                        available_outputs.remove(out_item)

                    if obj_out in available_outputs:
                        available_outputs.remove(obj_out)
                    break

        # 2. Objets vraiment orphelins (Apparition spontanée)
        for obj_out in available_outputs:
            color_name = ARC_COLORS.get(obj_out.color_code, str(obj_out.color_code))
            transformations.append(f"Nouvel objet {obj_out.shape} ({color_name}) apparu.")

        return transformations