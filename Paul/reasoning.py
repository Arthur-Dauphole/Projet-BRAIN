import numpy as np

# English Colors
COLOR_NAMES = {
    0: 'Black', 1: 'Blue', 2: 'Red', 3: 'Green', 4: 'Yellow',
    5: 'Grey', 6: 'Magenta', 7: 'Orange', 8: 'Azure', 9: 'Maroon'
}

class ReasoningEngine:
    def __init__(self):
        pass

    def get_object_name(self, obj):
        """Tente de nommer la forme via l'objet GeometryObject."""
        pixels = obj.cells 
        if not pixels: return "Nothing"
        
        rs = [p[0] for p in pixels]
        cs = [p[1] for p in pixels]
        h = max(rs) - min(rs) + 1
        w = max(cs) - min(cs) + 1
        area = len(pixels)
        
        if h == w and area == h*w: return "Square"
        if h != w and area == h*w: 
            if h==1 or w==1: return "Line"
            return "Rectangle"
        
        return "Arbitrary Shape"

    def normalize(self, pixels):
        """Ramène la forme en (0,0)."""
        if not pixels: return []
        min_r = min(p[0] for p in pixels)
        min_c = min(p[1] for p in pixels)
        return sorted([(r - min_r, c - min_c) for r, c in pixels])

    def detect_transformation(self, obj_in, obj_out):
        """Compare deux GeometryObjects et retourne description EN ANGLAIS."""
        
        c_in_name = COLOR_NAMES.get(obj_in.color_code, 'Unknown')
        c_out_name = COLOR_NAMES.get(obj_out.color_code, 'Unknown')
        name = self.get_object_name(obj_in)
        
        shape_in = self.normalize(obj_in.cells)
        shape_out = self.normalize(obj_out.cells)
        
        # 1. IDENTICAL
        if shape_in == shape_out and obj_in.cells == obj_out.cells and obj_in.color_code == obj_out.color_code:
            return f"{c_in_name} {name} stayed identical."

        # 2. COLOR CHANGE
        if shape_in == shape_out and obj_in.cells == obj_out.cells:
            return f"{c_in_name} {name} became {c_out_name} (Color)."

        # 3. TRANSLATION
        if shape_in == shape_out:
            r_in, c_in = min(list(obj_in.cells))
            r_out, c_out = min(list(obj_out.cells))
            dr, dc = r_out - r_in, c_out - c_in
            return f"{c_in_name} {name} underwent Translation ({dr:+d}, {dc:+d})."

        # 4. ROTATION
        pixels_in = list(obj_in.cells)
        for angle in [90, 180, 270]:
            rotated = []
            for r, c in pixels_in:
                rel_r, rel_c = r, c 
                if angle == 90: nr, nc = c, -r
                elif angle == 180: nr, nc = -r, -c
                elif angle == 270: nr, nc = -c, r
                rotated.append((nr, nc))
            
            if self.normalize(rotated) == shape_out:
                return f"{c_in_name} {name} underwent Rotation {angle}°."

        # 5. SYMMETRY
        sym_v = self.normalize([(p[0], -p[1]) for p in pixels_in])
        if sym_v == shape_out:
            return f"{c_in_name} {name} underwent Vertical Symmetry."
            
        sym_h = self.normalize([(-p[0], p[1]) for p in pixels_in])
        if sym_h == shape_out:
            return f"{c_in_name} {name} underwent Horizontal Symmetry."

        # 6. FILL (OLD LOGIC - kept for simple cases)
        set_in = obj_in.cells
        set_out = obj_out.cells
        if set_in.issubset(set_out) and len(set_out) > len(set_in):
            return f"{c_in_name} {name} was filled (Fill)."

        return f"{c_in_name} {name} changed into a new shape."

    def compare_grids(self, objects_in, objects_out, grid_out):
        log = []
        matched_indices = set()
        
        # Stockage des paires matchées pour analyse ultérieure (Remplissage)
        matched_pairs = [] # Liste de tuples (obj_in, obj_out)
        
        # 1. MATCHING DES OBJETS EXISTANTS
        for obj_in in objects_in:
            best_match = None
            min_dist = 9999
            
            cin_r = sum(p[0] for p in obj_in.cells)/len(obj_in.cells)
            cin_c = sum(p[1] for p in obj_in.cells)/len(obj_in.cells)
            
            for i, obj_out in enumerate(objects_out):
                if i in matched_indices: continue
                
                cout_r = sum(p[0] for p in obj_out.cells)/len(obj_out.cells)
                cout_c = sum(p[1] for p in obj_out.cells)/len(obj_out.cells)
                
                dist = (cin_r - cout_r)**2 + (cin_c - cout_c)**2
                if obj_in.color_code == obj_out.color_code: dist -= 5
                
                # Bonus inclusion
                if obj_in.cells.issubset(obj_out.cells): dist -= 10
                
                if dist < min_dist:
                    min_dist = dist
                    best_match = i
            
            if best_match is not None:
                matched_indices.add(best_match)
                obj_out_match = objects_out[best_match]
                
                # On stocke la paire
                matched_pairs.append((obj_in, obj_out_match))
                
                # On détecte la transfo basique
                desc = self.detect_transformation(obj_in, obj_out_match)
                log.append(desc)
            else:
                c_name = COLOR_NAMES.get(obj_in.color_code, 'Unknown')
                log.append(f"{c_name} object disappeared.")
        
        # 2. ANALYSE DES NOUVEAUX OBJETS (ET DÉTECTION DU REMPLISSAGE COMPLEXE)
        grid_h, grid_w = len(grid_out), len(grid_out[0])
        
        for i, obj_out in enumerate(objects_out):
            if i not in matched_indices:
                # C'est un nouvel objet (ex: le Gris)
                is_filling = False
                
                # On regarde s'il rentre dans les TROUS d'un objet existant matché (ex: le Magenta)
                for m_in, m_out in matched_pairs:
                    # On calcule les trous de l'objet contenant (Output)
                    holes = set(m_out.get_internal_holes(grid_w, grid_h))
                    
                    # Si le nouvel objet est DANS les trous
                    if obj_out.cells.issubset(holes):
                        container_name = COLOR_NAMES.get(m_out.color_code, 'Unknown')
                        content_name = COLOR_NAMES.get(obj_out.color_code, 'Unknown')
                        
                        log.append(f"{container_name} object was filled with {content_name} (Fill).")
                        is_filling = True
                        break
                
                if not is_filling:
                    c_name = COLOR_NAMES.get(obj_out.color_code, 'Unknown')
                    log.append(f"New {c_name} object appeared.")
                
        return log