"""
Reasoning engine to detect transformations between input and output states.
"""
import numpy as np

class Transformation:
    def __init__(self, obj_in, obj_out):
        self.obj_in = obj_in
        self.obj_out = obj_out
        self.actions = []
        self.analyze()

    def analyze(self):
        # 1. Détection de Translation
        x1, y1 = self.obj_in.top_left
        x2, y2 = self.obj_out.top_left
        dx, dy = x2 - x1, y2 - y1
        
        if dx != 0 or dy != 0:
            self.actions.append(f"Translation de ({dx}, {dy})")

        # 2. Détection de Changement de Couleur
        c1 = self.obj_in.color_code
        c2 = self.obj_out.color_code
        
        if c1 != c2:
            self.actions.append(f"Changement couleur ({c1} -> {c2})")

        if not self.actions:
            self.actions.append("Identique")

class ReasoningEngine:
    def compare_grids(self, objects_input, objects_output):
        """
        Tente de relier les objets de l'entrée à ceux de la sortie.
        Stratégie simple : On relie les objets qui ont la même 'shape_signature'.
        """
        transformations = []
        
        # Pour chaque objet de l'entrée...
        for obj_in in objects_input:
            match_found = False
            
            # ... on cherche un objet correspondant dans la sortie
            for obj_out in objects_output:
                # Si la forme est géométriquement identique
                if obj_in.shape_signature == obj_out.shape_signature:
                    # On crée un lien (Transformation)
                    trans = Transformation(obj_in, obj_out)
                    transformations.append(trans)
                    match_found = True
                    break # On arrête de chercher pour cet objet (Hypothèse simplifiée)
            
            if not match_found:
                transformations.append(f"Objet {obj_in.shape} (Couleur {obj_in.color_code}) a disparu.")

        return transformations