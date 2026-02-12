"""
SYSTÈME HYBRIDE DE RÉSOLUTION DE GRILLES ARC
- Combine l'approche simple pour les problèmes basiques
- Utilise l'approche avancée pour les problèmes complexes
- Garde la compatibilité avec tous les modules existants
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
import copy
from scipy import ndimage

# ============================================
# PARTIE 1: DÉTECTEUR DE FORMES GÉOMÉTRIQUES
# ============================================

class GeometryDetector:
    """Détecte tous les objets géométriques présents dans une grille."""
    
    def __init__(self):
        self.grid = None
        self.objects = []
        
    def load_grid(self, grid):
        """Charge une grille pour l'analyser."""
        self.grid = np.array(grid)
        self.objects = []
        
    def extract_objects(self):
        """
        Extrait tous les objets distincts de la grille.
        Un objet = pixels de même couleur connectés (8-connectivité).
        
        Returns:
            list: Liste de (color, positions) pour chaque objet
        """
        objects = []
        
        # Pour chaque couleur non-noire
        for color in range(1, 10):
            # Créer un masque binaire pour cette couleur
            mask = (self.grid == color).astype(int)
            
            if not mask.any():
                continue
            
            # Trouver les composantes connexes (8-connectivité)
            labeled, num_features = ndimage.label(mask, structure=np.ones((3, 3)))
            
            # Pour chaque composante
            for obj_id in range(1, num_features + 1):
                positions = np.argwhere(labeled == obj_id)
                objects.append({
                    'color': color,
                    'positions': positions,
                    'size': len(positions)
                })
        
        return objects
    
    def detect_all(self, grid):
        """
        Détecte tous les objets géométriques dans une grille.
        
        Returns:
            dict: Résumé de l'analyse avec tous les objets détectés
        """
        self.load_grid(grid)
        
        # Extraire tous les objets
        objects = self.extract_objects()
        
        if not objects:
            return {
                "total_objects": 0,
                "objects": [],
                "summary": "Grille vide"
            }
        
        # Classifier chaque objet
        classified_objects = []
        for obj in objects:
            classified = self.classify_object(obj)
            classified_objects.append(classified)
        
        # Créer un résumé
        type_counts = Counter([obj['type'] for obj in classified_objects])
        
        return {
            "total_objects": len(classified_objects),
            "objects": classified_objects,
            "summary": dict(type_counts),
            "colors_used": list(set(obj['color'] for obj in classified_objects))
        }
    
    def classify_object(self, obj):
        """Classifie un objet selon sa géométrie."""
        positions = obj['positions']
        color = obj['color']
        
        result = {
            'color': color,
            'size': obj['size']
        }
        
        # Essayer segment
        segment = self.detect_line_segment(positions)
        if segment:
            result['type'] = 'segment'
            result.update(segment)
            return result
        
        # Essayer rectangle
        rectangle = self.detect_rectangle(positions)
        if rectangle:
            result['type'] = rectangle['shape']
            result.update(rectangle)
            return result
        
        # Par défaut : blob/cluster
        min_row = positions[:, 0].min()
        max_row = positions[:, 0].max()
        min_col = positions[:, 1].min()
        max_col = positions[:, 1].max()
        
        result['type'] = 'blob'
        result['bounding_box'] = {
            'top_left': (min_row, min_col),
            'bottom_right': (max_row, max_col),
            'height': max_row - min_row + 1,
            'width': max_col - min_col + 1
        }
        
        return result
    
    def detect_line_segment(self, positions):
        """Détecte si un ensemble de positions forme un segment de ligne."""
        if len(positions) < 2:
            return None
        
        # Vérifier alignement horizontal
        rows = positions[:, 0]
        if len(set(rows)) == 1:
            cols = sorted(positions[:, 1])
            if all(cols[i+1] - cols[i] == 1 for i in range(len(cols)-1)):
                return {
                    "orientation": "horizontal",
                    "length": len(positions),
                    "start": tuple(positions[0]),
                    "end": tuple(positions[-1])
                }
        
        # Vérifier alignement vertical
        cols = positions[:, 1]
        if len(set(cols)) == 1:
            rows = sorted(positions[:, 0])
            if all(rows[i+1] - rows[i] == 1 for i in range(len(rows)-1)):
                return {
                    "orientation": "vertical",
                    "length": len(positions),
                    "start": tuple(positions[0]),
                    "end": tuple(positions[-1])
                }
        
        return None
    
    def detect_rectangle(self, positions):
        """Détecte si c'est un rectangle ou un carré."""
        if len(positions) < 4:
            return None
        
        min_row = positions[:, 0].min()
        max_row = positions[:, 0].max()
        min_col = positions[:, 1].min()
        max_col = positions[:, 1].max()
        
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        
        # Rectangle plein
        expected_size = height * width
        if len(positions) == expected_size:
            # Vérifier que c'est bien un rectangle plein
            rect_positions = set()
            for r in range(min_row, max_row + 1):
                for c in range(min_col, max_col + 1):
                    rect_positions.add((r, c))
            
            actual_positions = set(map(tuple, positions))
            if rect_positions == actual_positions:
                is_square = (height == width)
                return {
                    "shape": "square" if is_square else "rectangle",
                    "filled": True,
                    "height": height,
                    "width": width,
                    "top_left": (min_row, min_col),
                    "bottom_right": (max_row, max_col)
                }
        
        return None

# ============================================
# PARTIE 2: GESTION DES COULEURS ET VISUALISATION
# ============================================

class ColorMapper:
    """Mappe les codes de couleur (0-9) vers des couleurs hexadécimales."""
    COLOR_MAP = {
        0: "#000000",  # Noir (fond)
        1: "#0074D9",  # Bleu
        2: "#FF4136",  # Rouge
        3: "#2ECC40",  # Vert
        4: "#FFDC00",  # Jaune
        5: "#AAAAAA",  # Gris
        6: "#F012BE",  # Magenta
        7: "#FF851B",  # Orange
        8: "#7FDBFF",  # Cyan
        9: "#870C25",  # Marron
    }
    
    @staticmethod
    def hex(color_code: int) -> str:
        return ColorMapper.COLOR_MAP.get(color_code, "#FFFFFF")
    
    @staticmethod
    def name(color_code: int) -> str:
        names = {
            0: "Noir", 1: "Bleu", 2: "Rouge", 3: "Vert", 4: "Jaune",
            5: "Gris", 6: "Magenta", 7: "Orange", 8: "Cyan", 9: "Marron"
        }
        return names.get(color_code, f"Couleur {color_code}")

def grid_to_rgb(grid: List[List[int]]) -> np.ndarray:
    """Convertit une grille en tableau RGB."""
    h, w = len(grid), len(grid[0])
    rgb_array = np.zeros((h, w, 3))
    
    for y in range(h):
        for x in range(w):
            hex_color = ColorMapper.hex(grid[y][x])
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) / 255.0 
                       for i in (0, 2, 4))
            rgb_array[y, x] = rgb
    
    return rgb_array

# ============================================
# PARTIE 3: EXTRACTION DE RÈGLES HYBRIDE
# ============================================

class AdvancedGridRuleExtractor:
    """Extrait les règles de transformation en combinant approche simple et avancée."""
    
    def __init__(self, use_shape_detection: bool = True, mode: str = 'auto'):
        """
        Args:
            use_shape_detection: Si True, utilise la détection de formes
            mode: 'simple', 'advanced', ou 'auto' (choix automatique)
        """
        self.color_changes = {}  # {old_color: new_color}
        self.connections = {}    # {color: type} où type = 'H', 'V', ou 'D'
        self.translations = {}   # {color: (dx, dy)}
        self.propagations = {}   # règles de propagation
        self.position_based_changes = {}  # changements basés sur la position
        self.color_pairs = set()  # stocke toutes les paires de couleurs
        self.use_shape_detection = use_shape_detection
        self.mode = mode
        
        if use_shape_detection:
            self.shape_detector = GeometryDetector()
        else:
            self.shape_detector = None
    
    def extract_from_examples(self, train_examples: List[Dict]) -> None:
        """Extrait les règles à partir des exemples d'entraînement."""
        print("=== Extraction des règles depuis les exemples ===")
        
        # Réinitialiser les collections
        self.color_pairs = set()
        self.color_changes = {}
        self.connections = {}
        self.translations = {}
        self.propagations = {}
        self.position_based_changes = {}
        
        # Si mode auto, décider quelle approche utiliser
        if self.mode == 'auto':
            self.mode = self._detect_best_mode(train_examples)
            print(f"Mode sélectionné automatiquement: {self.mode}")
        
        # Appliquer l'approche sélectionnée
        if self.mode == 'simple':
            self._simple_extract_from_examples(train_examples)
        else:  # 'advanced'
            self._advanced_extract_from_examples(train_examples)
    
    def _detect_best_mode(self, train_examples: List[Dict]) -> str:
        """Détecte automatiquement le meilleur mode pour ce problème."""
        if not train_examples:
            return 'simple'
        
        # Analyser la complexité des exemples
        total_pixels = 0
        color_changes = 0
        has_translations = False
        has_propagations = False
        
        for example in train_examples:
            input_grid = example["input"]
            output_grid = example["output"]
            h, w = len(input_grid), len(input_grid[0])
            
            total_pixels += h * w
            
            # Compter les changements de couleur
            for i in range(h):
                for j in range(w):
                    if input_grid[i][j] != output_grid[i][j]:
                        color_changes += 1
            
            # Détecter les translations simples
            if self._has_simple_translation(input_grid, output_grid):
                has_translations = True
            
            # Détecter les propagations
            if self._has_propagation(input_grid, output_grid):
                has_propagations = True
        
        # Règles de décision
        if color_changes / total_pixels < 0.3 and not has_translations and not has_propagations:
            return 'simple'
        else:
            return 'advanced'
    
    def _has_simple_translation(self, input_grid, output_grid):
        """Détecte les translations simples (tous les pixels décalés de la même manière)."""
        h, w = len(input_grid), len(input_grid[0])
        
        input_points = defaultdict(set)
        output_points = defaultdict(set)
        
        for i in range(h):
            for j in range(w):
                c_in = input_grid[i][j]
                c_out = output_grid[i][j]
                if c_in != 0:
                    input_points[c_in].add((i, j))
                if c_out != 0:
                    output_points[c_out].add((i, j))
        
        for color in input_points:
            if color in output_points:
                if len(input_points[color]) == len(output_points[color]):
                    return True
        return False
    
    def _has_propagation(self, input_grid, output_grid):
        """Détecte si il y a des propagations (plus de pixels en sortie qu'en entrée)."""
        h, w = len(input_grid), len(input_grid[0])
        
        non_zero_input = sum(1 for i in range(h) for j in range(w) if input_grid[i][j] != 0)
        non_zero_output = sum(1 for i in range(h) for j in range(w) if output_grid[i][j] != 0)
        
        return non_zero_output > non_zero_input * 1.5  # 50% de plus
    
    def _simple_extract_from_examples(self, train_examples: List[Dict]) -> None:
        """Approche simple (ancien code)."""
        print("Utilisation de l'approche simple...")
        
        for idx, example in enumerate(train_examples):
            print(f"\n--- Analyse de l'exemple {idx+1} ---")
            
            input_grid = example["input"]
            output_grid = example["output"]
            
            # Détecter les changements de couleur
            self._simple_extract_color_changes(input_grid, output_grid)
            
            # Pour chaque couleur présente en sortie
            self._simple_analyze_transformations(input_grid, output_grid)
    
    def _simple_extract_color_changes(self, input_grid: List[List[int]], 
                                     output_grid: List[List[int]]) -> None:
        """Extrait les changements de couleur (même position)."""
        h = len(input_grid)
        w = len(input_grid[0])
        
        for i in range(h):
            for j in range(w):
                old_color = input_grid[i][j]
                new_color = output_grid[i][j]
                
                if old_color != 0 and old_color != new_color:
                    if old_color in self.color_changes:
                        if self.color_changes[old_color] != new_color:
                            print(f"  Attention: conflit pour {old_color} -> {self.color_changes[old_color]} ou {new_color}")
                    self.color_changes[old_color] = new_color
                    print(f"  Changement couleur: {old_color} -> {new_color}")
    
    def _simple_analyze_transformations(self, input_grid: List[List[int]], 
                                      output_grid: List[List[int]]) -> None:
        """Analyse les transformations (connexions et translations)."""
        h = len(input_grid)
        w = len(input_grid[0])
        
        input_points_by_color = defaultdict(set)
        output_points_by_color = defaultdict(set)
        
        for i in range(h):
            for j in range(w):
                c_in = input_grid[i][j]
                c_out = output_grid[i][j]
                if c_in != 0:
                    input_points_by_color[c_in].add((i, j))
                if c_out != 0:
                    output_points_by_color[c_out].add((i, j))
        
        for color_out in output_points_by_color.keys():
            possible_input_colors = []
            
            for old_color, new_color in self.color_changes.items():
                if new_color == color_out:
                    possible_input_colors.append(old_color)
            
            if color_out not in self.color_changes.values():
                possible_input_colors.append(color_out)
            
            input_points_for_color = set()
            for c_in in possible_input_colors:
                if c_in in input_points_by_color:
                    input_points_for_color.update(input_points_by_color[c_in])
            
            output_points = output_points_by_color[color_out]
            
            if not input_points_for_color:
                continue
            
            if not self._simple_try_detect_translation(input_points_for_color, 
                                                     output_points, color_out):
                self._simple_try_detect_connection(input_points_for_color,
                                                 output_points, color_out)
    
    def _simple_try_detect_translation(self, input_points: Set[Tuple[int, int]],
                                     output_points: Set[Tuple[int, int]], 
                                     color: int) -> bool:
        """Détecte si c'est une translation. Retourne True si détectée."""
        if len(input_points) != len(output_points) or not input_points:
            return False
        
        ref_in = next(iter(input_points))
        possible_displacements = set()
        
        for p_out in output_points:
            dx = p_out[0] - ref_in[0]
            dy = p_out[1] - ref_in[1]
            possible_displacements.add((dx, dy))
        
        for dx, dy in possible_displacements:
            match_all = True
            for p_in in input_points:
                moved = (p_in[0] + dx, p_in[1] + dy)
                if moved not in output_points:
                    match_all = False
                    break
            
            if match_all:
                for p_out in output_points:
                    original = (p_out[0] - dx, p_out[1] - dy)
                    if original not in input_points:
                        match_all = False
                        break
            
            if match_all:
                self.translations[color] = (dx, dy)
                print(f"  Translation pour {color}: ({dx}, {dy})")
                return True
        
        return False
    
    def _simple_try_detect_connection(self, input_points: Set[Tuple[int, int]],
                                    output_points: Set[Tuple[int, int]],
                                    color: int) -> None:
        """Détecte le type de connexion entre les points."""
        base_points = input_points.intersection(output_points)
        added_points = output_points - input_points
        
        if not added_points or len(base_points) < 2:
            return
        
        base_list = list(base_points)
        for i in range(len(base_list)):
            for j in range(i+1, len(base_list)):
                p1 = base_list[i]
                p2 = base_list[j]
                
                if p1[0] == p2[0]:  # Horizontal
                    min_y = min(p1[1], p2[1])
                    max_y = max(p1[1], p2[1])
                    line_points = {(p1[0], y) for y in range(min_y, max_y+1)}
                    
                    if line_points.issubset(output_points):
                        self.connections[color] = 'H'
                        print(f"  Connexion horizontale pour {color}")
                        return
                
                elif p1[1] == p2[1]:  # Vertical
                    min_x = min(p1[0], p2[0])
                    max_x = max(p1[0], p2[0])
                    line_points = {(x, p1[1]) for x in range(min_x, max_x+1)}
                    
                    if line_points.issubset(output_points):
                        self.connections[color] = 'V'
                        print(f"  Connexion verticale pour {color}")
                        return
    
    def _advanced_extract_from_examples(self, train_examples: List[Dict]) -> None:
        """Approche avancée (nouveau code)."""
        print("Utilisation de l'approche avancée...")
        
        # Collecter toutes les paires de couleurs
        print("\nPhase 1: Collecte des paires de couleurs...")
        for idx, example in enumerate(train_examples):
            input_grid = example["input"]
            output_grid = example["output"]
            h, w = len(input_grid), len(input_grid[0])
            
            for i in range(h):
                for j in range(w):
                    old_color = input_grid[i][j]
                    new_color = output_grid[i][j]
                    if old_color != 0 and old_color != new_color:
                        self.color_pairs.add((old_color, new_color))
        
        # Construire la bijection globale
        print("\nPhase 2: Construction de la bijection...")
        self._build_color_mapping_advanced()
        
        # Analyser les transformations
        print("\nPhase 3: Analyse des transformations...")
        for idx, example in enumerate(train_examples):
            input_grid = example["input"]
            output_grid = example["output"]
            
            # Vérifier si c'est un pattern par colonne
            h, w = len(input_grid), len(input_grid[0])
            is_column_pattern = True
            for i in range(1, h):
                if input_grid[i] != input_grid[0] or output_grid[i] != output_grid[0]:
                    is_column_pattern = False
                    break
            
            if is_column_pattern:
                print(f"  Exemple {idx+1}: Pattern par colonne détecté")
                # Stocker les transformations par colonne comme référence
                for j in range(w):
                    old_color = input_grid[0][j]
                    new_color = output_grid[0][j]
                    if old_color != new_color:
                        self.position_based_changes[('column', j)] = (old_color, new_color)
            
            # Analyser les autres transformations
            self._advanced_analyze_transformations(input_grid, output_grid)
    
    def _build_color_mapping_advanced(self):
        """Construit le mapping de couleurs avancé."""
        if not self.color_pairs:
            return
            
        print("\n  Construction du mapping de couleurs avancé...")
        
        # Détecter les paires réciproques (bijections)
        forward_pairs = set(self.color_pairs)
        backward_pairs = set((new, old) for (old, new) in self.color_pairs)
        reciprocal_pairs = forward_pairs.intersection(backward_pairs)
        
        # Si plus de la moitié des paires sont réciproques, c'est probablement une bijection
        if len(reciprocal_pairs) >= len(self.color_pairs) / 2:
            print("  Pattern bijectif détecté! Construction de paires réciproques...")
            
            # Créer des paires uniques
            unique_pairs = set()
            for old, new in self.color_pairs:
                # Ne garder que la version "croissante" pour éviter les doublons
                if (new, old) not in unique_pairs:
                    unique_pairs.add((old, new))
            
            # Construire un mapping bidirectionnel
            self.color_changes = {}
            for old, new in unique_pairs:
                self.color_changes[old] = new
                # Si c'est une bijection, on ajoute aussi l'inverse
                if (new, old) in self.color_pairs:
                    self.color_changes[new] = old
        else:
            # Sinon, utiliser l'approche par fréquence avec résolution de conflits
            print("  Utilisation de l'approche par fréquence avec résolution de conflits...")
            pair_counts = {}
            for old, new in self.color_pairs:
                pair_counts[(old, new)] = pair_counts.get((old, new), 0) + 1
            
            # Trier par fréquence (plus fréquent d'abord)
            sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Construire le mapping avec résolution de conflits
            forward_map = {}
            backward_map = {}
            
            for (old, new), count in sorted_pairs:
                # Vérifier les conflits
                if old in forward_map:
                    existing_new = forward_map[old]
                    existing_count = pair_counts.get((old, existing_new), 0)
                    
                    if count > existing_count:
                        # Remplacer par la transformation plus fréquente
                        print(f"  Résolution conflit: {old} -> {new} (fréquence {count}) au lieu de {old} -> {existing_new} (fréquence {existing_count})")
                        forward_map[old] = new
                        # Mettre à jour backward_map
                        if existing_new in backward_map and backward_map[existing_new] == old:
                            del backward_map[existing_new]
                        backward_map[new] = old
                else:
                    forward_map[old] = new
                    if new not in backward_map:
                        backward_map[new] = old
            
            self.color_changes = forward_map
        
        # Afficher le mapping final
        print("  Mapping final des couleurs:")
        for old, new in sorted(self.color_changes.items()):
            print(f"    {old} -> {new}")
    
    def _advanced_analyze_transformations(self, input_grid: List[List[int]], 
                                         output_grid: List[List[int]]) -> None:
        """Analyse les transformations avec détection de propagations."""
        h = len(input_grid)
        w = len(input_grid[0])
        
        input_points_by_color = defaultdict(set)
        output_points_by_color = defaultdict(set)
        
        # Collecter les points par couleur
        for i in range(h):
            for j in range(w):
                c_in = input_grid[i][j]
                c_out = output_grid[i][j]
                if c_in != 0:
                    input_points_by_color[c_in].add((i, j))
                if c_out != 0:
                    output_points_by_color[c_out].add((i, j))
        
        # Analyser chaque couleur en sortie
        for color_out in output_points_by_color.keys():
            possible_input_colors = []
            
            # Chercher les couleurs d'entrée qui peuvent devenir cette couleur de sortie
            for old_color, new_color in self.color_changes.items():
                if new_color == color_out:
                    possible_input_colors.append(old_color)
            
            # Si cette couleur de sortie n'est pas le résultat d'un changement,
            # elle peut provenir de la même couleur d'entrée
            if color_out not in self.color_changes.values():
                possible_input_colors.append(color_out)
            
            input_points_for_color = set()
            for c_in in possible_input_colors:
                if c_in in input_points_by_color:
                    input_points_for_color.update(input_points_by_color[c_in])
            
            output_points = output_points_by_color[color_out]
            
            if not input_points_for_color:
                continue
            
            # Essayer de détecter une propagation avant une translation simple
            propagation_detected = self._try_detect_propagation(input_points_for_color, output_points, color_out)
            
            # Essayer de détecter une translation
            translation_detected = self._try_detect_translation(input_points_for_color, output_points, color_out)
            
            # Si aucune propagation ni translation, essayer les connexions
            if not propagation_detected and not translation_detected:
                self._try_detect_connection(input_points_for_color, output_points, color_out)
    
    def _try_detect_propagation(self, input_points: Set[Tuple[int, int]],
                               output_points: Set[Tuple[int, int]], 
                               color: int) -> bool:
        """Détecte les propagations (horizontales et verticales)."""
        if not input_points or not output_points:
            return False
        
        # Convertir en listes pour analyse
        input_list = list(input_points)
        output_list = list(output_points)
        
        # Regrouper par ligne et colonne
        input_by_row = defaultdict(list)
        input_by_col = defaultdict(list)
        
        for i, j in input_list:
            input_by_row[i].append(j)
            input_by_col[j].append(i)
        
        output_by_row = defaultdict(list)
        output_by_col = defaultdict(list)
        
        for i, j in output_list:
            output_by_row[i].append(j)
            output_by_col[j].append(i)
        
        propagation_detected = False
        
        # Détecter les propagations horizontales
        for row, input_cols in input_by_row.items():
            if row in output_by_row:
                output_cols = output_by_row[row]
                if len(output_cols) > len(input_cols):
                    # Vérifier si c'est une propagation complète
                    input_cols_sorted = sorted(input_cols)
                    output_cols_sorted = sorted(output_cols)
                    
                    # Calculer la plage
                    min_col = min(output_cols_sorted)
                    max_col = max(output_cols_sorted)
                    
                    # Vérifier si tous les pixels intermédiaires sont remplis
                    if all(col in output_cols for col in range(min_col, max_col + 1)):
                        # Propagation horizontale détectée
                        self.propagations[(color, 'horizontal', row)] = {
                            'type': 'horizontal',
                            'color': color,
                            'row': row,
                            'full_range': (min_col, max_col)
                        }
                        print(f"  Propagation horizontale pour couleur {color}, ligne {row}")
                        propagation_detected = True
        
        # Détecter les propagations verticales
        for col, input_rows in input_by_col.items():
            if col in output_by_col:
                output_rows = output_by_col[col]
                if len(output_rows) > len(input_rows):
                    # Vérifier si c'est une propagation complète
                    input_rows_sorted = sorted(input_rows)
                    output_rows_sorted = sorted(output_rows)
                    
                    # Calculer la plage
                    min_row = min(output_rows_sorted)
                    max_row = max(output_rows_sorted)
                    
                    # Vérifier si tous les pixels intermédiaires sont remplis
                    if all(row in output_rows for row in range(min_row, max_row + 1)):
                        # Propagation verticale détectée
                        self.propagations[(color, 'vertical', col)] = {
                            'type': 'vertical',
                            'color': color,
                            'col': col,
                            'full_range': (min_row, max_row)
                        }
                        print(f"  Propagation verticale pour couleur {color}, colonne {col}")
                        propagation_detected = True
        
        return propagation_detected
    
    def _try_detect_translation(self, input_points: Set[Tuple[int, int]],
                               output_points: Set[Tuple[int, int]], 
                               color: int) -> bool:
        """Détecte si c'est une translation. Retourne True si détectée."""
        if len(input_points) != len(output_points) or not input_points:
            return False
        
        ref_in = next(iter(input_points))
        possible_displacements = set()
        
        for p_out in output_points:
            dx = p_out[0] - ref_in[0]
            dy = p_out[1] - ref_in[1]
            possible_displacements.add((dx, dy))
        
        for dx, dy in possible_displacements:
            match_all = True
            for p_in in input_points:
                moved = (p_in[0] + dx, p_in[1] + dy)
                if moved not in output_points:
                    match_all = False
                    break
            
            if match_all:
                for p_out in output_points:
                    original = (p_out[0] - dx, p_out[1] - dy)
                    if original not in input_points:
                        match_all = False
                        break
            
            if match_all:
                self.translations[color] = (dx, dy)
                print(f"  Translation pour {color}: ({dx}, {dy})")
                return True
        
        return False
    
    def _try_detect_connection(self, input_points: Set[Tuple[int, int]],
                              output_points: Set[Tuple[int, int]],
                              color: int) -> None:
        """Détecte TOUS les types de connexion entre les points."""
        base_points = input_points.intersection(output_points)
        added_points = output_points - input_points
        
        if not added_points or len(base_points) < 2:
            return
        
        base_list = list(base_points)
        for i in range(len(base_list)):
            for j in range(i+1, len(base_list)):
                p1 = base_list[i]
                p2 = base_list[j]
                
                # Détection horizontale
                if p1[0] == p2[0]:  # Même ligne
                    min_y = min(p1[1], p2[1])
                    max_y = max(p1[1], p2[1])
                    line_points = {(p1[0], y) for y in range(min_y, max_y+1)}
                    
                    if line_points.issubset(output_points):
                        if color not in self.connections:
                            self.connections[color] = set()
                        self.connections[color].add('H')
                        print(f"  Connexion horizontale détectée pour couleur {color}")
                
                # Détection verticale
                if p1[1] == p2[1]:  # Même colonne
                    min_x = min(p1[0], p2[0])
                    max_x = max(p1[0], p2[0])
                    line_points = {(x, p1[1]) for x in range(min_x, max_x+1)}
                    
                    if line_points.issubset(output_points):
                        if color not in self.connections:
                            self.connections[color] = set()
                        self.connections[color].add('V')
                        print(f"  Connexion verticale détectée pour couleur {color}")
    
    def get_rules(self) -> Dict:
        """Retourne toutes les règles extraites."""
        # Convertir les sets en listes pour la sérialisation JSON
        connections_as_lists = {}
        for color, conn_set in self.connections.items():
            if isinstance(conn_set, set):
                connections_as_lists[color] = list(conn_set) if conn_set else []
            else:
                connections_as_lists[color] = conn_set
        
        return {
            "color_changes": self.color_changes,
            "connections": connections_as_lists,
            "translations": self.translations,
            "propagations": self.propagations,
            "position_based_changes": self.position_based_changes
        }


# # ============================================
# # PARTIE 3: EXTRACTION DE RÈGLES HYBRIDE (MODIFIÉ)
# # ============================================

# class AdvancedGridRuleExtractor:
#     """Extrait les règles de transformation en combinant approche simple et avancée."""
    
#     def __init__(self, use_shape_detection: bool = True, mode: str = 'auto'):
#         """
#         Args:
#             use_shape_detection: Si True, utilise la détection de formes
#             mode: 'simple', 'advanced', ou 'auto' (choix automatique)
#         """
#         self.color_changes = {}  # {old_color: new_color}
#         self.connections = {}    # {color: type} où type = 'H', 'V', ou 'D'
#         self.translations = {}   # {color: (dx, dy)}
#         self.propagations = {}   # règles de propagation
#         self.position_based_changes = {}  # changements basés sur la position
#         self.color_pairs = set()  # stocke toutes les paires de couleurs
#         self.point_connections = {}  # NOUVEAU: Stocke les paires de points à connecter
#         self.use_shape_detection = use_shape_detection
#         self.mode = mode
        
#         if use_shape_detection:
#             self.shape_detector = GeometryDetector()
#         else:
#             self.shape_detector = None
    
#     def extract_from_examples(self, train_examples: list[dict]) -> None:
#         """Extrait les règles à partir des exemples d'entraînement."""
#         print("=== Extraction des règles depuis les exemples ===")
        
#         # Réinitialiser les collections
#         self.color_pairs = set()
#         self.color_changes = {}
#         self.connections = {}
#         self.translations = {}
#         self.propagations = {}
#         self.position_based_changes = {}
#         self.point_connections = {}  # Réinitialiser
        
#         # Si mode auto, décider quelle approche utiliser
#         if self.mode == 'auto':
#             self.mode = self._detect_best_mode(train_examples)
#             print(f"Mode sélectionné automatiquement: {self.mode}")
        
#         # Appliquer l'approche sélectionnée
#         if self.mode == 'simple':
#             self._simple_extract_from_examples(train_examples)
#         else:  # 'advanced'
#             self._advanced_extract_from_examples(train_examples)
    
#     def _detect_best_mode(self, train_examples: list[dict]) -> str:
#         """Détecte automatiquement le meilleur mode pour ce problème."""
#         if not train_examples:
#             return 'simple'
        
#         # Analyser la complexité des exemples
#         total_pixels = 0
#         color_changes = 0
#         has_translations = False
#         has_propagations = False
#         has_simple_connections = False
        
#         for example in train_examples:
#             input_grid = example["input"]
#             output_grid = example["output"]
#             h, w = len(input_grid), len(input_grid[0])
            
#             total_pixels += h * w
            
#             # Compter les changements de couleur
#             for i in range(h):
#                 for j in range(w):
#                     if input_grid[i][j] != output_grid[i][j]:
#                         color_changes += 1
            
#             # Détecter les translations simples
#             if self._has_simple_translation(input_grid, output_grid):
#                 has_translations = True
            
#             # Détecter les propagations
#             if self._has_propagation(input_grid, output_grid):
#                 has_propagations = True
            
#             # Détecter les connexions simples entre points
#             if self._has_simple_connections(input_grid, output_grid):
#                 has_simple_connections = True
        
#         # Règles de décision
#         if has_simple_connections and not has_translations and not has_propagations:
#             return 'simple'
#         elif color_changes / total_pixels < 0.3 and not has_translations and not has_propagations:
#             return 'simple'
#         else:
#             return 'advanced'
    
#     def _has_simple_connections(self, input_grid, output_grid):
#         """Détecte si il y a des connexions simples entre points."""
#         h, w = len(input_grid), len(input_grid[0])
        
#         # Compter le nombre de points en entrée
#         input_points = {}
#         for i in range(h):
#             for j in range(w):
#                 color = input_grid[i][j]
#                 if color != 0:
#                     if color not in input_points:
#                         input_points[color] = []
#                     input_points[color].append((i, j))
        
#         # Vérifier pour chaque couleur si les points sont connectés en sortie
#         for color, points in input_points.items():
#             if len(points) == 2:
#                 # Deux points à connecter
#                 p1, p2 = points
#                 # Vérifier si en sortie ils sont connectés
#                 line_points = self._get_line_points(p1, p2)
#                 all_connected = True
#                 for point in line_points:
#                     i, j = point
#                     if 0 <= i < h and 0 <= j < w:
#                         if output_grid[i][j] != color and output_grid[i][j] != 0:
#                             all_connected = False
#                             break
#                     else:
#                         all_connected = False
#                         break
                
#                 if all_connected:
#                     return True
        
#         return False
    
#     def _get_line_points(self, p1, p2):
#         """Retourne tous les points entre p1 et p2 (ligne horizontale, verticale ou diagonale)."""
#         i1, j1 = p1
#         i2, j2 = p2
#         points = []
        
#         # Ligne horizontale
#         if i1 == i2:
#             start_j = min(j1, j2)
#             end_j = max(j1, j2)
#             for j in range(start_j, end_j + 1):
#                 points.append((i1, j))
        
#         # Ligne verticale
#         elif j1 == j2:
#             start_i = min(i1, i2)
#             end_i = max(i1, i2)
#             for i in range(start_i, end_i + 1):
#                 points.append((i, j1))
        
#         # Ligne diagonale (pente 1 ou -1)
#         elif abs(i1 - i2) == abs(j1 - j2):
#             step_i = 1 if i2 > i1 else -1
#             step_j = 1 if j2 > j1 else -1
#             i, j = i1, j1
#             while (i != i2 + step_i) and (j != j2 + step_j):
#                 points.append((i, j))
#                 i += step_i
#                 j += step_j
        
#         return points
    
#     def _simple_extract_from_examples(self, train_examples: list[dict]) -> None:
#         """Approche simple (ancien code)."""
#         print("Utilisation de l'approche simple...")
        
#         for idx, example in enumerate(train_examples):
#             print(f"\n--- Analyse de l'exemple {idx+1} ---")
            
#             input_grid = example["input"]
#             output_grid = example["output"]
            
#             # Détecter les changements de couleur
#             self._simple_extract_color_changes(input_grid, output_grid)
            
#             # Détecter les connexions entre points
#             self._detect_point_connections(input_grid, output_grid)
            
#             # Pour chaque couleur présente en sortie
#             self._simple_analyze_transformations(input_grid, output_grid)
    
#     def _detect_point_connections(self, input_grid: list[list[int]], 
#                                  output_grid: list[list[int]]) -> None:
#         """Détecte les connexions entre points simples."""
#         h, w = len(input_grid), len(input_grid[0])
        
#         # Pour chaque couleur
#         for color in range(1, 10):
#             input_points = []
            
#             # Collecter les points de cette couleur en entrée
#             for i in range(h):
#                 for j in range(w):
#                     if input_grid[i][j] == color:
#                         input_points.append((i, j))
            
#             # Si on a exactement 2 points, vérifier s'ils sont connectés en sortie
#             if len(input_points) == 2:
#                 p1, p2 = input_points
#                 line_points = self._get_line_points(p1, p2)
                
#                 # Vérifier si tous les points de la ligne sont de la bonne couleur en sortie
#                 is_connected = True
#                 for i, j in line_points:
#                     if 0 <= i < h and 0 <= j < w:
#                         if output_grid[i][j] != color and output_grid[i][j] != 0:
#                             is_connected = False
#                             break
#                     else:
#                         is_connected = False
#                         break
                
#                 if is_connected:
#                     self.point_connections[color] = (p1, p2)
#                     print(f"  Connexion détectée entre {p1} et {p2} pour couleur {color}")
    
#     def _simple_try_detect_connection(self, input_points: set[tuple[int, int]],
#                                     output_points: set[tuple[int, int]],
#                                     color: int) -> None:
#         """Détecte le type de connexion entre les points."""
#         base_points = input_points.intersection(output_points)
#         added_points = output_points - input_points
        
#         if not added_points or len(base_points) < 2:
#             return
        
#         base_list = list(base_points)
#         for i in range(len(base_list)):
#             for j in range(i+1, len(base_list)):
#                 p1 = base_list[i]
#                 p2 = base_list[j]
                
#                 # Vérifier si c'est une ligne horizontale
#                 if p1[0] == p2[0]:  # Même ligne
#                     min_y = min(p1[1], p2[1])
#                     max_y = max(p1[1], p2[1])
#                     line_points = {(p1[0], y) for y in range(min_y, max_y+1)}
                    
#                     if line_points.issubset(output_points):
#                         self.connections[color] = 'H'
#                         print(f"  Connexion horizontale pour {color}")
#                         return
                
#                 # Vérifier si c'est une ligne verticale
#                 elif p1[1] == p2[1]:  # Même colonne
#                     min_x = min(p1[0], p2[0])
#                     max_x = max(p1[0], p2[0])
#                     line_points = {(x, p1[1]) for x in range(min_x, max_x+1)}
                    
#                     if line_points.issubset(output_points):
#                         self.connections[color] = 'V'
#                         print(f"  Connexion verticale pour {color}")
#                         return
                
#                 # Vérifier si c'est une ligne diagonale
#                 elif abs(p1[0] - p2[0]) == abs(p1[1] - p2[1]):  # Diagonale
#                     # Générer tous les points de la diagonale
#                     step_i = 1 if p2[0] > p1[0] else -1
#                     step_j = 1 if p2[1] > p1[1] else -1
#                     current = p1
#                     line_points = set()
#                     while current != p2:
#                         line_points.add(current)
#                         current = (current[0] + step_i, current[1] + step_j)
#                     line_points.add(p2)
                    
#                     if line_points.issubset(output_points):
#                         self.connections[color] = 'D'
#                         print(f"  Connexion diagonale pour {color}")
#                         return

# ============================================
# PARTIE 4: TRANSFORMATEUR HYBRIDE (MODIFIÉ)
# ============================================

class AdvancedGridTransformer:
    """Applique les règles de transformation avec support hybride."""
    
    def __init__(self, rules: dict):
        self.color_changes = rules.get("color_changes", {})
        # Convertir les connexions
        connections = rules.get("connections", {})
        self.connections = {}
        for color, conn_type in connections.items():
            if isinstance(conn_type, list):
                self.connections[color] = set(conn_type)
            elif isinstance(conn_type, set):
                self.connections[color] = conn_type
            else:
                self.connections[color] = {conn_type}
        self.translations = rules.get("translations", {})
        self.propagations = rules.get("propagations", {})
        self.position_based_changes = rules.get("position_based_changes", {})
        self.point_connections = rules.get("point_connections", {})  # NOUVEAU
    
    def apply_rules(self, input_grid: list[list[int]]) -> list[list[int]]:
        """
        Applique les règles avec une approche hybride :
        1. Si bijection disponible, l'utiliser
        2. Sinon, essayer les transformations par position
        3. Ensuite, les propagations, connexions, translations
        """
        grid = copy.deepcopy(input_grid)
        h = len(grid)
        w = len(grid[0])
        
        # D'abord, appliquer les connexions entre points (priorité haute)
        grid = self._apply_point_connections(grid)
        
        # Étape 1: Vérifier si on a une bijection globale complète
        if self.color_changes and len(self.color_changes) >= 4:
            print("  Utilisation de la bijection globale")
            for i in range(h):
                for j in range(w):
                    old_color = grid[i][j]
                    if old_color in self.color_changes:
                        grid[i][j] = self.color_changes[old_color]
            return grid
        
        # Étape 2: Sinon, essayer les transformations par colonne
        grid = self._apply_position_based_changes(grid)
        
        # Étape 3: Autres transformations
        grid = self._apply_color_changes(grid)
        grid = self._apply_propagations(grid)
        grid = self._apply_connections(grid)
        grid = self._apply_translations(grid)
        
        return grid
    
    def _apply_point_connections(self, grid: list[list[int]]) -> list[list[int]]:
        """Applique les connexions entre points simples."""
        if not self.point_connections:
            return grid
        
        h = len(grid)
        w = len(grid[0])
        
        print(f"  Application de {len(self.point_connections)} connexions de points")
        
        for color, (p1, p2) in self.point_connections.items():
            # Vérifier si les deux points sont présents dans la grille
            i1, j1 = p1
            i2, j2 = p2
            
            # Vérifier les limites
            if (0 <= i1 < h and 0 <= j1 < w and 
                0 <= i2 < h and 0 <= j2 < w):
                
                # Vérifier si les deux points sont de la bonne couleur
                if grid[i1][j1] == color and grid[i2][j2] == color:
                    # Tracer la ligne entre les points
                    self._draw_line_between_points(grid, p1, p2, color)
                    print(f"    Connexion tracée entre {p1} et {p2} pour couleur {color}")
        
        return grid
    
    def _draw_line_between_points(self, grid: list[list[int]], 
                                p1: tuple[int, int], p2: tuple[int, int], 
                                color: int) -> None:
        """Dessine une ligne entre deux points."""
        i1, j1 = p1
        i2, j2 = p2
        
        # Ligne horizontale
        if i1 == i2:
            start_j = min(j1, j2)
            end_j = max(j1, j2)
            for j in range(start_j, end_j + 1):
                if grid[i1][j] == 0:  # Ne pas écraser les couleurs existantes
                    grid[i1][j] = color
        
        # Ligne verticale
        elif j1 == j2:
            start_i = min(i1, i2)
            end_i = max(i1, i2)
            for i in range(start_i, end_i + 1):
                if grid[i][j1] == 0:  # Ne pas écraser les couleurs existantes
                    grid[i][j1] = color
        
        # Ligne diagonale (pente 1 ou -1)
        elif abs(i1 - i2) == abs(j1 - j2):
            step_i = 1 if i2 > i1 else -1
            step_j = 1 if j2 > j1 else -1
            i, j = i1, j1
            while (i != i2 + step_i) and (j != j2 + step_j):
                if grid[i][j] == 0:  # Ne pas écraser les couleurs existantes
                    grid[i][j] = color
                i += step_i
                j += step_j

# Ajouter la clé point_connections dans get_rules de AdvancedGridRuleExtractor
    def get_rules(self) -> dict:
        """Retourne toutes les règles extraites."""
        # Convertir les sets en listes pour la sérialisation JSON
        connections_as_lists = {}
        for color, conn_set in self.connections.items():
            if isinstance(conn_set, set):
                connections_as_lists[color] = list(conn_set) if conn_set else []
            else:
                connections_as_lists[color] = conn_set
        
        return {
            "color_changes": self.color_changes,
            "connections": connections_as_lists,
            "translations": self.translations,
            "propagations": self.propagations,
            "position_based_changes": self.position_based_changes,
            "point_connections": self.point_connections  # NOUVEAU
        }

# ============================================
# PARTIE 4: TRANSFORMATEUR HYBRIDE
# ============================================

class AdvancedGridTransformer:
    """Applique les règles de transformation avec support hybride."""
    
    def __init__(self, rules: Dict):
        self.color_changes = rules.get("color_changes", {})
        # Convertir les connexions
        connections = rules.get("connections", {})
        self.connections = {}
        for color, conn_type in connections.items():
            if isinstance(conn_type, list):
                self.connections[color] = set(conn_type)
            elif isinstance(conn_type, set):
                self.connections[color] = conn_type
            else:
                self.connections[color] = {conn_type}
        self.translations = rules.get("translations", {})
        self.propagations = rules.get("propagations", {})
        self.position_based_changes = rules.get("position_based_changes", {})
    
    def apply_rules(self, input_grid: List[List[int]]) -> List[List[int]]:
        """
        Applique les règles avec une approche hybride :
        1. Si bijection disponible, l'utiliser
        2. Sinon, essayer les transformations par position
        3. Ensuite, les propagations, connexions, translations
        """
        grid = copy.deepcopy(input_grid)
        h = len(grid)
        w = len(grid[0])
        
        # Étape 1: Vérifier si on a une bijection globale complète
        if self.color_changes and len(self.color_changes) >= 4:
            print("  Utilisation de la bijection globale")
            for i in range(h):
                for j in range(w):
                    old_color = grid[i][j]
                    if old_color in self.color_changes:
                        grid[i][j] = self.color_changes[old_color]
            return grid
        
        # Étape 2: Sinon, essayer les transformations par colonne
        grid = self._apply_position_based_changes(grid)
        
        # Étape 3: Autres transformations
        grid = self._apply_color_changes(grid)
        grid = self._apply_propagations(grid)
        grid = self._apply_connections(grid)
        grid = self._apply_translations(grid)
        
        return grid
    
    def _apply_color_changes(self, grid: List[List[int]]) -> List[List[int]]:
        """Applique les changements de couleur globaux."""
        h = len(grid)
        w = len(grid[0])
        
        for i in range(h):
            for j in range(w):
                if grid[i][j] in self.color_changes:
                    grid[i][j] = self.color_changes[grid[i][j]]
        
        return grid
    
    def _apply_position_based_changes(self, grid: List[List[int]]) -> List[List[int]]:
        """Applique les changements basés sur la position (colonne ou ligne)."""
        h = len(grid)
        w = len(grid[0])
        
        # Appliquer les transformations par colonne
        for (change_type, index), (old_color, new_color) in self.position_based_changes.items():
            if change_type == 'column' and 0 <= index < w:
                # Transformer toute la colonne
                for i in range(h):
                    if grid[i][index] == old_color:
                        grid[i][index] = new_color
        
        return grid
    
    def _apply_propagations(self, grid: List[List[int]]) -> List[List[int]]:
        """Applique les règles de propagation."""
        h = len(grid)
        w = len(grid[0])
        
        # Pour chaque règle de propagation
        for key, rule in self.propagations.items():
            color = rule['color']
            rule_type = rule['type']
            
            if rule_type == 'horizontal':
                row = rule['row']
                min_col, max_col = rule['full_range']
                
                # Remplir toute la ligne entre min_col et max_col
                if 0 <= row < h:
                    for col in range(min_col, max_col + 1):
                        if 0 <= col < w and grid[row][col] == 0:
                            grid[row][col] = color
            
            elif rule_type == 'vertical':
                col = rule['col']
                min_row, max_row = rule['full_range']
                
                # Remplir toute la colonne entre min_row et max_row
                if 0 <= col < w:
                    for row in range(min_row, max_row + 1):
                        if 0 <= row < h and grid[row][col] == 0:
                            grid[row][col] = color
        
        return grid
    
    def _apply_connections(self, grid: List[List[int]]) -> List[List[int]]:
        """Applique les règles de connexion."""
        h = len(grid)
        w = len(grid[0])
        
        # Appliquer tous les types de connexion pour chaque couleur
        for color, conn_types in self.connections.items():
            # Trouver tous les points de cette couleur
            points = [(i, j) for i in range(h) for j in range(w) 
                     if grid[i][j] == color]
            
            if len(points) < 2:
                continue
            
            # Pour chaque type de connexion détecté
            for conn_type in conn_types:
                # Pour chaque paire de points
                for k in range(len(points)):
                    for l in range(k+1, len(points)):
                        p1 = points[k]
                        p2 = points[l]
                        
                        if conn_type == 'H' and p1[0] == p2[0]:  # Horizontal
                            self._draw_horizontal_line(grid, p1, p2, color)
                        
                        elif conn_type == 'V' and p1[1] == p2[1]:  # Vertical
                            self._draw_vertical_line(grid, p1, p2, color)
        
        return grid
    
    def _draw_horizontal_line(self, grid: List[List[int]], 
                            p1: Tuple[int, int], p2: Tuple[int, int], 
                            color: int) -> None:
        """Dessine une ligne horizontale entre p1 et p2."""
        row = p1[0]
        start_col = min(p1[1], p2[1])
        end_col = max(p1[1], p2[1])
        
        for col in range(start_col, end_col + 1):
            if grid[row][col] == 0:  # Remplir seulement les cases vides
                grid[row][col] = color
    
    def _draw_vertical_line(self, grid: List[List[int]], 
                          p1: Tuple[int, int], p2: Tuple[int, int], 
                          color: int) -> None:
        """Dessine une ligne verticale entre p1 et p2."""
        col = p1[1]
        start_row = min(p1[0], p2[0])
        end_row = max(p1[0], p2[0])
        
        for row in range(start_row, end_row + 1):
            if grid[row][col] == 0:  # Remplir seulement les cases vides
                grid[row][col] = color
    
    def _apply_translations(self, grid: List[List[int]]) -> List[List[int]]:
        """
        Applique les translations.
        IMPORTANT: On applique les translations à TOUS les pixels de cette couleur.
        """
        h = len(grid)
        w = len(grid[0])
        
        # Créer une nouvelle grille pour cette translation
        new_grid = [[0 for _ in range(w)] for _ in range(h)]
        
        # D'abord, copier tout ce qui n'est pas concerné par une translation
        for i in range(h):
            for j in range(w):
                color = grid[i][j]
                should_translate = False
                
                # Vérifier si cette couleur a une translation
                for trans_color, (dx, dy) in self.translations.items():
                    if color == trans_color:
                        should_translate = True
                        break
                
                if not should_translate:
                    new_grid[i][j] = color
        
        # Ensuite, appliquer les translations
        for color, (dx, dy) in self.translations.items():
            for i in range(h):
                for j in range(w):
                    if grid[i][j] == color:
                        new_i = i + dy  # Note: i = y, j = x
                        new_j = j + dx
                        
                        # Si la nouvelle position est dans la grille
                        if 0 <= new_i < h and 0 <= new_j < w:
                            new_grid[new_i][new_j] = color
                        # Sinon, le pixel sort de la grille (disparaît)
        
        return new_grid

# ============================================
# PARTIE 5: FONCTIONS UTILITAIRES
# ============================================

def load_data_from_json(filepath: str) -> Dict:
    """Charge les données depuis un fichier JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_results_to_json(results: Dict, filepath: str) -> None:
    """Sauvegarde les résultats dans un fichier JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

def solve_single_test(train_examples: List[Dict], test_input: List[List[int]], 
                     use_shape_detection: bool = True, mode: str = 'auto') -> List[List[int]]:
    """
    Résout un seul test rapidement.
    
    Args:
        train_examples: Liste d'exemples d'entraînement
        test_input: Grille d'entrée du test
        use_shape_detection: Si True, utilise la détection de formes
        mode: 'simple', 'advanced', ou 'auto'
    
    Returns:
        Grille de sortie prédite
    """
    # Extraire les règles
    extractor = AdvancedGridRuleExtractor(
        use_shape_detection=use_shape_detection,
        mode=mode
    )
    extractor.extract_from_examples(train_examples)
    rules = extractor.get_rules()
    
    # Appliquer les règles
    transformer = AdvancedGridTransformer(rules)
    output = transformer.apply_rules(test_input)
    
    return output

def batch_process(json_file: str, output_file: str = "grid_results.json", 
                  use_shape_detection: bool = True, mode: str = 'auto'):
    """
    Traite un fichier JSON en batch.
    
    Args:
        json_file: Fichier d'entrée JSON
        output_file: Fichier de sortie JSON
        use_shape_detection: Si True, utilise la détection de formes
        mode: 'simple', 'advanced', ou 'auto'
    """
    print(f"Traitement batch de '{json_file}'...")
    
    data = load_data_from_json(json_file)
    
    # Extraire les règles
    extractor = AdvancedGridRuleExtractor(
        use_shape_detection=use_shape_detection,
        mode=mode
    )
    extractor.extract_from_examples(data["train"])
    rules = extractor.get_rules()
    
    transformer = AdvancedGridTransformer(rules)
    results = []
    
    # Chercher les tests
    test_key = "Test" if "Test" in data else "test"
    
    if test_key in data:
        for test_case in data[test_key]:
            input_grid = test_case["input"]
            output_grid = transformer.apply_rules(input_grid)
            
            results.append({
                "input": input_grid,
                "output": output_grid
            })
    
    save_results_to_json({"test_results": results}, output_file)
    print(f"✓ Terminé! {len(results)} test(s) sauvegardés dans '{output_file}'")

# ============================================
# FONCTIONS POUR LE BENCHMARK
# ============================================

def evaluate_on_dataset(data_directory: str, mode: str = 'auto', verbose: bool = False) -> Dict:
    """
    Évalue le système sur un ensemble de fichiers JSON.
    
    Args:
        data_directory: Répertoire contenant les fichiers JSON
        mode: 'simple', 'advanced', ou 'auto'
        verbose: Si True, affiche des détails
    
    Returns:
        Dict avec les résultats
    """
    import glob
    import os
    
    results = []
    
    json_files = glob.glob(os.path.join(data_directory, "*.json"))
    
    for filepath in json_files:
        try:
            data = load_data_from_json(filepath)
            filename = os.path.basename(filepath)
            
            if verbose:
                print(f"\nTraitement de {filename}...")
            
            # Extraire les règles
            extractor = AdvancedGridRuleExtractor(
                use_shape_detection=True,
                mode=mode
            )
            extractor.extract_from_examples(data["train"])
            rules = extractor.get_rules()
            
            transformer = AdvancedGridTransformer(rules)
            
            # Évaluer sur les tests
            test_key = "Test" if "Test" in data else "test"
            test_results = []
            correct_tests = 0
            total_tests = 0
            
            if test_key in data:
                for test_case in data[test_key]:
                    total_tests += 1
                    input_grid = test_case["input"]
                    
                    # Générer la prédiction
                    predicted_output = transformer.apply_rules(input_grid)
                    
                    # Vérifier si l'output attendu existe pour évaluation
                    if "output" in test_case:
                        expected_output = test_case["output"]
                        is_correct = (np.array(predicted_output) == np.array(expected_output)).all()
                        
                        if is_correct:
                            correct_tests += 1
                        
                        test_results.append({
                            'predicted': predicted_output,
                            'expected': expected_output,
                            'correct': is_correct
                        })
            
            accuracy = correct_tests / total_tests if total_tests > 0 else 0
            
            results.append({
                'filename': filename,
                'success': True,
                'accuracy': accuracy,
                'correct_tests': correct_tests,
                'total_tests': total_tests,
                'rules': {
                    'color_changes_count': len(rules.get('color_changes', {})),
                    'connections_count': len(rules.get('connections', {})),
                    'translations_count': len(rules.get('translations', {}))
                }
            })
            
            if verbose:
                print(f"  ✓ {correct_tests}/{total_tests} tests corrects ({accuracy*100:.1f}%)")
            
        except Exception as e:
            results.append({
                'filename': os.path.basename(filepath),
                'success': False,
                'error': str(e)
            })
            if verbose:
                print(f"  ✗ Erreur: {e}")
    
    return results

# ============================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("SYSTÈME HYBRIDE DE RÉSOLUTION DE GRILLES ARC")
    print("=" * 70)
    print("Mode: simple (problèmes basiques) | advanced (problèmes complexes) | auto (détection automatique)")
    print("=" * 70)
    
    # Mode interactif ou batch?
    mode = input("\nMode (1=Interactif, 2=Batch, 3=Benchmark): ").strip()
    
    if mode == "1":
        # Mode interactif
        json_file = input("Nom du fichier JSON (défaut: grid_data.json): ").strip()
        if not json_file:
            json_file = "grid_data.json"
        
        mode_input = input("Mode de résolution (simple/advanced/auto, défaut: auto): ").strip()
        if not mode_input:
            mode_input = "auto"
        
        shape_input = input("Activer la détection de formes? (oui/non, défaut: oui): ").strip().lower()
        use_shape_detection = shape_input in ["", "oui", "o", "yes", "y"]
        
        try:
            data = load_data_from_json(json_file)
            
            # Extraire les règles
            extractor = AdvancedGridRuleExtractor(
                use_shape_detection=use_shape_detection,
                mode=mode_input
            )
            extractor.extract_from_examples(data["train"])
            rules = extractor.get_rules()
            
            print(f"\n=== RÈGLES EXTRACTES ===")
            print(f"Changements de couleur: {rules['color_changes']}")
            print(f"Connexions: {rules['connections']}")
            print(f"Translations: {rules['translations']}")
            
            # Appliquer aux tests
            transformer = AdvancedGridTransformer(rules)
            
            test_key = "Test" if "Test" in data else "test"
            if test_key in data:
                for test_idx, test_case in enumerate(data[test_key]):
                    print(f"\n=== TEST {test_idx + 1} ===")
                    
                    input_grid = test_case["input"]
                    output_grid = transformer.apply_rules(input_grid)
                    
                    print("Input:")
                    for row in input_grid:
                        print("  " + str(row))
                    
                    print("\nOutput prédit:")
                    for row in output_grid:
                        print("  " + str(row))
            
        except Exception as e:
            print(f"Erreur: {e}")
    
    elif mode == "2":
        # Mode batch
        json_file = input("Nom du fichier JSON d'entrée: ").strip()
        if not json_file:
            json_file = "grid_data.json"
        
        output_file = input("Nom du fichier JSON de sortie (défaut: grid_results.json): ").strip()
        if not output_file:
            output_file = "grid_results.json"
        
        mode_input = input("Mode de résolution (simple/advanced/auto, défaut: auto): ").strip()
        if not mode_input:
            mode_input = "auto"
        
        try:
            batch_process(json_file, output_file, mode=mode_input)
        except Exception as e:
            print(f"Erreur: {e}")
    
    elif mode == "3":
        # Mode benchmark
        data_dir = input("Répertoire des fichiers JSON: ").strip()
        mode_input = input("Mode de résolution (simple/advanced/auto, défaut: auto): ").strip()
        if not mode_input:
            mode_input = "auto"
        
        results = evaluate_on_dataset(data_dir, mode=mode_input, verbose=True)
        
        # Calculer les statistiques
        successful = [r for r in results if r.get('success')]
        if successful:
            avg_accuracy = sum(r.get('accuracy', 0) for r in successful) / len(successful) * 100
            total_tests = sum(r.get('total_tests', 0) for r in successful)
            total_correct = sum(r.get('correct_tests', 0) for r in successful)
            
            print(f"\n=== STATISTIQUES ===")
            print(f"Fichiers traités: {len(results)}")
            print(f"Fichiers réussis: {len(successful)}")
            print(f"Tests totaux: {total_tests}")
            print(f"Tests corrects: {total_correct}")
            print(f"Précision moyenne: {avg_accuracy:.1f}%")
    
    else:
        print("Mode non reconnu.")
