import json
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import copy

class GridRuleExtractor:
    """
    Extrait les règles de transformation à partir d'exemples input/output.
    """
    
    def __init__(self):
        self.color_changes = {}  # {old_color: new_color}
        self.connections = {}    # {color: type} où type = 'H', 'V', ou 'D'
        self.translations = {}   # {color: (dx, dy)}
        
    def extract_from_examples(self, train_examples: List[Dict]) -> None:
        """
        Extrait les règles à partir des exemples d'entraînement.
        """
        print("=== Extraction des règles depuis les exemples ===")
        
        for idx, example in enumerate(train_examples):
            print(f"\n--- Analyse de l'exemple {idx+1} ---")
            
            input_grid = example["input"]
            output_grid = example["output"]
            
            # Détecter les changements de couleur (même position)
            self._extract_color_changes(input_grid, output_grid)
            
            # Pour chaque couleur présente en sortie
            self._analyze_transformations(input_grid, output_grid)
    
    def _extract_color_changes(self, input_grid: List[List[int]], 
                               output_grid: List[List[int]]) -> None:
        """
        Extrait les changements de couleur (même position).
        """
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
                            # On garde la dernière règle (pourrait être amélioré)
                    self.color_changes[old_color] = new_color
                    print(f"  Changement couleur: {old_color} -> {new_color}")
    
    def _analyze_transformations(self, input_grid: List[List[int]], 
                                output_grid: List[List[int]]) -> None:
        """
        Analyse les transformations (connexions et translations).
        """
        h = len(input_grid)
        w = len(input_grid[0])
        
        # Collecter tous les points par couleur
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
        
        # Pour chaque couleur présente en sortie
        for color_out in output_points_by_color.keys():
            # Trouver la couleur d'entrée correspondante
            possible_input_colors = []
            
            # 1. Couleurs qui ont changé en color_out
            for old_color, new_color in self.color_changes.items():
                if new_color == color_out:
                    possible_input_colors.append(old_color)
            
            # 2. La couleur elle-même (si pas de changement)
            if color_out not in self.color_changes.values():
                possible_input_colors.append(color_out)
            
            # Points d'entrée correspondants
            input_points_for_color = set()
            for c_in in possible_input_colors:
                if c_in in input_points_by_color:
                    input_points_for_color.update(input_points_by_color[c_in])
            
            output_points = output_points_by_color[color_out]
            
            if not input_points_for_color:
                continue
            
            # Essayer de détecter une translation
            if not self._try_detect_translation(input_points_for_color, 
                                              output_points, color_out):
                # Si pas de translation, essayer de détecter une connexion
                self._try_detect_connection(input_points_for_color,
                                          output_points, color_out)
    
    def _try_detect_translation(self, input_points: Set[Tuple[int, int]],
                               output_points: Set[Tuple[int, int]], 
                               color: int) -> bool:
        """
        Détecte si c'est une translation. Retourne True si détectée.
        """
        if len(input_points) != len(output_points) or not input_points:
            return False
        
        # Tester tous les déplacements possibles
        ref_in = next(iter(input_points))
        possible_displacements = set()
        
        for p_out in output_points:
            dx = p_out[0] - ref_in[0]
            dy = p_out[1] - ref_in[1]
            possible_displacements.add((dx, dy))
        
        # Tester chaque déplacement
        for dx, dy in possible_displacements:
            match_all = True
            for p_in in input_points:
                moved = (p_in[0] + dx, p_in[1] + dy)
                if moved not in output_points:
                    match_all = False
                    break
            
            if match_all:
                # Vérifier que tous les points de sortie sont atteints
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
        """
        Détecte le type de connexion entre les points.
        """
        # Points qui existent dans input ET output (points de base)
        base_points = input_points.intersection(output_points)
        added_points = output_points - input_points
        
        if not added_points:
            return  # Pas de nouveaux points
        
        if len(base_points) < 2:
            return  # Besoin d'au moins 2 points pour une connexion
        
        # Tester toutes les paires de points de base
        base_list = list(base_points)
        for i in range(len(base_list)):
            for j in range(i+1, len(base_list)):
                p1 = base_list[i]
                p2 = base_list[j]
                
                # Vérifier alignement horizontal
                if p1[0] == p2[0]:
                    min_y = min(p1[1], p2[1])
                    max_y = max(p1[1], p2[1])
                    line_points = {(p1[0], y) for y in range(min_y, max_y+1)}
                    
                    if line_points.issubset(output_points):
                        self.connections[color] = 'H'
                        print(f"  Connexion horizontale pour {color}")
                        return
                
                # Vérifier alignement vertical
                elif p1[1] == p2[1]:
                    min_x = min(p1[0], p2[0])
                    max_x = max(p1[0], p2[0])
                    line_points = {(x, p1[1]) for x in range(min_x, max_x+1)}
                    
                    if line_points.issubset(output_points):
                        self.connections[color] = 'V'
                        print(f"  Connexion verticale pour {color}")
                        return
                
                # Vérifier alignement diagonal
                elif abs(p1[0] - p2[0]) == abs(p1[1] - p2[1]):
                    dx = 1 if p2[0] > p1[0] else -1
                    dy = 1 if p2[1] > p1[1] else -1
                    steps = abs(p2[0] - p1[0])
                    line_points = {(p1[0] + k*dx, p1[1] + k*dy) 
                                   for k in range(steps+1)}
                    
                    if line_points.issubset(output_points):
                        self.connections[color] = 'D'
                        print(f"  Connexion diagonale pour {color}")
                        return
    
    def get_rules(self) -> Dict:
        """
        Retourne toutes les règles extraites.
        """
        return {
            "color_changes": self.color_changes,
            "connections": self.connections,
            "translations": self.translations
        }


class GridTransformer:
    """
    Applique les règles de transformation à une grille.
    """
    
    def __init__(self, rules: Dict):
        self.color_changes = rules.get("color_changes", {})
        self.connections = rules.get("connections", {})
        self.translations = rules.get("translations", {})
    
    def apply_rules(self, input_grid: List[List[int]]) -> List[List[int]]:
        """
        Applique les règles dans l'ordre:
        1. Changements de couleur
        2. Connexions
        3. Translations
        """
        # Étape 1: Copie profonde
        grid = copy.deepcopy(input_grid)
        h = len(grid)
        w = len(grid[0])
        
        # Étape 2: Changements de couleur
        grid = self._apply_color_changes(grid)
        
        # Étape 3: Connexions
        grid = self._apply_connections(grid)
        
        # Étape 4: Translations
        grid = self._apply_translations(grid)
        
        return grid
    
    def _apply_color_changes(self, grid: List[List[int]]) -> List[List[int]]:
        """
        Applique les changements de couleur.
        """
        h = len(grid)
        w = len(grid[0])
        
        for i in range(h):
            for j in range(w):
                if grid[i][j] in self.color_changes:
                    grid[i][j] = self.color_changes[grid[i][j]]
        
        return grid
    
    def _apply_connections(self, grid: List[List[int]]) -> List[List[int]]:
        """
        Applique les règles de connexion.
        """
        h = len(grid)
        w = len(grid[0])
        
        # Pour chaque couleur avec une règle de connexion
        for color, conn_type in self.connections.items():
            # Trouver tous les points de cette couleur
            points = [(i, j) for i in range(h) for j in range(w) 
                     if grid[i][j] == color]
            
            if len(points) < 2:
                continue
            
            # Pour chaque paire de points
            for k in range(len(points)):
                for l in range(k+1, len(points)):
                    p1 = points[k]
                    p2 = points[l]
                    
                    # Vérifier si ces deux points peuvent être connectés
                    # selon le type de connexion
                    if conn_type == 'H' and p1[0] == p2[0]:  # Horizontal
                        self._draw_horizontal_line(grid, p1, p2, color)
                    
                    elif conn_type == 'V' and p1[1] == p2[1]:  # Vertical
                        self._draw_vertical_line(grid, p1, p2, color)
                    
                    elif conn_type == 'D':  # Diagonale
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        if abs(dx) == abs(dy):  # Vraie diagonale
                            self._draw_diagonal_line(grid, p1, p2, color)
        
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
    
    def _draw_diagonal_line(self, grid: List[List[int]], 
                          p1: Tuple[int, int], p2: Tuple[int, int], 
                          color: int) -> None:
        """Dessine une ligne diagonale entre p1 et p2."""
        dx = 1 if p2[0] > p1[0] else -1
        dy = 1 if p2[1] > p1[1] else -1
        steps = abs(p2[0] - p1[0])
        
        for step in range(steps + 1):
            row = p1[0] + step * dx
            col = p1[1] + step * dy
            if grid[row][col] == 0:  # Remplir seulement les cases vides
                grid[row][col] = color
    
    def _apply_translations(self, grid: List[List[int]]) -> List[List[int]]:
        """
        Applique les translations.
        """
        h = len(grid)
        w = len(grid[0])
        
        # Pour chaque couleur avec une translation
        for color, (dx, dy) in self.translations.items():
            # Créer une nouvelle grille pour cette translation
            new_grid = copy.deepcopy(grid)
            
            # Déplacer les points de cette couleur
            for i in range(h):
                for j in range(w):
                    if grid[i][j] == color:
                        new_i = i + dx
                        new_j = j + dy
                        
                        # Effacer l'ancienne position
                        new_grid[i][j] = 0
                        
                        # Mettre à la nouvelle position si dans les limites
                        if 0 <= new_i < h and 0 <= new_j < w:
                            new_grid[new_i][new_j] = color
            
            grid = new_grid
        
        return grid


def load_data_from_json(filepath: str) -> Dict:
    """
    Charge les données depuis un fichier JSON.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_results_to_json(results: Dict, filepath: str) -> None:
    """
    Sauvegarde les résultats dans un fichier JSON.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


def main():
    """
    Fonction principale pour tester avec un fichier JSON.
    """
    # Charger les données
    print("Chargement des données...")
    data = load_data_from_json("Timothée\GridTestPAM_JSON")
    
    # Extraire les règles
    print("\nExtraction des règles...")
    extractor = GridRuleExtractor()
    extractor.extract_from_examples(data["train"])
    rules = extractor.get_rules()
    
    print(f"\n=== RÈGLES EXTRAITES ===")
    print(f"Changements de couleur: {rules['color_changes']}")
    print(f"Connexions: {rules['connections']}")
    print(f"Translations: {rules['translations']}")
    
    # Appliquer aux grilles de test
    print(f"\n=== APPLICATION AUX TESTS ===")
    
    transformer = GridTransformer(rules)
    results = []
    
    for test_idx, test_case in enumerate(data["Test"]):
        print(f"\nTest {test_idx + 1}:")
        
        input_grid = test_case["input"]
        print("Input:")
        for row in input_grid:
            print("  " + str(row))
        
        output_grid = transformer.apply_rules(input_grid)
        print("\nOutput prédit:")
        for row in output_grid:
            print("  " + str(row))
        
        results.append({
            "input": input_grid,
            "output": output_grid
        })
    
    # Sauvegarder les résultats
    save_results_to_json({"test_results": results}, "Timothée\grid_results.json")
    print(f"\nRésultats sauvegardés dans 'grid_results.json'")


if __name__ == "__main__":
    main()