import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import copy

# --- Gestion des couleurs ---
class ColorMapper:
    """
    Mappe les codes de couleur (0-9) vers des couleurs hexadécimales.
    """
    # Palette de couleurs pour les chiffres 0-9
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
        """Retourne la couleur hexadécimale pour un code couleur."""
        return ColorMapper.COLOR_MAP.get(color_code, "#FFFFFF")  # Blanc par défaut
    
    @staticmethod
    def name(color_code: int) -> str:
        """Retourne le nom de la couleur."""
        names = {
            0: "Noir", 1: "Bleu", 2: "Rouge", 3: "Vert", 4: "Jaune",
            5: "Gris", 6: "Magenta", 7: "Orange", 8: "Cyan", 9: "Marron"
        }
        return names.get(color_code, f"Couleur {color_code}")

# --- Fonctions de visualisation ---
def grid_to_rgb(grid: List[List[int]]) -> np.ndarray:
    """
    Convertit une grille numérique en tableau RGB.
    """
    h, w = len(grid), len(grid[0])
    rgb_array = np.zeros((h, w, 3))
    
    for y in range(h):
        for x in range(w):
            # Convertir le code couleur en RGB
            hex_color = ColorMapper.hex(grid[y][x])
            # Convertir hex en RGB (sans matplotlib pour éviter les dépendances circulaires)
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) / 255.0 
                       for i in (0, 2, 4))
            rgb_array[y, x] = rgb
    
    return rgb_array

def visualize_grids(input_grid: List[List[int]], 
                   output_grid: List[List[int]], 
                   title_left: str = "INPUT",
                   title_right: str = "OUTPUT",
                   rules_text: str = ""):
    """
    Visualise deux grilles côte à côte avec des règles.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # --- Grille d'entrée ---
    ax1.imshow(grid_to_rgb(input_grid))
    ax1.set_title(title_left, fontsize=14, fontweight='bold', pad=20)
    
    # --- Grille de sortie ---
    ax2.imshow(grid_to_rgb(output_grid))
    ax2.set_title(title_right, fontsize=14, fontweight='bold', pad=20)
    
    # Ajouter les règles en bas si fournies
    if rules_text:
        # Calculer la hauteur totale de la figure
        fig_height = fig.get_figheight()
        
        # Ajouter une zone de texte en bas de la figure
        fig.text(0.5, 0.02, rules_text, 
                ha='center', va='bottom',
                fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Ajuster les marges pour faire de la place pour le texte
        plt.subplots_adjust(bottom=0.25)
    
    # Grille pour les deux axes
    for ax in [ax1, ax2]:
        h, w = len(input_grid), len(input_grid[0])
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1)
        ax.tick_params(which='both', 
                      bottom=False, left=False, 
                      labelbottom=False, labelleft=False)
    
    # Légende des couleurs
    legend_elements = []
    colors_in_grid = set()
    
    for row in input_grid + output_grid:
        colors_in_grid.update(set(row))
    
    colors_in_grid = sorted(list(colors_in_grid))
    for color in colors_in_grid:
        if color != 0:  # Ne pas inclure le fond
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, 
                            facecolor=ColorMapper.hex(color),
                            label=f"{color}: {ColorMapper.name(color)}")
            )
    
    if legend_elements:
        fig.legend(handles=legend_elements, 
                  loc='upper center', 
                  ncol=min(5, len(legend_elements)),
                  bbox_to_anchor=(0.5, 0.95))
    
    plt.suptitle("Analyse des Grilles - Système de Règles", 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05 if rules_text else 0, 1, 0.95])
    plt.show()

def visualize_rules_summary(rules: Dict):
    """
    Crée une visualisation des règles extraites.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    rules_text = "=== RÈGLES EXTRACTES ===\n\n"
    
    # Changements de couleur
    rules_text += "CHANGEMENTS DE COULEUR:\n"
    if rules.get('color_changes'):
        for old_color, new_color in rules['color_changes'].items():
            rules_text += f"  {old_color} → {new_color}\n"
    else:
        rules_text += "  Aucun\n"
    
    # Connexions
    rules_text += "\nCONNEXIONS:\n"
    if rules.get('connections'):
        for color, conn_type in rules['connections'].items():
            type_name = {'H': 'Horizontale', 'V': 'Verticale', 'D': 'Diagonale'}.get(conn_type, conn_type)
            rules_text += f"  Couleur {color}: {type_name}\n"
    else:
        rules_text += "  Aucune\n"
    
    # Translations
    rules_text += "\nTRANSLATIONS:\n"
    if rules.get('translations'):
        for color, (dx, dy) in rules['translations'].items():
            rules_text += f"  Couleur {color}: ({dx}, {dy})\n"
    else:
        rules_text += "  Aucune\n"
    
    # Afficher le texte
    ax.text(0.1, 0.5, rules_text, 
            fontsize=12, 
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Règles Extraites des Exemples", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# --- Classes d'extraction et transformation (inchangées) ---
class GridRuleExtractor:
    """Extrait les règles de transformation à partir d'exemples input/output."""
    
    def __init__(self):
        self.color_changes = {}
        self.connections = {}
        self.translations = {}
        
    def extract_from_examples(self, train_examples: List[Dict]) -> None:
        """Extrait les règles à partir des exemples d'entraînement."""
        print("=== Extraction des règles depuis les exemples ===")
        
        for idx, example in enumerate(train_examples):
            print(f"\n--- Analyse de l'exemple {idx+1} ---")
            
            input_grid = example["input"]
            output_grid = example["output"]
            
            # Détecter les changements de couleur
            self._extract_color_changes(input_grid, output_grid)
            
            # Pour chaque couleur présente en sortie
            self._analyze_transformations(input_grid, output_grid)
    
    def _extract_color_changes(self, input_grid: List[List[int]], 
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
    
    def _analyze_transformations(self, input_grid: List[List[int]], 
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
            
            if not self._try_detect_translation(input_points_for_color, 
                                              output_points, color_out):
                self._try_detect_connection(input_points_for_color,
                                          output_points, color_out)
    
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
                
                elif abs(p1[0] - p2[0]) == abs(p1[1] - p2[1]):  # Diagonale
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
        """Retourne toutes les règles extraites."""
        return {
            "color_changes": self.color_changes,
            "connections": self.connections,
            "translations": self.translations
        }

class GridTransformer:
    """Applique les règles de transformation à une grille."""
    
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
        grid = copy.deepcopy(input_grid)
        h = len(grid)
        w = len(grid[0])
        
        # Étape 1: Changements de couleur
        grid = self._apply_color_changes(grid)
        
        # Étape 2: Connexions
        grid = self._apply_connections(grid)
        
        # Étape 3: Translations
        grid = self._apply_translations(grid)
        
        return grid
    
    def _apply_color_changes(self, grid: List[List[int]]) -> List[List[int]]:
        h = len(grid)
        w = len(grid[0])
        
        for i in range(h):
            for j in range(w):
                if grid[i][j] in self.color_changes:
                    grid[i][j] = self.color_changes[grid[i][j]]
        
        return grid
    
    def _apply_connections(self, grid: List[List[int]]) -> List[List[int]]:
        h = len(grid)
        w = len(grid[0])
        
        for color, conn_type in self.connections.items():
            points = [(i, j) for i in range(h) for j in range(w) 
                     if grid[i][j] == color]
            
            if len(points) < 2:
                continue
            
            for k in range(len(points)):
                for l in range(k+1, len(points)):
                    p1 = points[k]
                    p2 = points[l]
                    
                    if conn_type == 'H' and p1[0] == p2[0]:
                        self._draw_horizontal_line(grid, p1, p2, color)
                    
                    elif conn_type == 'V' and p1[1] == p2[1]:
                        self._draw_vertical_line(grid, p1, p2, color)
                    
                    elif conn_type == 'D':
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        if abs(dx) == abs(dy):
                            self._draw_diagonal_line(grid, p1, p2, color)
        
        return grid
    
    def _draw_horizontal_line(self, grid: List[List[int]], 
                            p1: Tuple[int, int], p2: Tuple[int, int], 
                            color: int) -> None:
        row = p1[0]
        start_col = min(p1[1], p2[1])
        end_col = max(p1[1], p2[1])
        
        for col in range(start_col, end_col + 1):
            if grid[row][col] == 0:
                grid[row][col] = color
    
    def _draw_vertical_line(self, grid: List[List[int]], 
                          p1: Tuple[int, int], p2: Tuple[int, int], 
                          color: int) -> None:
        col = p1[1]
        start_row = min(p1[0], p2[0])
        end_row = max(p1[0], p2[0])
        
        for row in range(start_row, end_row + 1):
            if grid[row][col] == 0:
                grid[row][col] = color
    
    def _draw_diagonal_line(self, grid: List[List[int]], 
                          p1: Tuple[int, int], p2: Tuple[int, int], 
                          color: int) -> None:
        dx = 1 if p2[0] > p1[0] else -1
        dy = 1 if p2[1] > p1[1] else -1
        steps = abs(p2[0] - p1[0])
        
        for step in range(steps + 1):
            row = p1[0] + step * dx
            col = p1[1] + step * dy
            if grid[row][col] == 0:
                grid[row][col] = color
    
    def _apply_translations(self, grid: List[List[int]]) -> List[List[int]]:
        h = len(grid)
        w = len(grid[0])
        
        for color, (dx, dy) in self.translations.items():
            new_grid = copy.deepcopy(grid)
            
            for i in range(h):
                for j in range(w):
                    if grid[i][j] == color:
                        new_i = i + dx
                        new_j = j + dy
                        
                        new_grid[i][j] = 0
                        
                        if 0 <= new_i < h and 0 <= new_j < w:
                            new_grid[new_i][new_j] = color
            
            grid = new_grid
        
        return grid

# --- Fonctions de gestion des fichiers ---
def load_data_from_json(filepath: str) -> Dict:
    """Charge les données depuis un fichier JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_results_to_json(results: Dict, filepath: str) -> None:
    """Sauvegarde les résultats dans un fichier JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

# --- Fonction principale avec visualisation ---
def main(visualize: bool = True, json_file: str = "Timothée\GridTestPAM_JSON.json"):
    """
    Fonction principale avec option de visualisation.
    
    Args:
        visualize: Si True, affiche les visualisations
        json_file: Chemin vers le fichier JSON contenant les données
    """
    # Charger les données
    print("Chargement des données...")
    try:
        data = load_data_from_json(json_file)
    except FileNotFoundError:
        print(f"Erreur: Fichier '{json_file}' non trouvé.")
        return
    
    # Extraire les règles
    print("\nExtraction des règles...")
    extractor = GridRuleExtractor()
    extractor.extract_from_examples(data["train"])
    rules = extractor.get_rules()
    
    print(f"\n=== RÈGLES EXTRACTES ===")
    print(f"Changements de couleur: {rules['color_changes']}")
    print(f"Connexions: {rules['connections']}")
    print(f"Translations: {rules['translations']}")
    
    # Visualisation des règles
    if visualize:
        visualize_rules_summary(rules)
    
    # Visualisation des exemples d'entraînement
    if visualize and "train" in data:
        print("\n=== VISUALISATION DES EXEMPLES D'ENTRAÎNEMENT ===")
        for idx, example in enumerate(data["train"]):
            print(f"\nExemple d'entraînement {idx + 1}:")
            
            # Créer le texte des règles pour cet exemple
            rules_text = f"Règles appliquées à cet exemple:\n"
            
            # Identifier les règles pertinentes pour cet exemple
            input_grid = example["input"]
            colors_present = set()
            for row in input_grid:
                colors_present.update(set(row))
            
            # Filtrer les règles pour les couleurs présentes
            for color in colors_present:
                if color in rules['color_changes']:
                    rules_text += f"  {color} → {rules['color_changes'][color]}\n"
                if color in rules['connections']:
                    conn_type = rules['connections'][color]
                    type_name = {'H': 'Horizontale', 'V': 'Verticale', 'D': 'Diagonale'}.get(conn_type, conn_type)
                    rules_text += f"  Connexion {type_name} pour {color}\n"
                if color in rules['translations']:
                    dx, dy = rules['translations'][color]
                    rules_text += f"  Translation ({dx}, {dy}) pour {color}\n"
            
            visualize_grids(
                input_grid=example["input"],
                output_grid=example["output"],
                title_left=f"EXEMPLE {idx+1} - INPUT",
                title_right=f"EXEMPLE {idx+1} - OUTPUT ATTENDU",
                rules_text=rules_text
            )
    
    # Appliquer aux grilles de test
    print(f"\n=== APPLICATION AUX TESTS ===")
    
    transformer = GridTransformer(rules)
    results = []
    
    test_key = "Test" if "Test" in data else "test"
    
    if test_key in data:
        for test_idx, test_case in enumerate(data[test_key]):
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
            
            # Visualisation du test
            if visualize:
                visualize_grids(
                    input_grid=input_grid,
                    output_grid=output_grid,
                    title_left=f"TEST {test_idx+1} - INPUT",
                    title_right=f"TEST {test_idx+1} - OUTPUT PRÉDIT",
                    rules_text="Règles appliquées automatiquement"
                )
    
    # Sauvegarder les résultats
    save_results_to_json({"test_results": results}, "Timothée\grid_results.json")
    print(f"\nRésultats sauvegardés dans 'grid_results.json'")

# --- Point d'entrée principal ---
if __name__ == "__main__":
    # Demander à l'utilisateur s'il veut des visualisations
    import sys
    
    visualize_input = input("Voulez-vous afficher les visualisations? (oui/non): ").strip().lower()
    visualize = visualize_input in ["oui", "o", "yes", "y"]
    
    # Demander le nom du fichier JSON
    json_file = input("Nom du fichier JSON (défaut: grid_data.json): ").strip()
    if not json_file:
        json_file = "grid_data.json"
    
    # Exécuter le programme
    try:
        main(visualize=visualize, json_file=json_file)
    except Exception as e:
        print(f"Erreur lors de l'exécution: {e}")
        print("Assurez-vous que matplotlib est installé: pip install matplotlib")
        sys.exit(1)