import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# 1. CONNAISSANCE : Dictionnaire des couleurs ARC
def get_arc_color_name(color_id):
    arc_palette = {
        0: "noir (fond)", 1: "bleu", 2: "rouge", 3: "vert", 4: "jaune",
        5: "gris", 6: "rose", 7: "orange", 8: "azur", 9: "marron"
    }
    return arc_palette.get(color_id, "inconnue")

# 2. CONNAISSANCE : Identification de la couleur dominante
def detect_shape_color(grid):
    unique_colors = np.unique(grid)
    shape_colors = [c for c in unique_colors if c != 0]
    
    if len(shape_colors) == 0: return "aucune (grille vide)"
    # On prend la couleur la plus présente si multicolore, ou l'unique couleur
    return get_arc_color_name(shape_colors[0])

# 3. OUTIL : Affichage visuel
def plot_result(grid, color_name):
    arc_colors = [
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ]
    cmap = colors.ListedColormap(arc_colors)
    norm = colors.BoundaryNorm(np.arange(-0.5, 10, 1), cmap.N)

    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.grid(True, which='both', color='gray', linewidth=0.5)
    plt.title(f"Détection : La forme est {color_name}")
    plt.show()

# 4. MAIN : Chargement JSON et exécution
def run_color_test(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # On récupère la grille d'entrée du test
    test_grid = np.array(data['test'][0]['input'])
    
    # Analyse
    color_detected = detect_shape_color(test_grid)
    print(f"✅ Analyse terminée : {color_detected}")
    
    # Visualisation
    plot_result(test_grid, color_detected)

if __name__ == "__main__":
    # Remplace par le nom de ton fichier JSON
    run_color_test("test_color.json")