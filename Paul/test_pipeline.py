import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb
from perception import PerceptionSystem
from color import ColorMapper
from reasoning import ReasoningEngine

# --- 1. DÉFINITION DES GRILLES (Input et Output attendu de l'énoncé) ---
grid_in = [
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 6],
    [0, 0, 0, 0, 6],
    [0, 0, 6, 6, 6],
    [0, 0, 0, 0, 0]
]

grid_out_expected = [
    [0, 0, 4, 4, 4],
    [0, 0, 4, 4, 4],
    [0, 0, 8, 0, 0],
    [0, 0, 8, 0, 0],
    [8, 8, 8, 0, 0]
]

# --- 2. TRAITEMENT (Perception + Raisonnement) ---
ps = PerceptionSystem()
objs_in = ps.extract_objects(grid_in)
objs_out = ps.extract_objects(grid_out_expected)

engine = ReasoningEngine()
transformations = engine.compare_grids(objs_in, objs_out)

# --- 3. FONCTIONS DE VISUALISATION ---

def grid_to_rgb(grid):
    h, w = len(grid), len(grid[0])
    rgb_array = np.zeros((h, w, 3))
    for y in range(h):
        for x in range(w):
            rgb_array[y, x] = to_rgb(ColorMapper.hex(int(grid[y][x])))
    return rgb_array

def show_full_analysis(g_in, g_out, objs_found, transforms):
    # On garde une figure large pour éviter que les textes se chevauchent
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # --- 1. Affichage Input ---
    ax1.imshow(grid_to_rgb(g_in))
    ax1.set_title("INPUT (Détection)", fontsize=14, fontweight='bold', pad=20)
    
    detected_text = "Objets détectés :\n" + "\n".join([f"- {o.shape} ({ColorMapper.name(o.color_code)})" for o in objs_found])
    
    # Placement SOUS la grille input
    ax1.text(0.5, -0.15, detected_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='center',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- 2. Affichage Output ---
    ax2.imshow(grid_to_rgb(g_out))
    ax2.set_title("OUTPUT (Cible)", fontsize=14, fontweight='bold', pad=20)
    
    trans_text = "Transformations détectées :\n"
    for t in transforms:
        if isinstance(t, str): trans_text += f"- {t}\n"
        else: trans_text += f"- {t.obj_in.shape}: {' + '.join(t.actions)}\n"
    
    # Placement SOUS la grille output (identique à l'input)
    ax2.text(0.5, -0.15, trans_text, transform=ax2.transAxes, 
             verticalalignment='top', horizontalalignment='center',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # --- 3. Réglages des marges de la fenêtre ---
    # On laisse de l'espace en bas (bottom=0.3) pour que les textes ne soient pas coupés
    plt.subplots_adjust(top=0.85, bottom=0.3, left=0.1, right=0.9, wspace=0.3)
    
    fig.canvas.manager.set_window_title('Analyse BRAIN - Système de Raisonnement')

    # Cosmétique des grilles
    for ax in [ax1, ax2]:
        h, w = len(g_in), len(g_in[0])
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    plt.show()

# --- 4. LANCEMENT ---
show_full_analysis(grid_in, grid_out_expected, objs_in, transformations)