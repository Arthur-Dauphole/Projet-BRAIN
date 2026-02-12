#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_arc_grids.py

Visualisation de grilles ARC (format JSON) pour annexe de rapport.
Génère une figure PDF vectorielle avec :
- Paires input/output d'entraînement
- Paire de test : input + grille vide (cible à prédire, dimensions identiques à l'output réel)
- Flèches entre input et output / grille vide

Utilisation :
    python visualize_arc_grids.py chemin/vers/tache.json

Style : IEEE single column, couleurs ARC, pas de chiffres.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch

# ----------------------------------------------------------------------
# Configuration du style (identique aux figures précédentes)
# ----------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.format": "pdf",
})

# ----------------------------------------------------------------------
# Palette de couleurs officielle ARC (0 = blanc, 1-9 couleurs vives)
# ----------------------------------------------------------------------
COLOR_MAP = {
    0: "#ffffff",  # blanc
    1: "#0072B2",  # bleu
    2: "#D55E00",  # vermillon
    3: "#009E73",  # vert
    4: "#E69F00",  # orange
    5: "#CC79A7",  # violet
    6: "#56B4E9",  # bleu ciel
    7: "#F0E442",  # jaune
    8: "#999999",  # gris
    9: "#000000",  # noir
}

def grid_to_rgb(grid):
    """
    Convertit une grille d'entiers (0-9) en tableau RGB (uint8).
    """
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            val = grid[i, j]
            hex_color = COLOR_MAP.get(val, "#ffffff")
            rgb_float = to_rgb(hex_color)
            rgb[i, j] = np.array(rgb_float) * 255
    return rgb

def plot_grid(ax, grid, title=""):
    """Dessine une grille colorée sans annotations."""
    h, w = grid.shape
    rgb = grid_to_rgb(grid)
    
    ax.imshow(rgb, interpolation='nearest', aspect='equal')
    
    # Bordures de cellules (gris clair)
    for i in range(h + 1):
        ax.axhline(i - 0.5, color='#888888', linewidth=0.3)
    for j in range(w + 1):
        ax.axvline(j - 0.5, color='#888888', linewidth=0.3)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, pad=2)
    return ax

def plot_empty_grid(ax, height, width, title=""):
    """
    Dessine une grille vide (toutes les cases à 0/blanc) de dimensions (height, width).
    """
    empty_grid = np.zeros((height, width), dtype=int)
    plot_grid(ax, empty_grid, title=title)

def add_arrow(fig, ax_left, ax_right):
    """
    Ajoute une flèche horizontale entre deux subplots.
    """
    bbox_left = ax_left.get_position()
    bbox_right = ax_right.get_position()
    
    x_start = bbox_left.x1
    y_center = (bbox_left.y0 + bbox_left.y1) / 2
    
    x_end = bbox_right.x0
    y_center_end = (bbox_right.y0 + bbox_right.y1) / 2
    
    arrow = FancyArrowPatch(
        (x_start, y_center), (x_end, y_center_end),
        transform=fig.transFigure,
        arrowstyle='-|>', mutation_scale=12,
        facecolor='black', edgecolor='black',
        linewidth=0.8,
        shrinkA=5, shrinkB=5
    )
    fig.patches.append(arrow)

def load_and_visualize(json_path, output_dir="figures"):
    """
    Charge un fichier JSON ARC et génère une figure avec :
    - Train : input → output
    - Test  : input → grille vide (mêmes dimensions que l'output réel)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    train_pairs = data.get('train', [])
    test_pairs = data.get('test', [])
    
    n_train = len(train_pairs)
    n_test = len(test_pairs)
    n_cols = 2
    n_rows = n_train + n_test
    
    fig_height = max(2.0, n_rows * 1.2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5, fig_height))
    fig.set_layout_engine('constrained')
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # --- Paires d'entraînement ---
    for i, pair in enumerate(train_pairs):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        plot_grid(axes[i, 0], input_grid, title=f"Train {i+1} input")
        plot_grid(axes[i, 1], output_grid, title=f"Train {i+1} output")
        add_arrow(fig, axes[i, 0], axes[i, 1])
    
    # --- Paire de test ---
    for i, pair in enumerate(test_pairs):
        row_idx = n_train + i
        input_grid = np.array(pair['input'])
        # Récupération des dimensions de l'output réel (non affiché)
        output_shape = np.array(pair['output']).shape
        
        plot_grid(axes[row_idx, 0], input_grid, title=f"Test {i+1} input")
        plot_empty_grid(axes[row_idx, 1], 
                        height=output_shape[0], 
                        width=output_shape[1], 
                        title=f"Test {i+1} target")
        add_arrow(fig, axes[row_idx, 0], axes[row_idx, 1])
    
    # Sauvegarde
    base_name = Path(json_path).stem
    output_path = Path(output_dir) / f"{base_name}_grids.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"✅ Figure sauvegardée : {output_path}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_arc_grids.py <fichier_json>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    load_and_visualize(json_file)