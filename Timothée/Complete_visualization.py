"""
GÉNÉRATEUR DE VISUALISATION COMPLÈTE - Toutes les grilles dans une seule image
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def load_arc_colors():
    """Retourne la palette de couleurs ARC standard."""
    return {
        0: [0, 0, 0],         # noir
        1: [255, 255, 255],   # blanc
        2: [255, 0, 0],       # rouge
        3: [0, 255, 0],       # vert
        4: [0, 0, 255],       # bleu
        5: [255, 255, 0],     # jaune
        6: [255, 0, 255],     # magenta
        7: [0, 255, 255],     # cyan
        8: [128, 128, 128],   # gris
        9: [255, 128, 0]      # orange
    }

def grid_to_image(grid, arc_colors):
    """Convertit une grille ARC en image numpy RGB."""
    h = len(grid)
    w = len(grid[0])
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            color_val = grid[i][j]
            if color_val in arc_colors:
                img[i, j] = arc_colors[color_val]
            else:
                img[i, j] = [0, 0, 0]  # noir par défaut
    
    return img

def create_complete_visualization(benchmark_results, output_dir="complete_visualizations", max_problems=10):
    """
    Crée une grande image pour chaque problème montrant TOUT :
    - Tous les exemples d'entraînement (input/output)
    - Tous les tests (input/prédiction/expected)
    
    Args:
        benchmark_results: Liste des résultats du benchmark
        output_dir: Dossier de sortie
        max_problems: Nombre maximum de problèmes à visualiser
    """
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    arc_colors = load_arc_colors()
    
    # Filtrer les résultats réussis
    successful_results = [r for r in benchmark_results if r.get('success')]
    
    if not successful_results:
        print("Aucun résultat réussi à visualiser.")
        return
    
    print(f"Création des visualisations complètes pour {min(len(successful_results), max_problems)} problèmes...")
    
    for result_idx, result in enumerate(successful_results[:max_problems]):
        try:
            filename = result['filename']
            accuracy = result.get('accuracy', 0) * 100
            
            print(f"\nTraitement de {filename} ({accuracy:.1f}%)...")
            
            # Charger les données originales pour avoir les exemples d'entraînement
            # Vous devez avoir le chemin d'accès original
            if 'filepath' in result:
                filepath = result['filepath']
            else:
                # Essayer de deviner le chemin
                filepath = f"Arthur_2/BRAIN_PROJECT/data/{filename}"
            
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    original_data = json.load(f)
                
                # Récupérer tous les exemples d'entraînement
                train_examples = original_data.get('train', [])
                test_examples = original_data.get('test', [])
            else:
                print(f"  ⚠️ Fichier original non trouvé: {filepath}")
                train_examples = []
                test_examples = []
            
            # Récupérer les prédictions du benchmark
            test_results = result.get('test_results', [])
            
            # Calculer le nombre total de grilles à afficher
            n_train = len(train_examples)
            n_tests = len(test_examples)
            
            # Pour chaque exemple d'entraînement: input + output (2 grilles)
            # Pour chaque test: input + prédiction + expected (3 grilles)
            total_grids = (n_train * 2) + (n_tests * 3)
            
            if total_grids == 0:
                print(f"  ⚠️ Aucune donnée à afficher pour {filename}")
                continue
            
            # Déterminer la disposition optimale
            # On veut environ 5 colonnes maximum
            n_cols = min(5, max(3, int(np.sqrt(total_grids))))
            n_rows = (total_grids + n_cols - 1) // n_cols
            
            # Créer une grande figure
            fig_width = n_cols * 4
            fig_height = n_rows * 4
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            # Titre principal
            plt.suptitle(f"PROBLÈME: {filename}\nPrécision: {accuracy:.1f}%", 
                        fontsize=20, fontweight='bold', y=0.98)
            
            # Compteur pour la position dans la grille
            grid_counter = 0
            
            # 1. Afficher les exemples d'entraînement
            for i, example in enumerate(train_examples):
                # Input
                ax = plt.subplot(n_rows, n_cols, grid_counter + 1)
                if 'input' in example:
                    img = grid_to_image(example['input'], arc_colors)
                    ax.imshow(img)
                ax.set_title(f"Train {i+1} - Input", fontsize=10, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                grid_counter += 1
                
                # Output
                ax = plt.subplot(n_rows, n_cols, grid_counter + 1)
                if 'output' in example:
                    img = grid_to_image(example['output'], arc_colors)
                    ax.imshow(img)
                ax.set_title(f"Train {i+1} - Output", fontsize=10, fontweight='bold', color='blue')
                ax.set_xticks([])
                ax.set_yticks([])
                grid_counter += 1
            
            # 2. Afficher les tests avec prédictions
            for i in range(n_tests):
                # Input du test
                ax = plt.subplot(n_rows, n_cols, grid_counter + 1)
                if i < len(test_examples) and 'input' in test_examples[i]:
                    img = grid_to_image(test_examples[i]['input'], arc_colors)
                    ax.imshow(img)
                ax.set_title(f"Test {i+1} - Input", fontsize=10, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                grid_counter += 1
                
                # Prédiction
                ax = plt.subplot(n_rows, n_cols, grid_counter + 1)
                if i < len(test_results) and 'predicted' in test_results[i]:
                    img = grid_to_image(test_results[i]['predicted'], arc_colors)
                    ax.imshow(img)
                    
                    # Vérifier si la prédiction est correcte
                    is_correct = test_results[i].get('correct', False)
                    title_color = 'green' if is_correct else 'red'
                    status = "✓" if is_correct else "✗"
                else:
                    title_color = 'orange'
                    status = "?"
                
                ax.set_title(f"Test {i+1} - Prédiction {status}", 
                           fontsize=10, fontweight='bold', color=title_color)
                ax.set_xticks([])
                ax.set_yticks([])
                grid_counter += 1
                
                # Expected (si disponible)
                ax = plt.subplot(n_rows, n_cols, grid_counter + 1)
                if i < len(test_examples) and 'output' in test_examples[i]:
                    img = grid_to_image(test_examples[i]['output'], arc_colors)
                    ax.imshow(img)
                    ax.set_title(f"Test {i+1} - Expected", fontsize=10, fontweight='bold', color='purple')
                else:
                    ax.text(0.5, 0.5, 'No reference', 
                           ha='center', va='center', fontsize=12, color='gray')
                    ax.set_title(f"Test {i+1} - No ref", fontsize=10, color='gray')
                
                ax.set_xticks([])
                ax.set_yticks([])
                grid_counter += 1
            
            # Ajuster l'espacement
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            
            # Sauvegarder
            safe_name = filename.replace('.json', '').replace(' ', '_')
            output_path = os.path.join(output_dir, f"COMPLETE_{safe_name}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  ✓ Image complète générée: {output_path}")
            
        except Exception as e:
            print(f"  ✗ Erreur avec {result.get('filename', 'unknown')}: {e}")
            continue
    
    print(f"\n✅ Toutes les visualisations complètes sont dans: {os.path.abspath(output_dir)}")
    
    # Créer une image RÉCAPITULATIVE avec toutes les miniatures
    create_summary_recap(successful_results[:max_problems], output_dir, arc_colors)

def create_summary_recap(results, output_dir, arc_colors):
    """Crée une image récapitulative avec une miniature pour chaque problème."""
    if not results:
        return
    
    n_results = len(results)
    n_cols = min(5, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        
        # Prendre la première prédiction comme miniature
        test_results = result.get('test_results', [])
        if test_results and 'predicted' in test_results[0]:
            img = grid_to_image(test_results[0]['predicted'], arc_colors)
            ax.imshow(img)
        
        # Titre avec nom de fichier et précision
        filename = result['filename']
        accuracy = result.get('accuracy', 0) * 100
        short_name = filename[:20] + '...' if len(filename) > 20 else filename
        
        # Couleur selon la précision
        color = 'green' if accuracy >= 80 else 'orange' if accuracy >= 50 else 'red'
        
        ax.set_title(f"{short_name}\n{accuracy:.1f}%", 
                    fontsize=8, color=color, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Cadre coloré
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
    
    # Cacher les axes non utilisés
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    # Titre principal
    avg_accuracy = np.mean([r.get('accuracy', 0) * 100 for r in results])
    plt.suptitle(f"RÉCAPITULATIF - {len(results)} problèmes (moyenne: {avg_accuracy:.1f}%)",
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Sauvegarder
    output_path = os.path.join(output_dir, "RECAP_ALL_PROBLEMS.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Récapitulatif généré: {output_path}")