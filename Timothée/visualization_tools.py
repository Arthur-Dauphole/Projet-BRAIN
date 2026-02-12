# """
# visualization_tools.py - Outils de visualisation pour ARC
# """

# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import json

# def grid_to_rgb(grid):
#     """Convertit une grille ARC (valeurs 0-9) en image RGB."""
#     arc_colors = {
#         0: (0, 0, 0),        # noir
#         1: (255, 255, 255),  # blanc
#         2: (255, 0, 0),      # rouge
#         3: (0, 255, 0),      # vert
#         4: (0, 0, 255),      # bleu
#         5: (255, 255, 0),    # jaune
#         6: (255, 0, 255),    # magenta
#         7: (0, 255, 255),    # cyan
#         8: (128, 128, 128),  # gris
#         9: (255, 128, 0)     # orange
#     }
    
#     h = len(grid)
#     w = len(grid[0])
#     rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
    
#     for i in range(h):
#         for j in range(w):
#             color_val = grid[i][j]
#             rgb_array[i, j] = arc_colors.get(color_val, (0, 0, 0))
    
#     return rgb_array

# def generate_single_problem_visualization(result, output_dir):
#     """Génère la visualisation pour un seul problème."""
#     filename = result['filename']
#     accuracy = result.get('accuracy', 0) * 100
#     test_results = result.get('test_results', [])
    
#     if not test_results:
#         return False
    
#     # Créer une figure avec tous les tests
#     n_tests = len(test_results)
#     fig, axes = plt.subplots(n_tests, 3, figsize=(12, 4 * n_tests))
    
#     if n_tests == 1:
#         axes = [axes]
    
#     fig.suptitle(f'{filename}\nPrécision: {accuracy:.1f}%', 
#                 fontsize=16, fontweight='bold')
    
#     for test_idx, test_result in enumerate(test_results):
#         # Récupérer les grilles
#         input_grid = test_result.get('input')
#         predicted_grid = test_result.get('predicted')
#         expected_grid = test_result.get('expected')
        
#         # Déterminer les couleurs et titres
#         if expected_grid is not None:
#             is_correct = test_result.get('correct', False)
#             titles = ['Input', 'Prédit', 'Attendu']
#             colors = ['black', 'green' if is_correct else 'red', 'blue']
#         else:
#             titles = ['Input', 'Prédit', 'Pas de référence']
#             colors = ['black', 'orange', 'gray']
        
#         # Afficher chaque grille
#         for col_idx, (grid, title, color) in enumerate([
#             (input_grid, titles[0], colors[0]),
#             (predicted_grid, titles[1], colors[1]),
#             (expected_grid, titles[2], colors[2])
#         ]):
#             ax = axes[test_idx][col_idx] if n_tests > 1 else axes[col_idx]
            
#             if grid is not None:
#                 ax.imshow(grid_to_rgb(grid))
            
#             ax.set_title(title, color=color, fontweight='bold', fontsize=12)
#             ax.set_xticks([])
#             ax.set_yticks([])
            
#             # Cadre coloré
#             for spine in ax.spines.values():
#                 spine.set_edgecolor(color)
#                 spine.set_linewidth(2)
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     # Sauvegarder
#     safe_name = filename.replace('.json', '').replace(' ', '_')
#     save_path = os.path.join(output_dir, f"{safe_name}.png")
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close(fig)
    
#     return save_path

# def generate_all_visualizations(benchmark, output_dir="grid_visualizations"):
#     """Génère toutes les visualisations."""
#     os.makedirs(output_dir, exist_ok=True)
    
#     successful_results = [r for r in benchmark.results if r.get('success')]
    
#     print(f"\nGénération des visualisations pour {len(successful_results)} problèmes...")
    
#     for result in successful_results:
#         try:
#             save_path = generate_single_problem_visualization(result, output_dir)
#             if save_path:
#                 print(f"  ✓ {result['filename']}")
#         except Exception as e:
#             print(f"  ✗ Erreur avec {result['filename']}: {e}")
    
#     print(f"\nVisualisations sauvegardées dans: {os.path.abspath(output_dir)}")

"""
GÉNÉRATEUR D'IMAGE GÉANTE - Toutes les grilles de tous les problèmes dans une seule image
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.gridspec import GridSpec

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

def create_giant_visualization(benchmark_results, output_dir="giant_visualization", max_problems=50):
    """
    Crée une seule image géante qui montre TOUTES les grilles de TOUS les problèmes.
    Chaque problème a sa propre section avec :
    - Tous les exemples d'entraînement (input/output)
    - Tous les tests (input/prédiction/expected)
    
    Args:
        benchmark_results: Liste des résultats du benchmark
        output_dir: Dossier de sortie
        max_problems: Nombre maximum de problèmes à inclure
    """
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    arc_colors = load_arc_colors()
    
    # Filtrer les résultats réussis
    successful_results = [r for r in benchmark_results if r.get('success')]
    
    if not successful_results:
        print("Aucun résultat réussi à visualiser.")
        return
    
    print(f"Préparation de l'image géante avec {min(len(successful_results), max_problems)} problèmes...")
    
    # Limiter le nombre de problèmes
    results_to_show = successful_results[:max_problems]
    
    # Calculer le nombre total de grilles pour déterminer la taille de l'image
    total_grids = 0
    problem_info = []
    
    for result in results_to_show:
        filename = result['filename']
        accuracy = result.get('accuracy', 0) * 100
        
        # Charger les données originales
        try:
            if 'filepath' in result:
                filepath = result['filepath']
            else:
                filepath = f"Arthur_2/BRAIN_PROJECT/data/{filename}"
            
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    original_data = json.load(f)
                
                train_examples = original_data.get('train', [])
                test_examples = original_data.get('test', [])
                
                # Nombre de grilles pour ce problème
                n_train_grids = len(train_examples) * 2  # input + output
                n_test_grids = len(test_examples) * 3    # input + prédiction + expected
                
                total_grids += n_train_grids + n_test_grids
                
                problem_info.append({
                    'filename': filename,
                    'accuracy': accuracy,
                    'train_examples': train_examples,
                    'test_examples': test_examples,
                    'test_results': result.get('test_results', []),
                    'total_grids': n_train_grids + n_test_grids
                })
            else:
                print(f"  ⚠️ Fichier non trouvé: {filepath}")
        except Exception as e:
            print(f"  ⚠️ Erreur avec {filename}: {e}")
    
    if total_grids == 0:
        print("Aucune donnée à afficher.")
        return
    
    print(f"Total de grilles à afficher: {total_grids}")
    
    # Calculer la disposition optimale
    # On veut environ 8 colonnes pour une bonne lisibilité
    n_cols = 8
    n_rows = (total_grids + n_cols - 1) // n_cols
    
    # Créer une figure GÉANTE
    # Taille basée sur le nombre de lignes et colonnes
    fig_width = n_cols * 2.5  # 2.5 pouces par colonne
    fig_height = n_rows * 2.5  # 2.5 pouces par ligne
    
    # Limiter la taille maximum
    fig_width = min(fig_width, 50)
    fig_height = min(fig_height, 100)
    
    print(f"Création de l'image géante: {fig_width:.1f} x {fig_height:.1f} pouces")
    print(f"Disposition: {n_rows} lignes x {n_cols} colonnes")
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Titre principal
    avg_accuracy = np.mean([p['accuracy'] for p in problem_info])
    plt.suptitle(f"VISUALISATION COMPLÈTE DE TOUS LES PROBLÈMES ARC\n"
                f"{len(problem_info)} problèmes - Précision moyenne: {avg_accuracy:.1f}%\n"
                f"Chaque problème montre: [Entraînement: Input→Output] [Tests: Input→Prédiction→Expected]",
                fontsize=24, fontweight='bold', y=0.99)
    
    # Utiliser GridSpec pour un contrôle précis
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
    
    # Position courante dans la grille
    current_row = 0
    current_col = 0
    
    # Pour chaque problème
    for problem_idx, problem in enumerate(problem_info):
        filename = problem['filename']
        accuracy = problem['accuracy']
        
        # Titre du problème (en texte dans la première cellule)
        if current_col < n_cols:
            ax = fig.add_subplot(gs[current_row, current_col])
            ax.text(0.5, 0.5, f"{filename}\n{accuracy:.1f}%", 
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
            ax.axis('off')
            current_col += 1
        
        # Afficher les exemples d'entraînement
        for i, example in enumerate(problem['train_examples']):
            if current_col >= n_cols:
                current_col = 0
                current_row += 1
            
            # Input
            if current_row < n_rows:
                ax = fig.add_subplot(gs[current_row, current_col])
                if 'input' in example:
                    img = grid_to_image(example['input'], arc_colors)
                    ax.imshow(img)
                    ax.set_title(f"Train {i+1} In", fontsize=6)
                ax.axis('off')
                current_col += 1
            
            if current_col >= n_cols:
                current_col = 0
                current_row += 1
            
            # Output
            if current_row < n_rows:
                ax = fig.add_subplot(gs[current_row, current_col])
                if 'output' in example:
                    img = grid_to_image(example['output'], arc_colors)
                    ax.imshow(img)
                    ax.set_title(f"Train {i+1} Out", fontsize=6, color='blue')
                ax.axis('off')
                current_col += 1
        
        # Afficher les tests avec prédictions
        test_results = problem['test_results']
        for i in range(len(problem['test_examples'])):
            # Input du test
            if current_col >= n_cols:
                current_col = 0
                current_row += 1
            
            if current_row < n_rows:
                ax = fig.add_subplot(gs[current_row, current_col])
                if i < len(problem['test_examples']) and 'input' in problem['test_examples'][i]:
                    img = grid_to_image(problem['test_examples'][i]['input'], arc_colors)
                    ax.imshow(img)
                    ax.set_title(f"Test {i+1} In", fontsize=6)
                ax.axis('off')
                current_col += 1
            
            # Prédiction
            if current_col >= n_cols:
                current_col = 0
                current_row += 1
            
            if current_row < n_rows:
                ax = fig.add_subplot(gs[current_row, current_col])
                if i < len(test_results) and 'predicted' in test_results[i]:
                    img = grid_to_image(test_results[i]['predicted'], arc_colors)
                    ax.imshow(img)
                    
                    # Vérifier si correct
                    is_correct = test_results[i].get('correct', False)
                    status = "✓" if is_correct else "✗"
                    color = 'green' if is_correct else 'red'
                    ax.set_title(f"Test {i+1} Préd {status}", fontsize=6, color=color)
                ax.axis('off')
                current_col += 1
            
            # Expected
            if current_col >= n_cols:
                current_col = 0
                current_row += 1
            
            if current_row < n_rows:
                ax = fig.add_subplot(gs[current_row, current_col])
                if i < len(problem['test_examples']) and 'output' in problem['test_examples'][i]:
                    img = grid_to_image(problem['test_examples'][i]['output'], arc_colors)
                    ax.imshow(img)
                    ax.set_title(f"Test {i+1} Exp", fontsize=6, color='purple')
                ax.axis('off')
                current_col += 1
        
        # Ajouter une ligne vide entre les problèmes pour la lisibilité
        if current_col < n_cols and current_col > 0:
            # Remplir le reste de la ligne avec des cellules vides
            while current_col < n_cols:
                ax = fig.add_subplot(gs[current_row, current_col])
                ax.axis('off')
                current_col += 1
        
        current_col = 0
        current_row += 1
        
        # Afficher la progression
        if (problem_idx + 1) % 10 == 0:
            print(f"  Progression: {problem_idx + 1}/{len(problem_info)} problèmes traités")
    
    # Ajuster la disposition
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    
    # Sauvegarder l'image géante
    output_path = os.path.join(output_dir, "GIANT_VISUALIZATION_ALL_PROBLEMS.png")
    print(f"\nSauvegarde de l'image géante... (cela peut prendre un moment)")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Image géante sauvegardée: {output_path}")
    print(f"Taille: {fig_width:.1f} x {fig_height:.1f} pouces")
    
    # Créer aussi une version PDF (meilleure pour le zoom)
    output_path_pdf = os.path.join(output_dir, "GIANT_VISUALIZATION_ALL_PROBLEMS.pdf")
    print("Création de la version PDF...")
    
    # Réduire la taille pour le PDF
    fig_width_pdf = min(fig_width, 30)
    fig_height_pdf = min(fig_height, 60)
    
    fig2 = plt.figure(figsize=(fig_width_pdf, fig_height_pdf))
    plt.suptitle(f"VISUALISATION COMPLÈTE - {len(problem_info)} problèmes ARC", 
                fontsize=16, fontweight='bold', y=0.99)
    
    # Recréer avec moins de détails pour le PDF
    current_row = 0
    current_col = 0
    n_cols_pdf = 6
    
    for problem in problem_info:
        # Uniquement les prédictions principales
        if current_row < 100:  # Limiter à 100 lignes
            if problem['test_results']:
                test = problem['test_results'][0]
                if 'predicted' in test:
                    ax = fig2.add_subplot(gs[current_row, current_col])
                    img = grid_to_image(test['predicted'], arc_colors)
                    ax.imshow(img)
                    
                    # Titre court
                    short_name = problem['filename'][:15] + '...' if len(problem['filename']) > 15 else problem['filename']
                    ax.set_title(f"{short_name}\n{problem['accuracy']:.1f}%", fontsize=6)
                    ax.axis('off')
                    
                    current_col += 1
                    if current_col >= n_cols_pdf:
                        current_col = 0
                        current_row += 1
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(output_path_pdf, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"✅ Version PDF sauvegardée: {output_path_pdf}")
    print(f"\n📊 Résumé:")
    print(f"   - Problèmes inclus: {len(problem_info)}")
    print(f"   - Précision moyenne: {avg_accuracy:.1f}%")
    print(f"   - Fichiers PNG et PDF créés dans: {os.path.abspath(output_dir)}")
    print(f"   - Ouvrez le fichier PNG avec un visualiseur d'images qui supporte le zoom")