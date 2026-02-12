"""
RUNNER FINAL - Version avec imports corrects
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import json
import glob
from matplotlib.gridspec import GridSpec

# ============================================================
# IMPORTS AU NIVEAU MODULE (avant toute fonction)
# ============================================================

print("Chargement des modules ARC...")
try:
    # Importer au niveau module (pas dans une fonction)
    from Merge_ResolutionARC import AdvancedGridRuleExtractor, AdvancedGridTransformer, GeometryDetector
    from arc_benchmark import ARCBatchBenchmark
    print("✓ Modules ARC chargés avec succès")
except ImportError as e:
    print(f"❌ Erreur d'importation: {e}")
    print("Assurez-vous que Merge_ResolutionARC.py et arc_benchmark.py sont dans le même dossier.")
    # Créer des classes factices pour éviter les erreurs
    class AdvancedGridRuleExtractor:
        def __init__(self, use_shape_detection=True):
            pass
        def extract_from_examples(self, examples):
            pass
        def get_rules(self):
            return {}
    
    class AdvancedGridTransformer:
        def __init__(self, rules):
            pass
        def apply_rules(self, grid):
            return grid
    
    class GeometryDetector:
        def detect_all(self, grid):
            return {"summary": {}, "colors_used": []}
    
    class ARCBatchBenchmark:
        def __init__(self, use_shape_detection=True, verbose=False, mode='simple'):
            pass
        def run_benchmark(self, directory):
            pass
        @property
        def results(self):
            return []

# ============================================================
# FONCTIONS DE VISUALISATION
# ============================================================

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
    if grid is None or len(grid) == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    
    try:
        h = len(grid)
        w = len(grid[0])
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                color_val = grid[i][j]
                if color_val in arc_colors:
                    img[i, j] = arc_colors[color_val]
                else:
                    img[i, j] = [0, 0, 0]
        
        return img
    except:
        return np.zeros((10, 10, 3), dtype=np.uint8)

def create_multi_page_visualization(benchmark_results, data_directory, output_dir="visualization_results", max_problems=50):
    """
    Crée plusieurs pages d'images montrant les résultats.
    """
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    arc_colors = load_arc_colors()
    
    # Filtrer les résultats réussis
    successful_results = [r for r in benchmark_results if r.get('success')]
    
    if not successful_results:
        print("Aucun résultat réussi à visualiser.")
        return
    
    # Limiter le nombre de problèmes
    results_to_show = successful_results[:max_problems]
    
    # Préparer les informations de chaque problème
    problem_info = []
    
    print(f"Chargement des données pour {len(results_to_show)} problèmes...")
    
    for result in results_to_show:
        filename = result['filename']
        accuracy = result.get('accuracy', 0) * 100
        
        try:
            filepath = os.path.join(data_directory, filename)
            
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                
                train_examples = original_data.get('train', [])
                test_examples = original_data.get('test', original_data.get('Test', []))
                
                problem_info.append({
                    'filename': filename,
                    'accuracy': accuracy,
                    'train_examples': train_examples,
                    'test_examples': test_examples,
                    'test_results': result.get('test_results', [])
                })
                
                if len(problem_info) % 10 == 0:
                    print(f"  {len(problem_info)} problèmes chargés...")
            else:
                print(f"  ⚠️ Fichier non trouvé: {filepath}")
        except Exception as e:
            print(f"  ⚠️ Erreur avec {filename}: {e}")
    
    if not problem_info:
        print("Aucune donnée à afficher.")
        return
    
    print(f"\n✓ {len(problem_info)} problèmes prêts pour la visualisation")
    
    # Créer des pages avec 9 problèmes par page (3x3)
    problems_per_page = 9
    n_pages = (len(problem_info) + problems_per_page - 1) // problems_per_page
    
    print(f"Création de {n_pages} pages ({problems_per_page} problèmes par page)...")
    
    for page_num in range(n_pages):
        start_idx = page_num * problems_per_page
        end_idx = min((page_num + 1) * problems_per_page, len(problem_info))
        page_problems = problem_info[start_idx:end_idx]
        
        # Créer une grille 3x3 pour cette page
        n_rows = 3
        n_cols = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 6))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, problem in enumerate(page_problems):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Créer une grille interne pour ce problème
            inner_gs = GridSpec(3, 3, figure=fig, hspace=0.1, wspace=0.1)
            subplot_idx = idx
            
            # Calculer la position dans la figure principale
            row = idx // n_cols
            col = idx % n_cols
            
            # Titre du problème
            fig.text(col/n_cols + 0.02, 1 - (row/n_rows) - 0.03, 
                    f"{problem['filename'][:20]}...\n{problem['accuracy']:.1f}%", 
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.8),
                    transform=fig.transFigure)
            
            # Positions pour les 5 sous-graphiques
            positions = [
                (0.05, 0.7, 0.4, 0.25),   # Train Input
                (0.55, 0.7, 0.4, 0.25),   # Train Output
                (0.05, 0.4, 0.4, 0.25),   # Test Input
                (0.55, 0.4, 0.4, 0.25),   # Test Prediction
                (0.3, 0.05, 0.4, 0.25),   # Test Expected
            ]
            
            labels = ["Train In", "Train Out", "Test In", "Prédiction", "Expected"]
            
            # Collecter les grilles à afficher
            grids = []
            
            # Premier exemple d'entraînement
            if problem['train_examples']:
                grids.append(problem['train_examples'][0].get('input'))
                grids.append(problem['train_examples'][0].get('output'))
            else:
                grids.extend([None, None])
            
            # Premier test
            if problem['test_examples']:
                grids.append(problem['test_examples'][0].get('input'))
                
                # Prédiction
                if problem['test_results'] and 'predicted' in problem['test_results'][0]:
                    grids.append(problem['test_results'][0]['predicted'])
                else:
                    grids.append(None)
                
                # Expected
                grids.append(problem['test_examples'][0].get('output'))
            else:
                grids.extend([None, None, None])
            
            # Afficher les grilles
            for i, (grid, label) in enumerate(zip(grids, labels)):
                if grid is not None:
                    # Calculer la position absolue dans la figure
                    abs_x = col/n_cols + positions[i][0]/n_cols
                    abs_y = 1 - ((row+1)/n_rows) + positions[i][1]/n_rows
                    abs_width = positions[i][2]/n_cols
                    abs_height = positions[i][3]/n_rows
                    
                    sub_ax = fig.add_axes([abs_x, abs_y, abs_width, abs_height])
                    
                    img = grid_to_image(grid, arc_colors)
                    sub_ax.imshow(img)
                    
                    # Couleur du titre
                    if label == "Train Out":
                        title_color = 'blue'
                    elif label == "Prédiction":
                        if problem['test_results'] and problem['test_results'][0].get('correct', False):
                            title_color = 'green'
                            label = "Préd ✓"
                        else:
                            title_color = 'red'
                            label = "Préd ✗"
                    elif label == "Expected":
                        title_color = 'purple'
                    else:
                        title_color = 'black'
                    
                    sub_ax.set_title(label, fontsize=8, color=title_color, pad=2)
                    sub_ax.set_xticks([])
                    sub_ax.set_yticks([])
            
            # Cacher l'axe principal
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
        
        # Cacher les axes non utilisés
        for idx in range(len(page_problems), len(axes)):
            axes[idx].axis('off')
        
        # Titre de la page
        avg_acc = np.mean([p['accuracy'] for p in page_problems])
        plt.suptitle(f"RÉSULTATS ARC - Page {page_num + 1}/{n_pages}\n"
                    f"Problèmes {start_idx + 1} à {end_idx} - Précision moyenne: {avg_acc:.1f}%",
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        # Sauvegarder la page
        output_path = os.path.join(output_dir, f"RESULTS_PAGE_{page_num + 1:02d}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Page {page_num + 1}/{n_pages} sauvegardée")
    
    # Créer un récapitulatif simple
    create_summary_grid(problem_info, output_dir, arc_colors)
    
    return output_dir

def create_summary_grid(problem_info, output_dir, arc_colors):
    """Crée un récapitulatif avec une grille de miniatures."""
    n_problems = len(problem_info)
    n_cols = min(10, int(np.sqrt(n_problems * 2)))
    n_rows = (n_problems + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, problem in enumerate(problem_info):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Choisir une image à afficher (priorité à la prédiction)
        img_to_show = None
        if problem['test_results'] and 'predicted' in problem['test_results'][0]:
            img_to_show = problem['test_results'][0]['predicted']
        elif problem['train_examples'] and 'output' in problem['train_examples'][0]:
            img_to_show = problem['train_examples'][0]['output']
        elif problem['test_examples'] and 'output' in problem['test_examples'][0]:
            img_to_show = problem['test_examples'][0]['output']
        
        if img_to_show is not None:
            img = grid_to_image(img_to_show, arc_colors)
            ax.imshow(img)
        
        # Titre court avec précision
        short_name = problem['filename'].replace('.json', '')[:10]
        if short_name.endswith('...'):
            short_name = short_name[:-3]
        
        color = 'green' if problem['accuracy'] >= 80 else 'orange' if problem['accuracy'] >= 50 else 'red'
        ax.set_title(f"{short_name}\n{problem['accuracy']:.1f}%", 
                    fontsize=7, color=color, fontweight='bold', pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Cadre coloré
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
    
    # Cacher les axes non utilisés
    for idx in range(n_problems, len(axes)):
        axes[idx].axis('off')
    
    # Titre
    avg_acc = np.mean([p['accuracy'] for p in problem_info])
    plt.suptitle(f"RÉCAPITULATIF - {n_problems} problèmes ARC (moyenne: {avg_acc:.1f}%)",
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # Sauvegarder
    output_path = os.path.join(output_dir, "SUMMARY_ALL.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Récapitulatif sauvegardé: {output_path}")

# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def main():
    print("="*70)
    print("VISUALISATION DES RÉSULTATS ARC")
    print("="*70)
    
    # Chemin vers les données
    data_path = "C:\\Users\\timor\\.virtual_documents\\Projet-BRAIN\\Arthur_2\\BRAIN_PROJECT\\data"
    
    if not os.path.exists(data_path):
        print(f"\n❌ ERREUR: Dossier '{data_path}' introuvable!")
        print("Assurez-vous d'être dans le bon répertoire.")
        return
    
    print(f"✓ Dossier data trouvé: {data_path}")
    
    # Lancer le benchmark
    print(f"\n" + "="*70)
    print("EXÉCUTION DU BENCHMARK")
    print("="*70)
    
    benchmark = ARCBatchBenchmark(use_shape_detection=True, verbose=False, mode='simple')
    benchmark.run_benchmark(data_path)
    
    # Afficher les résultats
    print(f"\n" + "="*70)
    print("RÉSULTATS")
    print("="*70)
    
    successful = [r for r in benchmark.results if r.get('success')]
    
    if successful:
        total = len(benchmark.results)
        success_count = len(successful)
        avg_accuracy = sum(r.get('accuracy', 0) for r in successful) / success_count * 100
        
        print(f"Total: {total} problèmes")
        print(f"Réussis: {success_count} ({success_count/total*100:.1f}%)")
        print(f"Précision moyenne: {avg_accuracy:.1f}%")
    else:
        print("Aucun problème réussi.")
        return
    
    # Générer les visualisations
    print(f"\n" + "="*70)
    print("CRÉATION DES VISUALISATIONS")
    print("="*70)
    
    output_dir = create_multi_page_visualization(
        benchmark.results,
        data_path,
        output_dir="results_visualization",
        max_problems=50
    )
    
    print(f"\n" + "="*70)
    print("TERMINÉ !")
    print("="*70)
    print(f"\n✅ Visualisations créées avec succès !")
    print(f"\n📁 Ouvrez le dossier '{output_dir}' pour voir:")
    print(f"   - RESULTS_PAGE_01.png, RESULTS_PAGE_02.png, etc.")
    print(f"   - SUMMARY_ALL.png (récapitulatif de toutes les prédictions)")
    print(f"\n📊 Chaque page montre 9 problèmes avec:")
    print(f"   • Train In / Train Out (entraînement)")
    print(f"   • Test In / Prédiction / Expected (test)")
    print(f"   • ✓ en vert si correct, ✗ en rouge si incorrect")

# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    main()