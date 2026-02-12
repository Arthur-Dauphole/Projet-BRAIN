"""
RUNNER CORRIGÉ
"""

import os

# Assurez-vous d'être dans le bon dossier
print("Dossier courant:", os.getcwd())

# Liste des fichiers dans le dossier
print("\nFichiers disponibles:")
for f in os.listdir("."):
    if f.endswith(".py"):
        print(f"  - {f}")

# Chemin vers vos données (relatif au dossier courant)
data_path = "C:\\Users\\timor\\.virtual_documents\\Projet-BRAIN\\Arthur_2\\BRAIN_PROJECT\\data"

# VÉRIFIER AVANT TOUT !
print(f"\nVérification du dossier data: {data_path}")
if not os.path.exists(data_path):
    print(f"\n ERREUR: Dossier '{data_path}' introuvable!")
    print("Contenu du répertoire courant:")
    for item in os.listdir('.'):
        print(f"  - {item}")
    print("\nCherchez le dossier 'Arthur_2' dans le répertoire ci-dessus.")
    exit(1)
else:
    print(f"✓ Dossier trouvé: {data_path}")

# Importer directement (ils doivent être dans le même dossier)
from Merge_ResolutionARC import *
from arc_benchmark import ARCBatchBenchmark

# Lancer le benchmark
print(f"\n Lancement du benchmark avec {data_path}")
benchmark = ARCBatchBenchmark(use_shape_detection=True, verbose=True)
benchmark.run_benchmark(data_path)

# APRÈS benchmark.run_benchmark(full_data_path)

# Analyser pourquoi ça échoue
print("\n" + "="*60)
print("ANALYSE DES ÉCHECS")
print("="*60)

failed_results = [r for r in benchmark.results if not r.get('success')]
print(f"Nombre d'échecs: {len(failed_results)}")

for result in failed_results[:5]:  # Afficher les 5 premiers
    print(f"\n• {result['filename']}:")
    print(f"  Erreur: {result.get('error', 'Pas de message d\'erreur')}")

# Si certains réussissent, analyser aussi
successful_results = [r for r in benchmark.results if r.get('success')]
if successful_results:
    print(f"\nNombre de réussites: {len(successful_results)}")
    for result in successful_results[:3]:
        print(f"\n✓ {result['filename']}:")
        print(f"  Précision: {result.get('accuracy', 0)*100:.1f}%")
        print(f"  Tests: {result.get('correct_tests', 0)}/{result.get('total_tests', 0)}")
        print(f"  Règles: {result.get('rule_types', [])}")

# GÉNÉRER LES RAPPORTS ET VISUALISATIONS
print("\n" + "="*60)
print("GÉNÉRATION DES RAPPORTS ET VISUALISATIONS")
print("="*60)

# 1. Générer le rapport principal
print("\n1. Génération du rapport principal...")
benchmark.generate_report("resultats")

# 2. Générer les visualisations de grilles
print("\n2. Génération des visualisations de grilles...")
try:
    # Essayer d'abord avec la méthode de la classe
    benchmark.generate_grid_visualizations("grid_visualizations")
    print("✓ Visualisations générées via la méthode de classe")
except Exception as e:
    print(f"⚠️  Erreur avec la méthode de classe: {e}")
    
    # Fallback: créer une fonction simple
    print("Tentative avec une fonction simple...")
    def create_simple_visualizations(benchmark, output_dir="grid_visualizations"):
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        
        os.makedirs(output_dir, exist_ok=True)
        
        successful_results = [r for r in benchmark.results if r.get('success')]
        
        if not successful_results:
            print("Aucun résultat à visualiser.")
            return
        
        print(f"Création des visualisations pour {len(successful_results)} problèmes...")
        
        # Palette de couleurs ARC
        arc_colors = {
            0: [0, 0, 0],        # noir
            1: [255, 255, 255],  # blanc
            2: [255, 0, 0],      # rouge
            3: [0, 255, 0],      # vert
            4: [0, 0, 255],      # bleu
            5: [255, 255, 0],    # jaune
            6: [255, 0, 255],    # magenta
            7: [0, 255, 255],    # cyan
            8: [128, 128, 128],  # gris
            9: [255, 128, 0]     # orange
        }
        
        def grid_to_image(grid):
            h = len(grid)
            w = len(grid[0])
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    color = arc_colors.get(grid[i][j], [0, 0, 0])
                    img[i, j] = color
            return img
        
        # Limiter à 10 problèmes pour éviter trop d'images
        for idx, result in enumerate(successful_results[:10]):
            filename = result['filename']
            accuracy = result.get('accuracy', 0) * 100
            
            # Prendre le premier test
            test_results = result.get('test_results', [])
            if not test_results:
                continue
                
            test = test_results[0]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Input
            if 'input' in test:
                axes[0].imshow(grid_to_image(test['input']))
                axes[0].set_title('Input')
                axes[0].axis('off')
            
            # Prédit
            if 'predicted' in test:
                axes[1].imshow(grid_to_image(test['predicted']))
                color = 'green' if test.get('correct', False) else 'red'
                axes[1].set_title('Prédit', color=color)
                axes[1].axis('off')
            
            # Attendu
            if 'expected' in test and test['expected'] is not None:
                axes[2].imshow(grid_to_image(test['expected']))
                axes[2].set_title('Attendu')
                axes[2].axis('off')
            else:
                axes[2].text(0.5, 0.5, 'Pas de référence', 
                           ha='center', va='center', fontsize=12)
                axes[2].axis('off')
            
            plt.suptitle(f"{filename} - Précision: {accuracy:.1f}%")
            plt.tight_layout()
            
            # Sauvegarder
            safe_name = filename.replace('.json', '').replace(' ', '_')
            save_path = os.path.join(output_dir, f"{safe_name}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ {filename}")
        
        print(f"\nVisualisations sauvegardées dans: {os.path.abspath(output_dir)}")
    
    # Appeler la fonction simple
    create_simple_visualizations(benchmark, "grid_visualizations")

print("\n" + "="*60)
print("FIN DU BENCHMARK")
print("="*60)