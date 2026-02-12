"""
RUNNER POUR VISUALISATIONS COMPLÈTES
"""

import os
from Merge_ResolutionARC import *
from arc_benchmark import ARCBatchBenchmark
# Importez la fonction ci-dessus si vous l'avez mise dans un fichier séparé
from Complete_visualization import create_complete_visualization

print("Dossier courant:", os.getcwd())

# Vérification du dossier data
data_path = "C:\\Users\\timor\\.virtual_documents\\Projet-BRAIN\\Arthur_2\\BRAIN_PROJECT\\data"
if not os.path.exists(data_path):
    print(f"ERREUR: Dossier '{data_path}' introuvable!")
    exit(1)

print("Chargement des modules...")
from Merge_ResolutionARC import *
from arc_benchmark import ARCBatchBenchmark

# Lancer le benchmark
print(f"\nLancement du benchmark avec {data_path}")
benchmark = ARCBatchBenchmark(use_shape_detection=True, verbose=True)
benchmark.run_benchmark(data_path)

# Générer les visualisations COMPLÈTES
print("\n" + "="*60)
print("GÉNÉRATION DES VISUALISATIONS COMPLÈTES")
print("="*60)

# Créer la fonction inline si elle n'est pas importée
# (Copiez-collez la fonction create_complete_visualization ici si besoin)

# Appeler la fonction
create_complete_visualization(
    benchmark.results, 
    output_dir="complete_visualizations",
    max_problems=20  # Limitez à 20 problèmes pour ne pas surcharger
)

print("\n" + "="*60)
print("TERMINÉ !")
print("="*60)
print("Ouvrez le dossier 'complete_visualizations' pour voir les grandes images.")
print("Chaque image montre TOUS les exemples d'entraînement et TOUS les tests.")