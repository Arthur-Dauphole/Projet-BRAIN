"""
SIMPLE RUNNER - Version la plus simple
"""

import os

# Assurez-vous d'être dans le bon dossier
print("Dossier courant:", os.getcwd())

# Liste des fichiers dans le dossier
print("\nFichiers disponibles:")
for f in os.listdir("."):
    if f.endswith(".py"):
        print(f"  - {f}")

# Importer directement (ils doivent être dans le même dossier)
from Merge_ResolutionARC import *
from arc_benchmark import ARCBatchBenchmark

# Chemin vers vos données (relatif au dossier courant)
data_path = "Arthur_2/BRAIN_PROJECT/data"

# Vérifier
if not os.path.exists(data_path):
    print(f"\n ERREUR: Dossier '{data_path}' introuvable!")
    print("Placez-vous dans le dossier qui contient 'Arthur_2'")
    exit(1)

# Lancer le benchmark
print(f"\n Lancement du benchmark avec {data_path}")
benchmark = ARCBatchBenchmark(use_shape_detection=True, verbose=True)
benchmark.run_benchmark(data_path)

# Après benchmark.run_benchmark(full_data_path)

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
benchmark.generate_report("resultats")
benchmark.generate_grid_visualizations("grid_visualizations")
