import json
import os
import glob
from perception import PerceptionSystem
from reasoning import ReasoningEngine
import matplotlib.pyplot as plt
import numpy as np

class BrainBatchRunner:
    def __init__(self, tasks_path):
        self.tasks_path = tasks_path
        self.ps = PerceptionSystem()
        self.engine = ReasoningEngine()
        self.results = []

    def run_all(self):
            task_files = glob.glob(os.path.join(self.tasks_path, "*.json"))
            correct = 0
            
            for file_path in task_files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # --- CORRECTION ICI ---
                # Si data est une liste, on prend le premier élément
                if isinstance(data, list):
                    example = data[0]
                else:
                    example = data
                
                try:
                    grid_in = example['train'][0]['input']
                    grid_target = example['train'][0]['output']
                except (KeyError, IndexError) as e:
                    print(f"Format invalide dans {file_path}: {e}")
                    continue
                # -----------------------

                # Pipeline BRAIN
                objs_in = self.ps.extract_objects(grid_in)
                objs_out = self.ps.extract_objects(grid_target)
                
                # Analyse
                transformations = self.engine.compare_grids(objs_in, objs_out, grid_target)
                
                # Succès si au moins une règle logique est extraite
                success = len(transformations) > 0 
                if success: correct += 1
                
                self.results.append({
                    "name": os.path.basename(file_path),
                    "input": grid_in,
                    "target": grid_target,
                    "success": success
                })
                
            accuracy = (correct / len(task_files)) * 100 if task_files else 0
            return accuracy, self.results

def plot_batch_results(accuracy, results):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(10, n * 3))
    fig.suptitle(f"BRAIN Batch Results: {accuracy:.1f}% accuracy", fontsize=16)

    for i, res in enumerate(results):
        # Affichage Input
        axes[i, 0].imshow(res['input'], cmap='nipy_spectral', vmin=0, vmax=9)
        axes[i, 0].set_title(f"Input: {res['name']}")
        
        # Affichage Target (avec indicateur de succès)
        color = 'green' if res['success'] else 'red'
        axes[i, 1].imshow(res['target'], cmap='nipy_spectral', vmin=0, vmax=9)
        axes[i, 1].set_title(f"Target (Predicted: {'✓' if res['success'] else '✗'})", color=color)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # 1. Vérifie bien que tes JSON sont dans le dossier "tasks" au même niveau que ce script
    # Sinon, mets le chemin complet : "/Users/paullefrais/Documents/.../tasks"
    folder = "/Users/paullefrais/Documents/ISAE SUPAERO/Cours Supaero/2A/Projet R&D Brain/Projet-BRAIN-VSCODE/Projet-BRAIN/Paul/tasks" 
    
    runner = BrainBatchRunner(folder)
    accuracy, results = runner.run_all()
    
    print(f"Nombre de tâches analysées : {len(results)}")
    print(f"Précision globale : {accuracy:.1f}%")
    
    if len(results) > 0:
        # Appelle la fonction de visualisation (assure-toi qu'elle est définie au-dessus)
        plot_batch_results(accuracy, results)
    else:
        print("Erreur : Aucun fichier JSON trouvé dans le dossier 'tasks'.")

def plot_batch_results(accuracy, results):
    n = len(results)
    if n == 0: return
    
    # On crée une grille : 2 colonnes (Input / Expected)
    cols = 2
    rows = n
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows))
    fig.suptitle(f"BRAIN Batch Results: {accuracy:.1f}% accuracy", fontsize=16, fontweight='bold', y=0.99)

    # Si n=1, axes n'est pas une matrice 2D, on le transforme
    if n == 1: axes = np.expand_dims(axes, axis=0)

    for i, res in enumerate(results):
        # Affichage Input
        axes[i, 0].imshow(res['input'], cmap='nipy_spectral', vmin=0, vmax=9)
        axes[i, 0].set_title(f"Input: {res['name']}", fontsize=10)
        
        # Affichage Target
        axes[i, 1].imshow(res['target'], cmap='nipy_spectral', vmin=0, vmax=9)
        color = 'green' if res['success'] else 'red'
        status = "Predicted ✓" if res['success'] else "Predicted ✗"
        axes[i, 1].set_title(f"{status}", color=color, fontweight='bold')

        # Cosmétique : enlever les axes et ajouter une grille
        for j in range(2):
            axes[i, j].set_xticks(np.arange(-0.5, len(res['input'][0]), 1), minor=True)
            axes[i, j].set_yticks(np.arange(-0.5, len(res['input']), 1), minor=True)
            axes[i, j].grid(which='minor', color='white', linewidth=0.5)
            axes[i, j].tick_params(which='both', labelbottom=False, labelleft=False, length=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()