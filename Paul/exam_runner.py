import matplotlib.pyplot as plt
import numpy as np
import textwrap
import json
import os
import glob
from perception import PerceptionSystem
from reasoning import ReasoningEngine

# --- PARTIE 1 : MOTEUR D'EXAMEN (Inchangé) ---
def run_exam(folder_path="tasks_exam"):
    ps = PerceptionSystem()
    engine = ReasoningEngine()
    task_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))
    results = []
    score = 0
    total = len(task_files)
    print(f"--- DÉBUT DE L'EXAMEN : {total} ÉPREUVES ---")

    for file_path in task_files:
        with open(file_path, 'r') as f: data = json.load(f)
        grid_in = data['train'][0]['input']
        grid_out = data['train'][0]['output']
        expected_keyword = data.get('expected_logic', 'Inconnu')
        
        objs_in = ps.extract_objects(grid_in)
        objs_out = ps.extract_objects(grid_out)
        detected_transformations = engine.compare_grids(objs_in, objs_out, grid_out)
        
        system_output_str = ""
        for item in detected_transformations:
            if isinstance(item, str): system_output_str += item + " "
            else: system_output_str += " ".join(item.actions) + " "
        
        success = expected_keyword.lower() in system_output_str.lower()
        if success: score += 1
        
        results.append({
            "input": grid_in, "target": grid_out,
            "success": success, "expected": expected_keyword,
            "detected": system_output_str.strip()
        })

    final_accuracy = (score / total) * 100
    return results, final_accuracy

# --- PARTIE 2 : VISUALISATION BLINDÉE ---
def show_exam_report(results, accuracy):
    rows_layout = 2
    cols_layout = 5 
    
    # DPI élevé pour la netteté
    fig, axes = plt.subplots(rows_layout, cols_layout * 2, figsize=(20, 12), dpi=120)
    
    SUCCESS_COLOR = '#27ae60'  # Vert
    FAILURE_COLOR = '#c0392b'  # Rouge
    BG_COLOR = '#f4f6f7'       # Gris fond
    
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f"Tests connaissances systeme BRAIN : {accuracy:.1f}% ({len([r for r in results if r['success']])}/{len(results)})", 
                 fontsize=20, fontweight='bold', color='#2c3e50', y=0.97)

    for i, res in enumerate(results):
        if i >= 10: break
        
        r = i // cols_layout
        c_start = (i % cols_layout) * 2
        ax_in = axes[r, c_start]
        ax_out = axes[r, c_start+1]
        
        # --- 1. TITRE PRINCIPAL (Position très haute) ---
        # y=1.35 assure qu'il est loin au-dessus
        ax_in.text(1.10, 1.5, f"TEST {i+1}", transform=ax_in.transAxes,
                   ha='center', va='bottom', fontsize=14, fontweight='bold', color='#34495e')

        # --- 2. GRILLES ---
        # On utilise zorder=1 pour les couleurs
        ax_in.imshow(res['input'], cmap='nipy_spectral', vmin=0, vmax=9, aspect='equal', zorder=1)
        ax_out.imshow(res['target'], cmap='nipy_spectral', vmin=0, vmax=9, aspect='equal', zorder=1)
        
        # --- 3. SOUS-TITRES ---
        ax_in.set_title("Input", fontsize=10, color='#7f8c8d', pad=8)
        
        res_color = SUCCESS_COLOR if res['success'] else FAILURE_COLOR
        icon = "✓" if res['success'] else "✗"
        ax_out.set_title(f"Attendu :\n{res['expected']} {icon}", fontsize=8, fontweight='bold', color=res_color, pad=8)
        
        # --- 4. LA GRILLE MANUELLE (LE FIX) ---
        # Au lieu de ax.grid(), on trace des lignes nous-mêmes.
        # Pour une grille 5x5, les séparations sont à 0.5, 1.5, 2.5, 3.5, 4.5
        separators = np.arange(0.5, 5, 1)
        
        for ax in [ax_in, ax_out]:
            # Lignes Verticales
            for x in separators:
                ax.axvline(x, color='white', linewidth=1, zorder=1)
            # Lignes Horizontales
            for y in separators:
                ax.axhline(y, color='white', linewidth=1, zorder=1)
            
            # On nettoie les axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Cadre de couleur autour
            for spine in ax.spines.values():
                if ax == ax_out:
                    spine.set_linewidth(3)
                    spine.set_edgecolor(res_color)
                else:
                    spine.set_linewidth(1)
                    spine.set_edgecolor('#bdc3c7')

        # --- 5. TEXTE D'ANALYSE ---
        det_txt = res['detected'] if res['detected'] else "Rien détecté"
        wrapped_text = "\n".join(textwrap.wrap(det_txt, width=28))
        
        # Position basse (y=-0.20)
        ax_in.text(1.10, -0.20, wrapped_text, 
                   transform=ax_in.transAxes, 
                   ha='center', va='top', fontsize=9, color='#2c3e50',
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bdc3c7", alpha=0.9))

    # Masquer les vides
    total_slots = rows_layout * cols_layout
    for j in range(len(results), total_slots):
        r = j // cols_layout
        c = (j % cols_layout) * 2
        axes[r, c].axis('off')
        axes[r, c+1].axis('off')

    # Espacement large
    plt.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.12, wspace=0.3, hspace=0.6)
    plt.show()

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg') 
    res, acc = run_exam()
    show_exam_report(res, acc)