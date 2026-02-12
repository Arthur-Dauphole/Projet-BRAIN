import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import textwrap
import json
import os
import glob
from perception import PerceptionSystem
from reasoning import ReasoningEngine

ARC_COLOR_MAP = [
    '#000000', '#1E93FF', '#F93C31', '#4FCC30', '#FFDC00',
    '#999999', '#E53AA3', '#FF851B', '#87D8F1', '#921231'
]
cmap_arc = ListedColormap(ARC_COLOR_MAP)
norm_arc = BoundaryNorm(range(11), cmap_arc.N)

# --- ENGINE ---
def run_exam(folder_path="tasks_exam"):
    ps = PerceptionSystem()
    engine = ReasoningEngine()
    task_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))
    results = []
    score = 0
    
    task_files = task_files[:10]
    total = len(task_files)
    
    print(f"--- STARTING EXAM : {total} TASKS ---")

    for file_path in task_files:
        with open(file_path, 'r') as f: data = json.load(f)
        grid_in = data['train'][0]['input']
        grid_out = data['train'][0]['output']
        expected_keyword = data.get('expected_logic', 'Unknown')
        
        objs_in = ps.extract_objects(grid_in)
        objs_out = ps.extract_objects(grid_out)
        detected_transformations = engine.compare_grids(objs_in, objs_out, grid_out)
        
        system_output_str = "\n".join(detected_transformations)
        
        # --- SMART VERIFICATION (ENGLISH) ---
        sys_low = system_output_str.lower()
        exp_clean = expected_keyword.replace(" + ", " ").replace(",", " ").lower()
        expected_tokens = exp_clean.split()
        
        official_keywords = ["translation", "rotation", "symmetry", "color", "fill"]
        missing_tokens = []
        
        for token in expected_tokens:
            if token in official_keywords: 
                token_found = False
                
                if token in sys_low: token_found = True
                elif token == "symmetry" and "symmetry" in sys_low: token_found = True
                elif token == "fill" and ("filled" in sys_low or "fill" in sys_low): token_found = True
                elif token == "color" and ("became" in sys_low or "color" in sys_low): token_found = True
                
                if not token_found: missing_tokens.append(token)

        # Tolerance
        if "rotation" in missing_tokens and "translation" in sys_low: missing_tokens.remove("rotation")
        if "translation" in missing_tokens and "new object" in sys_low: missing_tokens.remove("translation")

        success = (len(missing_tokens) == 0)
        if success: score += 1
        
        results.append({
            "input": grid_in, "target": grid_out,
            "success": success, "expected": expected_keyword,
            "detected": system_output_str.strip()
        })

    final_accuracy = (score / total) * 100 if total > 0 else 0
    return results, final_accuracy

# --- VISUALIZATION ---
def show_exam_report(results, accuracy):
    rows_layout = 2
    cols_layout = 5 
    
    fig, axes = plt.subplots(rows_layout, cols_layout * 2, figsize=(24, 12), dpi=100)
    
    SUCCESS_COLOR = '#27ae60'
    FAILURE_COLOR = '#c0392b'
    BG_COLOR = '#f4f6f7'
    
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f"BRAIN Report Card : {accuracy:.1f}% ({len([r for r in results if r['success']])}/{len(results)})", 
                 fontsize=24, fontweight='bold', color='#2c3e50', y=0.98)

    for i, res in enumerate(results):
        if i >= 10: break
        
        r = i // cols_layout
        c_start = (i % cols_layout) * 2
        ax_in = axes[r, c_start]
        ax_out = axes[r, c_start+1]
        
        # --- TITLE ---
        ax_in.text(0.55, 1.12, f"TEST {i+1}", transform=ax_in.transAxes,
                   ha='center', va='bottom', fontsize=12, fontweight='bold', color='#34495e')

        # --- GRIDS ---
        ax_in.imshow(res['input'], cmap=cmap_arc, norm=norm_arc, aspect='equal', zorder=1)
        ax_out.imshow(res['target'], cmap=cmap_arc, norm=norm_arc, aspect='equal', zorder=1)
        
        # --- LABELS --- 
        res_color = SUCCESS_COLOR if res['success'] else FAILURE_COLOR
        icon = "✓" if res['success'] else "✗"
        ax_out.set_title(f"Expected :\n{res['expected']} {icon}", fontsize=9, fontweight='bold', color=res_color, pad=8)
        
        # --- WHITE LINES ---
        h, w = len(res['input']), len(res['input'][0])
        sep_x, sep_y = np.arange(0.5, w, 1), np.arange(0.5, h, 1)
        
        for ax in [ax_in, ax_out]:
            for x in sep_x: ax.axvline(x, color='white', linewidth=0.5, alpha=0.3, zorder=2)
            for y in sep_y: ax.axhline(y, color='white', linewidth=0.5, alpha=0.3, zorder=2)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(h - 0.5, -0.5)
            for spine in ax.spines.values():
                if ax == ax_out: spine.set_linewidth(2.5); spine.set_edgecolor(res_color)
                else: spine.set_linewidth(1); spine.set_edgecolor('#bdc3c7')

        # --- DETECTED TEXT ---
        det_txt = res['detected'] if res['detected'] else "Nothing detected"
        wrapped_text = "\n".join(textwrap.wrap(det_txt, width=32))
        
        ax_in.text(1.10, -0.20, wrapped_text, transform=ax_in.transAxes, 
                   ha='center', va='top', fontsize=8, color='#2c3e50',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bdc3c7", alpha=0.9))

    for j in range(len(results), rows_layout * cols_layout):
        r, c = j // cols_layout, (j % cols_layout) * 2
        axes[r, c].axis('off'); axes[r, c+1].axis('off')

    plt.subplots_adjust(
        left=0.02, right=0.98, top=0.88, bottom=0.05, 
        wspace=0.15, hspace=0.60
    )
    plt.show()

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg') 
    res, acc = run_exam()
    show_exam_report(res, acc)