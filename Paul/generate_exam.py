import json
import os
import shutil

output_dir = "tasks_exam"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

tasks = []

# --- 1. TRANSLATION (3 cas) ---
# Simple droite
t1_in = [[0,0,0,0,0],[0,2,0,0,0],[0,2,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
t1_out= [[0,0,0,0,0],[0,0,0,2,0],[0,0,0,2,0],[0,0,0,0,0],[0,0,0,0,0]]
tasks.append({"name": "trans_right", "expected_logic": "Translation", "input": t1_in, "output": t1_out})

# Diagonale
t2_in = [[3,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
t2_out= [[0,0,0,0,0],[0,0,0,0,0],[0,0,3,0,0],[0,0,0,0,0],[0,0,0,0,0]]
tasks.append({"name": "trans_diag", "expected_logic": "Translation", "input": t2_in, "output": t2_out})

# Bloc complet
t3_in = [[0,0,0,0,0],[0,4,4,0,0],[0,4,4,0,0],[0,0,0,0,0],[0,0,0,0,0]]
t3_out= [[0,0,0,0,0],[0,0,0,0,0],[0,0,4,4,0],[0,0,4,4,0],[0,0,0,0,0]]
tasks.append({"name": "trans_block", "expected_logic": "Translation", "input": t3_in, "output": t3_out})

# --- 2. COULEUR (2 cas) ---
t4_in = [[0,0,0,0,0],[0,1,1,1,0],[0,1,0,1,0],[0,1,1,1,0],[0,0,0,0,0]] 
t4_out= [[0,0,0,0,0],[0,2,2,2,0],[0,2,0,2,0],[0,2,2,2,0],[0,0,0,0,0]] 
tasks.append({"name": "color_swap", "expected_logic": "Couleur", "input": t4_in, "output": t4_out})

t5_in = [[5,5,5,5,5],[5,0,0,0,5],[5,0,0,0,5],[5,0,0,0,5],[5,5,5,5,5]]
t5_out= [[2,2,2,2,2],[2,0,0,0,2],[2,0,0,0,2],[2,0,0,0,2],[2,2,2,2,2]]
tasks.append({"name": "color_frame", "expected_logic": "Couleur", "input": t5_in, "output": t5_out})

# --- 3. ROTATION & SYMÉTRIE (Centrés pour éviter la translation) ---
# Rotation 90 d'une barre autour du centre (2,2)
# Barre horizontale (1,2)-(2,2)-(3,2) devient Verticale (2,1)-(2,2)-(2,3)
t6_in = [[0,0,0,0,0],[0,0,0,0,0],[0,1,1,1,0],[0,0,0,0,0],[0,0,0,0,0]]
t6_out= [[0,0,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,0,0]]
tasks.append({"name": "rot_90_center", "expected_logic": "Rotation", "input": t6_in, "output": t6_out})

# Symétrie Miroir (L'objet est collé à l'axe pour minimiser le mouvement visuel)
t7_in = [[0,0,0,0,0],[0,2,2,0,0],[0,2,0,0,0],[0,2,0,0,0],[0,0,0,0,0]]
t7_out= [[0,0,0,0,0],[0,0,2,2,0],[0,0,0,2,0],[0,0,0,2,0],[0,0,0,0,0]]
tasks.append({"name": "mirror_h", "expected_logic": "Symétrie", "input": t7_in, "output": t7_out})

# Rotation 180 (Forme en S centrée)
t8_in = [[0,0,0,0,0],[0,0,3,3,0],[0,3,3,0,0],[0,0,0,0,0],[0,0,0,0,0]]
t8_out= [[0,0,0,0,0],[0,0,3,3,0],[0,3,3,0,0],[0,0,0,0,0],[0,0,0,0,0]] # 180 d'un S centré = identique, mauvais test.
# Changeons pour un T :
t8_in_T = [[0,0,0,0,0],[0,3,3,3,0],[0,0,3,0,0],[0,0,0,0,0],[0,0,0,0,0]]
t8_out_T= [[0,0,0,0,0],[0,0,3,0,0],[0,3,3,3,0],[0,0,0,0,0],[0,0,0,0,0]]
tasks.append({"name": "rot_180", "expected_logic": "Rotation", "input": t8_in_T, "output": t8_out_T})

# --- 4. RÈGLES / REMPLISSAGE (2 cas) ---
t9_in = [[0,0,0,0,0],[0,2,2,2,0],[0,2,0,2,0],[0,2,2,2,0],[0,0,0,0,0]]
t9_out= [[0,0,0,0,0],[0,2,2,2,0],[0,2,1,2,0],[0,2,2,2,0],[0,0,0,0,0]]
tasks.append({"name": "fill_hole", "expected_logic": "Règle", "input": t9_in, "output": t9_out})

t10_in= [[0,0,0,0,0],[0,0,0,0,0],[0,4,0,4,0],[0,0,0,0,0],[0,0,0,0,0]]
t10_out=[[0,0,0,0,0],[0,0,0,0,0],[0,4,4,4,0],[0,0,0,0,0],[0,0,0,0,0]]
tasks.append({"name": "complete_line", "expected_logic": "Règle", "input": t10_in, "output": t10_out})

# Génération
for i, t in enumerate(tasks):
    filename = os.path.join(output_dir, f"task_{i+1:02d}_{t['name']}.json")
    with open(filename, 'w') as f:
        data = { "train": [{"input": t['input'], "output": t['output']}], "expected_logic": t['expected_logic'] }
        json.dump(data, f, indent=2)

print(f"✅ 10 fichiers calibrés (5x5) générés dans '{output_dir}'")