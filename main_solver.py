import json
import os
import numpy as np
import re
from openai import OpenAI
from geometric_engine import GeometricDetectionEngine
import matplotlib.pyplot as plt
from matplotlib import colors


# ============================================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================

def load_arc_task(file_path):
    """Charge un fichier JSON ARC et sépare les exemples de l'énoncé."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Premier exemple d'entraînement
    train_input = np.array(data['train'][0]['input'])
    train_output = np.array(data['train'][0]['output'])
    
    # Premier énoncé de test (énoncé à résoudre)
    test_input = np.array(data['test'][0]['input'])
    
    return train_input, train_output, test_input

def serialize_for_llm(analysis_result):
    """Transforme les objets complexes en dictionnaire JSON simple pour le LLM."""
    clean_output = {"rectangles": [], "lines": []}

    def point_to_list(p):
        return [p.x, p.y]

    # Traitement des Rectangles
    for rect in analysis_result['rectangles']:
        item = {
            "id": f"rect_{id(rect)}",
            "type": "rectangle",
            "color": int(rect.color),
            "position": {
                "top_left": [rect.bounding_box.min_x, rect.bounding_box.min_y],
                "center": point_to_list(rect.bounding_box.center)
            },
            "size": {"width": rect.bounding_box.width, "height": rect.bounding_box.height},
            "is_filled": rect.properties.get('is_filled', False),
            "is_square": rect.properties.get('is_square', False)
        }
        clean_output["rectangles"].append(item)

    # Traitement des Lignes
    for line in analysis_result['lines']:
        endpoints = [point_to_list(p) for p in line.properties.get('endpoints', [])]
        item = {
            "id": f"line_{id(line)}",
            "type": "line",
            "color": int(line.color),
            "direction": line.properties.get('direction'),
            "length": line.properties.get('length'),
            "endpoints": endpoints
        }
        clean_output["lines"].append(item)

    return clean_output


def json_to_grid(json_data, grid_size=(10, 10)):
    """
    Transforme la description JSON du LLM en une grille NumPy exploitable.
    """
    # Créer une grille vide (couleur 0 par défaut)
    grid = np.zeros(grid_size, dtype=int)
    
    # 1. Dessiner les rectangles
    for rect in json_data.get('rectangles', []):
        x, y = rect['position']['top_left']
        w, h = rect['size']['width'], rect['size']['height']
        color = rect['color']
        
        # On remplit la zone du rectangle (en vérifiant de ne pas sortir de la grille)
        grid[x:x+w, y:y+h] = color
        
    # 2. Dessiner les lignes
    for line in json_data.get('lines', []):
        color = line['color']
        for pt in line.get('endpoints', []):
            # Ici on simplifie en dessinant les points des endpoints
            # Pour une ligne complète, il faudrait une boucle de remplissage
            grid[pt[0], pt[1]] = color
            
    return grid

# ============================================================================
# POINT D'ENTRÉE DU PROGRAMME
# ============================================================================


def plot_arc_grid(grid, title="Grille ARC"):
    """Affiche une grille avec les couleurs officielles ARC."""
    # Palette officielle ARC (0 à 9)
    arc_colors = [
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ]
    cmap = colors.ListedColormap(arc_colors)
    norm = colors.BoundaryNorm(np.arange(-0.5, 10, 1), cmap.N)

    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.grid(True, which='both', color='gray', linewidth=0.5)
    
    # Affichage des chiffres dans les cases (optionnel)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            plt.text(j, i, str(grid[i, j]), ha='center', va='center', 
                     color='white' if grid[i, j] in [0, 1, 2, 9] else 'black')

    plt.title(title)
    plt.xticks(np.arange(-0.5, grid.shape[1], 1), [])
    plt.yticks(np.arange(-0.5, grid.shape[0], 1), [])
    plt.show()

def deduce_and_solve_arc_task(train_input_grid, train_output_grid, test_input_grid):
    """Boucle complète : Perception -> Prompt -> LLM -> Solution."""
    
    # 1. PERCEPTION
    print("👁️  Phase 1/3: Analyse géométrique des grilles...")
    engine = GeometricDetectionEngine(background_color=0)
    
    tr_in_json = json.dumps(serialize_for_llm(engine.detect_all_shapes(train_input_grid)), indent=2)
    tr_out_json = json.dumps(serialize_for_llm(engine.detect_all_shapes(train_output_grid)), indent=2)
    te_in_json = json.dumps(serialize_for_llm(engine.detect_all_shapes(test_input_grid)), indent=2)
    
    # 2. CONSTRUCTION DU PROMPT
    print("🧠 Phase 2/3: Construction du Prompt Maître...")
    
    system_prompt = (
        "Tu es un solveur ARC-AGI. Analyse la transformation entre l'ENTREE et la SORTIE de l'EXEMPLE. "
        "Applique cette règle à l'ENONCÉ. Ta réponse doit être EXCLUSIVEMENT un objet JSON valide. "
        "Pas de texte, pas de blabla."
    )

    user_message = f"""
    --- EXEMPLE ---
    ENTREE: {tr_in_json}
    SORTIE: {tr_out_json}

    --- ENONCÉ ---
    ENTREE: {te_in_json}
    
    Génère le JSON de la SORTIE de l'ENONCÉ.
    """
    
    # 3. APPEL OLLAMA (GEMMA)
    print("⏳ Phase 3/3: Soumission à Gemma (via Ollama)...")
    try:
        client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        
        response = client.chat.completions.create(
            model="gemma3:1b", # Vérifie que c'est bien le nom dans 'ollama list'
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1 # On baisse la créativité pour plus de logique
        )

        llm_response_text = response.choices[0].message.content
        
        # Extraction du JSON par Regex (pour ignorer le texte superflu)
        json_match = re.search(r"(\{.*\})", llm_response_text, re.DOTALL)
        
        if json_match:
            clean_json_str = json_match.group(1)
            try:
                solved_json = json.loads(clean_json_str)
                print("✅ Résolution réussie. JSON extrait avec succès.")
                return solved_json
            except json.JSONDecodeError:
                print("❌ ERREUR: Le texte extrait n'est pas un JSON valide.")
        else:
            print("❌ ERREUR: Aucun JSON trouvé dans la réponse du LLM.")
            
        return {"error": "Invalid output", "raw": llm_response_text}

    except Exception as e:
        print(f"❌ ERREUR CRITIQUE: {e}")
        return {"error": str(e)}
    
# ============================================================================
# POINT D'ENTRÉE DU PROGRAMME
# ============================================================================

if __name__ == "__main__":
    # 1. Chemin vers ton fichier
    task_file = "/Users/paullefrais/Documents/ISAE SUPAERO/Cours Supaero/2A/Projet R&D Brain/Projet-BRAIN-VSCODE/Projet-BRAIN/Paul/03560426.json"

    if not os.path.exists(task_file):
        print(f"❌ Fichier non trouvé : {task_file}")
    else:
        print(f"🚀 Chargement de la tâche : {os.path.basename(task_file)}")
        
        # CHARGEMENT DES DONNÉES
        tr_in, tr_out, te_in = load_arc_task(task_file)

        # 2. Exécution du Solver
        resultat_json = deduce_and_solve_arc_task(tr_in, tr_out, te_in)
        
        # 3. Visualisation et Sauvegarde
        if "error" not in resultat_json:
            # On génère la grille avec la taille de l'énoncé
            grid_result = json_to_grid(resultat_json, grid_size=te_in.shape)
            
            # Affichage terminal
            print("\n🎨 GRILLE GÉNÉRÉE PAR L'IA (NUMPY) :")
            print(grid_result)
            
            # Sauvegarde du JSON
            output_filename = "solution_brain.json"
            with open(output_filename, "w") as f:
                json.dump(resultat_json, f, indent=4)
            print(f"💾 Résultat sauvegardé dans {output_filename}")

            # --- AFFICHAGE VISUEL (MATPLOTLIB) ---
            print("📊 Ouverture de la fenêtre de visualisation...")
            plot_arc_grid(te_in, title="Entrée (Test Input)")
            plot_arc_grid(grid_result, title="Solution générée par BRAIN (Gemma)")
        else:
            print(f"❌ Échec de la résolution : {resultat_json.get('error')}")