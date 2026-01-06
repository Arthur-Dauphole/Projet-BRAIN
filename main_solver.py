import json
import os
import numpy as np

def load_arc_task(file_path):
    """
    Charge un fichier JSON ARC et sépare les exemples de l'énoncé.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # On prend généralement le premier exemple d'entraînement
    train_input = np.array(data['train'][0]['input'])
    train_output = np.array(data['train'][0]['output'])
    
    # On prend le premier énoncé de test (celui à résoudre)
    test_input = np.array(data['test'][0]['input'])
    
    return train_input, train_output, test_input


def serialize_for_llm(analysis_result):

    """
    Transforme les objets complexes (Set, Point, Enum) en dictionnaire simple.
    """
    clean_output = {
        "rectangles": [],
        "lines": []
    }

    # Helper pour convertir un Point en liste [x, y]
    def point_to_list(p):
        return [p.x, p.y]

    # 1. Traitement des Rectangles
    for rect in analysis_result['rectangles']:
        item = {
            "id": f"rect_{id(rect)}", # ID unique temporaire
            "type": "rectangle",
            "color": int(rect.color), # Convertir numpy int en int standard
            "position": {
                "top_left": [rect.bounding_box.min_x, rect.bounding_box.min_y],
                "center": point_to_list(rect.bounding_box.center)
            },
            "size": {"width": rect.bounding_box.width, "height": rect.bounding_box.height},
            "is_filled": rect.properties.get('is_filled', False),
            "is_square": rect.properties.get('is_square', False)
        }
        clean_output["rectangles"].append(item)

    # 2. Traitement des Lignes
    for line in analysis_result['lines']:
        # Conversion des endpoints (qui sont des objets Point)
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

# ============================================================================
# MOTEUR D'INFÉRENCE (LLM)
# ============================================================================

    # Une grille de test (Reprise de ton exemple)
    test_grid = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0], # Rectangle Bleu (1)
        [0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 2, 2], # Ligne Rouge (2)
        [0, 1, 1, 1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0, 0, 2, 2],
        [0, 0, 3, 3, 3, 3, 0, 0], # Ligne Verte (3)
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])

    resultat = solve_arc_task(test_grid)
    
    print("\n" + "="*50)
    print("RÉSULTAT DU SOLVER NEURO-SYMBOLIQUE")
    print("="*50)
    print(resultat)


    # ============================================================================
# MOTEUR D'INFÉRENCE (LLM) - Mise à jour
# ============================================================================

def deduce_and_solve_arc_task(train_input_grid, train_output_grid, test_input_grid):
    
    # 1. PERCEPTION SUR LES GRILLES (Ton code)
    print("👁️  Phase 1/3: Analyse géométrique des 3 grilles...")
    engine = GeometricDetectionEngine(background_color=0)
    
    # Analyse de l'exemple d'entraînement (Input)
    train_input_raw = engine.detect_all_shapes(train_input_grid)
    train_input_json = json.dumps(serialize_for_llm(train_input_raw), indent=2)
    
    # Analyse de l'exemple d'entraînement (Output)
    train_output_raw = engine.detect_all_shapes(train_output_grid)
    train_output_json = json.dumps(serialize_for_llm(train_output_raw), indent=2)

    # Analyse de l'énoncé à résoudre (Test Input)
    test_input_raw = engine.detect_all_shapes(test_input_grid)
    test_input_json = json.dumps(serialize_for_llm(test_input_raw), indent=2)
    
    # 2. CONSTRUCTION DU PROMPT MAÎTRE
    print("🧠 Phase 2/3: Construction du Prompt Maître pour déduction...")
    
    system_prompt = """
    Tu es un solveur de problèmes ARC-AGI. Ton travail est d'abord de DÉDUIRE la règle 
    de transformation entre la scène ENTREE et la scène SORTIE de l'EXEMPLE.
    Ensuite, tu dois APPLIQUER cette règle à la scène ENONCÉ.
    
    Ton unique réponse doit être une description JSON (strictement) 
    de la scène finale résolue (SORTIE TEST). 
    Ne génère aucun commentaire ou texte explicatif.
    """

    user_message = f"""
    --- DÉDUCTION DE LA RÈGLE (EXEMPLE) ---

    SCÈNE ENTRÉE EXEMPLE:
    {train_input_json}

    SCÈNE SORTIE EXEMPLE (Résolution):
    {train_output_json}

    --- APPLICATION DE LA RÈGLE (ÉNONCÉ) ---
    
    SCÈNE ENTRÉE ÉNONCÉ:
    {test_input_json}
    
    Génère le JSON de la SCÈNE SORTIE ÉNONCÉ.
    """
    
    # 3. RAISONNEMENT (LLM)
    print("⏳ Phase 3/3: Soumission au LLM. Déduction et Application en cours...")
    
    client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

    try:
        response = client.chat.completions.create(
            model="llama3", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1 # Toujours basse pour la logique
        )
        llm_response_text = response.choices[0].message.content
        
        # Tentative de parser le JSON retourné par le LLM (l'étape finale!)
        try:
            solved_json = json.loads(llm_response_text)
            print("✅ Résolution réussie. JSON de sortie parsé.")
            return solved_json
        except json.JSONDecodeError:
            print("❌ ERREUR: Le LLM n'a pas retourné un JSON valide.")
            return {"error": "LLM output was not valid JSON", "raw_output": llm_response_text}


    except Exception as e:
        return {"error": f"Erreur de connexion au LLM : {e}"}

# ============================================================================
# MAIN - EXEMPLE OPÉRATIONNEL
# ============================================================================

if __name__ == "__main__":
    
    # --- SIMULATION D'UNE TÂCHE ARC-AGI : Déplacer le carré ---
    
    # EXEMPLE D'ENTRAÎNEMENT: Carré Bleu (1) en haut à gauche -> Carré Bleu en bas à droite
    train_input = np.array([
        [0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    
    train_output = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ])
    
    # ÉNONCÉ À RÉSOUDRE: Carré Rouge (2) au centre -> ??? (Doit aller dans le coin)
    test_input = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0],
    ])

    resultat = deduce_and_solve_arc_task(train_input, train_output, test_input)
    
    print("\n" + "="*50)
    print("RÉSULTAT DU SOLVER (JSON de la SCÈNE SORTIE ÉNONCÉ)")
    print("="*50)
    
    # Affiche le résultat joliment
    if isinstance(resultat, dict) and 'error' in resultat:
        print(f"Erreur: {resultat['error']}")
        if 'raw_output' in resultat:
            print("\nSortie brute du LLM :")
            print(resultat['raw_output'])
    else:
        print(json.dumps(resultat, indent=2))
        
    # Idéalement, ici, tu aurais une fonction pour reconstruire la grille NumPy 
    # à partir du JSON du LLM et vérifier si elle est correcte.