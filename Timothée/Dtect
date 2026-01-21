import json
from collections import defaultdict

def extract_rules(train_examples):
    """
    Extrait les règles des exemples d'entraînement.
    Retourne trois dictionnaires:
    - color_changes: {ancienne_couleur: nouvelle_couleur}
    - connections: {couleur: type} où type est 'H', 'V', ou 'D'
    - translations: {couleur: (dx, dy)}
    """
    color_changes = {}
    connections = {}
    translations = {}
    
    for idx, example in enumerate(train_examples):
        input_grid = example["input"]
        output_grid = example["output"]
        
        print(f"\n=== Analyse de l'exemple {idx+1} ===")
        
        # Dimensions
        h = len(input_grid)
        w = len(input_grid[0])
        
        # Étape 1: Détecter les changements de couleur
        for i in range(h):
            for j in range(w):
                old = input_grid[i][j]
                new = output_grid[i][j]
                if old != 0 and old != new:
                    if old in color_changes and color_changes[old] != new:
                        print(f"  Conflit: {old} -> {color_changes[old]} mais aussi -> {new}")
                    else:
                        color_changes[old] = new
                        print(f"  Changement de couleur: {old} -> {new}")
        
        # Étape 2: Pour chaque couleur de sortie, analyser les transformations
        # Construire les ensembles de points
        input_points = defaultdict(set)
        output_points = defaultdict(set)
        
        for i in range(h):
            for j in range(w):
                c_in = input_grid[i][j]
                c_out = output_grid[i][j]
                if c_in != 0:
                    input_points[c_in].add((i, j))
                if c_out != 0:
                    output_points[c_out].add((i, j))
        
        # Pour chaque couleur de sortie
        for c_out in output_points.keys():
            print(f"\n  Analyse pour la couleur de sortie {c_out}:")
            
            # Trouver la couleur d'entrée correspondante
            c_in_candidates = []
            # 1. Si une couleur a été changée en c_out
            for old, new in color_changes.items():
                if new == c_out:
                    c_in_candidates.append(old)
            # 2. La couleur c_out elle-même (si elle n'a pas été changée)
            if c_out not in color_changes.values():
                c_in_candidates.append(c_out)
            
            # Construire P_in (points d'entrée correspondants)
            P_in = set()
            for c_in in c_in_candidates:
                if c_in in input_points:
                    P_in.update(input_points[c_in])
            
            P_out = output_points[c_out]
            
            if not P_in:
                print(f"    Aucun point d'entrée correspondant trouvé pour {c_out}")
                continue
            
            print(f"    Points d'entrée: {sorted(P_in)}")
            print(f"    Points de sortie: {sorted(P_out)}")
            
            # Étape 3: Vérifier si c'est une translation
            translation_found = False
            if P_in and P_out and len(P_in) == len(P_out):
                # Prendre un point de référence
                ref_in = next(iter(P_in))
                # Chercher un déplacement possible
                possible_displacements = []
                for p_out in P_out:
                    dx = p_out[0] - ref_in[0]
                    dy = p_out[1] - ref_in[1]
                    possible_displacements.append((dx, dy))
                
                # Tester chaque déplacement
                for dx, dy in possible_displacements:
                    match_all = True
                    for p_in in P_in:
                        moved = (p_in[0] + dx, p_in[1] + dy)
                        if moved not in P_out:
                            match_all = False
                            break
                    if match_all:
                        translations[c_out] = (dx, dy)
                        translation_found = True
                        print(f"    Translation trouvée: ({dx}, {dy})")
                        break
            
            # Étape 4: Si pas de translation, chercher des connexions
            if not translation_found and P_out:
                # Points de base: intersection entre P_in et P_out
                base_points = P_in.intersection(P_out)
                added_points = P_out - P_in
                
                print(f"    Points de base: {sorted(base_points)}")
                print(f"    Points ajoutés: {sorted(added_points)}")
                
                if added_points:
                    # Chercher une paire de points de base alignés
                    base_list = list(base_points)
                    for i in range(len(base_list)):
                        for j in range(i+1, len(base_list)):
                            p1 = base_list[i]
                            p2 = base_list[j]
                            
                            # Vérifier l'alignement
                            if p1[0] == p2[0]:  # Même ligne -> horizontal
                                # Tous les points entre p1 et p2 sur la même ligne
                                min_y = min(p1[1], p2[1])
                                max_y = max(p1[1], p2[1])
                                line_points = [(p1[0], y) for y in range(min_y, max_y+1)]
                                
                                # Vérifier si tous ces points sont dans P_out
                                if all(p in P_out for p in line_points):
                                    connections[c_out] = 'H'
                                    print(f"    Connexion horizontale entre {p1} et {p2}")
                                    break
                            
                            elif p1[1] == p2[1]:  # Même colonne -> vertical
                                # Tous les points entre p1 et p2 sur la même colonne
                                min_x = min(p1[0], p2[0])
                                max_x = max(p1[0], p2[0])
                                line_points = [(x, p1[1]) for x in range(min_x, max_x+1)]
                                
                                # Vérifier si tous ces points sont dans P_out
                                if all(p in P_out for p in line_points):
                                    connections[c_out] = 'V'
                                    print(f"    Connexion verticale entre {p1} et {p2}")
                                    break
                            
                            elif abs(p1[0] - p2[0]) == abs(p1[1] - p2[1]):  # Diagonale
                                # Tous les points entre p1 et p2 sur la diagonale
                                dx = 1 if p2[0] > p1[0] else -1
                                dy = 1 if p2[1] > p1[1] else -1
                                steps = abs(p2[0] - p1[0])
                                line_points = [(p1[0] + k*dx, p1[1] + k*dy) for k in range(steps+1)]
                                
                                # Vérifier si tous ces points sont dans P_out
                                if all(p in P_out for p in line_points):
                                    connections[c_out] = 'D'
                                    print(f"    Connexion diagonale entre {p1} et {p2}")
                                    break
                    
                    # Si on n'a pas trouvé de connexion avec les points de base,
                    # on peut essayer avec les points de P_in qui ne sont pas dans P_out?
                    # Pour l'instant, on garde cette approche
    
    return color_changes, connections, translations

# Données d'entraînement
train_data = {
  "train": [
    {
      "input": [
        [2, 2, 2, 2, 2, 0, 0],
        [2, 0, 0, 0, 2, 0, 0],
        [2, 0, 2, 0, 2, 0, 0],
        [2, 0, 0, 0, 2, 0, 0],
        [2, 2, 2, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
      ],
      "output": [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 2, 2],
        [0, 0, 2, 0, 0, 0, 2],
        [0, 0, 2, 0, 2, 0, 2],
        [0, 0, 2, 0, 0, 0, 2],
        [0, 0, 2, 2, 2, 2, 2]
      ]
    },
    {
      "input": [
        [0, 7, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 8],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
      ],
      "output": [
        [0, 9, 0, 0, 0, 0, 0],
        [0, 9, 0, 0, 8, 0, 0],
        [0, 9, 0, 0, 0, 8, 0],
        [0, 9, 0, 0, 0, 0, 8],
        [0, 9, 0, 0, 0, 0, 0],
        [0, 9, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
      ]
    },
    {
      "input": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 2, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
      ],
      "output": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 3, 3, 3, 3, 3, 3, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
      ]
    }
  ]
}

# Extraire les règles
color_changes, connections, translations = extract_rules(train_data["train"])

print("\n=== RÈGLES EXTRACTES ===")
print(f"Changements de couleur: {color_changes}")
print(f"Connexions: {connections}")
print(f"Translations: {translations}")

# Fonction pour appliquer les règles à une grille d'entrée

def apply_rules(input_grid, color_changes, connections, translations):
    """
    Applique les règles à une grille d'entrée.
    L'ordre d'application:
    1. Changements de couleur
    2. Connexions
    3. Translations
    """
    # Copie de la grille d'entrée
    h = len(input_grid)
    w = len(input_grid[0])
    grid = [row[:] for row in input_grid]
    
    # Étape 1: Changements de couleur
    for i in range(h):
        for j in range(w):
            if grid[i][j] in color_changes:
                grid[i][j] = color_changes[grid[i][j]]
    
    # Étape 2: Connexions
    # Pour chaque couleur avec une règle de connexion
    for color, conn_type in connections.items():
        # Trouver tous les points de cette couleur
        points = [(i, j) for i in range(h) for j in range(w) if grid[i][j] == color]
        
        if len(points) >= 2:
            # Pour chaque paire de points
            for k in range(len(points)):
                for l in range(k+1, len(points)):
                    p1 = points[k]
                    p2 = points[l]
                    
                    # Vérifier l'alignement selon le type de connexion
                    if conn_type == 'V' and p1[1] == p2[1]:  # Vertical
                        min_x = min(p1[0], p2[0])
                        max_x = max(p1[0], p2[0])
                        for x in range(min_x, max_x + 1):
                            if grid[x][p1[1]] == 0:  # Remplir seulement si vide
                                grid[x][p1[1]] = color
                    
                    elif conn_type == 'H' and p1[0] == p2[0]:  # Horizontal
                        min_y = min(p1[1], p2[1])
                        max_y = max(p1[1], p2[1])
                        for y in range(min_y, max_y + 1):
                            if grid[p1[0]][y] == 0:  # Remplir seulement si vide
                                grid[p1[0]][y] = color
                    
                    elif conn_type == 'D':  # Diagonale
                        # Vérifier si c'est une diagonale parfaite
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        if abs(dx) == abs(dy):
                            step_x = 1 if dx > 0 else -1
                            step_y = 1 if dy > 0 else -1
                            steps = abs(dx)
                            for s in range(steps + 1):
                                x = p1[0] + s * step_x
                                y = p1[1] + s * step_y
                                if grid[x][y] == 0:  # Remplir seulement si vide
                                    grid[x][y] = color
    
    # Étape 3: Translations
    # Pour chaque couleur avec une règle de translation
    for color, (dx, dy) in translations.items():
        # Créer une nouvelle grille temporaire
        new_grid = [[0 for _ in range(w)] for _ in range(h)]
        
        # Copier toutes les couleurs
        for i in range(h):
            for j in range(w):
                if 0 <= i < h and 0 <= j < w:
                    new_grid[i][j] = grid[i][j]
        
        # Appliquer la translation
        for i in range(h):
            for j in range(w):
                if grid[i][j] == color:
                    new_i = i + dx
                    new_j = j + dy
                    # Effacer l'ancienne position
                    new_grid[i][j] = 0
                    # Mettre à la nouvelle position si dans les limites
                    if 0 <= new_i < h and 0 <= new_j < w:
                        new_grid[new_i][new_j] = color
        
        grid = new_grid
    
    return grid

# Test avec un exemple simple
test_input = [
    [0, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 8],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]

print("Grille d'entrée:")
for row in test_input:
    print(row)

output = apply_rules(test_input, color_changes, connections, translations)

print("\nGrille de sortie après application des règles:")
for row in output:
    print(row)