import json
import os
import shutil
import random
import time

output_dir = "tasks_exam"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

GRID_SIZE = 15

# --- UTILITAIRES ---
def get_blank():
    return [[0]*GRID_SIZE for _ in range(GRID_SIZE)]

def is_valid(r, c):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE

def place_pixels(grid, pixels, color):
    for r, c in pixels:
        if is_valid(r, c):
            grid[r][c] = color

def get_contour(pixels):
    """Retourne uniquement les pixels du bord d'une forme."""
    pixels_set = set(pixels)
    border = []
    for r, c in pixels:
        neighbors = 0
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            if (r+dr, c+dc) in pixels_set:
                neighbors += 1
        if neighbors < 4: border.append((r, c))
    return border

# --- MOTEUR DE GÉNÉRATION ---
def generate_random_blob(occupied_set, min_pixels=4, max_pixels=8):
    for _ in range(50): 
        start_r = random.randint(0, GRID_SIZE - 4)
        start_c = random.randint(0, GRID_SIZE - 4)
        
        target_size = random.randint(min_pixels, max_pixels)
        blob = {(start_r, start_c)}
        
        failed_growth = False
        while len(blob) < target_size:
            candidates = set()
            for r, c in blob:
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if is_valid(nr, nc):
                        if (nr, nc) not in blob and \
                           all((nr+dx, nc+dy) not in occupied_set for dx in [-1,0,1] for dy in [-1,0,1]):
                            candidates.add((nr, nc))
            if not candidates: failed_growth = True; break
            blob.add(random.choice(list(candidates)))
        
        if failed_growth: continue
        return list(blob)
    return None

def generate_scene(num_objects_range=(2, 5)): # Un peu plus d'objets pour le fun
    num_objs = random.randint(*num_objects_range)
    occupied = set()
    objects = [] 
    for _ in range(num_objs):
        blob_pixels = generate_random_blob(occupied)
        if blob_pixels:
            color = random.randint(1, 9)
            used_colors = {o['color'] for o in objects}
            if len(used_colors) < 9:
                while color in used_colors: color = random.randint(1, 9)
            objects.append({'pixels': blob_pixels, 'color': color})
            for p in blob_pixels: occupied.add(p)
    return objects, occupied

# --- GÉNÉRATEURS DE TESTS MULTI-OBJETS ---

def get_active_indices(objects):
    """Retourne une liste aléatoire d'indices d'objets à transformer (1 à tous)."""
    n = len(objects)
    if n == 0: return []
    num_active = random.randint(1, n)
    indices = list(range(n))
    random.shuffle(indices)
    return set(indices[:num_active])

def try_gen_translation():
    for _ in range(10):
        objects, _ = generate_scene()
        if not objects: continue
        
        active_indices = get_active_indices(objects)
        
        grid_in, grid_out = get_blank(), get_blank()
        occupied_out = set()
        
        # On place d'abord l'input
        for obj in objects:
            place_pixels(grid_in, obj['pixels'], obj['color'])
            
        # On calcule les mouvements pour l'output
        success_count = 0
        temp_objects_out = []
        
        # On traite les objets : actifs tentent de bouger, inactifs restent
        # Pour éviter les collisions complexes, on place d'abord les inactifs
        for i, obj in enumerate(objects):
            if i not in active_indices:
                for p in obj['pixels']: occupied_out.add(p)
                temp_objects_out.append(obj) # Reste identique

        # Ensuite on essaie de bouger les actifs
        for i, obj in enumerate(objects):
            if i in active_indices:
                # Essai de mouvement
                moved = False
                for _ in range(10): # 10 essais de vecteurs
                    dr, dc = random.randint(-4, 4), random.randint(-4, 4)
                    if dr == 0 and dc == 0: continue
                    
                    new_pixels = [(r+dr, c+dc) for r, c in obj['pixels']]
                    
                    if all(is_valid(r,c) and (r,c) not in occupied_out for r,c in new_pixels):
                        # Validé
                        for p in new_pixels: occupied_out.add(p)
                        temp_objects_out.append({'pixels': new_pixels, 'color': obj['color']})
                        moved = True
                        success_count += 1
                        break
                
                if not moved:
                    # Si pas de place, il reste sur place (devient inactif de facto)
                    for p in obj['pixels']: occupied_out.add(p)
                    temp_objects_out.append(obj)

        if success_count == 0: continue # Il faut au moins un mouvement
        
        # Rendu final
        for obj in temp_objects_out:
            place_pixels(grid_out, obj['pixels'], obj['color'])
                
        return grid_in, grid_out, "Translation"
    return None

def try_gen_color():
    for _ in range(10):
        objects, _ = generate_scene()
        if not objects: continue
        
        active_indices = get_active_indices(objects)
        grid_in, grid_out = get_blank(), get_blank()
        
        for i, obj in enumerate(objects):
            place_pixels(grid_in, obj['pixels'], obj['color'])
            
            final_color = obj['color']
            if i in active_indices:
                new_c = random.randint(1, 9)
                while new_c == obj['color']: new_c = random.randint(1, 9)
                final_color = new_c
            
            place_pixels(grid_out, obj['pixels'], final_color)
            
        return grid_in, grid_out, "Color"
    return None

def try_gen_rotation():
    for _ in range(10):
        objects, _ = generate_scene()
        if not objects: continue
        
        active_indices = get_active_indices(objects)
        
        grid_in, grid_out = get_blank(), get_blank()
        occupied_out = set()
        
        for obj in objects: place_pixels(grid_in, obj['pixels'], obj['color'])
        
        # Placement des inactifs
        for i, obj in enumerate(objects):
            if i not in active_indices:
                for p in obj['pixels']: occupied_out.add(p)

        temp_out = []
        success = False
        
        for i, obj in enumerate(objects):
            if i in active_indices:
                pixels = obj['pixels']
                rs, cs = [p[0] for p in pixels], [p[1] for p in pixels]
                cx, cy = (min(rs)+max(rs))//2, (min(cs)+max(cs))//2
                
                # On essaie une rotation valide
                rotated = False
                for angle in [90, 180, 270]:
                    new_pixels = []
                    valid = True
                    for r, c in pixels:
                        rel_r, rel_c = r - cx, c - cy
                        if angle == 90: nr, nc = rel_c, -rel_r
                        elif angle == 180: nr, nc = -rel_r, -rel_c
                        elif angle == 270: nr, nc = -rel_c, rel_r
                        ar, ac = cx + nr, cy + nc
                        if not is_valid(ar, ac) or (ar, ac) in occupied_out:
                            valid = False; break
                        new_pixels.append((ar, ac))
                    
                    if valid and sorted(new_pixels) != sorted(pixels):
                        for p in new_pixels: occupied_out.add(p)
                        place_pixels(grid_out, new_pixels, obj['color'])
                        rotated = True
                        success = True
                        break
                
                if not rotated: # Fallback
                    place_pixels(grid_out, pixels, obj['color'])
            else:
                place_pixels(grid_out, obj['pixels'], obj['color'])
        
        if not success: continue
        return grid_in, grid_out, "Rotation"
    return None

def try_gen_symmetry():
    for _ in range(10):
        objects, _ = generate_scene()
        if not objects: continue
        
        active_indices = get_active_indices(objects)
        grid_in, grid_out = get_blank(), get_blank()
        occupied_out = set()
        
        for obj in objects: place_pixels(grid_in, obj['pixels'], obj['color'])
        
        for i, obj in enumerate(objects):
            if i not in active_indices:
                for p in obj['pixels']: occupied_out.add(p)

        success = False
        for i, obj in enumerate(objects):
            if i in active_indices:
                pixels = obj['pixels']
                rs, cs = [p[0] for p in pixels], [p[1] for p in pixels]
                cr, cc = (min(rs)+max(rs))/2, (min(cs)+max(cs))/2
                
                reflected = False
                for axis in ["Vertical", "Horizontal"]:
                    new_pixels = []
                    valid = True
                    for r, c in pixels:
                        if axis == "Vertical": nr, nc = r, int(2*cc - c)
                        else: nr, nc = int(2*cr - r), c
                        if not is_valid(nr, nc) or (nr, nc) in occupied_out:
                            valid = False; break
                        new_pixels.append((nr, nc))
                    
                    if valid and sorted(new_pixels) != sorted(pixels):
                        for p in new_pixels: occupied_out.add(p)
                        place_pixels(grid_out, new_pixels, obj['color'])
                        reflected = True
                        success = True
                        break
                
                if not reflected:
                    place_pixels(grid_out, pixels, obj['color'])
            else:
                place_pixels(grid_out, obj['pixels'], obj['color'])
                
        if not success: continue
        return grid_in, grid_out, "Symmetry"
    return None

def try_gen_fill():
    # Maintenant on applique Fill à plusieurs objets
    for _ in range(10):
        objects, _ = generate_scene()
        if not objects: continue
        
        active_indices = get_active_indices(objects)
        grid_in, grid_out = get_blank(), get_blank()
        
        # Pour Fill, pas de conflit de position, c'est sur place
        for i, obj in enumerate(objects):
            pixels = obj['pixels']
            color = obj['color']
            
            # Tous les objets apparaissent dans l'Input
            # Si Actif -> Input = Contour, Output = Plein
            # Si Inactif -> Input = Plein, Output = Plein
            
            if i in active_indices:
                contour = get_contour(pixels)
                # Si l'objet est trop petit (tout est contour), on ne peut pas le remplir
                if len(contour) < len(pixels):
                    # INPUT : Contour
                    place_pixels(grid_in, contour, color)
                    # OUTPUT : Plein (Contour + Intérieur avec autre couleur ?)
                    # Dans ARC souvent : Contour reste couleur X, intérieur devient couleur Y
                    fill_c = random.randint(1, 9)
                    while fill_c == color: fill_c = random.randint(1, 9)
                    
                    # On dessine tout en fill_c puis on remet le contour en color
                    place_pixels(grid_out, pixels, fill_c)
                    place_pixels(grid_out, contour, color)
                else:
                    # Fallback si trop petit
                    place_pixels(grid_in, pixels, color)
                    place_pixels(grid_out, pixels, color)
            else:
                place_pixels(grid_in, pixels, color)
                place_pixels(grid_out, pixels, color)
                
        return grid_in, grid_out, "Fill"
    return None

# --- BOUCLE PRINCIPALE ---
categories = [try_gen_translation, try_gen_color, try_gen_rotation, try_gen_symmetry, try_gen_fill]
count = 0
idx = 1

batch_id = str(time.time())
with open("current_batch.json", "w") as f:
    json.dump({"batch_id": batch_id}, f)

print(f"--- Generating 10 Multi-Object Tasks (Batch ID: {batch_id[-6:]}) ---")

while count < 10:
    func = random.choice(categories)
    result = func()
    if result:
        g_in, g_out, logic = result
        task_data = {"train": [{"input": g_in, "output": g_out}], "expected_logic": logic}
        with open(os.path.join(output_dir, f"task_{idx:02d}.json"), 'w') as f:
            json.dump(task_data, f, indent=2)
        print(f"Task {idx:02d} : {logic}")
        count += 1
        idx += 1

print(f"✅ Done.")