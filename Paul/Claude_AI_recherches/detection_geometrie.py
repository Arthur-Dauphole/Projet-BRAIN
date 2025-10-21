import json
import numpy as np
from collections import Counter
from scipy import ndimage

class GeometryDetector:
    """Détecte tous les objets géométriques présents dans une grille."""
    
    def __init__(self):
        self.grid = None
        self.objects = []
        
    def load_grid(self, grid):
        """Charge une grille pour l'analyser."""
        self.grid = np.array(grid)
        self.objects = []
        
    def extract_objects(self):
        """
        Extrait tous les objets distincts de la grille.
        Un objet = pixels de même couleur connectés (8-connectivité).
        
        Returns:
            list: Liste de (color, positions) pour chaque objet
        """
        objects = []
        
        # Pour chaque couleur non-noire
        for color in range(1, 10):
            # Créer un masque binaire pour cette couleur
            mask = (self.grid == color).astype(int)
            
            if not mask.any():
                continue
            
            # Trouver les composantes connexes (8-connectivité)
            labeled, num_features = ndimage.label(mask, structure=np.ones((3, 3)))
            
            # Pour chaque composante
            for obj_id in range(1, num_features + 1):
                positions = np.argwhere(labeled == obj_id)
                objects.append({
                    'color': color,
                    'positions': positions,
                    'size': len(positions)
                })
        
        return objects
    
    def detect_line_segment(self, positions):
        """Détecte si un ensemble de positions forme un segment de ligne."""
        if len(positions) < 2:
            return None
        
        # Vérifier alignement horizontal
        rows = positions[:, 0]
        if len(set(rows)) == 1:
            cols = sorted(positions[:, 1])
            if all(cols[i+1] - cols[i] == 1 for i in range(len(cols)-1)):
                return {
                    "orientation": "horizontal",
                    "length": len(positions),
                    "start": tuple(positions[0]),
                    "end": tuple(positions[-1])
                }
        
        # Vérifier alignement vertical
        cols = positions[:, 1]
        if len(set(cols)) == 1:
            rows = sorted(positions[:, 0])
            if all(rows[i+1] - rows[i] == 1 for i in range(len(rows)-1)):
                return {
                    "orientation": "vertical",
                    "length": len(positions),
                    "start": tuple(positions[0]),
                    "end": tuple(positions[-1])
                }
        
        # Vérifier diagonale
        sorted_positions = positions[positions[:, 0].argsort()]
        diffs = np.diff(sorted_positions, axis=0)
        
        # Diagonale descendante (↘)
        if len(diffs) > 0 and np.all((diffs[:, 0] == 1) & (diffs[:, 1] == 1)):
            return {
                "orientation": "diagonal_down_right",
                "length": len(positions),
                "start": tuple(sorted_positions[0]),
                "end": tuple(sorted_positions[-1])
            }
        
        # Diagonale montante (↗)
        if len(diffs) > 0 and np.all((diffs[:, 0] == 1) & (diffs[:, 1] == -1)):
            return {
                "orientation": "diagonal_up_right",
                "length": len(positions),
                "start": tuple(sorted_positions[0]),
                "end": tuple(sorted_positions[-1])
            }
        
        # Diagonale descendante vers la gauche (↙)
        sorted_by_col = positions[positions[:, 1].argsort()]
        diffs_col = np.diff(sorted_by_col, axis=0)
        if len(diffs_col) > 0 and np.all((diffs_col[:, 0] == 1) & (diffs_col[:, 1] == -1)):
            return {
                "orientation": "diagonal_down_left",
                "length": len(positions),
                "start": tuple(sorted_by_col[0]),
                "end": tuple(sorted_by_col[-1])
            }
        
        return None
    
    def detect_rectangle(self, positions):
        """Détecte si c'est un rectangle ou un carré."""
        if len(positions) < 4:
            return None
        
        min_row = positions[:, 0].min()
        max_row = positions[:, 0].max()
        min_col = positions[:, 1].min()
        max_col = positions[:, 1].max()
        
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        
        # Rectangle plein
        expected_size = height * width
        if len(positions) == expected_size:
            # Vérifier que c'est bien un rectangle plein
            rect_positions = set()
            for r in range(min_row, max_row + 1):
                for c in range(min_col, max_col + 1):
                    rect_positions.add((r, c))
            
            actual_positions = set(map(tuple, positions))
            if rect_positions == actual_positions:
                is_square = (height == width)
                return {
                    "shape": "square" if is_square else "rectangle",
                    "filled": True,
                    "height": height,
                    "width": width,
                    "top_left": (min_row, min_col),
                    "bottom_right": (max_row, max_col)
                }
        
        # Rectangle contour
        perimeter_size = 2 * (height + width) - 4
        if len(positions) == perimeter_size and height > 2 and width > 2:
            is_square = (height == width)
            return {
                "shape": "square" if is_square else "rectangle",
                "filled": False,
                "height": height,
                "width": width,
                "top_left": (min_row, min_col),
                "bottom_right": (max_row, max_col)
            }
        
        return None
    
    def classify_object(self, obj):
        """Classifie un objet selon sa géométrie."""
        positions = obj['positions']
        color = obj['color']
        
        result = {
            'color': color,
            'size': obj['size']
        }
        
        # Essayer segment
        segment = self.detect_line_segment(positions)
        if segment:
            result['type'] = 'segment'
            result.update(segment)
            return result
        
        # Essayer rectangle
        rectangle = self.detect_rectangle(positions)
        if rectangle:
            result['type'] = rectangle['shape']
            result.update(rectangle)
            return result
        
        # Par défaut : blob/cluster
        min_row = positions[:, 0].min()
        max_row = positions[:, 0].max()
        min_col = positions[:, 1].min()
        max_col = positions[:, 1].max()
        
        result['type'] = 'blob'
        result['bounding_box'] = {
            'top_left': (min_row, min_col),
            'bottom_right': (max_row, max_col),
            'height': max_row - min_row + 1,
            'width': max_col - min_col + 1
        }
        
        return result
    
    def detect_all(self, grid):
        """
        Détecte tous les objets géométriques dans une grille.
        
        Returns:
            dict: Résumé de l'analyse avec tous les objets détectés
        """
        self.load_grid(grid)
        
        # Extraire tous les objets
        objects = self.extract_objects()
        
        if not objects:
            return {
                "total_objects": 0,
                "objects": [],
                "summary": "Grille vide"
            }
        
        # Classifier chaque objet
        classified_objects = []
        for obj in objects:
            classified = self.classify_object(obj)
            classified_objects.append(classified)
        
        # Créer un résumé
        type_counts = Counter([obj['type'] for obj in classified_objects])
        
        return {
            "total_objects": len(classified_objects),
            "objects": classified_objects,
            "summary": dict(type_counts),
            "colors_used": list(set(obj['color'] for obj in classified_objects))
        }


def analyze_json_file(json_file_path):
    """Analyse toutes les grilles d'un fichier JSON."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("="*70)
        print(f"📁 ANALYSE DU FICHIER : {json_file_path}")
        print("="*70)
        print()
        
        detector = GeometryDetector()
        
        # Analyser les exemples train
        if 'train' in data:
            print("🎓 EXEMPLES D'ENTRAÎNEMENT")
            print("-"*70)
            
            for i, example in enumerate(data['train'], 1):
                print(f"\n📋 Exemple {i} - INPUT:")
                result = detector.detect_all(example['input'])
                print_detection_result(result)
                
                if 'output' in example:
                    print(f"\n📋 Exemple {i} - OUTPUT:")
                    result = detector.detect_all(example['output'])
                    print_detection_result(result)
            
            print()
        
        # Analyser les exemples test
        if 'test' in data:
            print("🧪 EXEMPLES DE TEST")
            print("-"*70)
            
            for i, example in enumerate(data['test'], 1):
                print(f"\n📋 Test {i} - INPUT:")
                result = detector.detect_all(example['input'])
                print_detection_result(result)
                
                if 'output' in example:
                    print(f"\n📋 Test {i} - OUTPUT:")
                    result = detector.detect_all(example['output'])
                    print_detection_result(result)
        
        print()
        print("="*70)
        
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier '{json_file_path}' n'existe pas.")
    except json.JSONDecodeError:
        print("❌ Erreur : Le fichier n'est pas un JSON valide.")
    except Exception as e:
        print(f"❌ Erreur : {e}")


def print_detection_result(result):
    """Affiche le résultat de détection de manière lisible."""
    if result["total_objects"] == 0:
        print(f"   ➜ {result['summary']}")
        return
    
    print(f"   ➜ Nombre total d'objets : {result['total_objects']}")
    print(f"   ➜ Résumé : {result['summary']}")
    print(f"   ➜ Couleurs utilisées : {result['colors_used']}")
    print()
    
    # Détails de chaque objet
    for i, obj in enumerate(result['objects'], 1):
        print(f"   Objet {i}:")
        print(f"      • Type : {obj['type']}")
        print(f"      • Couleur : {obj['color']}")
        print(f"      • Taille : {obj['size']} pixels")
        
        if obj['type'] == 'segment':
            print(f"      • Orientation : {obj['orientation']}")
            print(f"      • Longueur : {obj['length']}")
            print(f"      • De {obj['start']} à {obj['end']}")
            
        elif obj['type'] in ['rectangle', 'square']:
            print(f"      • Rempli : {'Oui' if obj['filled'] else 'Non (contour)'}")
            print(f"      • Dimensions : {obj['height']}x{obj['width']}")
            print(f"      • Position : {obj['top_left']} → {obj['bottom_right']}")
            
        elif obj['type'] == 'blob':
            bb = obj['bounding_box']
            print(f"      • Boîte englobante : {bb['height']}x{bb['width']}")
            print(f"      • Position : {bb['top_left']} → {bb['bottom_right']}")
        
        print()


if __name__ == "__main__":
    # Exemple d'utilisation
    fichier = '/Users/paullefrais/Documents/ISAE SUPAERO/Cours Supaero/2A/Projet R&D Brain/Projet-BRAIN-VSCODE/ARC-AGI-master/data/training/a5f85a15.json'
    analyze_json_file(fichier)