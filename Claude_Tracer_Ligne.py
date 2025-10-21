import json
import numpy as np
from copy import deepcopy

class LinePrimitive:
    """Primitive qui apprend à tracer des lignes droites entre des points de même couleur."""
    
    def __init__(self):
        self.learned = False
        self.primitive_name = "draw_line"
        self.color_to_connect = None
        
    def detect_two_points(self, grid):
        """
        Détecte s'il y a exactement deux points de même couleur non-noir dans la grille.
        
        Returns:
            tuple: (color, point1, point2) ou None
        """
        grid = np.array(grid)
        
        # Chercher toutes les couleurs non-noires (≠ 0)
        for color in range(1, 10):
            positions = np.argwhere(grid == color)
            
            if len(positions) == 2:
                return (color, tuple(positions[0]), tuple(positions[1]))
        
        return None
    
    def are_aligned(self, p1, p2):
        """Vérifie si deux points sont alignés (même ligne, même colonne, ou diagonale)."""
        # Même ligne
        if p1[0] == p2[0]:
            return True, "horizontal"
        # Même colonne
        if p1[1] == p2[1]:
            return True, "vertical"
        # Diagonale (pente = 1 ou -1)
        if abs(p1[0] - p2[0]) == abs(p1[1] - p2[1]):
            return True, "diagonal"
        
        return False, None
    
    def draw_line(self, grid, p1, p2, color):
        """Trace une ligne entre deux points avec la couleur donnée."""
        grid = np.array(grid)
        result = grid.copy()
        
        x1, y1 = p1
        x2, y2 = p2
        
        # S'assurer que p1 est avant p2
        if x1 > x2 or (x1 == x2 and y1 > y2):
            x1, y1, x2, y2 = x2, y2, x1, y1
        
        # Ligne horizontale
        if x1 == x2:
            for y in range(min(y1, y2), max(y1, y2) + 1):
                result[x1, y] = color
        
        # Ligne verticale
        elif y1 == y2:
            for x in range(x1, x2 + 1):
                result[x, y1] = color
        
        # Diagonale
        else:
            steps = max(abs(x2 - x1), abs(y2 - y1)) + 1
            for i in range(steps):
                x = x1 + i * (1 if x2 > x1 else -1 if x2 < x1 else 0)
                y = y1 + i * (1 if y2 > y1 else -1 if y2 < y1 else 0)
                result[x, y] = color
        
        return result.tolist()
    
    def learn_from_example(self, input_grid, output_grid):
        """
        Apprend la primitive à partir d'un exemple input/output.
        
        Returns:
            bool: True si la primitive a été apprise avec succès
        """
        # Détecter deux points dans l'input
        detection = self.detect_two_points(input_grid)
        
        if detection is None:
            return False
        
        color, p1, p2 = detection
        
        # Vérifier qu'ils sont alignés
        aligned, direction = self.are_aligned(p1, p2)
        
        if not aligned:
            return False
        
        # Tracer la ligne théorique
        predicted_output = self.draw_line(input_grid, p1, p2, color)
        
        # Vérifier si ça correspond à l'output réel
        if np.array_equal(predicted_output, output_grid):
            self.learned = True
            self.color_to_connect = color
            print(f"✅ Primitive apprise : tracer une ligne {direction} de couleur {color}")
            print(f"   Entre les points {p1} et {p2}")
            return True
        
        return False
    
    def apply(self, input_grid):
        """
        Applique la primitive apprise sur une nouvelle grille.
        
        Returns:
            list: Grille avec la ligne tracée, ou None si impossible
        """
        if not self.learned:
            print("❌ La primitive n'a pas encore été apprise !")
            return None
        
        # Détecter deux points dans la nouvelle grille
        detection = self.detect_two_points(input_grid)
        
        if detection is None:
            print("❌ Aucune paire de points détectée dans cette grille")
            return None
        
        color, p1, p2 = detection
        
        # Vérifier qu'ils sont alignés
        aligned, direction = self.are_aligned(p1, p2)
        
        if not aligned:
            print("❌ Les points ne sont pas alignés")
            return None
        
        # Tracer la ligne
        result = self.draw_line(input_grid, p1, p2, color)
        print(f"✅ Ligne {direction} tracée entre {p1} et {p2} avec la couleur {color}")
        
        return result


def train_and_test_primitive(json_file_path):
    """
    Entraîne la primitive sur les exemples train et teste sur les exemples test.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("="*70)
        print(f"📁 Fichier : {json_file_path}")
        print("="*70)
        print()
        
        primitive = LinePrimitive()
        
        # Phase d'apprentissage
        print("🎓 PHASE D'APPRENTISSAGE")
        print("-"*70)
        
        if 'train' not in data or len(data['train']) == 0:
            print("❌ Aucun exemple d'entraînement trouvé")
            return
        
        # Essayer d'apprendre sur le premier exemple
        first_example = data['train'][0]
        input_grid = first_example['input']
        output_grid = first_example['output']
        
        print(f"Tentative d'apprentissage sur l'exemple 1...")
        success = primitive.learn_from_example(input_grid, output_grid)
        
        if not success:
            print("❌ Impossible d'apprendre cette primitive sur cet exemple")
            print("   (Pas deux points alignés ou pattern différent)")
            return
        
        print()
        
        # Phase de test sur les autres exemples train
        print("🔍 VALIDATION SUR LES AUTRES EXEMPLES D'ENTRAÎNEMENT")
        print("-"*70)
        
        for i, example in enumerate(data['train'][1:], 2):
            print(f"\nTest sur exemple d'entraînement {i}:")
            predicted = primitive.apply(example['input'])
            
            if predicted and np.array_equal(predicted, example['output']):
                print("   ✅ Prédiction correcte !")
            elif predicted:
                print("   ❌ Prédiction incorrecte")
            
        print()
        
        # Phase de test
        print("🧪 PHASE DE TEST")
        print("-"*70)
        
        if 'test' in data and len(data['test']) > 0:
            for i, example in enumerate(data['test'], 1):
                print(f"\nTest {i}:")
                predicted = primitive.apply(example['input'])
                
                if predicted:
                    print(f"\n📊 Grille prédite :")
                    for row in predicted:
                        print(f"   {row}")
                    
                    if 'output' in example:
                        if np.array_equal(predicted, example['output']):
                            print("   ✅ Solution correcte !")
                        else:
                            print("   ❌ Solution incorrecte")
                            print(f"\n📊 Solution attendue :")
                            for row in example['output']:
                                print(f"   {row}")
        else:
            print("Aucun exemple de test trouvé")
        
        print()
        print("="*70)
        
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier '{json_file_path}' n'existe pas.")
    except json.JSONDecodeError:
        print("❌ Erreur : Le fichier n'est pas un JSON valide.")
    except Exception as e:
        print(f"❌ Erreur : {e}")


if __name__ == "__main__":
    # Exemple d'utilisation
    fichier = "line_mixed.json"
    train_and_test_primitive(fichier)