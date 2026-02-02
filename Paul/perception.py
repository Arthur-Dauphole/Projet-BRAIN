from geometry import GeometryObject

class PerceptionSystem:
    def extract_objects(self, grid):
        visited = set()
        objects = []
        height, width = len(grid), len(grid[0])

        for y in range(height):
            for x in range(width):
                if grid[y][x] != 0 and (x, y) not in visited:
                    color = grid[y][x]
                    stack, cells = [(x, y)], []
                    
                    while stack:
                        cx, cy = stack.pop()
                        if (cx, cy) in visited: continue
                        visited.add((cx, cy))
                        cells.append((cx, cy))
                        
                        # On regarde les 4 cardinaux + les 4 diagonales
                        # Cela permet de souder les pixels qui se touchent par un coin
                        neighbors = [
                            (1,0), (-1,0), (0,1), (0,-1),   # Haut/Bas/Gauche/Droite
                            (1,1), (1,-1), (-1,1), (-1,-1)  # Diagonales
                        ]
                        
                        for dx, dy in neighbors:
                            nx, ny = cx + dx, cy + dy
                            
                            # On vérifie qu'on reste dans la grille et que c'est la même couleur
                            if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == color:
                                stack.append((nx, ny))
                    
                    obj = GeometryObject(cells, color)
                    obj.detect_shape()
                    objects.append(obj)
        return objects