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
                        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == color:
                                stack.append((nx, ny))
                    
                    obj = GeometryObject(cells, color)
                    obj.detect_shape()
                    objects.append(obj)
        return objects
    