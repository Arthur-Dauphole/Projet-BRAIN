"""
Perceptual system for extracting structured objects from ARC-style grids.
"""

from geometry import GeometryObject

class PerceptionSystem:
    """
    Performs visual segmentation of the grid into coherent objects.
    """

    def extract_objects(self, grid):
        """
        Scan the grid and extract all connected components
        of identical color as GeometryObject instances.
        """
        visited = set()
        objects = []

        height = len(grid)
        width = len(grid[0])

        for y in range(height):
            for x in range(width):

                if grid[y][x] != 0 and (x, y) not in visited:

                    color = grid[y][x]
                    stack = [(x, y)]
                    cells = []

                    while stack:
                        cx, cy = stack.pop()
                        if (cx, cy) in visited:
                            continue

                        visited.add((cx, cy))
                        cells.append((cx, cy))

                        # Explore 4-connected neighbors
                        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                if grid[ny][nx] == color:
                                    stack.append((nx, ny))

                    obj = GeometryObject(cells, color)
                    obj.detect_shape()
                    objects.append(obj)

        return objects