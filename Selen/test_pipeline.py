import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from perception import PerceptionSystem
from colors import ColorMapper

# Test grid
grid = [
 [0,0,0,0,0,0,1,1,1,0],
 [0,1,0,0,0,0,1,1,1,0],   
 [0,0,0,1,1,0,1,1,1,0],   
 [0,0,0,0,0,0,0,0,0,0],
 [0,0,1,1,0,0,0,0,0,0],   
 [0,0,1,1,0,0,0,0,0,0],
 [0,0,0,0,0,1,1,1,0,0],   
 [0,1,1,1,0,0,0,0,0,0],
 [0,1,1,1,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0],
]

# Perception
ps = PerceptionSystem()
objects = ps.extract_objects(grid)

# Recolor objects based on geometry
new_grid = [row.copy() for row in grid]
for obj in objects:
    if obj.shape == "rectangle":
        new_color = 4  # Yellow
    elif obj.shape == "square":
        new_color = 2  # Red
    elif obj.shape == "point":
        new_color = 8  # Cyan
    elif obj.shape == "line":
        new_color = 6  # Magenta
    else:
        new_color = obj.color_code

    for x, y in obj.cells:
        new_grid[y][x] = new_color

# Convert numeric grid to RGB
def grid_to_rgb(grid):
    h, w = len(grid), len(grid[0])
    rgb_array = np.zeros((h, w, 3))
    for y in range(h):
        for x in range(w):
            rgb_array[y, x] = to_rgb(ColorMapper.hex(grid[y][x]))
    return rgb_array

# Display function with grid lines
def show_grid_with_lines(grid, title):
    rgb = grid_to_rgb(grid)
    h, w, _ = rgb.shape
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title)
    plt.show()

# Display
show_grid_with_lines(grid, "Original Grid")
show_grid_with_lines(new_grid, "After Geometric Reasoning")