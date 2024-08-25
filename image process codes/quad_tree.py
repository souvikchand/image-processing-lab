import matplotlib.pyplot as plt
import numpy as np

# Example of recursively subdividing a square using a quadtree
def plot_quadtree(x, y, size, depth):
    if depth == 0:
        # Draw a square at this level
        rect = plt.Rectangle((x, y), size, size, fill=False)
        plt.gca().add_patch(rect)
    else:
        # Recursively subdivide into four quadrants
        size /= 2
        plot_quadtree(x, y, size, depth - 1)        # Top-left quadrant
        plot_quadtree(x + size, y, size, depth - 1) # Top-right quadrant
        plot_quadtree(x, y + size, size, depth - 1) # Bottom-left quadrant
        plot_quadtree(x + size, y + size, size, depth - 1) # Bottom-right quadrant

# Example usage
plt.figure(figsize=(6, 6))
plt.title('Example of Quadtree Partitioning')
plot_quadtree(0, 0, 4, 2)  # Starting with a 4x4 square, depth 2
plt.xlim(-0.5, 4.5)
plt.ylim(-0.5, 4.5)
plt.gca().invert_yaxis()
plt.show()
