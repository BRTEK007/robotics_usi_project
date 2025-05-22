from matplotlib import pyplot as plt
import matplotlib.colors as mcolors


def show_array_image(array):
    """
    Visualize a 2D NumPy array with values 0, 1, and 2 as an image.
    """
    # Define a custom color map
    cmap = mcolors.ListedColormap(["white", "black", "red"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(array, cmap=cmap, norm=norm)
    plt.axis("off")
    plt.title("Array Visualization")
    plt.show()


def visualize_grid_only(grid, path=None, title="Grid Visualization"):
    """
    Visualizes the grid with:
    - 2 (walls): red
    - 1 (explored): green
    - 0 (unknown): black

    Optionally overlays a path (list of (y, x) tuples).
    """
    height, width = grid.shape
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define colors
    color_map = {2: "red", 1: "green", 0: "black", -1: "blue", -2: "yellow"}

    for y in range(height):
        for x in range(width):
            val = grid[y, x]
            color = color_map.get(val, "white")
            rect = plt.Rectangle(
                (x, height - y - 1),  # Invert Y axis for visualization
                1,
                1,
                facecolor=color,
                edgecolor="gray",
                linewidth=0.5,
            )
            ax.add_patch(rect)

    # Draw path if provided
    if path:
        for i in range(len(path) - 1):
            y1, x1 = path[i]
            y2, x2 = path[i + 1]
            dx, dy = x2 - x1, y1 - y2  # y-axis is inverted in display
            ax.arrow(
                x1 + 0.5,
                height - y1 - 0.5,
                dx * 0.8,
                dy * 0.8,
                head_width=0.3,
                head_length=0.3,
                fc="white",
                ec="white",
                zorder=10,
            )

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(False)
    plt.show()


def visualize_grid_with_cells_and_path(
    grid, cell_map, path=None, title="Grid with Cells and Sweep Path"
):
    """Visualizes the grid, cell decomposition, and an optional sweep path."""
    height, width = grid.shape
    fig, ax = plt.subplots(figsize=(8, 8))
    unique_ids = np.unique(cell_map[cell_map >= 0])
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
    color_map = {cid: colors[i % len(colors)] for i, cid in enumerate(unique_ids)}

    for y in range(height):
        for x in range(width):
            facecolor = (
                "black" if grid[y, x] == 1 else color_map.get(cell_map[y, x], "white")
            )
            rect = plt.Rectangle(
                (x, height - y - 1),
                1,
                1,
                facecolor=facecolor,
                edgecolor="gray",
                linewidth=0.5,
            )
            ax.add_patch(rect)

    if path:
        for i in range(len(path) - 1):
            y1, x1 = path[i]
            y2, x2 = path[i + 1]
            dx, dy = x2 - x1, y1 - y2
            ax.arrow(
                x1 + 0.5,
                height - y1 - 0.5,
                dx * 0.8,
                dy * 0.8,
                head_width=0.3,
                head_length=0.3,
                fc="white",
                ec="white",
                zorder=10,
            )

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xticks(np.arange(0, width + 1, 1))
    ax.set_yticks(np.arange(0, height + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, which="both", color="lightgray", linewidth=0.5)
    plt.show()
