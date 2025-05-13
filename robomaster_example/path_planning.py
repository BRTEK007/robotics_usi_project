import numpy as np
import heapq
import matplotlib.pyplot as plt
from collections import deque


class Cell:
    """Represents a rectangular group of walkable pixels in the grid."""

    def __init__(self, cell_id, pixels):
        self.id = cell_id
        self.min_y = self.max_y = self.min_x = self.max_x = None
        self.update_bounds(pixels)

    @property
    def corners(self):
        return [
            (self.min_y, self.min_x),
            (self.min_y, self.max_x),
            (self.max_y, self.min_x),
            (self.max_y, self.max_x),
        ]

    def update_bounds(self, pixels):
        ys, xs = zip(*pixels)
        self.min_y = min(ys) if self.min_y is None else min(self.min_y, *ys)
        self.max_y = max(ys) if self.max_y is None else max(self.max_y, *ys)
        self.min_x = min(xs) if self.min_x is None else min(self.min_x, *xs)
        self.max_x = max(xs) if self.max_x is None else max(self.max_x, *xs)

    def contains(self, y, x):
        return self.min_y <= y <= self.max_y and self.min_x <= x <= self.max_x

    def sweep_path(self, start=None, goal=None):
        """Generates a sweep path that covers all pixels in the cell, moving toward the goal."""
        pixels = {
            (y, x)
            for y in range(self.min_y, self.max_y + 1)
            for x in range(self.min_x, self.max_x + 1)
        }

        if goal is None:
            goal = (self.max_y, self.max_x)
        if start is None:
            start = (self.min_y, self.min_x)

        distances = {p: abs(p[0] - goal[0]) + abs(p[1] - goal[1]) for p in pixels}
        visited = set()
        path = [start]
        current = start
        visited.add(current)
        direction = None

        while len(visited) < len(pixels):
            y, x = current
            neighbors = []
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                n = (ny, nx)
                if n in pixels and n not in visited:
                    neighbors.append(((dy, dx), distances[n], n))

            if not neighbors:
                unvisited = [p for p in pixels if p not in visited]
                if not unvisited:
                    break

                unvisited.sort(
                    key=lambda p: (abs(p[0] - y) + abs(p[1] - x), -distances[p])
                )
                target = unvisited[0]
                sub_path = a_star_grid(
                    grid=None,
                    start=current,
                    goal=target,
                    is_walkable=is_walkable_in_grid,
                )
                if not sub_path:
                    break
                for step in sub_path:
                    visited.add(step)
                    path.append(step)
                current = target
                direction = None
                continue

            neighbors.sort(
                key=lambda item: (
                    -item[1],
                    0 if direction and item[0] == direction else 1,
                )
            )

            next_dir, _, next_cell = neighbors[0]
            visited.add(next_cell)
            path.append(next_cell)
            current = next_cell
            direction = next_dir

        return path[1:]


class GridDecomposer:
    """Decomposes a 2D binary grid into non-overlapping vertical cells of walkable regions."""

    def __init__(self, grid, max_cell_height=None):
        self.grid = grid
        self.max_cell_height = max_cell_height or len(grid)
        self.height, self.width = grid.shape
        self.cell_map = -np.ones_like(grid, dtype=int)
        self.cells = {}
        self._decompose()

    def _decompose(self):
        """Performs vertical decomposition of the grid into cells of limited height."""
        cell_id = 0
        prev_segments = {}

        for x in range(self.width):
            current_segments = {}
            y = 0
            while y < self.height:
                if self.grid[y, x] == 0:
                    start_y = y
                    while y < self.height and self.grid[y, x] == 0:
                        y += 1
                    end_y = y
                    segment_height = end_y - start_y
                    num_splits = (
                        segment_height + self.max_cell_height - 1
                    ) // self.max_cell_height

                    for i in range(num_splits):
                        sy = start_y + i * self.max_cell_height
                        ey = min(sy + self.max_cell_height, end_y)
                        seg_key = (sy, ey)
                        cid = prev_segments.get(seg_key, cell_id)
                        current_segments[seg_key] = cid
                        pixels = [(yy, x) for yy in range(sy, ey)]
                        self.cell_map[sy:ey, x] = cid

                        if cid not in self.cells:
                            self.cells[cid] = Cell(cid, pixels)
                            cell_id += 1
                        else:
                            self.cells[cid].update_bounds(pixels)
                else:
                    y += 1
            prev_segments = current_segments

    def build_adjacency(self):
        """Builds an adjacency map between neighboring cells."""
        adjacency = {cid: set() for cid in self.cells}
        for y in range(self.height):
            for x in range(self.width):
                cid = self.cell_map[y, x]
                if cid < 0:
                    continue
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        nid = self.cell_map[ny, nx]
                        if nid >= 0 and nid != cid:
                            adjacency[cid].add(nid)
        return {cid: sorted(neighs) for cid, neighs in adjacency.items()}

    def order_cells_by_gradient(self, start_cell_id, distances):
        """Orders cells by moving through neighbors with highest distance from the start."""
        visited = {start_cell_id}
        order = [start_cell_id]
        current = start_cell_id

        while len(visited) < len(self.cells):
            neighbors = self.adjacency.get(current, [])
            unvisited_neighbors = [n for n in neighbors if n not in visited]

            if unvisited_neighbors:
                next_cell = max(unvisited_neighbors, key=lambda cid: distances[cid])
                visited.add(next_cell)
                order.append(next_cell)
                current = next_cell
            else:
                path = self._bfs_to_best_unvisited(current, visited, distances)
                if not path:
                    break
                for cid in path[1:]:
                    visited.add(cid)
                    order.append(cid)
                current = path[-1]

        return order

    def _bfs_to_best_unvisited(self, start, visited, distances):
        """Finds a short path to the best unvisited cell using BFS and gradient scores."""
        queue = deque([(start, [start])])
        seen = {start}
        best_paths, min_depth = [], None

        while queue:
            current, path = queue.popleft()
            if current not in visited:
                depth = len(path) - 1
                if min_depth is None or depth == min_depth:
                    best_paths.append(path)
                    min_depth = depth
                elif depth > min_depth:
                    break
                continue
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return max(best_paths, key=lambda p: distances[p[-1]]) if best_paths else None


def compute_cell_distances(adjacency, start_id):
    """Computes shortest distances from a start cell to all others using BFS."""
    distances = {start_id: 0}
    queue = deque([start_id])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, []):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances


def find_goal_toward_next_cell(cell_a, cell_b):
    """Finds the pixel in cell A that borders the adjacent cell B for transition."""
    if cell_b.min_x == cell_a.max_x + 1 or cell_b.max_x == cell_a.min_x - 1:
        y_start = max(cell_a.min_y, cell_b.min_y)
        goal_x = cell_a.max_x if cell_b.min_x == cell_a.max_x + 1 else cell_a.min_x
        return (y_start, goal_x)
    assert False, "Unsupported cell adjacency"


def find_nearest_goal_in_cell(current_pixel, goal_cell):
    """Finds the closest boundary pixel in the goal cell from the current position."""
    goal_x = (
        goal_cell.min_x
        if abs(current_pixel[1] - goal_cell.min_x)
        < abs(current_pixel[1] - goal_cell.max_x)
        else goal_cell.max_x
    )
    goal_y = (
        goal_cell.min_y
        if abs(current_pixel[0] - goal_cell.min_y)
        < abs(current_pixel[0] - goal_cell.max_y)
        else goal_cell.max_y
    )
    return goal_y, goal_x


def is_walkable_in_grid(grid, pixel):
    """Checks if a pixel is within bounds and walkable (i.e., grid value is 0)."""
    y, x = pixel
    return 0 <= y < 20 and 0 <= x < 20 and grid[y, x] == 0


def a_star_grid(grid, start, goal, is_walkable):
    """Runs A* pathfinding on the grid from start to goal using the walkability function."""
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return list(reversed(path))[1:]

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_cell = (current[0] + dy, current[1] + dx)
            if not is_walkable(grid, next_cell):
                continue
            new_cost = cost_so_far[current] + 1
            if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                cost_so_far[next_cell] = new_cost
                came_from[next_cell] = current
                heapq.heappush(
                    frontier, (new_cost + heuristic(goal, next_cell), next_cell)
                )
    assert False, "A* path not found"


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


# Example usage
example_grid = np.zeros((20, 20))
example_grid[4:6, 3:6] = 1
example_grid[16:19, 14:15] = 1
example_grid[16:17, 15:16] = 1
example_grid[16:18, 12:14] = 1
example_grid[4:7, 12:16] = 1

decomposer = GridDecomposer(example_grid)
decomposer.adjacency = decomposer.build_adjacency()

distances = compute_cell_distances(decomposer.adjacency, 0)
order = decomposer.order_cells_by_gradient(start_cell_id=0, distances=distances)
order.append(0)

path = []
visited = np.zeros(len(decomposer.cells))
start = (19, 0)

for i in range(len(order) - 1):
    cell_idx = order[i]
    if not visited[cell_idx]:
        visited[cell_idx] = 1
        goal = find_goal_toward_next_cell(
            decomposer.cells[cell_idx], decomposer.cells[order[i + 1]]
        )
        cell_path = decomposer.cells[cell_idx].sweep_path(start=start, goal=goal)
        path.extend(cell_path)
        current = path[-1]
        start = (
            find_nearest_goal_in_cell(current, decomposer.cells[order[i + 1]])
            if i < len(order) - 2
            else (19, 0)
        )
        path.extend(a_star_grid(example_grid, current, start, is_walkable_in_grid))
    else:
        for j in range(i + 1, len(order) - 1):
            if not visited[order[j]]:
                current = path[-1]
                start = find_nearest_goal_in_cell(current, decomposer.cells[order[j]])
                path.extend(
                    a_star_grid(example_grid, current, start, is_walkable_in_grid)
                )
                break

visualize_grid_with_cells_and_path(example_grid, decomposer.cell_map, path)
