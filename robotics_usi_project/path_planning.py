import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt
from collections import deque


class Cell:
    """Represents a rectangular group of walkable pixels in the grid."""

    def __init__(self, cell_id, pixels):
        self.id = cell_id
        self.min_y = self.max_y = self.min_x = self.max_x = None
        self.update_bounds(pixels)

    def update_bounds(self, pixels):
        ys, xs = zip(*pixels)
        self.min_y = min(ys) if self.min_y is None else min(self.min_y, *ys)
        self.max_y = max(ys) if self.max_y is None else max(self.max_y, *ys)
        self.min_x = min(xs) if self.min_x is None else min(self.min_x, *xs)
        self.max_x = max(xs) if self.max_x is None else max(self.max_x, *xs)

    def contains(self, y, x):
        return self.min_y <= y <= self.max_y and self.min_x <= x <= self.max_x

    def sweep_path(self, a_start_func, start=None, goal=None):
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
                sub_path = a_start_func(
                    start=current,
                    goal=target,
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

    def find_goal_towards_next_cell(self, cell_b):
        """Finds the pixel in cell that borders the adjacent cell B for transition."""
        if cell_b.min_x == self.max_x + 1 or cell_b.max_x == self.min_x - 1:
            y_start = max(self.min_y, cell_b.min_y)
            goal_x = self.max_x if cell_b.min_x == self.max_x + 1 else self.min_x
            return (y_start, goal_x)
        assert False, "Unsupported cell adjacency"

    def find_nearest_goal_in_cell(self, current_pixel):
        """Finds the closest boundary pixel in the cell from the current position."""
        goal_x = (
            self.min_x
            if abs(current_pixel[1] - self.min_x) < abs(current_pixel[1] - self.max_x)
            else self.max_x
        )
        goal_y = (
            self.min_y
            if abs(current_pixel[0] - self.min_y) < abs(current_pixel[0] - self.max_y)
            else self.max_y
        )
        return goal_y, goal_x


class GridDecomposer:
    """Decomposes a 2D binary grid into non-overlapping vertical cells of walkable regions."""

    def __init__(self, grid, max_cell_height=None):
        self.grid = grid
        self.max_cell_height = max_cell_height or len(grid)
        self.height, self.width = grid.shape
        self.cell_map = -np.ones_like(grid, dtype=int)
        self.cells = {}
        self._decompose()
        self.adjacency = self._build_adjacency()

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

    def _build_adjacency(self):
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

    def get_cell_id_from_point(self, point):
        """Returns the cell ID that the point (y, x) belongs to."""
        y, x = point
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.cell_map[y, x]
        else:
            raise ValueError(f"Point {point} is out of bounds.")

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

    def compute_cell_distances(self, start_id):
        """Computes shortest distances from a start cell to all others using BFS."""
        distances = {start_id: 0}
        queue = deque([start_id])
        while queue:
            current = queue.popleft()
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances


class PathPlanner:

    def __init__(self, occupancy_grid, room_size, robot_large_side):
        cell_size = (robot_large_side + 0.2) / 8
        self.room_size = room_size
        self.cell_size = cell_size
        self.grid = self._downsample_to_robot_grid(occupancy_grid)
        self.grid = self._dilatate_walls_and_unknowns(4)

    def _dilatate_walls_and_unknowns(self, num_dilatation_cells):
        mask = np.isin(self.grid, [0, 2]).astype(np.uint8)
        kernel = np.ones((1 + num_dilatation_cells, 1 + num_dilatation_cells), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        dilated_arr = self.grid.copy()
        dilated_arr[(dilated_mask == 1) & (self.grid == 1)] = 0
        mask_2s = (self.grid == 2).astype(np.uint8)
        dilated_2s = cv2.dilate(mask_2s, kernel, iterations=1)
        dilated_arr[(dilated_2s == 1)] = 2
        return dilated_arr

    def _downsample_to_robot_grid(self, occupancy_grid):
        """
        Downsamples a high-resolution map to a grid where each cell corresponds to the robot's size.
        Uses physical position mapping to avoid precision loss. Priority: 2 > 0 > 1.
        """
        target_cols = int(self.room_size[0] / self.cell_size)
        target_rows = int(self.room_size[1] / self.cell_size)

        result = np.zeros((target_rows, target_cols), dtype=int)

        occupancy_cell_size = self.room_size[0] / occupancy_grid.shape[0]

        for i in range(target_rows):
            for j in range(target_cols):
                top_left_x = int((i * self.cell_size) / occupancy_cell_size)
                top_left_y = int((j * self.cell_size) / occupancy_cell_size)
                bottom_right_x = int(((i + 1) * self.cell_size) / occupancy_cell_size)
                bottom_right_y = int(((j + 1) * self.cell_size) / occupancy_cell_size)

                block = occupancy_grid[
                    top_left_x:bottom_right_x, top_left_y:bottom_right_y
                ]

                # print("block size: " + str(block.size))
                if np.sum(block == 2) > block.size * 0.015:
                    result[i, j] = 2
                elif np.sum(block == 1) > block.size * 0.75:
                    result[i, j] = 1

        return result

    def _calculate_cell_from_physical(self, rm_physical):
        """Calculates the cell position of a real point"""
        x = int((-rm_physical[0] + self.room_size[0] / 2) / self.cell_size)
        y = int((rm_physical[1] + self.room_size[1] / 2) / self.cell_size)
        return (y, x)

    def _is_valid(self, pixel):
        """Checks if a pixel is within bounds and walkable (i.e., grid value is 0)."""
        height, width = self.grid.shape
        y, x = pixel
        return 0 <= y < height and 0 <= x < width

    def compute_a_star_path(self, start, goal):
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
                return list(reversed(path))
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_cell = (current[0] + dy, current[1] + dx)
                if (
                    not self._is_valid(next_cell)
                    or self.grid[next_cell[0], next_cell[1]] == 2
                ):
                    continue
                new_cost = cost_so_far[current] + 1
                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    came_from[next_cell] = current
                    heapq.heappush(
                        frontier, (new_cost + heuristic(goal, next_cell), next_cell)
                    )
        assert False, "A* path not found"

    def compute_bfs_path_to_nearest_frontier(self, start_point):
        """
        Performs BFS from the start position to find the closest free cell (1)
        that is adjacent to at least one unknown cell (0). Avoids walls (2).

        """
        start_cell = self._calculate_cell_from_physical(start_point)
        visited = np.zeros_like(self.grid, dtype=bool)
        parent = dict()

        queue = deque()
        queue.append(start_cell)
        visited[start_cell] = True

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            y, x = queue.popleft()

            if self.grid[y, x] == 1:
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if self._is_valid((ny, nx)) and self.grid[ny, nx] == 0:
                        path = [(y, x)]
                        while (y, x) != start_cell:
                            y, x = parent[(y, x)]
                            path.append((y, x))
                        path.reverse()
                        return path

            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if (
                    self._is_valid((ny, nx))
                    and not visited[ny, nx]
                    and self.grid[ny, nx] != 2
                ):
                    visited[ny, nx] = True
                    parent[(ny, nx)] = (y, x)
                    queue.append((ny, nx))
        return None

    def plan_full_coverage_path(self, start):

        decomposer = GridDecomposer(self.grid)

        start = (19, 0)
        cell_id = decomposer.get_cell_id_from_point(start)
        distances = decomposer.compute_cell_distances(cell_id)
        order = decomposer.order_cells_by_gradient(
            start_cell_id=cell_id, distances=distances
        )
        order.append(cell_id)

        path = []
        visited = np.zeros(len(decomposer.cells))
        pos = start

        for i in range(len(order) - 1):
            cell_idx = order[i]
            if not visited[cell_idx]:
                visited[cell_idx] = 1
                goal = decomposer.cells[cell_idx].find_goal_towards_next_cell(
                    decomposer.cells[order[i + 1]]
                )
                cell_path = decomposer.cells[cell_idx].sweep_path(
                    self.compute_a_star_path, start=pos, goal=goal
                )
                path.extend(cell_path)
                current = path[-1]
                pos = (
                    decomposer.cells[order[i + 1]].find_nearest_goal_in_cell(current)
                    if i < len(order) - 2
                    else (19, 0)
                )
                path.extend(self.compute_a_star_path(current, pos))
            else:
                for j in range(i + 1, len(order) - 1):
                    if not visited[order[j]]:
                        current = path[-1]
                        pos = decomposer.cells[order[j]].find_nearest_goal_in_cell(
                            current
                        )
                        path.extend(self.compute_a_star_path(current, pos))
                        break
        return path


class FourNeighborPath:
    def __init__(self, path):
        self.len = len(path)
        self.path = self._simplify_path(path)

    def _simplify_path(self, path):
        """
        Given a list of (y, x) coordinates representing a path moving
        in 4-neighbor steps, returns a simplified path containing only:
        The start point, turns and the end point.
        """
        if len(path) < 2:
            return path[:]

        simplified = [path[0]]
        prev_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])

        for i in range(2, len(path)):
            curr_dir = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            if curr_dir != prev_dir:
                simplified.append(path[i - 1])
                prev_dir = curr_dir

        simplified.append(path[-1])
        return simplified

    def obtain_physical_path(self, pathPlanner, rm_physical_start):
        """
        Convert simplified path grid coordinates (y,x) to physical coordinates,
        using the robot's known physical position at the first cell in the path as reference.

        """

        (y, x) = pathPlanner._calculate_cell_from_physical(rm_physical_start)

        x0_phys = -((x + 0.5) * pathPlanner.cell_size) + pathPlanner.room_size[0] / 2
        y0_phys = ((y + 0.5) * pathPlanner.cell_size) - pathPlanner.room_size[1] / 2

        y0_grid, x0_grid = self.path[0]

        physical_path = []
        for y, x in self.path:
            dx_cells = x - x0_grid
            dy_cells = y - y0_grid

            x_phys = x0_phys - dx_cells * pathPlanner.cell_size
            y_phys = y0_phys + dy_cells * pathPlanner.cell_size

            physical_path.append((x_phys, y_phys))

        return physical_path
