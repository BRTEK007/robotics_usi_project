from math import cos, sin, ceil, radians
import numpy as np
from collections import deque
import pygame
from .robot_state import RobotState

def draw_rotated_rect(
    screen, screen_pos, angle, dimensions, color=(255, 0, 0), width=2
):
    """Draws rotated square"""

    corners = [
        (-dimensions[0] / 2, -dimensions[1] / 2),
        (dimensions[0] / 2, -dimensions[1] / 2),
        (dimensions[0] / 2, dimensions[1] / 2),
        (-dimensions[0] / 2, dimensions[1] / 2),
    ]

    angle = -angle  # flip the angle TODO why?

    points = []
    for corner in corners:
        x = corner[0] * cos(angle) - corner[1] * sin(angle) + screen_pos[0]
        y = corner[0] * sin(angle) + corner[1] * cos(angle) + screen_pos[1]
        points.append((x, y))

    pygame.draw.polygon(screen, color, points, width=width)


### HELPER FUNCTION FOR ROTATED SQUARE END


class OccupancyGrid:
    # For occupancy grid we can use pygame texture and color coding for occupancy
    # this already gives us functionality for drawing lines and shapes in the grid for connectivity
    # the texture can also be converted into numpy array
    def __init__(self, physical_size, physical_cell_size):
        """ "physical_size: tuple (width, height) size in meters"""
        tex_w = int(ceil(physical_size[0] / physical_cell_size))
        tex_h = int(ceil(physical_size[1] / physical_cell_size))
        self.texture = pygame.Surface((tex_w, tex_h))
        self.texture.fill((75, 75, 75))
        self.texture_walls = pygame.Surface((tex_w, tex_h), pygame.SRCALPHA)
        self.texture_walls.fill((0, 0, 0, 0))

        self._physical_size = physical_size
        self._physical_cell_size = physical_cell_size
        self._tex_size = (tex_w, tex_h)

        self._wall_scans_buffer = deque(maxlen=2)  # circular buffer of wall scans

    def _room_to_grid_pos(self, room_pos):
        """Converts room coordinates to grid coordinates."""
        tex_x = int(
            round(room_pos[0] / self._physical_cell_size + self._tex_size[0] / 2.0)
        )
        tex_y = int(
            round(room_pos[1] / self._physical_cell_size + self._tex_size[1] / 2.0)
        )
        return (self._tex_size[0] - tex_x, tex_y)  # TODO why does x has to be flipped?

    def _is_grid_pos_valid(self, grid_pos):
        """Returns True if grid_pos is withing the grid dimensions."""
        return (
            grid_pos[0] >= 0
            and grid_pos[1] >= 0
            and grid_pos[0] < self._tex_size[0]
            and grid_pos[1] < self._tex_size[1]
        )

    def mark_sensor_reading(self, scanner_pos, scanned_pos, hit_wall):
        """Mark a wall in the grid, if sensor didn't hit the wall"""
        grid_scanned_pos = self._room_to_grid_pos(scanned_pos)
        grid_scanner_pos = self._room_to_grid_pos(scanner_pos)
        if not self._is_grid_pos_valid(grid_scanned_pos) or not self._is_grid_pos_valid(
            grid_scanner_pos
        ):
            return

        # draw the ray before the wall
        pygame.draw.line(self.texture, (0, 255, 0), grid_scanner_pos, grid_scanned_pos)

        if not hit_wall:
            return
        # draw wall
        self.texture_walls.set_at(grid_scanned_pos, (255, 0, 0))

        # draw the wall segment
        self._wall_scans_buffer.append(grid_scanned_pos)
        if (
            len(self._wall_scans_buffer) >= 2
            and pygame.math.Vector2(self._wall_scans_buffer[0]).distance_to(
                self._wall_scans_buffer[1]
            )
            <= 4
        ):
            pygame.draw.line(
                self.texture_walls,
                (255, 0, 0),
                self._wall_scans_buffer[0],
                self._wall_scans_buffer[1],
                1,
            )

    def mark_rm_path(self, rm_pose):
        """Mark a robot path in the grid"""
        grid_pos = self._room_to_grid_pos((rm_pose[0], rm_pose[1]))
        if not self._is_grid_pos_valid(grid_pos):
            return

        draw_rotated_rect(
            self.texture,
            grid_pos,
            rm_pose[2],
            (
                int(round(RoomMapper.RM_DIMS[0] / self._physical_cell_size)),
                int(round(RoomMapper.RM_DIMS[1] / self._physical_cell_size)),
            ),
            (0, 255, 0),
            0,
        )

    def to_numpy_array(self):
        """Returns an numpy array representing the occupancy grid"""
        arr_texture = pygame.surfarray.pixels3d(self.texture)
        arr_texture_walls = pygame.surfarray.pixels3d(self.texture_walls)
        is_known = (
            (arr_texture[:, :, 1] == 255)
            & (arr_texture[:, :, 0] == 0)
            & (arr_texture[:, :, 2] == 0)
        )
        is_wall = (
            (arr_texture_walls[:, :, 0] == 255)
            & (arr_texture_walls[:, :, 1] == 0)
            & (arr_texture_walls[:, :, 2] == 0)
        )
        output = np.zeros(self._tex_size, dtype=np.uint8)
        output[is_known] = 1
        output[is_wall] = 2
        return output


class RoomMapper:
    RM_DIMS = (0.44, 0.22)  # robomaster dimensions (x,y)

    ROOM_SIZE = (8, 8)  # room size in m, assuming robot starts in the center

    SENSOR_LOCAL_POSES = [  # (x,y,angle)
        (-0.107, -0.102, radians(-30)),  # back right
        (0.107, -0.102, 0),  # front right
        (-0.107, 0.102, radians(30)),  # back left
        (0.107, 0.102, 0),  # front left
    ]

    """Create a map of the room based on the measurments from the robot."""

    def __init__(self, logger):
        self._logger = logger
        self.rm_pose_list = deque(
            maxlen=1
        )  # circular buffer of positions robomaster travelled to
        self.room_size = RoomMapper.ROOM_SIZE  # expected physical size of the room
        self.occupancy_grid = OccupancyGrid(self.room_size, 0.025)
        self.room_center = None  # refrence point for the center of the world

    def _scanned_pos(self, rm_pose, sensor_pose, scan_dist):
        """
        Calculates the position of the scanned object.
        Returns tuple scanner_pos, and scanned_pos ((x,y), (x,y))
        """
        rm_x, rm_y, rm_angle = rm_pose[0], rm_pose[1], rm_pose[2]

        # scanner world coordinates
        s_x = rm_x + sensor_pose[0] * cos(rm_angle) - sensor_pose[1] * sin(rm_angle)
        s_y = rm_y + sensor_pose[0] * sin(rm_angle) + sensor_pose[1] * cos(rm_angle)
        s_angle = rm_angle + sensor_pose[2]

        # print(s_angle)

        # scanned point world coordinates
        px = s_x + cos(s_angle) * scan_dist
        py = s_y + sin(s_angle) * scan_dist

        return ((s_x, s_y), (px, py))

    def update(self, measurment):
        """Update map based on measurment"""
        if measurment.pose is None:
            return

        if self.room_center is None:
            self.room_center = (
                measurment.pose
            )  # first measurment, establish the refrence point

        self.rm_pose_list.append(measurment.pose)

        # self.logger.info(f"{measurment.pose[0]}, {measurment.pose[1]}, {measurment.pose[2]}")

        self.occupancy_grid.mark_rm_path(measurment.pose)

        for i in range(0, 4):
            scanned_dist = measurment.sensor_data[i]

            if scanned_dist is None:
                continue

            SENSOR_RANGE = (
                0.9  # The range in which we trust the sensor to work correctly
            )

            if scanned_dist < SENSOR_RANGE:  # hit a wall
                scanner_pos, scan_pos = self._scanned_pos(
                    measurment.pose, RoomMapper.SENSOR_LOCAL_POSES[i], scanned_dist
                )
                self.occupancy_grid.mark_sensor_reading(scanner_pos, scan_pos, True)
            else:  # didn't hit a wall
                scanner_pos, scan_pos = self._scanned_pos(
                    measurment.pose, RoomMapper.SENSOR_LOCAL_POSES[i], SENSOR_RANGE
                )
                self.occupancy_grid.mark_sensor_reading(scanner_pos, scan_pos, False)


class MappingMonitor:
    SCREEN_DIMS = (900, 900)
    MAP_DIMS = (750, 750)
    """Draws room mapping to the screen."""

    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode(MappingMonitor.SCREEN_DIMS)
        pygame.display.set_caption("Room live feedback")
        # self.world_to_screen_scaling = 200 # meters in world to pixels on the screen
        self._displaying_live = True # displays live map building, False -> shows path planner

    def _draw_occupancy_grid(self, occ_grid, screen_pos, screen_size):
        """
        screen_pos: tuple (x, y)
        scren_size: tuple (width, height)
        """
        scaled_free = pygame.transform.scale(occ_grid.texture, screen_size)
        self._screen.blit(scaled_free, screen_pos)
        scaled_walls = pygame.transform.scale(occ_grid.texture_walls, screen_size)
        self._screen.blit(scaled_walls, screen_pos)

    def draw_and_update_state(self, room_mapper, grid, robot_grid_pos, robot_state, path):
        """Visualizes room mapper and controlls. Returns new robot state."""
        next_robot_state = robot_state

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v:
                    self._displaying_live = not self._displaying_live
                
                if robot_state == RobotState.WAIT_FOR_ORDER_NO_MAP:
                    if event.key == pygame.K_1:
                        next_robot_state = RobotState.ROTATE_360
                    elif event.key == pygame.K_2:
                        next_robot_state = RobotState.WALL_DETECTION


        if self._displaying_live:
            self._draw_mapper(room_mapper)
        else:
            self._draw_planner(grid, path, robot_grid_pos)

        # map outline
        pygame.draw.rect(self._screen, (255, 255, 0), 
                         ((MappingMonitor.SCREEN_DIMS[0] - MappingMonitor.MAP_DIMS[0])//2 -2,
                          (MappingMonitor.SCREEN_DIMS[1] - MappingMonitor.MAP_DIMS[1])//2 -2,
                          MappingMonitor.MAP_DIMS[0] + 4, MappingMonitor.MAP_DIMS[1] + 4),
                          width=2)

        font = pygame.font.Font(None, 25)
        text = font.render(f"Current state: {robot_state.value}", True, (255, 255, 255))
        self._screen.blit(text, (1, 1))
        text = font.render(f"Press V to switch views.", True, (255, 255, 255))
        self._screen.blit(text, (1, 1 + 20))

        if robot_state == RobotState.WAIT_FOR_ORDER_NO_MAP:
            text = font.render(f"Press 1 for grid scanning | Press 2 for wall following scan.", True, (255, 255, 0))
            self._screen.blit(text, (1, 1 + 20*2))
        elif robot_state == RobotState.WAIT_FOR_ORDER_FULL_MAP:
            text = font.render(f"Press 1 for navigation mode. | Press 2 to begin sweep.", True, (255, 255, 0))
            self._screen.blit(text, (1, 1 + 20*2))

        pygame.display.flip()

        return next_robot_state
        

    def _draw_mapper(self, room_mapper):
        self._screen.fill((0, 0, 0))
        

        if len(room_mapper.rm_pose_list) < 1:
            return

        world_to_screen_scaling = min(
            MappingMonitor.MAP_DIMS[0] / room_mapper.room_size[0],
            MappingMonitor.MAP_DIMS[1] / room_mapper.room_size[1],
        )

        self._draw_occupancy_grid(
            room_mapper.occupancy_grid,
            (
                MappingMonitor.SCREEN_DIMS[0] / 2
                + (room_mapper.room_center[0] - 0.5 * room_mapper.room_size[0])
                * world_to_screen_scaling,
                MappingMonitor.SCREEN_DIMS[1] / 2
                + (room_mapper.room_center[1] - 0.5 * room_mapper.room_size[1])
                * world_to_screen_scaling,
            ),
            (
                room_mapper.room_size[0] * world_to_screen_scaling,
                room_mapper.room_size[1] * world_to_screen_scaling,
            ),
        )

        for i, path_pos in enumerate(room_mapper.rm_pose_list):
            px, py, theta = path_pos[0], path_pos[1], path_pos[2]

            screen_x = (
                MappingMonitor.SCREEN_DIMS[0] / 2
                + (px - room_mapper.room_center[0]) * world_to_screen_scaling
            )
            screen_y = (
                MappingMonitor.SCREEN_DIMS[1] / 2
                + (py - room_mapper.room_center[1]) * world_to_screen_scaling
            )

            l = len(room_mapper.rm_pose_list)

            color = (255, 255, 255)
            width = 2

            if i == l - 1:
                color = (255, 255, 255)
                width = 0

            draw_rotated_rect(
                self._screen,
                (MappingMonitor.SCREEN_DIMS[0] - screen_x, screen_y),
                theta,
                (
                    RoomMapper.RM_DIMS[0] * world_to_screen_scaling,
                    RoomMapper.RM_DIMS[1] * world_to_screen_scaling,
                ),
                color,
                width,
            )


    def _draw_planner(self, grid, path, robot_grid_pos):
        self._screen.fill((0, 0, 0))

        if grid is None:
            pygame.draw.rect(self._screen, (75, 75, 75), 
                         ((MappingMonitor.SCREEN_DIMS[0] - MappingMonitor.MAP_DIMS[0])//2,
                          (MappingMonitor.SCREEN_DIMS[1] - MappingMonitor.MAP_DIMS[1])//2,
                          MappingMonitor.MAP_DIMS[0], MappingMonitor.MAP_DIMS[1] ))
            font = pygame.font.Font(None, 60)
            text = font.render(f"Preview not availible.", True, (255, 125, 0))
            self._screen.blit(text, (200, 200))
            return

        colors = {
            0: (75, 75, 75),    # Unknown
            1: (0, 255, 0),    # Free
            2: (255, 0, 0)     # Wall
        }

        rgb_array = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        for val, color in colors.items():
            rgb_array[grid == val] = color

        surface = pygame.surfarray.make_surface(rgb_array)


        if path is not None and path.ndim != 0: # Draw path to the surface
            for point in path:
                surface.set_at(point, (255, 255, 0))

        if robot_grid_pos is not None:
            surface.set_at(robot_grid_pos, (0, 0, 255))


        surface = pygame.transform.rotate(surface, -90) # rotate 90 degrees right
        surface = pygame.transform.flip(surface, True, False) # flip by Y axis
        surface = pygame.transform.scale(surface, (MappingMonitor.MAP_DIMS[0], MappingMonitor.MAP_DIMS[1])) # scale the texture
        self._screen.blit(surface, (
            (MappingMonitor.SCREEN_DIMS[0]-MappingMonitor.MAP_DIMS[0])//2,
            (MappingMonitor.SCREEN_DIMS[1]-MappingMonitor.MAP_DIMS[1])//2))


class MeasurmentData:
    def __init__(self, pose, sensor_data):
        self.pose = pose  # robot pose
        self.sensor_data = sensor_data  # sensor measurments
