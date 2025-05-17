import rclpy
from rclpy.node import Node
from transforms3d._gohlketransforms import euler_from_quaternion
from sensor_msgs.msg import Range
import numpy as np
from math import pi, cos, sin, degrees, ceil, radians

from rclpy.task import Future
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from collections import deque

import sys

### ROOM MONITOR MODULE BEGIN
import pygame

### HELPER FUNCTION FOR ROTATED SQUARE BEGIN

def draw_rotated_rect(screen, screen_pos, angle, dimensions, color = (255, 0, 0), width = 2):
    """Draws rotated square"""
    
    corners = [
        (-dimensions[0]/2,-dimensions[1]/2),
        (dimensions[0]/2,-dimensions[1]/2),
        (dimensions[0]/2,dimensions[1]/2),
        (-dimensions[0]/2,dimensions[1]/2)
    ]

    angle = -angle # flip the angle TODO why?

    points = []
    for corner in corners:
        x = corner[0] * cos(angle) - corner[1] * sin(angle) + screen_pos[0]
        y = corner[0] * sin(angle) + corner[1] * cos(angle) + screen_pos[1]
        points.append((x,y))

    pygame.draw.polygon(screen, color, points, width = width)

### HELPER FUNCTION FOR ROTATED SQUARE END

class OccupancyGrid:
    # For occupancy grid we can use pygame texture and color coding for occupancy
    # this already gives us functionality for drawing lines and shapes in the grid for connectivity
    # the texture can also be converted into numpy array
    def __init__(self, physical_size, physical_cell_size):
        """"physical_size: tuple (width, height) size in meters"""
        tex_w = int(ceil(physical_size[0] / physical_cell_size))
        tex_h = int(ceil(physical_size[1] / physical_cell_size))
        self.texture = pygame.Surface((tex_w, tex_h))
        self.texture_walls = pygame.Surface((tex_w, tex_h), pygame.SRCALPHA)
        self.texture_walls.fill((0,0,0,0))

        self.physical_size = physical_size
        self.physical_cell_size = physical_cell_size
        self.tex_size = (tex_w, tex_h)
        
        self.wall_scans_buffer = deque(maxlen=2)      # circular buffer of wall scans

    def room_to_grid_pos(self, room_pos):
        """Converts room coordinates to grid coordinates."""
        tex_x = int(round(room_pos[0]/self.physical_cell_size + self.tex_size[0]/2.0))
        tex_y = int(round(room_pos[1]/self.physical_cell_size + self.tex_size[1]/2.0))
        return (self.tex_size[0] - tex_x, tex_y) # TODO why does x has to be flipped?

    def is_grid_pos_valid(self, grid_pos):
        """Returns True if grid_pos is withing the grid dimensions."""
        return grid_pos[0] >= 0 and grid_pos[1] >= 0 and grid_pos[0] < self.tex_size[0] and grid_pos[1] < self.tex_size[1]

    def mark_sensor_reading(self, scanner_pos, scanned_pos, hit_wall):
        """Mark a wall in the grid, if sensor didn't hit the wall"""
        grid_scanned_pos = self.room_to_grid_pos(scanned_pos)
        grid_scanner_pos = self.room_to_grid_pos(scanner_pos)
        if not self.is_grid_pos_valid(grid_scanned_pos) or not self.is_grid_pos_valid(grid_scanner_pos):
            return

        # draw the ray before the wall
        pygame.draw.line(self.texture, (0, 255, 0), grid_scanner_pos, grid_scanned_pos)

        if not hit_wall:
            return
        # draw wall
        self.texture_walls.set_at(grid_scanned_pos, (255, 0, 0))
        
        # draw the wall segment
        self.wall_scans_buffer.append(grid_scanned_pos) 
        if len(self.wall_scans_buffer) >= 2 and pygame.math.Vector2(self.wall_scans_buffer[0]).distance_to(self.wall_scans_buffer[1]) <= 4:
            pygame.draw.line(self.texture_walls, (255, 0, 0), self.wall_scans_buffer[0], self.wall_scans_buffer[1], 1)


    def mark_rm_path(self, rm_pose):
        """Mark a robot path in the grid"""
        grid_pos = self.room_to_grid_pos((rm_pose[0], rm_pose[1]))
        if not self.is_grid_pos_valid(grid_pos):
            return
        
        draw_rotated_rect(self.texture, grid_pos, rm_pose[2], 
                          (int(round(RoomMapper.RM_DIMS[0]/self.physical_cell_size)),int(round(RoomMapper.RM_DIMS[1]/self.physical_cell_size))),
                          (0, 255, 0),
                          0)


class RoomMapper:
    RM_DIMS = (0.215 * 1.1, 0.101 * 1.1)  # robomaster dimensions (x,y)

    ROOM_SIZE = (12, 12) # room size in m, assuming robot starts in the center

    SENSOR_LOCAL_POSES = [ # (x,y,angle)
        (-0.107, -0.102, radians(-30)), # back right
        (0.107, -0.102, 0), # front right
        (-0.107, 0.102, radians(30)), # back left
        (0.107, 0.102, 0), # front left
    ]

    """Create a map of the room based on the measurments from the robot."""
    def __init__(self, logger):
        self.map = None
        self.logger = logger
        self.rm_pose_list = deque(maxlen=1)      # circular buffer of positions robomaster travelled to
        #self.scan_pose_list = []    # list of scanned positions
        self.room_size = RoomMapper.ROOM_SIZE   # expected physical size of the room
        self.occupancy_grid = OccupancyGrid(self.room_size, 0.025)
        self.room_center = None    # refrence point for the center of the world
    
    def scanned_pos(self, rm_pose, sensor_pose, scan_dist):
        """
        Calculates the position of the scanned object.
        Returns tuple scanner_pos, and scanned_pos ((x,y), (x,y))
        """
        rm_x, rm_y, rm_angle = rm_pose[0], rm_pose[1], rm_pose[2]
        
        # scanner world coordinates
        s_x = rm_x+ sensor_pose[0]*cos(rm_angle) - sensor_pose[1]*sin(rm_angle)
        s_y = rm_y+ sensor_pose[0]*sin(rm_angle) + sensor_pose[1]*cos(rm_angle)
        s_angle = rm_angle + sensor_pose[2]

        print(s_angle)

        # scanned point world coordinates
        px = s_x + cos(s_angle)*scan_dist
        py = s_y + sin(s_angle)*scan_dist

        return ((s_x, s_y),(px, py))

    def update(self, measurment):
        """Update map based on measurment"""
        if measurment.pose is None:
            return
        
        if self.room_center is None:
            self.room_center = measurment.pose # first measurment, establish the refrence point

        self.rm_pose_list.append(measurment.pose)

        #self.logger.info(f"{measurment.pose[0]}, {measurment.pose[1]}, {measurment.pose[2]}")

        self.occupancy_grid.mark_rm_path(measurment.pose)
        
        for i in range(0, 4):
            scanned_dist = measurment.sensor_data[i]

            if scanned_dist is None:
                continue

            SENSOR_RANGE = 0.9 # The range in which we trust the sensor to work correctly

            if scanned_dist < SENSOR_RANGE: # hit a wall
                scanner_pos, scan_pos = self.scanned_pos(measurment.pose, RoomMapper.SENSOR_LOCAL_POSES[i], scanned_dist)
                self.occupancy_grid.mark_sensor_reading(scanner_pos, scan_pos, True)
            else: # didn't hit a wall
                scanner_pos, scan_pos = self.scanned_pos(measurment.pose, RoomMapper.SENSOR_LOCAL_POSES[i], SENSOR_RANGE)
                self.occupancy_grid.mark_sensor_reading(scanner_pos, scan_pos, False)


class MappingMonitor:
    SCREEN_DIMS = (1000, 1000)
    """Draws room mapping to the screen."""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(MappingMonitor.SCREEN_DIMS)
        pygame.display.set_caption("Room live feedback")
        #self.world_to_screen_scaling = 200 # meters in world to pixels on the screen

    def draw_occupancy_grid(self, occ_grid, screen_pos, screen_size):
        """
        screen_pos: tuple (x, y)
        scren_size: tuple (width, height)
        """
        scaled_free = pygame.transform.scale(occ_grid.texture, screen_size)
        self.screen.blit(scaled_free, screen_pos)
        scaled_walls = pygame.transform.scale(occ_grid.texture_walls, screen_size)
        self.screen.blit(scaled_walls, screen_pos)


    def draw(self, room_mapper):
        """Visualizes room mapper."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        self.screen.fill((0, 0, 0))

        if len(room_mapper.rm_pose_list) < 1:
            return

        world_to_screen_scaling = min(MappingMonitor.SCREEN_DIMS[0]/room_mapper.room_size[0], MappingMonitor.SCREEN_DIMS[1]/room_mapper.room_size[1])

        self.draw_occupancy_grid(room_mapper.occupancy_grid, 
        (MappingMonitor.SCREEN_DIMS[0]/2 + (room_mapper.room_center[0] -0.5*room_mapper.room_size[0]) * world_to_screen_scaling,
         MappingMonitor.SCREEN_DIMS[1]/2 + (room_mapper.room_center[1] -0.5*room_mapper.room_size[1]) * world_to_screen_scaling), 
        (room_mapper.room_size[0] * world_to_screen_scaling, room_mapper.room_size[1] * world_to_screen_scaling))



        for i, path_pos in enumerate(room_mapper.rm_pose_list):
            px,py,theta = path_pos[0], path_pos[1], path_pos[2]

            screen_x = MappingMonitor.SCREEN_DIMS[0]/2 + (px-room_mapper.room_center[0])*world_to_screen_scaling
            screen_y = MappingMonitor.SCREEN_DIMS[1]/2 + (py-room_mapper.room_center[1])*world_to_screen_scaling
        
            l = len(room_mapper.rm_pose_list)

            color = (255, 255, 255)
            width = 2

            if i == l-1:
                color = (255, 255, 255)
                width = 0

            draw_rotated_rect(self.screen, 
                              (MappingMonitor.SCREEN_DIMS[0] - screen_x, screen_y),
                            theta, 
                            (RoomMapper.RM_DIMS[0] * world_to_screen_scaling, RoomMapper.RM_DIMS[1]*world_to_screen_scaling),
                            color, width)

        pygame.display.flip()

class MeasurmentData:
    def __init__(self, pose, sensor_data):
        self.pose = pose                 # robot pose
        self.sensor_data = sensor_data   # sensor measurments
### ROOM MONITOR MODULE END


########## PARTE FINAL SEGUNDA ENTREGA PARA TENERLO GUARDADO

class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")

        self.pose2d = None

        self.vel_publisher = self.create_publisher(Twist, "cmd_vel", 10)

        self.odom_subscriber = self.create_subscription(
            Odometry, "odom", self.odom_callback, 10
        )

        self.ranges = [None] * 4
        self.sensor_subs = [
            self.create_subscription(Range, f"range_{i}", self.make_sensor_callback(i), 10)
            for i in range(4)
        ]

        self.done_future = Future()
        self.threshold_distance = 0.3
        self.avoidance_threshold = 0.35
        self.forward_speed = 0.45
        self.angular_speed = 0.5
        self.state = "move"

        self.wall_angle = None
        self.tolerance = 0.05
        self.wall_ideal_distance = None

        self.room_monitor = MappingMonitor()
        self.room_mapper = RoomMapper(logger=self.get_logger())

    def make_sensor_callback(self, index):

        def callback(msg):
            self.ranges[index] = msg.range

        return callback

    def start(self):
        self.timer = self.create_timer(0.1, self.control_loop)
        self.timer_mapper_loop = self.create_timer(0.05, self.mapping_loop)
        self.timer_monitor_loop = self.create_timer(0.1, self.monitor_loop)

    def stop(self):
        cmd_vel = Twist()
        self.vel_publisher.publish(cmd_vel)

    def odom_callback(self, msg):
        odom_pose = msg.pose.pose

        self.pose2d = self.pose3d_to_2d(odom_pose)

        # self.get_logger().info(
        #     "odometry: received pose (x: {:.2f}, y: {:.2f}, theta: {:.2f})".format(
        #         *self.pose2d
        #     ),
        #     throttle_duration_sec=0.5,
        # )

    def pose3d_to_2d(self, pose3):
        quaternion = (
            pose3.orientation.x,
            pose3.orientation.y,
            pose3.orientation.z,
            pose3.orientation.w,
        )

        roll, pitch, yaw = euler_from_quaternion(quaternion)

        pose2 = (
            pose3.position.x,
            pose3.position.y,
            roll,
        )

        return pose2
    
    def calculate_turn_direction(self, front_right, front_left, back_right, back_left):
        difference = (
            - front_right
            - back_right
            + front_left
            + back_left
        )
        self.move_right = (
            difference > 0 # if abs(difference) > 0.25 else np.random.rand() < 0.5
        )
        
    def is_wall(self, front_left, front_right, back_left, back_right, tolerance=0.15):
        values = [front_left, front_right, back_left, back_right]
        close = [d < self.avoidance_threshold * 1.5 for d in values]
        num_close = sum(close)

        left_diff = front_left - back_left
        right_diff = front_right - back_right

        possible_corner = abs(left_diff) > 0.3 or abs(right_diff) > 0.3

        return (
            num_close > 3 and (
                abs(front_left - back_left) < tolerance or
                abs(front_right - back_right) < tolerance or
                possible_corner
            )
        )


    def rotate(
        self,
        left_sensor,
        right_sensor,
        back_right,
        back_left,
    ):
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = (
            -self.angular_speed if self.move_right else self.angular_speed
        )

        self.vel_publisher.publish(cmd_vel)
        if (
            left_sensor >= self.avoidance_threshold
            and right_sensor >= self.avoidance_threshold
            and back_right >= self.avoidance_threshold
            and back_left >= self.avoidance_threshold
        ):
            return "free_way"
        elif self.is_wall(left_sensor, right_sensor, back_left, back_right):
            self.wall_angle = self.pose2d[2]
            self.stop()
            return "wall_detected"
    
        return "continuar_girando"

    def monitor_loop(self):
        """Calls the mapping monitor to draw to the screen."""
        self.room_monitor.draw(room_mapper=self.room_mapper)

    def mapping_loop(self):
        """Updates the room mapper based on measurments from scanners."""

        self.room_mapper.update(
            MeasurmentData(pose = self.pose2d, sensor_data=self.ranges))


    def control_loop(self):
        if None in self.ranges:
            return

        cmd = Twist()
        front_left = self.ranges[3]
        front_right = self.ranges[1]
        corrected_back_right = self.ranges[0] - 0.25 * np.cos(np.deg2rad(30))
        corrected_back_left = self.ranges[2] - 0.25 * np.cos(np.deg2rad(30))

        if self.state == "move":
            if (
                front_right > self.threshold_distance
                and front_left > self.threshold_distance
                and corrected_back_right > self.threshold_distance
                and corrected_back_left > self.threshold_distance
            ):
                cmd.linear.x = self.forward_speed
                self.vel_publisher.publish(cmd)
            else:
                # difference = (
                #     - front_right
                #     - corrected_back_right
                #     + front_left
                #     + corrected_back_left
                # )
                # self.move_right = (
                #     difference > 0 # if abs(difference) > 0.25 else np.random.rand() < 0.5
                # )
                self.calculate_turn_direction(front_right, front_left, corrected_back_right, corrected_back_left)
                self.state = "avoid"

        elif self.state == "avoid":
            state = self.rotate(
                front_left, front_right, corrected_back_right, corrected_back_left
            )

            if state == "wall_detected":
                self.get_logger().info("Wall infront")
                self.wall_angle = self.pose2d[2]
                self.state = "rotate90"
                self.get_logger().info(">>> Perpendicular a la pared")

                # pared detectada, hay que girar 90º ahora

            elif state == "free_way":
                # no tenemos nada delante por lo que ya podemos avanzar
                self.state = "move"
                self.get_logger().info("<<<<<< Objeto esquivado")

            else:
                # todavía esta girando
                # no hacemos nada
                self.get_logger().info("+++++++++ Seguimos girando")

        elif self.state == "rotate90":
            current_angle = self.pose2d[2]
            goal_angle = self.wall_angle - (pi/2)
            angle_diff = goal_angle - current_angle
            normalized_diff = (angle_diff + pi) % (2 * pi) - pi

            if abs(normalized_diff) < 0.01:
                self.stop()
                self.get_logger().info("Parallel to the wall")
                # para medir la distancia perpendicular a la pared
                self.wall_ideal_distance = self.ranges[2] *  np.cos(np.deg2rad(60))
                self.get_logger().info("Ideal distance: " + str(self.wall_ideal_distance))
                self.state = "wall_follow"

            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 1.0 * normalized_diff
            self.vel_publisher.publish(cmd_vel)

        elif self.state == "wall_follow":
            # para medir la distancia perpendicular a la pared
            left_back = self.ranges[2] *  np.cos(np.deg2rad(60))

            cmd_vel = Twist()
            cmd_vel.linear.x = self.forward_speed

            if front_left < self.avoidance_threshold or front_right < self.avoidance_threshold:
                self.state = "avoid_front_obstacle"
                self.get_logger().info("Obstáculo frontal detectado. Cambiando a avoid_front_obstacle.")

            elif left_back > self.avoidance_threshold:
                self.state = "follow_corner"
                self.get_logger().info("Pared perdida, vamos a girar para recuperarla")

            elif left_back > self.wall_ideal_distance + self.tolerance:
                cmd_vel.linear.x /= 3
                cmd_vel.angular.z = self.angular_speed / 2
                self.get_logger().info("Demasiado lejos de la pared. Corrigiendo a la izquierda.")
            elif left_back < self.wall_ideal_distance - self.tolerance:
                cmd_vel.linear.x /= 3
                cmd_vel.angular.z = -self.angular_speed / 2
                self.get_logger().info("Demasiado cerca de la pared. Corrigiendo a la derecha.")
            else:
                cmd_vel.angular.z = 0.0
                self.get_logger().info("Distancia correcta. Avanzando paralelo.")

            self.vel_publisher.publish(cmd_vel)

        elif self.state == "avoid_front_obstacle":
            left_back = self.ranges[2] * np.cos(np.deg2rad(60))

            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = -self.angular_speed

            # self.get_logger().info("Left sensor: " + str(front_left))
            # self.get_logger().info("Right sensor: " + str(front_right))
            # self.get_logger().info("Back right: " + str(corrected_back_right))
            # self.get_logger().info("Back left: " + str(corrected_back_left))

            # values = [front_left, front_right, corrected_back_left, corrected_back_right]
            # self.get_logger().info("Diff: " + str(max(values) - min(values)))

            if (front_left > self.avoidance_threshold and front_right > self.avoidance_threshold and
                left_back < self.wall_ideal_distance + self.tolerance):
                self.state = "wall_follow"
                self.get_logger().info("Obstáculo evitado. Volviendo a wall_follow.")

            elif self.is_wall(front_left, front_right, corrected_back_left, corrected_back_right):
                self.stop()
                self.calculate_turn_direction(front_right, front_left, corrected_back_right, corrected_back_left)
                self.state = "avoid"
                # self.wall_angle = self.pose2d[2]
                # self.state = "rotate90"
                # self.get_logger().info("Wall infront")

            self.vel_publisher.publish(cmd_vel)

        
        elif self.state == "follow_corner":
            left_back = self.ranges[2] * np.cos(np.deg2rad(60))

            cmd_vel = Twist()
            cmd_vel.linear.x = self.forward_speed / 2
            cmd_vel.angular.z = self.angular_speed * 2

            if (left_back < self.wall_ideal_distance + self.tolerance and
                front_left > self.avoidance_threshold and
                front_right > self.avoidance_threshold):
                self.state = "wall_follow"
                self.get_logger().info("Pared recuperada. Volviendo a wall_follow.")
            
            elif self.is_wall(front_left, front_right, corrected_back_left, corrected_back_right):
                self.stop()
                self.calculate_turn_direction(front_right, front_left, corrected_back_right, corrected_back_left)
                self.state = "avoid"
                # self.wall_angle = self.pose2d[2]
                # self.state = "rotate90"
                # self.get_logger().info("Wall infront")

            self.vel_publisher.publish(cmd_vel)


def main():
    rclpy.init(args=sys.argv)

    node = ControllerNode()
    node.start()

    rclpy.spin_until_future_complete(node, node.done_future)


if __name__ == "__main__":
    main()
