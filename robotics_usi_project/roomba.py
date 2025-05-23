import rclpy
import sys
from enum import Enum
from rclpy.node import Node
import numpy as np
from transforms3d._gohlketransforms import euler_from_quaternion
from sensor_msgs.msg import Range
from math import pi, sqrt, atan2

from rclpy.task import Future
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from .mapping import MappingMonitor, RoomMapper, MeasurmentData
from .path_planning import PathPlanner, FourNeighborPath
from .visualization_helpers import visualize_grid_only, visualize_grid_with_cells_and_path
from .robot_state import RobotState



class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")

        self.rotation_done = False
        self.returned_to_base = False

        # Robot parameters
        self.goal_angle = None
        self.pose2d = (0, 0, 0)
        self.vel_publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, "odom", self.odom_callback, 10
        )

        self.ranges = [None] * 4
        self.sensor_subs = [
            self.create_subscription(
                Range, f"range_{i}", self.make_sensor_callback(i), 10
            )
            for i in range(4)
        ]

        self.done_future = Future()

        # movement parameters
        self.threshold_distance = 0.3
        self.avoidance_threshold = 0.35
        self.forward_speed = 0.45
        self.angular_speed = 0.5
        self.last_angle = None
        self.scan_index = 0
        initial_angle = self.pose2d[2] % (2 * pi)
        self.target_angle = (initial_angle + 2 * pi - 0.05) % (2 * pi)
        self.state = RobotState.WAIT_FOR_ORDER_NO_MAP

        # Use this to enable wall following
        # self.state = RobotState.WALL_DETECTION

        self.wall_angle = None
        self.tolerance = 0.05
        self.wall_ideal_distance = None

        # room monitor parameters
        self.room_monitor = MappingMonitor()
        self.room_mapper = RoomMapper(logger=self.get_logger())
        self.base_pose = self.pose2d
        self.away_from_starting_pos = False

        self.path_planner = None

    def make_sensor_callback(self, index):
        """callback for the sensors"""

        def callback(msg):
            self.ranges[index] = msg.range

        return callback

    def start(self):
        """beginning of the program"""
        self.timer = self.create_timer(0.1, self.control_loop)
        self.timer_mapper_loop = self.create_timer(0.0125, self.mapping_loop)
        self.timer_monitor_loop = self.create_timer(0.025, self.monitor_loop)

    def stop(self):
        """ "stops the robot movement"""
        cmd_vel = Twist()
        self.vel_publisher.publish(cmd_vel)

    def odom_callback(self, msg):
        """callback for the odometry"""
        odom_pose = msg.pose.pose
        self.pose2d = self.pose3d_to_2d(odom_pose)

    def pose3d_to_2d(self, pose3):
        """transforms 3D positions into 2D"""
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
        """calculates the direction of the turn in the obstacle avoidance and corner following"""
        difference = -front_right - back_right + front_left + back_left
        self.move_right = difference > 0

    def is_wall(self, front_left, front_right, back_left, back_right, tolerance=0.15):
        """detects if the obstacle infront is a wall (if the 4 sensors are under a threshold at the same time)"""
        values = [front_left, front_right, back_left, back_right]
        close = [d < self.avoidance_threshold * 1.5 for d in values]
        num_close = sum(close)

        left_diff = front_left - back_left
        right_diff = front_right - back_right
        possible_corner = abs(left_diff) > 0.3 or abs(right_diff) > 0.3

        return num_close > 3 and (
            abs(front_left - back_left) < tolerance
            or abs(front_right - back_right) < tolerance
            or possible_corner
        )

    def rotate(
        self,
        left_sensor,
        right_sensor,
        back_right,
        back_left,
    ):
        """rotates the robot towards he obstacle in order to detect whether it is or not a wall"""
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

    def angle_relative_to_robot_facing_left(self, dx, dy):
        """
        Returns the angle (in radians) relative to the robot's orientation system,
        where 0 is facing left, pi/2 is down, pi is right, and -pi/2 (or 3*pi/2) is up.
        """
        # Standard atan2 gives angle where 0 is to the right (x-positive), positive is counter-clockwise
        standard_angle = atan2(dy, dx)

        # Shift angle system so that 0 is to the left instead of right
        # This means subtracting π to rotate the frame by 180°
        relative_angle = (standard_angle - pi) % (2 * pi)

        # Convert from [0, 2π) to [-π, π) for consistency with atan2-like outputs
        if relative_angle >= pi:
            relative_angle -= 2 * pi

        return relative_angle

    def update_goal_angle(self):
        current_pos = self.pose2d[:2]
        if self.current_target_index > len(self.path_to_follow) - 1:
            return
        target_pose = self.path_to_follow[self.current_target_index]
        self.get_logger().info(f"current_pos: {str(current_pos)}")
        self.get_logger().info(f"target_pos: {str(target_pose)}")
        dx = current_pos[0] - target_pose[0]
        dy = current_pos[1] - target_pose[1]
        # self.goal_angle = self.closest_90_deg_angle(dx, dy)
        self.goal_angle = self.angle_relative_to_robot_facing_left(dx, dy)

    def update_path(self):
        """Updates the path towards the nearest explored cell near to an unexplored one."""
        robot_large_side = max(RoomMapper.RM_DIMS)
        occupancy_grid = self.room_mapper.occupancy_grid.to_numpy_array().T
        self.path_planner = PathPlanner(
            occupancy_grid, RoomMapper.ROOM_SIZE, robot_large_side
        )
        path = self.path_planner.compute_bfs_path_to_nearest_frontier(
            start_point=self.pose2d[:2]
        )

        if path is None:
            self.path_to_follow = None
            return
        path = FourNeighborPath(path)
        self.path_to_follow = path.obtain_physical_path(
            self.path_planner, self.pose2d[:2]
        )

        self.get_logger().info("Path: " + str(self.path_to_follow))
        self.stop()
        np.save("arr2.npy", occupancy_grid)

    def monitor_loop(self):
        """Calls the mapping monitor to draw to the screen."""
        self.state = self.room_monitor.draw_and_update_state(room_mapper=self.room_mapper, path_planner=self.path_planner, robot_state=self.state)

    def mapping_loop(self):
        """Updates the room mapper based on measurments from scanners."""

        self.room_mapper.update(
            MeasurmentData(pose=self.pose2d, sensor_data=self.ranges)
        )

    def euclidean_distance(self, goal_pose, current_pose):
        """calculates the euclidean distance between 2 poses"""
        return sqrt(
            pow((goal_pose[0] - current_pose[0]), 2)
            + pow((goal_pose[1] - current_pose[1]), 2)
        )

    def rotate_with_angles(
        self,
        current_angle,
        goal_angle,
        constant=2.0,
        tolerance=0.01,
    ):
        """rotates the robot towards a desire angle"""
        angle_diff = goal_angle - current_angle
        normalized_diff = (angle_diff + pi) % (2 * pi) - pi
        if abs(normalized_diff) < tolerance:
            self.stop()
            return True

        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = np.clip(constant * normalized_diff, -1.5, 1.5)
        self.vel_publisher.publish(cmd_vel)
        return False

    def closest_90_deg_angle(self, dx, dy):
        """detects the closest angle to turn in the path folowwing (0º - left, 90º - down, 180º - right, 270º - up)"""
        ### ¡¡¡due to this orientation the robot must start facing left horizontally ###
        ### because the odometry is based on its initial position!!! ###
        self.get_logger().info(f"dx: {dx}, dy: {dy}")
        target_angle = atan2(dy, dx)
        self.get_logger().info(f"atan2: {target_angle}")
        orientations = [0, pi / 2, pi, -pi / 2]

        self.get_logger().info(str(orientations))
        self.get_logger().info(
            str(
                list(
                    map(
                        lambda angle: abs((angle - target_angle + pi) % (2 * pi) - pi),
                        orientations,
                    )
                )
            )
        )

        closest_angle = min(
            orientations,
            key=lambda angle: abs((angle - target_angle + pi) % (2 * pi) - pi),
        )
        return closest_angle

    def control_loop(self):
        """main function for controlling the robot behaviour - state machine"""

        # unavailable sensors
        if None in self.ranges:
            return

        # save the initial position of the robot to know when we have completed a swept in order to stp
        if (
            self.base_pose
            and self.euclidean_distance(self.base_pose, self.pose2d) < 0.2
            and self.state != RobotState.PATH_FOLLOWING
            and self.state != RobotState.ROTATE_360
            and self.state != RobotState.RETURN_TO_BASE
        ):
            # if we have completeed the initial swept it switches to 'nuevo' state
            if self.away_from_starting_pos:
                self.update_path()
                if self.path_to_follow:
                    self.get_logger().info("entro a path cuando no debo")
                    self.state = RobotState.PATH_FOLLOWING
                    self.current_target_index = 0
                    self.update_goal_angle()
                else:
                    self.get_logger().info("Total scan completed")
                    self.state = RobotState.RETURN_TO_BASE

        elif self.base_pose and not self.away_from_starting_pos:
            self.away_from_starting_pos = True

        # correction of the sensors data
        cmd = Twist()
        front_left = self.ranges[3]
        front_right = self.ranges[1]
        corrected_back_right = self.ranges[0] - 0.25 * np.cos(np.deg2rad(30))
        corrected_back_left = self.ranges[2] - 0.25 * np.cos(np.deg2rad(30))

        # initial state that looks for the wall
        if self.state == RobotState.WALL_DETECTION:
            # moves in straight line with a constant speed as long as it does not have anything under the distance threshold
            if (
                front_right > self.threshold_distance
                and front_left > self.threshold_distance
                and corrected_back_right > self.threshold_distance
                and corrected_back_left > self.threshold_distance
            ):
                cmd.linear.x = self.forward_speed
                self.vel_publisher.publish(cmd)
            # once it detects the obstacle it decides the turn direction and switch into 'avoid' state
            else:
                self.calculate_turn_direction(
                    front_right, front_left, corrected_back_right, corrected_back_left
                )
                self.state = RobotState.OBSTACLE_AVOIDANCE

        # state for avoiding obstacles in the middle of the room
        elif self.state == RobotState.OBSTACLE_AVOIDANCE:
            # rotates towards the obstacle
            state = self.rotate(
                front_left, front_right, corrected_back_right, corrected_back_left
            )

            # if it is a wall it switches to 'rotate_90' state
            if state == "wall_detected":
                if self.base_pose is None:
                    self.base_pose = self.pose2d
                self.get_logger().info("Wall infront")
                self.wall_angle = self.pose2d[2]
                self.state = RobotState.ROTATE_90
                self.get_logger().info(">>> Perpendicular a la pared")

            # if we have already avoided it we return to 'wall_detection' state continuing the search of the wall
            elif state == "free_way":
                self.state = RobotState.WallDetection
                self.get_logger().info("<<<<<< Objeto esquivado")

            # otherwise we are still turning
            else:
                self.get_logger().info("+++++++++ Seguimos girando")

        # state that rotates 90º in order to be parallel to the wall
        elif self.state == RobotState.ROTATE_90:
            current_angle = self.pose2d[2]
            goal_angle = self.wall_angle - (pi / 2)
            angle_diff = goal_angle - current_angle
            normalized_diff = (angle_diff + pi) % (2 * pi) - pi

            # once we have turned we switch to 'wall_following' state
            if abs(normalized_diff) < 0.01:
                self.stop()
                self.get_logger().info("Parallel to the wall")
                # perpendicular distance to the wall
                self.wall_ideal_distance = self.ranges[2] * np.cos(np.deg2rad(60))
                # self.get_logger().info(
                #     "Ideal distance: " + str(self.wall_ideal_distance)
                # )
                self.state = RobotState.WALL_FOLLOWING

            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 1.0 * normalized_diff
            self.vel_publisher.publish(cmd_vel)

        # state for wall following
        elif self.state == RobotState.WALL_FOLLOWING:
            # perpendicular distance to the wall
            left_back = self.ranges[2] * np.cos(np.deg2rad(60))

            cmd_vel = Twist()
            cmd_vel.linear.x = self.forward_speed

            # avoid front obstacles that are not walls
            if (
                front_left < self.avoidance_threshold
                or front_right < self.avoidance_threshold
            ):
                self.state = RobotState.FRONT_OBSTACLE_AVOIDANCE
                self.get_logger().info(
                    "Frontal obstacle detected. Switching to front_obstacle_avoidance."
                )

            # to keep the wall following in open corners
            elif left_back > self.avoidance_threshold:
                self.state = RobotState.CORNER_FOLLOWING
                self.get_logger().info("Wall lost, turning to recover it")

            # small correction in  for the wall following
            elif left_back > self.wall_ideal_distance + self.tolerance:
                cmd_vel.linear.x /= 3
                cmd_vel.angular.z = self.angular_speed / 2
                self.get_logger().info("Too far from the wall, turning left.")
            elif left_back < self.wall_ideal_distance - self.tolerance:
                cmd_vel.linear.x /= 3
                cmd_vel.angular.z = -self.angular_speed / 2
                self.get_logger().info("Too far from the wall, turning right.")

            # parallel to the wall
            else:
                cmd_vel.angular.z = 0.0
                self.get_logger().info("Correct distace, following the wall")

            self.vel_publisher.publish(cmd_vel)

        # state for avoiding front obstacke while following the wall
        elif self.state == RobotState.FRONT_OBSTACLE_AVOIDANCE:
            left_back = self.ranges[2] * np.cos(np.deg2rad(60))

            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            # always turn right in order to keep the obstable (treated as it was the wall) to the left
            cmd_vel.angular.z = -self.angular_speed

            # treat the obstacle like a wall and border it
            if (
                front_left > self.avoidance_threshold
                and front_right > self.avoidance_threshold
                and left_back < self.wall_ideal_distance + self.tolerance
            ):
                self.state = RobotState.WALL_FOLLOWING
                self.get_logger().info("Obstacle avoided, returning to wall_following.")

            # if it is a wall we switch to 'obstacle_avoidance' state
            elif self.is_wall(
                front_left, front_right, corrected_back_left, corrected_back_right
            ):
                self.stop()
                self.calculate_turn_direction(
                    front_right, front_left, corrected_back_right, corrected_back_left
                )
                self.state = RobotState.OBSTACLE_AVOIDANCE

            self.vel_publisher.publish(cmd_vel)

        # state for following an open corner, missed wall
        elif self.state == RobotState.CORNER_FOLLOWING:
            left_back = self.ranges[2] * np.cos(np.deg2rad(60))

            cmd_vel = Twist()
            cmd_vel.linear.x = self.forward_speed / 2
            cmd_vel.angular.z = self.angular_speed * 2

            # if we stil detect the wall we switch to 'wall_following' state
            if (
                left_back < self.wall_ideal_distance + self.tolerance
                and front_left > self.avoidance_threshold
                and front_right > self.avoidance_threshold
            ):
                self.state = RobotState.WALL_FOLLOWING
                self.get_logger().info("Pared recuperada. Volviendo a wall_following.")

            # if it is a wall we switch to 'obstacle_avoidance' state
            elif self.is_wall(
                front_left, front_right, corrected_back_left, corrected_back_right
            ):
                self.stop()
                self.calculate_turn_direction(
                    front_right, front_left, corrected_back_right, corrected_back_left
                )
                self.state = RobotState.OBSTACLE_AVOIDANCE

            self.vel_publisher.publish(cmd_vel)

        # state for path following
        elif self.state == RobotState.PATH_FOLLOWING:

            def termination_func():
                self.get_logger().info("Path completed")
                self.stop()
                initial_angle = self.pose2d[2] % (2 * pi)
                self.target_angle = (initial_angle + 2 * pi - 0.05) % (2 * pi)

                if self.returned_to_base:
                    self.state = RobotState.WAIT_FOR_ORDER_FULL_MAP
                    #self.done_future.set_result(True)

                self.get_logger().info("Scanning 360.")
                self.last_angle = self.pose2d[2]
                self.state = RobotState.SCAN_FORWARD

            # we need to adjust the coppelia coordinates no the path matrix

            self.path_tolerance = 0.05

            # invalid data
            if not self.path_to_follow or self.pose2d is None:
                self.get_logger().warn("Path is empty or pose unknown")
                return

            if self.current_target_index >= len(self.path_to_follow):
                termination_func()
                return

            current_pos = self.pose2d[:2]
            target_pose = self.path_to_follow[self.current_target_index]

            distance = self.euclidean_distance(target_pose, current_pos)

            # we just turn once we have reached the point
            if not self.rotation_done:
                current_theta = self.pose2d[2]
                self.get_logger().info("Goal_angle: " + str(self.goal_angle))

                self.rotation_done = self.rotate_with_angles(
                    current_angle=current_theta,
                    goal_angle=self.goal_angle,
                    constant=2.0,
                    tolerance=0.005,
                )

            # stop the movement when we are close enough to the desired position and start looking for the next one
            self.get_logger().info("distance: " + str(distance))
            if distance < self.path_tolerance:
                self.get_logger().info(f"Reached point {target_pose}")
                self.current_target_index += 1
                self.update_goal_angle()

                if self.current_target_index >= len(self.path_to_follow):
                    termination_func()
                self.rotation_done = False
                return

            # once the turn is completed, we move forward the goal point
            if self.rotation_done:
                cmd_vel = Twist()
                cmd_vel.linear.x = min(0.6, 2 * distance)
                cmd_vel.angular.z = 0.0
                self.vel_publisher.publish(cmd_vel)

        elif self.state == RobotState.ROTATE_360:
            self.get_logger().info("360-----------360")
            current_angle = self.pose2d[2] % (2 * pi)

            angle_diff = (self.target_angle - current_angle) % (2 * pi)
            # self.get_logger().info(str(angle_diff))

            if abs(angle_diff) < 0.01:
                self.stop()
                self.get_logger().info("360 completed")
                self.update_path()
                if self.path_to_follow:
                    self.state = RobotState.PATH_FOLLOWING
                    self.current_target_index = 0
                    self.update_goal_angle()
                else:
                    self.get_logger().info("Total scan completed")
                    self.state = RobotState.RETURN_TO_BASE

            else:
                cmd_vel = Twist()
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = np.clip(1.0 * angle_diff, -1.5, 1.5)
                self.vel_publisher.publish(cmd_vel)

        elif self.state == RobotState.RETURN_TO_BASE:
            robot_large_side = max(RoomMapper.RM_DIMS)
            occupancy_grid = self.room_mapper.occupancy_grid.to_numpy_array().T
            self.path_planner = PathPlanner(
                occupancy_grid, RoomMapper.ROOM_SIZE, robot_large_side
            )
            path = self.path_planner.compute_a_star_path(
                start=self.path_planner._calculate_cell_from_physical(self.pose2d[:2]),
                goal=self.path_planner._calculate_cell_from_physical(self.base_pose),
            )
            path = FourNeighborPath(path)
            self.path_to_follow = path.obtain_physical_path(
                self.path_planner, self.pose2d[:2]
            )
            self.get_logger().info("Path: " + str(self.path_to_follow))
            self.stop()
            self.state = RobotState.PATH_FOLLOWING
            self.current_target_index = 0
            self.update_goal_angle()
            self.returned_to_base = True
        elif self.state == RobotState.SCAN_FORWARD:
            if self.scan_index == 0:
                if self.rotate_with_angles(
                    self.pose2d[2],
                    self.last_angle - pi / 3,
                    constant=2,
                    tolerance=0.03,
                ):
                    self.scan_index += 1
            elif self.scan_index == 1:
                if self.rotate_with_angles(
                    self.pose2d[2],
                    self.last_angle + pi / 3,
                    constant=2,
                    tolerance=0.03,
                ):
                    self.scan_index += 1
            else:
                if self.rotate_with_angles(
                    self.pose2d[2], self.last_angle, constant=2, tolerance=0.03
                ):
                    self.scan_index = 0
                    self.stop()
                    self.get_logger().info("Forward scan completed")
                    self.update_path()
                    if self.path_to_follow:
                        self.state = RobotState.PATH_FOLLOWING
                        self.current_target_index = 0
                        self.update_goal_angle()
                    else:
                        self.get_logger().info("Total scan completed")
                        self.state = RobotState.RETURN_TO_BASE
        elif self.state == RobotState.WAIT_FOR_ORDER_NO_MAP:
            pass
        elif self.state == RobotState.WAIT_FOR_ORDER_FULL_MAP:
            pass


def main():
    rclpy.init(args=sys.argv)

    node = ControllerNode()
    node.start()

    rclpy.spin_until_future_complete(node, node.done_future)


if __name__ == "__main__":
    main()


### CONTROLLER MODULE END
