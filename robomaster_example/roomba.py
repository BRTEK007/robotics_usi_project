import rclpy
from rclpy.node import Node
from transforms3d._gohlketransforms import euler_from_quaternion
from sensor_msgs.msg import Range
import numpy as np
from math import pi

from rclpy.task import Future
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import sys

### ROOM MONITOR MODULE BEGIN
import pygame
class RoomMonitor:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Room live feedback")

    def update(self):
        #for event in pygame.event.get():
            #if event.type == pygame.QUIT:
                #pygame.quit()
                #exit()
        self.screen.fill((0, 0, 0))
        pygame.display.flip()
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
            self.create_subscription(Range, f"range_{i}", self.make_callback(i), 10)
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

        self.room_monitor = RoomMonitor()

    def make_callback(self, index):

        def callback(msg):
            self.ranges[index] = msg.range

        return callback

    def start(self):
        self.timer = self.create_timer(0.1, self.control_loop)

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

        self.room_monitor.update()



def main():
    rclpy.init(args=sys.argv)

    node = ControllerNode()
    node.start()

    rclpy.spin_until_future_complete(node, node.done_future)


if __name__ == "__main__":
    main()
