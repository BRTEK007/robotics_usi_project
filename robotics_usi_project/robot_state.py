from enum import Enum

# Possible states for the state machine controller
class RobotState(Enum):
    WALL_DETECTION = "wall detection"
    OBSTACLE_AVOIDANCE = "obstacle avoidance"
    ROTATE_90 = "rotating 90"
    WALL_FOLLOWING = "wall following"
    FRONT_OBSTACLE_AVOIDANCE = "front obstacle avoidance"
    CORNER_FOLLOWING = "corner following"
    PATH_FOLLOWING = "path following"
    ROTATE_360 = "rotating 360"
    RETURN_TO_BASE = "returning to base"
    SCAN_FORWARD = "scanning forward"
    WAIT_FOR_ORDER_NO_MAP = "waiting for orders (no map)"
    WAIT_FOR_ORDER_FULL_MAP = "waiting for orders (full map)"