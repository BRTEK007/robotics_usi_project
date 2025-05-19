# Robotics USI 2025 project

This package is ment to be used inside the robomaster usi workspace: https://github.com/idsia-robotics/robotics-lab-usi-robomaster

## Building package

`pixi install`\
`pixi shell`\
`colcon build --symlink-install`

## Runnning project

1) Run coppelia sim

`source install/setup.sh`\
`pixi run coppelia`

2) Run robomaster

`pixi shell`\
`ros2 launch robomaster_example ep_tof.launch name:=/rm0`

3) Run controller

`pixi shell`\
`ros2 launch robomaster_example controller.launch name:=/rm0`