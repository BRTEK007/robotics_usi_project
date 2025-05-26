# Robotics USI 2025 project

## Video demonstration

part one: https://www.youtube.com/watch?v=41ErA4UiOQw \
part two: https://www.youtube.com/watch?v=dbaJPfQZw5I

## The code

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
`ros2 launch robotics_usi_project ep_tof.launch name:=/rm0`

3) Run controller

`pixi shell`\
`ros2 launch robotics_usi_project controller.launch name:=/rm0`
