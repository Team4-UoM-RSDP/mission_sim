# Mission Sim

![OS](https://img.shields.io/ubuntu/v/ubuntu-wallpapers/noble)
![ROS_2](https://img.shields.io/ros/v/jazzy/rclcpp)

Digital twin of Team 4's robot and Gazebo simulation of the Robotic System Design Project mission.

### Installation
1. Clone this repository:
```
https://github.com/Team4-UoM-RSDP/mission_sim
```
2.  Move to the cloned directory:
```
cd mission_sim
```
3. Install ROS2 dependencies:
```
 rosdep install -i --from-path src --rosdistro $ROS_DISTRO -y
 ```
 4. Move to `mission_sim/src`
 ```
 cd src
 ```
 5. Build and source:
 ```
 colcon build
 source install/setup.bash
 ```
 
 ### Tools
 - To view a single robot model e.g. `mounted_d435i.urdf.xacro`:
 ```
 ros2 launch robot_description view_model.launch.py model:=view_mounted_d435i.urdf.xacro
 ```
 - To simulate just the myCobot 280pi in Gazebo with MoveIt2 path planning:
 ```
 bash src/robot_bringup/scripts/mycobot_280pi_gazebo_moveit.sh 
 ```

###### Team 4 – AERO62520 Robotic System Design Project 