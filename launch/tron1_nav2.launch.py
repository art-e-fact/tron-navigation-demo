"""Tron1 in Newton, walking with LimX's own RL controller, driven by Nav2.

    ros2 launch launch/tron1_nav2.launch.py [route:=routes/x.yaml] [rviz:=true] [headless:=true] [nav2:=false]

Everything comes from the route (artefacts_toolkit_navigation): the robot spawns
at its start, the Newton world is extruded from its map, Nav2 localises on that
map. Run from the repo root (route and map paths are relative to it).
"""

import math
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from artefacts_toolkit_navigation import load_route, map_to_world, route_launch_args

ROOT = Path(__file__).resolve().parents[1]
ROBOT_IP = "127.0.0.1"  # where the controller finds the robot: our sim
WALL_HEIGHT_M = 2.0
CROP_M = 15.0  # only the walls within this distance of the route


def share(package):
    return Path(get_package_share_directory(package))


def launch_setup(context):
    arg = lambda name: LaunchConfiguration(name).perform(context)
    route = load_route(arg("route") or None)  # none given: the route recorded last
    start = route_launch_args(route)  # {"x_pose", "y_pose", "yaw", "map"}
    world = map_to_world(route.map.file, wall_height=WALL_HEIGHT_M, crop=(route, CROP_M))

    sim = ExecuteProcess(
        cmd=["python", str(ROOT / "sim" / "tron1_sim.py"), "--world", world,
             "--x", start["x_pose"], "--y", start["y_pose"], "--yaw", start["yaw"]]
            + (["--headless"] if arg("headless") == "true" else []),
        output="screen",
    )

    robot_type, rl_type = os.environ["ROBOT_TYPE"], os.environ["RL_TYPE"]
    controllers_config = share("robot_controllers") / "config"
    robot_config = controllers_config / "pointfoot" / robot_type
    hw_config = share("robot_hw") / "config"
    policy = robot_config / "policy" / rl_type
    controller = Node(
        package="robot_hw", executable="pointfoot_node", name="robot_hw_node",
        arguments=[ROBOT_IP], output="screen",
        parameters=[
            {"robot_controllers_policy_file": str(policy / "policy.onnx"),
             "robot_controllers_encoder_file": str(policy / "encoder.onnx"),
             "use_gazebo": True},  # upstream's "in sim": start the controller without the joystick
            str(robot_config / "params.yaml"),
            str(controllers_config / "robot_controllers.yaml"),
            str(hw_config / "joystick.yaml"),
            str(hw_config / "robot_hw.yaml"),
        ],
    )

    # Mid-360 cloud
    scan = Node(
        package="pointcloud_to_laserscan", executable="pointcloud_to_laserscan_node",
        remappings=[("cloud_in", "/livox/lidar"), ("scan", "/scan")],
        parameters=[{"target_frame": "livox_frame", "min_height": -0.1, "max_height": 0.1,  # the 0 deg ring
                     "range_min": 0.2, "range_max": 20.0,
                     "angle_increment": math.radians(0.5), "scan_time": 0.1}],  # the sim's azimuth step, 10 Hz
    )

    nav2_bringup = share("nav2_bringup")
    nav2 = TimerAction(  # after the robot has stood up
        period=float(arg("nav2_delay")),
        condition=IfCondition(arg("nav2")),
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(nav2_bringup / "launch" / "bringup_launch.py")),
            launch_arguments={"map": str(Path(start["map"]).resolve()),
                              "params_file": str(ROOT / "config" / "nav2_tron1.yaml"),
                              "use_sim_time": "False", "autostart": "True"}.items(),
        )],
    )
    # A plain node: including nav2_bringup's rviz_launch.py would set the (global)
    # launch argument namespace:=navigation, which Nav2's bringup then picks up.
    rviz = Node(
        package="rviz2", executable="rviz2", condition=IfCondition(arg("rviz")),
        arguments=["-d", str(nav2_bringup / "rviz" / "nav2_default_view.rviz")],
    )
    return [sim, controller, scan, nav2, rviz]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("route", default_value="", description="route yaml; empty = the route recorded last"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("nav2", default_value="true"),
        DeclareLaunchArgument("nav2_delay", default_value="15.0"),
        OpaqueFunction(function=launch_setup),
    ])
