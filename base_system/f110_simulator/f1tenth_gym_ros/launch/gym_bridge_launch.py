# MIT License

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def generate_launch_description():
    ld = LaunchDescription()

    map_yaml_path = LaunchConfiguration('map_yaml_path')
    map_yaml_path_arg = DeclareLaunchArgument(
        'map_yaml_path', description='Path to map YAML file. Passed in via top-level launchfile.')
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='false', description='Run simulator RViz instance')

    sim_setup_params = os.path.join(
        get_package_share_directory('stack_master'), 'config', 'SIM', 'sim.yaml')

    config_dict = yaml.safe_load(open(sim_setup_params, 'r'))
    has_opp = config_dict['bridge']['ros__parameters']['num_agent'] > 1

    bridge_node = Node(
        package='f1tenth_gym_ros',
        executable='gym_bridge',
        name='bridge',
        parameters=[
            sim_setup_params,
            {'map_path': map_yaml_path},
            {'sim_params': os.path.join(
                get_package_share_directory('stack_master'), 'config', 'SIM', 'sim_params.yaml')},
            {'use_sim_time': False},
        ])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_sim',
        arguments=['-d', os.path.join(
            get_package_share_directory('stack_master'), 'config', 'SIM', 'sim.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')))

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        parameters=[
            {'yaml_filename': map_yaml_path},
            {'topic': 'map'},
            {'frame_id': 'map'},
            {'output': 'screen'},
            {'use_sim_time': False},
        ])

    nav_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[
            {'use_sim_time': False},
            {'autostart': True},
            {'node_names': ['map_server']},
        ])

    ego_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ego_robot_state_publisher',
        parameters=[{
            'robot_description': Command([
                'xacro ', os.path.join(
                    get_package_share_directory('f1tenth_gym_ros'), 'config', 'ego_racecar.xacro')]),
            'use_sim_time': False,
        }],
        remappings=[('/robot_description', 'ego_robot_description')])

    opp_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='opp_robot_state_publisher',
        parameters=[{
            'robot_description': Command([
                'xacro ', os.path.join(
                    get_package_share_directory('f1tenth_gym_ros'), 'config', 'opp_racecar.xacro')]),
            'use_sim_time': False,
        }],
        remappings=[('/robot_description', 'opp_robot_description')])

    ld.add_action(map_yaml_path_arg)
    ld.add_action(rviz_arg)
    ld.add_action(rviz_node)
    ld.add_action(bridge_node)
    ld.add_action(nav_lifecycle_node)
    ld.add_action(map_server_node)
    ld.add_action(ego_robot_publisher)
    if has_opp:
        ld.add_action(opp_robot_publisher)

    return ld
