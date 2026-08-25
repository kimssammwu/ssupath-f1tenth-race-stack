import os

import launch
import launch_ros.actions
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    main_param_dir = LaunchConfiguration(
        'main_param_dir',
        default=os.path.join(
            get_package_share_directory('lidarslam'),
            'param',
            'lidarslam.yaml'))

    rviz_param_dir = LaunchConfiguration(
        'rviz_param_dir',
        default=os.path.join(
            get_package_share_directory('lidarslam'),
            'rviz',
            'localization.rviz'))

    map_path = LaunchConfiguration('map_path')

    mapping = launch_ros.actions.Node(
        package='scanmatcher',
        executable='scanmatcher_node',
        parameters=[main_param_dir, {'map_path': map_path}],
        remappings=[('/input_cloud', '/livox/lidar')],
        output='screen')

    tf = launch_ros.actions.Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.27', '0', '0.07', '0', '0', '0.7171', '0.7171', 'base_link', 'livox_frame'])

    rviz = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_param_dir],
        condition=IfCondition(LaunchConfiguration('rviz')))

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            'main_param_dir',
            default_value=main_param_dir,
            description='Full path to main parameter file to load'),
        launch.actions.DeclareLaunchArgument(
            'map_path',
            default_value='/home/misys/forza_ws/race_stack/map.pcd',
            description='Localization target PCD path'),
        launch.actions.DeclareLaunchArgument(
            'rviz',
            default_value='false',
            description='Run rviz2 on the car (draw /initialpose arrow remotely instead)'),
        mapping,
        tf,
        rviz,
    ])
