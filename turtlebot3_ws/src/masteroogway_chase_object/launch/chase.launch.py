from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='masteroogway_chase_object',
            executable='detect_object',
            output='screen'
        ),
        Node(
            package='masteroogway_chase_object',
            executable='chase_object',
            output='screen'
        ),
    ])

