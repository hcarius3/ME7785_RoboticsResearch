from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='masteroogway_object_follower',
            executable='find_object',
            output='screen'
        ),
        Node(
            package='masteroogway_object_follower',
            executable='rotate_robot',
            output='screen'
        ),
    ])
