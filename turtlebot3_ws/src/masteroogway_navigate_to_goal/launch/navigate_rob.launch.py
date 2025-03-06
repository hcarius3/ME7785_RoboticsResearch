from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='masteroogway_navigate_to_goal',
            executable='getRobotGlobalPos',
            output='log'
        ),
        Node(
            package='masteroogway_navigate_to_goal',
            executable='goToGoal',
            output='screen'
        ),
    ])

