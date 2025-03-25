from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='masteroogway_go_to_goal',
            executable='getRobotGlobalPos',
            output='none'
        ),
        Node(
            package='masteroogway_go_to_goal',
            executable='detectObstacle',
            output='screen'
        ),
        Node(
            package='masteroogway_go_to_goal',
            executable='planPath',
            output='screen'
        ),
    ])

