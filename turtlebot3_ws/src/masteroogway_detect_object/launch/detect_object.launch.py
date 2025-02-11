# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='masteroogway_detect_object',
            executable='find_object',
            output='screen'
        ),
        Node(
            package='masteroogway_detect_object',
            executable='rotate_robot',
            output='screen'
        ),
    ])
