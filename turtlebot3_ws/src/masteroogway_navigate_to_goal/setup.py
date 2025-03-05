from setuptools import setup
# import os
from glob import glob

package_name = 'masteroogway_navigate_to_goal'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),  # Ensure launch files are installed
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='henDan',
    # maintainer_email='your_email@example.com',
    description='ROS2 package for parcour',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'goToGoal = masteroogway_navigate_to_goal.goToGoal:main',
            'detectObstacle = masteroogway_navigate_to_goal.detectObstacle:main',
            'getRobotGlobalPos = masteroogway_navigate_to_goal.getRobotGlobalPos:main',
            'planPath = masteroogway_navigate_to_goal.planPath:main',
            'odomListener = masteroogway_navigate_to_goal.odomListener:main',
            'testObstaclePublisher = masteroogway_navigate_to_goal.test_obstacle_publisher:main',
            'drawMap = masteroogway_navigate_to_goal.drawMap:main',
        ],
    },
)

