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
        ('share/' + package_name + '/msg', glob('msg/*.msg')),  # Include messages
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
            'detectObject = masteroogway_navigate_to_goal.detectObject:main',
            'getRobotGlobalPos = masteroogway_navigate_to_goal.getRobotGlobalPos:main',
            'planPath = masteroogway_navigate_to_goal.planPath:main',
        ],
    },
)

