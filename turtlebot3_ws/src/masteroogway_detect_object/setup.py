# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

from setuptools import setup
# import os
from glob import glob

package_name = 'masteroogway_detect_object'

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
    description='ROS2 package for object following',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'find_object = masteroogway_detect_object.find_object:main',
            'rotate_robot = masteroogway_detect_object.rotate_robot:main',
        ],
    },
)
