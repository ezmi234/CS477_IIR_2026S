from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'team01_solution'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README_team01.md', 'README_team_1.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team01',
    maintainer_email='team01@example.com',
    description='Integrated Team 01 solution for the CS477 IIR Picking Challenge.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            # Final integrated runtime.
            'contest_executor = team01_solution.contest_executor:main',
            'vision_server = team01_solution.vision.vision_server:main',
            'vision_client = team01_solution.vision.vision_client:main',
            'vision_snapshot_collector = team01_solution.vision.snapshot_collector:main',

            # Previous phase nodes kept for compatibility/debugging.
            'standby_node = team01_solution.standby_node:main',
            'instruction_parser = team01_solution.instruction_parser:main',
            'detection_node = team01_solution.detection_node:main',
            'grasp_node = team01_solution.grasp_node:main',
            'motion_node = team01_solution.motion_node:main',
            'motion_node_rrt = team01_solution.motion_node_rrt:main',
        ],
    },
)
