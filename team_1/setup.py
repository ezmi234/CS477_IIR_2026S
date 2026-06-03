from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'team_1'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README_team_1.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team_1',
    maintainer_email='team01@example.com',
    description='Final Team 1 RGB-D manipulation pipeline for the CS477 IIR Picking Challenge.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'contest_executor = team_1.contest_executor:main',
            'vision_server = team_1.vision.vision_server:main',
            'vision_client = team_1.vision.vision_client:main',
            'vision_snapshot_collector = team_1.vision.snapshot_collector:main',
        ],
    },
)
