from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'team_x'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    package_data={package_name: ['model/*.pt']},
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, [
            'package.xml',
            'README_team_x.md',
            'README_debug.md',
        ]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'docker'), glob('docker/*')),
        (os.path.join('share', package_name, 'model'),
         glob(os.path.join(package_name, 'model', '*.pt'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team_x',
    maintainer_email='ningan@example.com',
    description='Merged Team X contest pipeline for the CS477 IIR Picking Challenge.',
    license='BSD',
    entry_points={
        'console_scripts': [
            'contest_executor = team_x.contest_executor:main',
            'detection_server = team_x.detection_server:main',
            'vision_server = team_x.vision.vision_server:main',
            'vision_client = team_x.vision.vision_client:main',
            'vision_snapshot_collector = team_x.vision.snapshot_collector:main',
        ],
    },
)
