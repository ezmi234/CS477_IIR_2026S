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
            'check_yolo_backend = team_1.tools.check_yolo_backend:main',
            'sweep_yolo_threshold = team_1.tools.sweep_yolo_threshold:main',
            'evaluate_perception_with_ground_truth = team_1.tools.evaluate_perception_with_ground_truth:main',
            'evaluate_pick_with_ground_truth = team_1.tools.evaluate_pick_with_ground_truth:main',
            'grasp_diagnostics = team_1.tools.run_grasp_diagnostics:main',
            'calibrate_object_grasp = team_1.tools.calibrate_object_grasp:main',
            'random_trial_diagnostics = team_1.tools.run_random_trial_diagnostics:main',
            'run_full_competition_diagnostics = team_1.tools.run_full_competition_diagnostics:main',
        ],
    },
)
