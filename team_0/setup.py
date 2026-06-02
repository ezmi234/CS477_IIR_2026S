from glob import glob
import os

from setuptools import setup

package_name = 'team_0'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_data={package_name: ['model/*.pt']},
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, [
            'package.xml',
            'README_team_0.md',
            'README_debug.md',
        ]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'docker'), glob('docker/*')),
        (os.path.join('share', package_name, 'model'),
         glob(os.path.join(package_name, 'model', '*.pt'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team_0',
    maintainer_email='ningan@example.com',
    description='Contest runner for the CS477 IIR Picking Challenge.',
    license='BSD',
    entry_points={
        'console_scripts': [
            'contest_executor = team_0.contest_executor:main',
            'detection_server = team_0.detection_server:main',
        ],
    },
)
