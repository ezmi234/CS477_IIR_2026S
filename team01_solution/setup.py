from setuptools import find_packages, setup
from glob import glob
package_name = 'team01_solution'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cam',
    maintainer_email='camillegaudron37@gmail.com',
    description='Team 01 solution for CS477 robotics integration challenge',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'standby_node = team01_solution.standby_node:main',
            'instruction_parser = team01_solution.instruction_parser:main',
        ],
    },
)
