from setuptools import setup

package_name = 'assignment_3'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Daehyung Park',
    maintainer_email='pidipidi52@gmail.com',
    description='Assignment 1',
    license='Now allowed to share.',
    entry_points={
        'console_scripts': [
            'cartpole_p = assignment_3.gazebo_cartpole_p_v0:main',
            'cartpole_pd = assignment_3.gazebo_cartpole_pd_v0:main',
            'cartpole_pid = assignment_3.gazebo_cartpole_pid_v0:main',
            'move_joint = assignment_3.move_joint:main',
        ],
    },
)
