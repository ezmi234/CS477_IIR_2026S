from setuptools import setup

package_name = 'assignment_1'

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
    maintainer='daehyung',
    maintainer_email='daehyung@kaist.ac.kr',
    description='Assignment 2',
    license='Not allowed to share',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'move_joint = assignment_1.move_joint:main',
            'forward_kin = assignment_1.forward_kin:main',            
            'inverse_kin = assignment_1.inverse_kin:main',            
        ],
    },
)
