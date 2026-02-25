from setuptools import find_packages, setup

package_name = 'ObsAwarePlan'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Davide Nascivera',
    maintainer_email='davidenascivera@example.com',
    description='UWB Obstacle-Aware SLAM planner for PX4 drones.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'position_ekf   = ObsAwarePlan.postion_UWB:main',
            'slam_planner   = ObsAwarePlan.slam_planner_vel:main',
        ],
    },
)
