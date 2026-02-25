from setuptools import find_packages, setup

package_name = 'obs_aware_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dave',
    maintainer_email='d.nascivera@gmail.com',
    description='Obstacle-aware UWB SLAM for UAVs using EKF and Reactive Planner navigation',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'position_ekf = obs_aware_slam.postion_UWB:main',
        ],
    },
)
