from setuptools import find_packages, setup

package_name = 'drone_package'
submodules = 'drone_package/submodules'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kristijan Segulja',
    maintainer_email='kristijan.segulja22@gmail.com',
    description='Package for drone navigation and obstacle detection',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drone_agent_DQN = drone_package.drone_agent_DQN:main',
            'drone_agent_DoubleDQN = drone_package.drone_agent_DoubleDQN:main',
            'drone_agent_SAC = drone_package.drone_agent_SAC:main',
        ],
    },
)
