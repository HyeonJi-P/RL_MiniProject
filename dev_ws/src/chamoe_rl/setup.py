from setuptools import find_packages, setup

package_name = 'chamoe_rl'

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
    maintainer='root',
    maintainer_email='hyunji7674@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'chamoe_rl_result_graph = chamoe_rl.result_graph:main',
            'chamoe_rl_6action_graph = chamoe_rl.6action_graph:main',
            'chamoe_detect = chamoe_rl.chamoe_detect:main',
            'chamoe_rl_RL_environment = chamoe_rl.RL_environment:main',
            'chamoe_rl_gazebo_interface = chamoe_rl.DQN_gazebo_interface:main',
            'chamoe_rl_DQN_agent = chamoe_rl.DQN_agent:main',
            'chamoe_rl_DQN_test = chamoe_rl.DQN_test:main',
            'chamoe_rl_Q_agent = chamoe_rl.Q_agent:main',
            'chamoe_rl_Q_test = chamoe_rl.Q_test:main',
        ],
    },
)
