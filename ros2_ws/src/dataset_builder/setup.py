from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'dataset_builder'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'configs'), glob('dataset_builder/configs/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'pyyaml',  # import yaml
        'pytz',    # timezone handling in run_id generation
    ],
    zip_safe=False,
    maintainer='Dataset Builder',
    maintainer_email='user@example.com',
    description='Dataset builder for continuous robot data collection',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run_manager = dataset_builder.nodes.run_manager:main',
            'metadata_logger = dataset_builder.nodes.metadata_logger:main',
        ],
    },
)