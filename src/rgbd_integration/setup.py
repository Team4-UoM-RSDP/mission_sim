from setuptools import find_packages, setup

package_name = 'rgbd_integration'

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
    maintainer='Team 4',
    maintainer_email='174347826+bt-nav@users.noreply.github.com',
    description='Intel D435i RGB-D camera integration for the Elephant myCobot 280pi manipulator.',
    license='BSD-3-Clause',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'object_detection = rgbd_integration.object_detection:main'
        ],
    },
)
