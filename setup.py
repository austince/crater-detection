from codecs import open
from os import path
from subprocess import call
from setuptools import Command, find_packages, setup

from crater_detection import __version__

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, 'README.md'), encoding='utf-8') as readme_file:
    long_description = readme_file.read()

with open(path.join(this_dir, 'requirements.txt')) as reqs_file:
    reqs_list = reqs_file.read().splitlines()


class RunTests(Command):
    """Run all tests."""
    description = 'run tests'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run all tests!"""
        errno = call(['py.test',
                      '--cov=crater_detection',
                      '--cov-report=term-missing',
                      '--ignore=env'])
        raise SystemExit(errno)


setup(
    name='crater_detection',
    version=__version__,
    description='Basic lunar crater detection using Open CV',
    long_description=long_description,
    url='https://github.com/',
    author='austin ce',
    author_email='austin.cawley@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Other Audience',
        'Topic :: Utilities',
        'License :: Public Domain',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='cli',
    packages=find_packages(exclude=['docs', 'tests*']),
    install_requires=reqs_list,
    extras_require={
        'test': ['coverage', 'pytest', 'pytest-cov'],
    },
    entry_points={
        'console_scripts': [
            'crater-detect=crater_detection.cli:main',
        ],
    },
    cmdclass={'test': RunTests},
)
