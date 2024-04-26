from setuptools import setup, find_packages
import os

def read(file_name):
    with open(os.path.join(os.path.dirname(__file__), file_name), 'r') as f:
        return f.read()

setup(
    name='dreambooth_trainer',
    version='0.1.0',
    author='Haoyang Ma',
    author_email='hm235@duke.edu',
    description='A script for running the dreambooth_trainer.py script',
    long_description=read('README.md'),
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'dreambooth_trainer=models.dreambooth_trainer:main',
        ],
    },
    install_requires=[
        # Add your project dependencies here
        'torch',
        'diffusers',
        # etc.
    ],
)