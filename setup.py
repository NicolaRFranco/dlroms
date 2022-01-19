from distutils.core import setup
from setuptools import find_packages

setup(
    name='dlrom',
    version='0.1.0',
    packages=find_packages(include=['dlrom', 'dlrom.*']),
    install_requires=[
        'ipython>=7.15.0',
        'numpy>=1.18.1',
        'matplotlib>=3.2.1',
	'fenics==2019.1.0',
	'scipy>=1.4.1',
	'pytorch>=1.5.0'
    ]
)
