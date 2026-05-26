from setuptools import find_packages, setup

setup(
    name='minicache',
    packages=find_packages(include=['llmlib']),
    version='0.1.1',
    description='MiniCache library components for LLM caching experiments',
    author='Sarthak Chakraborty',
    install_requires=['langchain'],
    setup_requires=['pytest-runner'],
)
