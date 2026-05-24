from setuptools import find_packages, setup

setup(
    name='llmlib',
    packages=find_packages(include=['llmlib']),
    version='0.1.1',
    description='LLM Wrapper Library with Caching for AI Agents',
    author='Sarthak Chakraborty',
    install_requires=['langchain'],
    setup_requires=['pytest-runner'],
)