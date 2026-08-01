from setuptools import find_packages, setup

setup(
    name='cachespec',
    packages=find_packages(include=['llmlib']),
    version='0.1.1',
    description='CacheSpec library components for LLM caching experiments',
    author='Sarthak Chakraborty',
    install_requires=['langchain'],
    setup_requires=['pytest-runner'],
)
