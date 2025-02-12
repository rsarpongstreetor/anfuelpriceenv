from setuptools import setup, find_packages

setup(
    name="AnFuelpriceEnv",
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchrl',
        'tensordict',
        'pandas',
        'numpy',
        'pyyaml'
        # ... add any other dependencies here
    ],
    # ... other setup options if needed
)
