from setuptools import setup, find_packages
import torch

# Import the DataDicLoader class
from anfuelpriceenv.DataDic import DataDicLoader

# Load the data using DataDicLoader
loader = DataDicLoader()
DataDic = loader.load_data()

setup(
    name='anfuelpriceenv',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchrl',
        'tensordict',
        'pandas',
        'numpy',
        'requests'
    ],
    include_package_data=True,  # Ensure package data is included
    package_data={
        'anfuelpriceenv': ['DataDic.pt'],  # You might still need this if the file exists
    },
    # Function to be executed during installation
    data_files=[
        ('anfuelpriceenv/data', ['temp_file.pt']) # Include the downloaded file as data
    ]
)

import os
# Remove the temp file if it's not part of your package data
os.remove("temp_file.pt")
