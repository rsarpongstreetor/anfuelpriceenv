from setuptools import setup, find_packages
from DataDic.py import DataDicLoader  # Import the class

     setup(
         name='anfuelpriceenv',  # Package name
         version='0.1.0',  # Package version
         packages=find_packages(),  # Include all sub-packages
         install_requires=[  # List of dependencies
             'torch',
             'torchrl',
             'tensordict',
             'pandas',
             'numpy',
             # Add other necessary packages
         ],
          loader = DataDicLoader()  # Create an instance of the class
          DataDic.pt= loader.load_data()  # Call the load_data method to load the data
          
          include_package_data=True, # Include the DataDic.pt file
         
          package_data={
            'anfuelpriceenv': ['DataDic.pt'],
        },
     )
