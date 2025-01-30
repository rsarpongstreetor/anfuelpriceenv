from setuptools import setup, find_packages

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
         include_package_data=True, # Include the DataDic.pt file
         package_data={
            'anfuelpriceenv': ['DataDic.pt'],
        },
     )
