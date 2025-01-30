import requests
import torch
import pandas as pd
import os



class DataDicLoader:
    """
    A class to load the DataDic data from a remote source.
    """

    def __init__(self, data_path="https://drive.google.com/uc?id=1K7OBG-qZnVC4Sm7-zwLqIXTmNRLYe02e"):
        """
        Initializes the DataDicLoader with the data path.
        """
        self.data_path = data_path
        self.data_dic = None  # Initialize data_dic attribute

    def load_data(self):
        """
        Downloads the data, loads it using torch, and removes the temporary file.
        """
        response = requests.get(self.data_path)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open("temp_file.pt", 'wb') as f:
            f.write(response.content)

        with open("temp_file.pt", 'rb') as f:
            self.data_dic = torch.load(f, weights_only=False)  # Load the entire file, including custom classes
            
        os.remove("temp_file.pt")

        return self.data_dic # Return the loaded data_dic

# Example usage:
loader = DataDicLoader()  # Create an instance of the class
data_dic = loader.load_data()  # Call the load_data method to load the data

print(data_dic)  # Now you can access the loaded data_dic
