import requests
import torch

data_path="https://drive.google.com/uc?id=1K7OBG-qZnVC4Sm7-zwLqIXTmNRLYe02e" #fixed the link

response = requests.get(data_path)
response.raise_for_status() # Raise an exception for bad status codes

with open("temp_file.pt", 'wb') as f:
  f.write(response.content)

with open("temp_file.pt", 'rb') as f:
  DataDic = torch.load(f) # removed weights_only, it's not a valid argument
print(DataDic)
import os
os.remove("temp_file.pt")
print(DataDic) 