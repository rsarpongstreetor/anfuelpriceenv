from typing import Dict as TypingDict, Any, Union, List, Optional
import torch
import pandas as pd
import numpy as np


class DDataenv:
    def __init__(self, data_path: str, data_columns: List[str], data_type: Any = np.float32):
        self.data_path = data_path
        self.data_columns = data_columns
        self.data_type = data_type
        self.data = None

    def load_data(self) -> pd.DataFrame:
        with open(self.data_path, 'rb') as f:
            self.data = torch.load(f, weights_only=False)
            self.data = np.array(self.data)
            # Check if the array has at least 3 dimensions
        if len(self.data.shape) >= 3:
            self.data = self.data.reshape(self.data.shape[1], self.data.shape[2])
        else:
            # Handle cases where the loaded data has fewer dimensions
            # You might need to adjust this logic based on the actual structure of your data
            print("Warning: Loaded data has fewer than 3 dimensions. Reshape skipped.")

        if not isinstance(self.data, pd.DataFrame):
            self.data = pd.DataFrame(self.data, columns=self.data_columns)
        self.data = np.array(self.data).reshape(self.data.shape[0], self.data.shape[1])

        if not isinstance(self.data, pd.DataFrame):
            self.data = pd.DataFrame(self.data, columns=self.data_columns)

    def get_observation(self) -> TypingDict[str, Union[np.ndarray, TypingDict[str, float]]]:
        if self.data is None:
            self.load_data()
        random_row_index = np.random.choice(self.data.shape[0], 1, replace=False)[0]
        observation = self.data.iloc[random_row_index, :].to_numpy().astype(self.data_type)
        describe_data = self.data.describe()
        min_value = describe_data.loc['min']
        max_value = describe_data.loc['max']
        date_min = self.data['Date'].min() if np.issubdtype(self.data['Date'].dtype, np.number) else self.data['Date'].iloc[0]
        date_max = self.data['Date'].max() if np.issubdtype(self.data['Date'].dtype, np.number) else self.data['Date'].iloc[-1]

        # Structure observation as a dictionary (concisely)
        observation_dict = {
            'obsState&Fuel': observation[0:13],
            'Date': observation[-1],
            'rewardState&reward': observation[13:26],
            'actionState&action': observation[26:39],
            'obsState&Fuel_max': max_value[0:13].values,
            'obsState&Fuel_min': min_value[0:13].values,
            'Date_max': date_max,
            'Date_min': date_min,
            'rewardState&reward_max': max_value[13:26].values,
            'rewardState&reward_min': min_value[13:26].values,
            'actionState&action_max': max_value[26:39].values,
            'actionState&action_min': min_value[26:39].values,
        }
        return observation_dict