####Categorical action spec Configuration

import pandas as pd
import os

# Find the part of the code that loads the data.
# It might look something like this:
# data_path = "some_url_or_path_here"
# data = pd.read_csv(data_path)

# Replace the data loading logic with this:






# Recreate the base environment instance and the PyGBatchProcessor for training
import networkx as nx
import random
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool # Import global_mean_pool
from torch_geometric.nn import SplineConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
from torchrl.data import Unbounded, MultiCategorical, Categorical, Composite # Import Composite
from typing import Optional
import numpy as np
import torch # Import torch
from tensordict.nn import TensorDictModuleBase # Import TensorDictModuleBase


class FuelpriceenvfeatureGraph():

    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.graph["graph_attr_1"] = random.random() * 10
        self.graph.graph["graph_attr_2"] = random.random() * 5.

    def _load_data(self):
        from scipy.stats import zscore
        import pandas as pd
        import numpy as np
        import os # Import os



        local_data_path = '/content/anfuelpriceenv/Cleaneddata (5).csv'
        # Removed the file existence check as we are reading from a URL directly

        if os.path.exists(local_data_path):
            data = pd.read_csv(local_data_path)
            print(f"Successfully loaded data from {local_data_path}")
        else:
            # Handle the case where the local file is not found
            print(f"Error: Data file not found at {local_data_path}. Please ensure it was downloaded.")
            data = None # Or raise an error, depending on the original code's behavior

        try:
            csv_column_names = [
                'USD_SDR', 'OPEC', 'Brent', 'WTI', "('Date',)",
                "('EER_EPD2DC_PF4_Y05LA_DPG',)", "('EER_EPD2DXL0_PF4_RGC_DPG',)",
                "('EER_EPD2DXL0_PF4_Y35NY_DPG',)", "('EER_EPD2F_PF4_Y35NY_DPG',)",
                "('EER_EPJK_PF4_RGC_DPG',)", "('EER_EPLLPA_PF4_Y44MB_DPG',)",
                "('EER_EPMRR_PF4_Y05LA_DPG',)", "('EER_EPMRU_PF4_RGC_DPG',)",
                "('EER_EPMRU_PF4_Y35NY_DP',)" # Corrected column name here
            ]

            dffff = pd.read_csv(
                local_data_path,
                header=0,
                names=csv_column_names,
                parse_dates=["('Date',)"]
            )

            dffff = dffff.set_index("('Date',)")
            dffff = dffff.ffill()
            dffff.dropna(axis=0, how='any', inplace=True)

            numeric_cols_dffff = dffff.select_dtypes(include=np.number)
            # Handle columns with zero standard deviation gracefully
            abs_z_scores_dffff = numeric_cols_dffff.apply(lambda x: np.abs(zscore(x, ddof=0)) if x.std() != 0 else pd.Series(0, index=x.index))
            threshold = 3
            outliers_dffff = abs_z_scores_dffff > threshold
            dffff.loc[:, numeric_cols_dffff.columns][outliers_dffff] = 0
            print(f"Loading data (allow_repeat_data: {self.allow_repeat_data})...")

            feature_columns = [
                'USD_SDR', 'OPEC', 'Brent', 'WTI',
                "('EER_EPD2DC_PF4_Y05LA_DPG',)", "('EER_EPD2DXL0_PF4_RGC_DPG',)",
                "('EER_EPD2DXL0_PF4_Y35NY_DPG',)", "('EER_EPD2F_PF4_Y35NY_DPG',)",
                "('EER_EPJK_PF4_RGC_DPG',)", "('EER_EPLLPA_PF4_Y44MB_DPG',)",
                "('EER_EPMRR_PF4_Y05LA_DPG',)", "('EER_EPMRU_PF4_RGC_DPG',)",
                "('EER_EPMRU_PF4_Y35NY_DP',)" # Corrected column name here
            ]

            features_df = dffff[feature_columns].select_dtypes(include=np.number)
            numberr_np = features_df.values

            def returns(x):
              x = np.array(x)
              return x[1:, :] - x[:-1, :]
            RRRR = returns(numberr_np)

            def actionspace(x):
              x = np.array(x)
              differences = x[1:, :] - x[:-1, :]
              yxx = np.zeros_like(differences)
              yxx[differences > 0] = 2
              yxx[differences < 0] = 0
              yxx[differences == 0] = 1
              return yxx
            action = actionspace(numberr_np)

            Indep = np.hstack((RRRR, action))
            features_aligned_np = numberr_np[1:, :]
            self.combined_data = np.hstack([features_aligned_np, Indep])

            # Explicitly check for and handle NaNs and Infs before converting to tensor
            if np.isnan(self.combined_data).any() or np.isinf(self.combined_data).any():
                print("Warning: NaNs or Infs found in combined_data numpy array before tensor conversion.")
                if np.isnan(self.combined_data).any():
                     nan_indices = np.argwhere(np.isnan(self.combined_data))
                     print(f"  NaN indices: {nan_indices[:5]}...") # Print first 5 if many
                if np.isinf(self.combined_data).any():
                     inf_indices = np.argwhere(np.isinf(self.combined_data))
                     print(f"  Inf indices: {inf_indices[:5]}...") # Print first 5 if many

                print(f"  Shape of combined_data before replacement: {self.combined_data.shape}")
                print(f"  Sample of combined_data before replacement:\n{self.combined_data[:5, :5]}...") # Print sample

                # Manual replacement using boolean indexing
                self.combined_data[np.isnan(self.combined_data)] = 0.0
                self.combined_data[np.isinf(self.combined_data)] = 0.0

                if np.isnan(self.combined_data).any() or np.isinf(self.combined_data).any():
                     print("Error: NaNs or Infs still present after manual replacement in numpy array. Investigate data preprocessing.")
                     # Consider raising an error here if invalid values are critical
                     raise ValueError("Invalid values in combined_data after numpy cleaning.")
                else:
                     print("NaNs and Infs successfully replaced in numpy array.")


            # Print shape and sample after handling invalid values but before tensor conversion
            print(f"  Shape of combined_data after replacement (numpy): {self.combined_data.shape}")
            print(f"  Sample of combined_data after replacement (numpy):\n{self.combined_data[:5, :5]}...")


            # Convert combined_data to torch tensor and move to device
            print(f"Attempting to convert combined_data to torch tensor on device {self.device}...")
            self.combined_data = torch.as_tensor(self.combined_data, dtype=torch.float32, device=self.device)
            print("Conversion to torch tensor successful.")

            # Explicitly check for NaNs/Infs in the resulting tensor on the device
            if torch.isnan(self.combined_data).any() or torch.isinf(self.combined_data).any():
                 print("Error: NaNs or Infs found in the PyTorch tensor after conversion.")
                 # Option 1: Replace invalid values in the tensor
                 # self.combined_data = torch.nan_to_num(self.combined_data, nan=0.0, posinf=0.0, neginf=0.0)
                 # print("NaNs and Infs replaced in PyTorch tensor.")
                 raise ValueError("Invalid values in PyTorch tensor after conversion.")
            else:
                 print("No NaNs or Infs found in the PyTorch tensor.")


            # Calculate observation bounds after loading data and moving to device
            obs_dim = 13 # Define obs_dim here
            self.obs_min = torch.min(self.combined_data[:, :obs_dim], dim=0)[0].unsqueeze(-1).to(self.device)
            self.obs_max = torch.max(self.combined_data[:, :obs_dim], dim=0)[0].unsqueeze(-1).to(self.device)

            print(f"\nCombined dataset loaded and processed with shape: {self.combined_data.shape} on device {self.device}.")
            print(f"Observation bounds calculated: Min = {self.obs_min.cpu().numpy()}, Max = {self.obs_max.cpu().numpy()}") # Print on CPU


        except Exception as e:
            print(f"Error loading or preprocessing data: {e}")
            raise e

        print(f"Data loading complete.")

    def get_graph_observation(self):
        """
        Generates a PyTorch Geometric Data object using slices of self.combined_data.
        Uses self.combined_data[:, 0:12] as node features, the index as edge indices,
        and a weighted mean of self.combined_data[:, 13:26] and self.combined_data[:, 26:38] as graph attributes.

        Returns:
            data (Data): The graph data object.
            num_nodes (int): The number of nodes in the graph.
            num_edges (int): The number of edges in the graph.
        """
        if self.combined_data is None:
            raise ValueError("Data has not been loaded. Call _load_data() first.")

        # Use self.combined_data[:, 0:12] as node features
        x = self.combined_data[:, 0:12]
        num_nodes = x.size(0)

        # Use the index as edge indices (assuming a fully connected graph for now, or define edge logic based on your data)
        # This is a placeholder and might need adjustment based on how you want to define edges from your time-series data
        # For demonstration, let's create a simple sequential edge structure
        edge_index = torch.arange(num_nodes - 1).repeat(2, 1)
        edge_index[1, :] = edge_index[1, :] + 1
        num_edges = edge_index.size(1)


        # Calculate weighted mean of graph attributes
        graph_attributes_part1 = torch.mean(self.combined_data[:, 13:26], dim=0)
        graph_attributes_part2 = torch.mean(self.combined_data[:, 26:39], dim=0)
        # Assuming equal weights for simplicity (0.5 for each part)
        graph_attributes = 0.5 * graph_attributes_part1 + 0.5 * graph_attributes_part2

        data = Data(x=x, edge_index=edge_index, graph_attributes=graph_attributes)

        # Note: Edge attributes were not specified in the slicing, so they are not included here.
        # If you need edge attributes, you'll need to define how to derive them from self.combined_data.

        return data, num_nodes, num_edges

    def get_node_tensor(self, node):
        """
        This method is no longer used with the new data loading approach.
        """
        raise NotImplementedError("get_node_tensor is not used in this version.")

    def get_edge_tensor(self, edge_data):
        """
        This method is no longer used with the new data loading approach.
        """
        raise NotImplementedError("get_edge_tensor is not used in this version.")

    def get_graph_tensor(self, graph_data):
        """
        This method is no longer used with the new data loading approach.
        Graph attributes are now directly derived from self.combined_data.
        """
        raise NotImplementedError("get_graph_tensor is not used in this version.")





class AnFuelpriceEnv(EnvBase):
    def __init__(self, num_envs, device, seed, **kwargs):
        self.episode_length = kwargs.get('episode_length', 100)
        self.num_agents = 13 # Keep as 13 based on original definition and feature count
        self.allow_repeat_data = kwargs.get('allow_repeat_data', False)
        self.num_envs = num_envs
        self.current_data_index = torch.zeros(num_envs, dtype=torch.int64, device=device)

        self.graph_generator = FuelpriceenvfeatureGraph()

        self.device = device

        self.graph_generator.device = self.device
        self.graph_generator.allow_repeat_data = self.allow_repeat_data

        self.graph_generator._load_data()
        self.combined_data = self.graph_generator.combined_data

        # Corrected num_agents to be consistent with __init__
        self.num_agents = 13
        self.num_individual_actions = 3 # Defined inside the class
        self.num_individual_actions_features = 13 # Still 13 action features per agent


        # In the new single-graph-per-environment structure, num_nodes_per_graph is num_agents
        self.num_nodes_per_graph = self.num_agents
        # Re-calculate num_edges_per_graph based on a potential graph structure among agents
        # For simplicity, let's assume a fully connected graph among agents for now.
        # This might need adjustment based on the actual desired graph structure.
        self.num_edges_per_graph = self.num_agents * (self.num_agents - 1) if self.num_agents > 1 else 0


        # Node feature dimension needs to be defined
        # If each node (agent) has a feature vector, define its dimension here.
        # Based on the data loading, the first 13 columns are features.
        # If each agent's node features are these 13 values, then node_feature_dim is 13.
        self.node_feature_dim = 1 # Changed to 1 assuming each node feature is a single value from the first 13 columns

        # Define obs_dim before using it
        # obs_dim is the dimension of the observation space for a single agent, or the relevant feature dimensions.
        # Given the structure, the first 13 columns seem to be the node features.
        obs_dim = 13 # This is the number of node features per graph (which is num_agents)

        # Re-calculate observation bounds based on the new node feature slicing
        # The observation bounds should be for the node features, which are the first 13 columns.
        self.obs_min = torch.min(self.combined_data[:, :self.num_nodes_per_graph], dim=0)[0].unsqueeze(-1).to(self.device) # Corrected slicing
        self.obs_max = torch.max(self.combined_data[:, :self.num_nodes_per_graph], dim=0)[0].unsqueeze(-1).to(self.device) # Corrected slicing


        super().__init__(device=device, batch_size=[num_envs])

        self._make_specs()


    def _make_specs(self):
        # Modified state_spec structure to reflect single graph per env
        self.state_spec = Composite(
             {
                 ("agents", "observation"): Composite({ # Nest under "agents" key
                     "x": Unbounded( # Node features [num_envs, num_nodes_per_graph, node_feature_dim]
                         shape=torch.Size([self.num_envs, self.num_nodes_per_graph, self.node_feature_dim]),
                         dtype=torch.float32,
                         device=self.device
                     ),
                     "edge_index": Unbounded( # Edge indices [num_envs, 2, num_edges_per_graph]
                         shape=torch.Size([self.num_envs, 2, self.num_edges_per_graph]),
                         dtype=torch.int64,
                         device=self.device
                     ),
                     "graph_attributes": Unbounded( # Graph attributes [num_envs, graph_attr_dim]
                          shape=torch.Size([self.num_envs, (26-13) + (39-26)]), # Assuming this is the graph attr dim
                          dtype=torch.float32,
                          device=self.device
                     ),
                 }),
                 ("agents", "global_reward_in_state"): Unbounded( # Agent-wise global reward in state [num_envs, num_agents, 1]
                      shape=torch.Size([self.num_envs, self.num_agents, 1]),
                      dtype=torch.float32,
                      device=self.device
                 ),
                 # "env_batch" is not needed as a state key in the single-graph-per-env structure
                 # The batch tensor from PyG Batch will handle mapping nodes to environments.
                 # The env_batch tensor is generated internally by the PyGBatchProcessor.
             },
             # The batch size of the state spec is the number of environments
             batch_size=self.batch_size,
             device=self.device,
         )
        print(f"Modified State specification defined with single graph per env structure and batch shape {self.state_spec.shape}.")

        # Reconstruct the action spec based on the provided keys structure
        # Type of reconstructed actions_valid_td_flat: <class 'tensordict._td.TensorDict'>
        # Keys of reconstructed actions_valid_td_flat: _TensorDictKeysView([('agents', 'agent_0', 'action_feature_0'), ...])

        # Define the action spec for a single agent
        agent_action_spec_dict = {}
        for i in range(self.num_individual_actions_features): # 13 action features per agent
             agent_action_spec_dict[f'action_feature_{i}'] = Categorical(
                  n=self.num_individual_actions, # 3 discrete choices per feature
                  shape=torch.Size([]), # Scalar action for each feature
                  dtype=torch.int64,
                  device=self.device
             )
        agent_action_spec = Composite(agent_action_spec_dict, batch_size=[], device=self.device) # No batch size for unbatched agent spec

        # Define the overall unbatched environment action spec as a Composite of agent specs
        env_action_spec_dict = {}
        for i in range(self.num_agents): # 13 agents
             env_action_spec_dict[f'agent_{i}'] = agent_action_spec # Use the single agent spec
        self.action_spec_unbatched = Composite(
             {("agents",): Composite(env_action_spec_dict, batch_size=[], device=self.device)}, # Nest under "agents"
             batch_size=[], # No batch size for the overall unbatched env spec
             device=self.device
        )


        # The batched action spec for the environment is automatically derived by TensorDict
        # from the unbatched spec and env batch size.
        # We can access it via self.action_spec
        print("\nUnbatched Multi-Agent Action specification defined using nested Composites and Categorical.")
        print(f"Unbatched Environment action_spec: {self.action_spec_unbatched}")
        print(f"Batched Environment action_spec: {self.action_spec}")


        # Restored original reward spec
        self.reward_spec = Composite(
             {('agents', 'reward'): Unbounded(shape=torch.Size([self.num_envs, self.num_agents, 1]), dtype=torch.float32, device=self.device)},
             batch_size=[self.num_envs],
             device=self.device,
        )
        print(f"Restored Agent-wise Reward specification defined with batch shape {self.reward_spec.shape}.")

        # Modified done_spec to include agents dimension
        self.done_spec = Composite(
            {
                ("agents", "done"):  Categorical( # Nested under agents
                      n=2,
                      shape=torch.Size([self.num_envs, self.num_agents, 1]), # Include num_agents dimension
                      dtype=torch.bool,
                      device=self.device),

                ("agents", "terminated"): Categorical( # Nested under agents
                      n=2,
                      shape=torch.Size([self.num_envs, self.num_agents, 1]), # Include num_agents dimension
                      dtype=torch.bool,
                      device=self.device),
                ("agents", "truncated"):  Categorical( # Nested under agents
                     n=2,
                     shape=torch.Size([self.num_envs, self.num_agents, 1]), # Include num_agents dimension
                      dtype=torch.bool,
                      device=self.device),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )
        print(f"Modified Done specification defined with agent dimension and batch shape {self.done_spec.shape}.")

        self.state_spec.unlock_(recurse=True)
        self.action_spec.unlock_(recurse=True) # Keep action_spec unlocked as it is batched
        self.reward_spec.unlock_(recurse=True)
        self.done_spec.unlock_(recurse=True) # Keep done_spec unlocked


    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.current_data_index += 1

        # Calculate global termination and truncation
        global_terminated = self._is_terminal()
        global_truncated = (self.current_data_index >= self.episode_length)

        # Broadcast global termination and truncation to per-agent shape
        per_agent_terminated = global_terminated.unsqueeze(-1).unsqueeze(-1).expand(self.num_envs, self.num_agents, 1)
        per_agent_truncated = global_truncated.unsqueeze(-1).unsqueeze(-1).expand(self.num_envs, self.num_agents, 1)
        per_agent_done = per_agent_terminated | per_agent_truncated # Per-agent done is OR of terminated and truncated


        # The action is now expected to be in the input tensordict passed to step() by the collector
        # It should be at tensordict[('agents', 'action')]

        # Pass the input tensordict directly to _batch_reward
        # _batch_reward will need to extract the action from this tensordict
        reward_td = self._batch_reward(self.current_data_index, tensordict) # Pass the entire input tensordict


        # Call _get_state_at with the current env indices to get the next state
        next_state_tensordict = self._get_state_at(torch.arange(self.num_envs, device=self.device))

        # Explicitly construct the output tensordict with observation keys at the root
        # Graph observation keys are now nested under ("agents", "observation")
        output_tensordict = TensorDict({
            ("agents", "observation"): TensorDict({
                 "x": next_state_tensordict.get(("agents", "observation", "x")),
                 "edge_index": next_state_tensordict.get(("agents", "observation", "edge_index")),
                 "graph_attributes": next_state_tensordict.get(("agents", "observation", "graph_attributes")),
            }, batch_size=[self.num_envs], device=self.device), # Batch size for nested tensordict should be num_envs
            ("agents", "global_reward_in_state"): next_state_tensordict.get(("agents", "global_reward_in_state")),
            ('agents', 'reward'): reward_td.get(("agents", "reward")), # Get reward from reward_td

            # Include per-agent done, terminated, and truncated in the output tensordict under the 'agents' key
            ("agents", "terminated"): per_agent_terminated,
            ("agents", "truncated"): per_agent_truncated,
            ("agents", "done"): per_agent_done,

            # "action": actions, # Removed: action is added by the collector, not returned by _step
            # env_batch is no longer a state key
        }, batch_size=self.batch_size, device=self.device)

        # Debugging print to check observation keys and reward before returning
        print("\n--- Environment _step output tensordict (before return) ---")
        print(f"Output tensordict keys: {output_tensordict.keys(include_nested=True)}") # Added include_nested=True
        print(f"Output tensordict shape: {output_tensordict.shape}")
        # Also print keys and shapes of nested observation tensordict
        if ("agents", "observation") in output_tensordict.keys(include_nested=True): # Added include_nested=True
             print(f"  Nested ('agents', 'observation') keys: {output_tensordict.get(('agents', 'observation')).keys(include_nested=True)}") # Added include_nested=True
             print(f"  Nested ('agents', 'observation') shape: {output_tensordict.get(('agents', 'observation')).shape}")
        # Print the reward value
        if ('agents', 'reward') in output_tensordict.keys(include_nested=True):
             print(f"  Reward value in output tensordict: {output_tensordict.get(('agents', 'reward'))}")
        else:
             print("  Reward key ('agents', 'reward') not found in output tensordict.")
        # Print done/terminated shapes
        if ('agents', 'done') in output_tensordict.keys(include_nested=True):
             print(f"  ('agents', 'done') shape: {output_tensordict.get(('agents', 'done')).shape}")
        if ('agents', 'terminated') in output_tensordict.keys(include_nested=True):
             print(f"  ('agents', 'terminated') shape: {output_tensordict.get(('agents', 'terminated')).shape}")
        print("-------------------------------------------")


        # Set next state directly under "next" with the root observation structure
        # The 'next' tensordict should contain the observation and termination signals for the next state
        output_tensordict.set("next", TensorDict({
            ("agents", "observation"): next_state_tensordict.get(("agents", "observation")),
            ("agents", "global_reward_in_state"): next_state_tensordict.get(("agents", "global_reward_in_state")),
            # Include per-agent done, terminated, and truncated in the 'next' tensordict as well
            ("agents", "terminated"): per_agent_terminated,
            ("agents", "truncated"): per_agent_truncated,
            ("agents", "done"): per_agent_done, # Use per-agent done here
        }, batch_size=self.batch_size, device=self.device))


        return output_tensordict


    def _reset(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        if self.allow_repeat_data and self.combined_data is not None:
             max_start_index = self.combined_data.shape[0] - self.episode_length -1
             if max_start_index < 0:
                  self.current_data_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
                  print("Warning: Data length is less than episode_length + 1. Starting episodes from index 0.")
             else:
                  self.current_data_index = torch.randint(0, max_start_index + 1, (self.num_envs,), dtype=torch.int64, device=self.device)
        else:
             self.current_data_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        # Calculate initial global termination and truncation (should be False for reset)
        global_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        global_truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Broadcast initial global flags to per-agent shape
        per_agent_terminated = global_terminated.unsqueeze(-1).unsqueeze(-1).expand(self.num_envs, self.num_agents, 1)
        per_agent_truncated = global_truncated.unsqueeze(-1).unsqueeze(-1).expand(self.num_envs, self.num_agents, 1)
        per_agent_done = per_agent_terminated | per_agent_truncated


        # Call _get_state_at with the current env indices to get the initial state
        initial_state_tensordict = self._get_state_at(torch.arange(self.num_envs, device=self.device))

        # Explicitly construct the initial tensordict with observation keys at the root
        # Graph observation keys are now nested under ("agents", "observation")
        initial_tensordict = TensorDict({
            ("agents", "observation"): TensorDict({
                 "x": initial_state_tensordict.get(("agents", "observation", "x")),
                 "edge_index": initial_state_tensordict.get(("agents", "observation", "edge_index")),
                 "graph_attributes": initial_state_tensordict.get(("agents", "observation", "graph_attributes")),
            }, batch_size=[self.num_envs], device=self.device), # Batch size for nested tensordict should be num_envs
            ("agents", "global_reward_in_state"): initial_state_tensordict.get(("agents", "global_reward_in_state")),
            # 'temp_reward': torch.zeros(self.num_envs, self.num_agents, 1, dtype=torch.float32, device=self.device), # Initialize temporary reward

            # Include per-agent done, terminated, and truncated in the initial tensordict under the 'agents' key
            ("agents", "terminated"): per_agent_terminated,
            ("agents", "truncated"): per_agent_truncated,
            ("agents", "done"): per_agent_done, # Use per-agent done here

            # env_batch is no longer a state key
        }, batch_size=self.batch_size, device=self.device)

        # Debugging print to check observation keys before returning
        print("\n--- Environment _reset output tensordict ---")
        print(f"Output tensordict keys: {initial_tensordict.keys(include_nested=True)}") # Added include_nested=True
        # Also print keys of nested observation tensordict
        if ("agents", "observation") in initial_tensordict.keys(include_nested=True): # Added include_nested=True
             print(f"  Nested ('agents', 'observation') keys: {initial_tensordict.get(('agents', 'observation')).keys(include_nested=True)}") # Added include_nested=True
             print(f"  Nested ('agents', 'observation') shape: {initial_tensordict.get(('agents', 'observation')).shape}")
        # Print done/terminated shapes
        if ('agents', 'done') in initial_tensordict.keys(include_nested=True):
             print(f"  ('agents', 'done') shape: {initial_tensordict.get(('agents', 'done')).shape}")
        if ('agents', 'terminated') in initial_tensordict.keys(include_nested=True):
             print(f"  ('agents', 'terminated') shape: {initial_tensordict.get(('agents', 'terminated')).shape}")
        print("-------------------------------------------")


        return initial_tensordict


    def _get_state_at(self, env_ids: torch.Tensor) -> TensorDict:
         if self.combined_data is None:
              raise RuntimeError("Combined data not loaded. Ensure _load_data is called.")

         num_envs_subset = len(env_ids)
         num_agents = self.num_agents
         num_nodes_per_graph = self.num_nodes_per_graph # This is num_agents
         node_feature_dim = self.node_feature_dim # This is 1
         num_edges_per_graph = self.num_edges_per_graph # Based on graph structure among agents
         graph_attr_dim = (26-13) + (39-26) # Assuming this is the graph attr dim

         state_data_index = self.current_data_index[env_ids]

         out_of_bounds_mask = (state_data_index < 0) | (state_data_index >= self.combined_data.shape[0])

         # Initialize tensors for batched data (batch size is num_envs_subset)
         # x: [num_envs_subset, num_nodes_per_graph, node_feature_dim]
         # edge_index: [num_envs_subset, 2, num_edges_per_graph]
         # graph_attributes: [num_envs_subset, graph_attr_dim]
         # global_reward_in_state: [num_envs_subset, num_agents, 1]

         x_batch = torch.zeros(num_envs_subset, num_nodes_per_graph, node_feature_dim, device=self.device)
         edge_index_batch = torch.zeros(num_envs_subset, 2, num_edges_per_graph, dtype=torch.int64, device=self.device)
         graph_attributes_batch = torch.zeros(num_envs_subset, graph_attr_dim, device=self.device)
         global_reward_in_state = torch.zeros(num_envs_subset, num_agents, 1, dtype=torch.float32, device=self.device)


         for i in range(num_envs_subset):
             env_idx_in_subset = i
             data_index_for_env = state_data_index[env_idx_in_subset].item()


             if out_of_bounds_mask[env_idx_in_subset]:
                 # If out of bounds, keep the corresponding slices in batches as zeros (initialized)
                 pass
             else:
                 try:
                     # Extract data for the current data_index
                     data_slice = self.combined_data[data_index_for_env] # Shape [39]

                     # Node features (x) - Shape [num_nodes_per_graph, node_feature_dim] = [num_agents, 1]
                     # Assuming the first 13 values correspond to the 13 agents' node features.
                     x = data_slice[0:13].unsqueeze(-1).to(self.device) # Shape [13, 1]
                     x_batch[env_idx_in_subset, :, :] = x # Assign directly


                     # Edge indices (edge_index) - Shape [2, num_edges_per_graph]
                     # Create a fully connected graph among num_agents nodes
                     # This generates all possible directed edges
                     # Number of edges = num_agents * (num_agents - 1)
                     if num_agents > 1:
                          senders, receivers = torch.meshgrid(torch.arange(num_agents, device=self.device), torch.arange(num_agents, device=self.device), indexing='ij') # Corrected self_device to self.device
                          edge_index = torch.stack([senders.flatten(), receivers.flatten()], dim=0)
                          # Remove self-loops if necessary
                          edge_index = edge_index[:, edge_index[0] != edge_index[1]]
                          # Ensure edge_index has the expected number of edges, even if there are no self-loops
                          # Pad if necessary (though for fully connected > 1 agent, num_edges_per_graph is fixed)
                          if edge_index.shape[1] < num_edges_per_graph:
                               # This case shouldn't happen for num_agents > 1 and fully connected
                               pass # Or raise error/warning if unexpected
                          edge_index_batch[env_idx_in_subset, :, :] = edge_index # Shape [2, num_agents * (num_agents - 1)]
                     else:
                          # Handle single agent case (no edges)
                          edge_index_batch[env_idx_in_subset, :, :] = torch.empty(2, 0, dtype=torch.int64, device=self.device)


                     # Graph attributes (graph_attributes) - Shape [graph_attr_dim]
                     graph_attributes_part1 = data_slice[13:26]
                     graph_attributes_part2 = data_slice[26:39]

                     # Ensure slices are not empty before concatenating
                     if graph_attributes_part1.numel() > 0 and graph_attributes_part2.numel() > 0:
                          graph_attributes = torch.cat([graph_attributes_part1, graph_attributes_part2], dim=0).to(self.device) # Shape [26]
                     elif graph_attributes_part1.numel() > 0:
                          graph_attributes = graph_attributes_part1.to(self.device)
                          # Pad if needed to match graph_attr_dim
                          if graph_attributes.shape[0] < graph_attr_dim:
                              graph_attributes = torch.cat([graph_attributes, torch.zeros(graph_attr_dim - graph_attributes.shape[0], device=self.device)])
                     elif graph_attributes_part2.numel() > 0:
                          graph_attributes = graph_attributes_part2.to(self.device)
                          # Pad if needed to match graph_attr_dim
                          if graph_attributes.shape[0] < graph_attr_dim:
                              graph_attributes = torch.cat([graph_attributes, torch.zeros(graph_attr_dim - graph_attributes.shape[0], device=self.device)])
                     else:
                          graph_attributes = torch.zeros(graph_attr_dim, device=self.device)

                     graph_attributes = graph_attributes.to(self.device) # Ensure on device
                     graph_attributes_batch[env_idx_in_subset, :] = graph_attributes # Assign directly


                     # Global reward in state
                     returns_for_state = data_slice[13:26] # Shape [13]
                     # Assuming global reward in state is the sum of returns for all agents (features)
                     global_reward_in_state[env_idx_in_subset, :, :] = returns_for_state.unsqueeze(-1) # Shape [num_agents, 1]


                 except IndexError:
                    print(f"Warning: Data index {data_index_for_env} out of bounds during state generation in _get_state_at. Using dummy data (zeros).")
                    # Keep the corresponding slices in batches as zeros (initialized)
                    pass


         # Create the state tensordict with observation keys at the root
         # Graph observation keys are now nested under ("agents", "observation")
         state_td = TensorDict(
              {
                  ("agents", "observation"): TensorDict({ # Nest under "agents" key
                       "x": x_batch,
                       "edge_index": edge_index_batch,
                       "graph_attributes": graph_attributes_batch,
                  }, batch_size=[num_envs_subset], device=self.device), # Batch size for nested tensordict should be num_envs_subset
                  ("agents", "global_reward_in_state"): global_reward_in_state, # Still agent-wise
              },
              # Use the batch size of the subset of environments
              batch_size=[num_envs_subset],
              device=self.device,
          )

         return state_td


    def _generate_graph_data_at_index(self, data_index: int):
        # This method is no longer directly used by _get_state_at,
        # but might be kept if other parts of the code still use it for single Data object generation.
        # For the current subtask, we are focusing on modifying _get_state_at.
        if self.combined_data is None:
            raise ValueError("Data has not been loaded. Ensure _load_data is called.")

        if data_index < 0 or data_index >= self.combined_data.size(0):
            raise IndexError(f"Data index {data_index} is out of bounds for combined_data with size {self.combined_data.size(0)}")


        # This method should also generate a single graph per index, with agents as nodes.
        num_agents = self.num_agents
        num_nodes = num_agents # Nodes are agents
        node_feature_dim = self.node_feature_dim # Node feature dimension
        graph_attr_dim = (26-13) + (39-26)

        data_slice = self.combined_data[data_index] # Shape [39]

        # Node features (x) - Shape [num_nodes, node_feature_dim] = [num_agents, 1]
        x = data_slice[0:13].unsqueeze(-1).to(self.device) # Shape [13, 1]


        # Edge indices (edge_index) - Shape [2, num_edges]
        # Create a fully connected graph among num_agents nodes
        if num_agents > 1:
             senders, receivers = torch.meshgrid(torch.arange(num_agents, device=self.device), torch.arange(num_agents, device=self.device), indexing='ij')
             edge_index = torch.stack([senders.flatten(), receivers.flatten()], dim=0)
             # Remove self-loops
             edge_index = edge_index[:, edge_index[0] != edge_index[1]] # Corrected edge variable name
        else:
             edge_index = torch.empty(2, 0, dtype=torch.int64, device=self.device)

        num_edges = edge_index.size(1)


        # Graph attributes (graph_attributes) - Shape [graph_attr_dim]
        graph_attributes_part1 = data_slice[13:26]
        graph_attributes_part2 = data_slice[26:39]

        if graph_attributes_part1.numel() > 0 and graph_attributes_part2.numel() > 0:
             graph_attributes = torch.cat([graph_attributes_part1, graph_attributes_part2], dim=0).to(self.device)
        elif graph_attributes_part1.numel() > 0:
             graph_attributes = graph_attributes_part1.to(self.device)
             if graph_attributes.shape[0] < graph_attr_dim:
                 graph_attributes = torch.cat([graph_attributes, torch.zeros(graph_attr_dim - graph_attributes.shape[0], device=self.device)])
        elif graph_attributes_part2.numel() > 0:
             graph_attributes = graph_attributes_part2.to(self.device)
             if graph_attributes.shape[0] < graph_attr_dim:
                 graph_attributes = torch.cat([graph_attributes, torch.zeros(graph_attr_dim - graph_attributes.shape[0], device=self.device)])
        else:
             graph_attributes = torch.zeros(graph_attr_dim, device=self.device)


        graph_attributes = graph_attributes.to(self.device) # Ensure on device

        data = Data(x=x, edge_index=edge_index, graph_attributes=graph_attributes)

        return data


    def _set_seed(self, seed: Optional[int] = None) -> int:
        if seed is None:
             seed = torch.randint(0, 1000000, (1,)).item()
        torch.manual_seed(seed)
        np.random.seed(seed)
        return seed

    def _is_terminal(self) -> torch.Tensor:
        # This method seems to represent global termination, not per-agent.
        # Based on the data structure, it's likely always False as done/terminated come from episode length.
        # Returning a tensor of False with batch size [num_envs]
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


    def _batch_reward(self, data_indices: torch.Tensor, tensordict: TensorDict) -> TensorDict: # Changed actions type hint to TensorDict
        # data_indices shape: [num_envs] or [num_envs, num_steps]
        # tensordict is the input tensordict to _step, expected to contain actions under ('agents', 'action')

        # Determine the flat batch size based on data_indices batch size
        if len(data_indices.shape) == 2:
            num_envs_current_batch = data_indices.shape[0]
            num_steps_current_batch = data_indices.shape[1]
            flat_batch_size = num_envs_current_batch * num_steps_current_batch
            # Flatten data_indices for combined access
            data_indices_flat = data_indices.view(flat_batch_size)
        elif len(data_indices.shape) == 1:
            num_envs_current_batch = data_indices.shape[0]
            num_steps_current_batch = 1
            flat_batch_size = num_envs_current_batch
            data_indices_flat = data_indices # Already flat
        else:
            raise ValueError(f"Unexpected data_indices batch size dimensions: {len(data_indices.shape)}")


        num_agents = self.num_agents
        num_action_features = self.num_individual_actions_features # Should be 13


        # Ensure data_indices are within bounds, handle out-of-bounds gracefully
        valid_mask = (data_indices_flat >= 0) & (data_indices_flat < self.combined_data.shape[0]) # Shape [flat_batch_size]

        # Initialize rewards tensor with shape [flat_batch_size, num_agents, 1]
        rewards_flat = torch.zeros(flat_batch_size, num_agents, 1, dtype=torch.float32, device=self.device)

        # Calculate rewards only for valid environments/timesteps
        valid_indices_flat = torch.where(valid_mask)[0]

        if valid_indices_flat.numel() > 0:
            # Select the relevant slices from rewards and other tensors using valid_indices_flat
            valid_data_indices_flat = data_indices_flat[valid_indices_flat] # Shape [num_valid_flat]

            # Get the returns for the valid data points
            returns_data_valid_flat = self.combined_data[valid_data_indices_flat][:, 13:26] # Shape [num_valid_flat, 13]

            # Extract actions from the input tensordict for valid data points and restructure them
            # The input tensordict has batch size matching data_indices.
            # View the input tensordict to have batch size [flat_batch_size]
            tensordict_flat = tensordict.view(flat_batch_size, *tensordict.shape[len(data_indices.shape):]) # View to [flat_batch_size, ...]

            # Slice the flattened tensordict using the valid indices
            tensordict_valid_flat = tensordict_flat[valid_indices_flat].contiguous()

            try:
                 # Get the action tensor from the sliced tensordict
                 # Assuming action is under ('agents', 'action') and is a tensor with shape [num_valid_flat, num_agents]
                 # The action structure is now nested under ('agents', 'agent_X', 'action_feature_Y')
                 # We need to extract the action for each agent and each feature.

                 # Create a tensor to hold the action for each agent [num_valid_flat, num_agents]
                 # Assuming each agent has a single discrete action derived from its 13 features
                 # Let's simplify for now and assume the action is a single value per agent.
                 # This might need adjustment based on the actual action space definition and how the policy outputs actions.

                 # Based on the action spec reconstruction, the action for agent i is at ('agents', 'agent_i', 'action_feature_0') etc.
                 # If we need a single action value per agent for reward calculation, we need to decide how to combine the 13 action features.
                 # For simplicity, let's use the first action feature of each agent as the representative action for reward calculation.
                 actions_for_reward = torch.stack([
                      tensordict_valid_flat.get(('agents', f'agent_{i}', 'action_feature_0')) # Use the first feature
                      for i in range(num_agents)
                 ], dim=1) # Shape [num_valid_flat, num_agents]


                 if actions_for_reward.dim() != 2 or actions_for_reward.shape[-1] != num_agents:
                      print(f"Warning: Unexpected action tensor shape for reward calculation: {actions_for_reward.shape}. Expected [num_valid_flat, num_agents]. Using zeros for rewards.")
                      # Keep the corresponding slices in rewards_flat as zeros (initialized)
                      pass # Skip reward calculation for this batch
                 else:
                      # Create masks based on the actions
                      down_mask_valid_flat = (actions_for_reward == 0) # Shape [num_valid_flat, num_agents]
                      up_mask_valid_flat = (actions_for_reward == 2)   # Shape [num_valid_flat, num_agents]
                      hold_mask_valid_flat = (actions_for_reward == 1) # Shape [num_valid_flat, num_agents]

                      # Calculate rewards based on action vs returns direction using masks
                      # returns_data_valid_flat shape: [num_valid_flat, 13]
                      # The rewards should be calculated for each agent based on their action and their corresponding return.
                      # For agent j at step i, the return is returns_data_valid_flat[i, j].
                      # The action is actions_for_reward[i, j].

                      # Reward for agent j at step i:
                      # If action is 0 (down): -returns_data_valid_flat[i, j]
                      # If action is 2 (up): +returns_data_valid_flat[i, j]
                      # If action is 1 (hold): -0.01 * torch.abs(returns_data_valid_flat[i, j])

                      # Vectorized calculation:
                      reward_down_valid_flat = -returns_data_valid_flat * down_mask_valid_flat.float() # Shape [num_valid_flat, num_agents]
                      reward_up_valid_flat = returns_data_valid_flat * up_mask_valid_flat.float()     # Shape [num_valid_flat, num_agents]
                      reward_hold_valid_flat = -0.01 * torch.abs(returns_data_valid_flat) * hold_mask_valid_flat.float() # Shape [num_valid_flat, num_agents]

                      # Sum the rewards for each agent
                      agent_rewards_valid_flat = reward_down_valid_flat + reward_up_valid_flat + reward_hold_valid_flat # Shape [num_valid_flat, num_agents]

                      # Add the last dimension to match the expected output shape [flat_batch_size, num_agents, 1]
                      agent_rewards_valid_flat = agent_rewards_valid_flat.unsqueeze(-1) # Shape [num_valid_flat, num_agents, 1]

                      # Assign calculated rewards to the selected slice of the rewards_flat tensor
                      rewards_flat[valid_indices_flat] = agent_rewards_valid_flat # Assign to the slice

            except KeyError as e:
                 print(f"Error: Action key not found in the input tensordict to _batch_reward: {e}. Ensure action is nested correctly.")
                 # Keep the corresponding slices in rewards_flat as zeros (initialized)
                 pass
            except Exception as e:
                 print(f"An error occurred during reward calculation in _batch_reward: {e}")
                 # Keep the corresponding slices in rewards_flat as zeros (initialized)
                 pass


        # Reshape rewards_flat back to the original input batch size [num_envs, num_steps, num_agents, 1]
        # If data_indices has shape [num_envs], rewards_reshaped is [num_envs, num_agents, 1].
        # If data_indices has shape [num_envs, num_steps], rewards_reshaped is [num_envs, num_steps, num_agents, 1].
        if len(data_indices.shape) == 2:
             rewards_reshaped = rewards_flat.view(*data_indices.shape, num_agents, 1)
        else:
             rewards_reshaped = rewards_flat.view(*data_indices.shape, num_agents, 1) # Ensure correct shape even if num_steps is 1


        # Return rewards wrapped in a TensorDict with the expected key and original input batch size
        # The batch size of the output TensorDict should match the batch size of data_indices
        # Reward is nested under ('agents', 'reward')
        return TensorDict({("agents", "reward"): rewards_reshaped}, batch_size=data_indices.shape, device=self.device) # Use data_indices.shape for batch size

# Define variables before creating the environment instance
num_envs = 4  # Example value
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
seed = 42 # Example seed
episode_length = 64 # Example value
allow_repeat_data = False # Example value

# Instantiate the environment
base_env = AnFuelpriceEnv(num_envs=num_envs, device=device, seed=seed, episode_length=episode_length, allow_repeat_data=allow_repeat_data)
