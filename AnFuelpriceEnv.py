# MultiCategorical Configuration

import pandas as pd
import os

# Find the part of the code that loads the data.
# It might look something like this:
# data_path = "some_url_or_path_here"
# data = pd.read_csv(data_path)

# Replace the data loading logic with this:





# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.utils import check_env_specs # Keep check_env_specs

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
        self.num_nodes_per_graph = 13 # Initialize here

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
            # Use self.num_nodes_per_graph which is now an attribute of FuelpriceenvfeatureGraph
            self.obs_min = torch.min(self.combined_data[:, :self.num_nodes_per_graph], dim=0)[0].unsqueeze(-1).to(self.device)
            self.obs_max = torch.max(self.combined_data[:, :self.num_nodes_per_graph], dim=0)[0].unsqueeze(-1).to(self.device)


        except Exception as e:
             print(f"An error occurred during data loading: {e}")
             self.combined_data = None # Set to None to indicate failure
             raise # Re-raise the exception to stop execution


class AnFuelpriceEnv(EnvBase):
    def __init__(self, num_envs, device, seed, **kwargs):
        self.episode_length = kwargs.get('episode_length', 100)
        self.num_agents = 13 # Keep as 13 based on original definition and feature count
        self.allow_repeat_data = kwargs.get('allow_repeat_data', False)
        self.num_envs = num_envs
        self.current_data_index = torch.zeros(num_envs, dtype=torch.int64, device=device)

        self.graph_generator = FuelpriceenvfeatureGraph()

        self.device = device

        # Pass num_agents to graph_generator BEFORE loading data
        self.graph_generator.num_nodes_per_graph = self.num_agents

        self.graph_generator.device = self.device
        self.graph_generator.allow_repeat_data = self.allow_generator.allow_repeat_data
        
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
        # Define the state_spec to match the structure of the tensordict
        # that will be collected by a collector after a step. This tensordict
        # will contain the current state, action, reward, done, and the *next* state.
        self.state_spec = Composite(
             {
                 # Define the keys for the current state
                 ("agents", "observation"): Composite({ # Nested under "agents"
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
                 # Removed global_reward_in_state from state_spec
                 # Added top-level done, terminated, truncated keys for the current state
                 "done": Categorical(
                      n=2,
                      shape=torch.Size([self.num_envs, 1]),
                      dtype=torch.bool,
                      device=self.device),
                 "terminated": Categorical(
                      n=2,
                      shape=torch.Size([self.num_envs, 1]),
                      dtype=torch.bool,
                      device=self.device),
                 "truncated": Categorical(
                      n=2,
                      shape=torch.Size([self.num_envs, 1]),
                      dtype=torch.bool,
                      device=self.device),
                 # Define the keys for the next state, nested under "next"
                 "next": Composite({
                      ("agents", "observation"): Composite({ # Nested under "agents"
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
                      # Removed global_reward_in_state from next state spec
                      # Also include reward key under ('agents',) under 'next'
                      ('agents', 'reward'): Unbounded(shape=torch.Size([self.num_envs, self.num_agents, 1]), dtype=torch.float32, device=self.device),
                       # done, terminated, truncated keys under ('agents',) under 'next'
                       ("agents", "terminated"): Categorical(
                            n=2,
                            shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1] as done/terminated/truncated are per env
                            dtype=torch.bool,
                            device=self.device),
                       ("agents", "truncated"):  Categorical(
                            n=2,
                            shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1]
                             dtype=torch.bool,
                             device=self.device),
                       ("agents", "done"):  Categorical(
                            n=2,
                            shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1]
                             dtype=torch.bool,
                             device=self.device),
                     # Add top-level done, terminated, truncated keys to the "next" composite
                     "done": Categorical(
                          n=2,
                          shape=torch.Size([self.num_envs, 1]),
                          dtype=torch.bool,
                          device=self.device),
                     "terminated": Categorical(
                          n=2,
                          shape=torch.Size([self.num_envs, 1]),
                          dtype=torch.bool,
                          device=self.device),
                     "truncated": Categorical(
                          n=2,
                          shape=torch.Size([self.num_envs, 1]),
                          dtype=torch.bool,
                          device=self.device),
                 }),
             },
             # The batch size of the state spec is the number of environments
             batch_size=self.batch_size,
             device=self.device,
         )
        print(f"State specification defined with single graph per env structure and batch shape {self.state_spec.shape}.")


        # Corrected nvec to be a 1D tensor of size num_individual_actions_features with value num_individual_actions
        # This tensor defines the number of categories for each of the individual action features (13 features, each with 3 categories).
        # Ensure nvec is correctly shaped [self.num_individual_actions_features]
        # nvec should be a tensor of shape [self.num_individual_actions_features] where each element is self.num_individual_actions (3).
        # Explicitly creating a list and then a tensor to ensure 1D shape.
        nvec_list = [self.num_individual_actions] * self.num_individual_actions_features
        nvec_tensor = torch.tensor(nvec_list, dtype=torch.int64, device=self.device)


        # Corrected agent_action_spec shape to be for a single agent, only defining the action features.
        # The shape here should be just the shape of the action for a single agent, which is a tensor of size [num_individual_actions_features].
        # The MultiCategorical action space for a single agent should have shape [num_individual_actions_features]
        agent_action_spec = MultiCategorical(nvec=nvec_tensor, # Pass the correctly shaped nvec_tensor: [13]
                # The shape argument here defines the shape of the *unbatched* action tensor for a single agent.
                # It should match the shape of nvec if nvec defines categories for each element in the action tensor.
                shape=torch.Size([self.num_individual_actions_features,]), # Shape for a single agent's action: [13]
                dtype=torch.int64,
                device=self.device
            )


        # The batched action spec for the environment should have batch_size=[num_envs]
        # This is automatically derived by TensorDict from the unbatched spec and env batch size.
        # We can access it via self.action_spec
        print("\nUnbatched Multi-Agent Action specification defined using nested Composites and Categorical.")
        # The unbatched action spec should be for a single environment, but include the agents dimension.
        # The shape here should reflect the actions for all agents in a single environment: [num_agents, num_individual_actions_features].
        # The MultiCategorical needs to handle actions for all agents in an unbatched env.
        # The nvec for the unbatched env MultiCategorical should be the nvec_tensor repeated for each agent.
        # Reshape the repeated nvec tensor to [num_agents, num_individual_actions_features]
        nvec_unbatched = nvec_tensor.repeat(self.num_agents).view(self.num_agents, self.num_individual_actions_features)

        self.action_spec_unbatched = Composite(
              {("agents","action"): MultiCategorical(nvec=nvec_unbatched, # Use the correctly shaped nvec: [num_agents, num_individual_actions_features]
                                                      # The shape here defines the shape of the action tensor for a single environment: [num_agents, num_individual_actions_features]
                                                      shape=torch.Size([self.num_agents, self.num_individual_actions_features]),
                                                      dtype=torch.int64,
                                                      device=self.device)},
              batch_size=[], # Unbatched environment has no batch size at the root
              device=self.device
            )

        print(f"Unbatched Environment action_spec: {self.action_spec_unbatched}")
        print(f"Batched Environment action_spec: {self.action_spec}")


        # Restored original reward spec
        self.reward_spec = Composite(
             {('agents', 'reward'): Unbounded(shape=torch.Size([self.num_envs, self.num_agents, 1]), dtype=torch.float32, device=self.device)},
             batch_size=[self.num_envs],
             device=self.device,
        )
        print(f"Agent-wise Reward specification defined with batch shape {self.reward_spec.shape}.")

        # Define the done_spec to match the structure of the done keys
        # returned by _step.
        # Moved done, terminated, truncated to the top level of done_spec
        self.done_spec = Composite(
            {
                ("agents", "done"):  Categorical(
                      n=2,
                      shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1]
                      dtype=torch.bool,
                      device=self.device),

                ("agents", "terminated"): Categorical(
                      n=2,
                      shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1]
                      dtype=torch.bool,
                      device=self.device),
                ("agents", "truncated"):  Categorical(
                     n=2,
                     shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1]
                      dtype=torch.bool,
                      device=self.device),
                # Add top-level done, terminated, truncated keys to done_spec
                "done": Categorical(
                     n=2,
                     shape=torch.Size([self.num_envs, 1]),
                     dtype=torch.bool,
                     device=self.device),
                "terminated": Categorical(
                     n=2,
                     shape=torch.Size([self.num_envs, 1]),
                     dtype=torch.bool,
                     device=self.device),
                "truncated": Categorical(
                     n=2,
                     shape=torch.Size([self.num_envs, 1]),
                     dtype=torch.bool,
                     device=self.device),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )
        print(f"Done specification defined with batch shape {self.done_spec.shape}.")

        self.state_spec.unlock_(recurse=True)
        self.action_spec.unlock_(recurse=True) # Keep action_spec unlocked as it is batched
        self.reward_spec.unlock_(recurse=True)




    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.current_data_index += 1

        terminated = self._is_terminal()
        truncated = (self.current_data_index >= self.episode_length)

        # The action is now expected to be in the input tensordict passed to step() by the collector
        # It should be at tensordict[('agents', 'action')]
        actions=tensordict[('agents', 'action')]
        # Pass the input tensordict directly to _batch_reward
        # _batch_reward will need to extract the action from this tensordict
        reward_td = self._batch_reward(self.current_data_index, tensordict) # Pass the entire input tensordict


        # Call _get_state_at with the current env indices to get the next state
        next_state_tensordict = self._get_state_at(torch.arange(self.num_envs, device=self.device))

        # Update the next_state_tensordict with reward and done flags
        # These keys should be at the root level of the tensordict returned by _step
        next_state_tensordict.set(('agents', 'reward'), reward_td.get(('agents', 'reward')))
        # Set done, terminated, truncated at the top level AND under ('agents',)
        next_state_tensordict.set("terminated", terminated.unsqueeze(-1))
        next_state_tensordict.set("truncated", truncated.unsqueeze(-1))
        next_state_tensordict.set("done", terminated.unsqueeze(-1) | truncated.unsqueeze(-1))
        next_state_tensordict.set(("agents", "terminated"), terminated.unsqueeze(-1))
        next_state_tensordict.set(("agents", "truncated"), truncated.unsqueeze(-1))
        next_state_tensordict.set(("agents", "done"), terminated.unsqueeze(-1) | truncated.unsqueeze(-1))


        # Debugging print to check observation keys and reward before returning
        print("\n--- Environment _step output tensordict (before return) ---")
        print(f"Output tensordict keys: {next_state_tensordict.keys(include_nested=True)}") # Added include_nested=True
        print(f"Output tensordict shape: {next_state_tensordict.shape}")
        # Also print keys and shapes of nested observation tensordict
        if ("agents", "observation") in next_state_tensordict.keys(include_nested=True): # Added include_nested=True
             print(f"  Nested ('agents', 'observation') keys: {next_state_tensordict.get(('agents', 'observation')).keys(include_nested=True)}") # Added include_nested=True
             print(f"  Nested ('agents', 'observation') shape: {next_state_tensordict.get(('agents', 'observation')).shape}")
        # Print the reward value
        if ('agents', 'reward') in next_state_tensordict.keys(include_nested=True):
             print(f"  Reward value in output tensordict: {next_state_tensordict.get(('agents', 'reward'))}")
        else:
             print("  Reward key ('agents', 'reward') not found in output tensordict.")
        # Print done keys to verify their location
        if ("done") in next_state_tensordict.keys():
             print(f"  ('done') value: {next_state_tensordict.get(('done'))}")
        if ("terminated") in next_state_tensordict.keys():
             print(f"  ('terminated') value: {next_state_tensordict.get(('terminated'))}")
        if ("truncated") in next_state_tensordict.keys():
             print(f"  ('truncated') value: {next_state_tensordict.get(('truncated'))}")
        if ("agents", "done") in next_state_tensordict.keys(include_nested=True):
             print(f"  ('agents', 'done') value: {next_state_tensordict.get(('agents', 'done'))}")
        if ("agents", "terminated") in next_state_tensordict.keys(include_nested=True):
             print(f"  ('agents', 'terminated') value: {next_state_tensordict.get(('agents', 'terminated'))}")
        if ("agents", "truncated") in next_state_tensordict.keys(include_nested=True):
             print(f"  ('agents', 'truncated') value: {next_state_tensordict.get(('agents', 'truncated'))}")

        print("-------------------------------------------")




        # Return the next_state_tensordict with reward and done flags included
        # torchrl will automatically put this under the "next" key when collected
        return next_state_tensordict





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

        # Call _get_state_at with the current env indices to get the initial state
        initial_state_tensordict = self._get_state_at(torch.arange(self.num_envs, device=self.device))

        # The initial tensordict should contain the initial state keys at the root.
        # It should also include the done flags initialized to False.
        initial_tensordict = initial_state_tensordict.clone() # Start with the initial state structure
        # Set done, terminated, truncated at the top level AND under ('agents',)
        initial_tensordict.set("terminated", torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device))
        initial_tensordict.set("truncated", torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device))
        initial_tensordict.set("done", torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device))
        initial_tensordict.set(("agents", "terminated"), torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device))
        initial_tensordict.set(("agents", "truncated"), torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device))
        initial_tensordict.set(("agents", "done"), torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device))
        # 'temp_reward' is not needed in the initial tensordict


        # Debugging print to check observation keys before returning
        # print("\n--- Environment _reset output tensordict ---")
        # print(f"Output tensordict keys: {initial_tensordict.keys(include_nested=True)}") # Added include_nested=True
        # # Also print keys of nested observation tensordict
        # if ("agents", "observation") in initial_tensordict.keys(include_nested=True): # Added include_nested=True
        #      print(f"  Nested ('agents', 'observation') keys: {initial_tensordict.get(('agents', 'observation')).keys(include_nested=True)}") # Added include_nested=True
        # print(f"  Nested ('agents', 'observation') shape: {initial_tensordict.get(('agents', 'observation')).shape}")
        # print("-------------------------------------------")


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
         # Removed initialization of global_reward_in_state


         for i in range(num_envs_subset):
             env_idx_in_subset = i
             data_index_for_env = state_data_index[env_idx_in_subset]


             if out_of_bounds_mask[env_idx_in_subset]:
                 # If out of bounds, keep the corresponding slices in batches as zeros (initialized)
                 pass
             else:
                try:
                     # Extract data for the current data_index
                     data_slice = self.combined_data[data_index_for_env.item()] # Shape [39]

                     # Node features (x) - Shape [num_nodes_per_graph, node_feature_dim] = [num_agents, 1]
                     # Assuming the first 13 values correspond to the 13 agents' node features.
                     x = data_slice[0:13].unsqueeze(-1).to(self.device) # Shape [13, 1]
                     x_batch[env_idx_in_subset, :, :] = x # Assign directly


                     # Edge indices (edge_index) - Shape [2, num_edges_per_graph]
                     # Create a fully connected graph among num_agents nodes
                     # This generates all possible directed edges
                     # Number of edges = num_agents * (num_agents - 1)
                     if num_agents > 1:
                          senders, receivers = torch.meshgrid(torch.arange(num_agents, device=self.device), torch.arange(num_agents, device=self.device), indexing='ij')
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
                          graph_attributes = torch.cat([graph_attributes_part1, graph_attributes_part2], dim=0).to(self.device)
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


                except Exception as e:
                     print(f"An error occurred during state data extraction for env {env_ids[i].item()} at data index {data_index_for_env.item()}: {e}")
                     # Keep the corresponding slices in batches as zeros (initialized)
                     pass # Continue to the next environment

         # Construct the output TensorDict for state at the current data index
         state_tensordict = TensorDict({
             ("agents", "observation"): TensorDict({
                  "x": x_batch,
                  "edge_index": edge_index_batch,
                  "graph_attributes": graph_attributes_batch,
             }, batch_size=[num_envs_subset], device=self.device),
             # Removed global_reward_in_state from state tensordict construction
             # Added top-level done, terminated, truncated keys
             "done": torch.zeros(num_envs_subset, 1, dtype=torch.bool, device=self.device),
             "terminated": torch.zeros(num_envs_subset, 1, dtype=torch.bool, device=self.device),
             "truncated": torch.zeros(num_envs_subset, 1, dtype=torch.bool, device=self.device),
             # Added done, terminated, truncated keys under ('agents',) as well
             ("agents", "done"): torch.zeros(num_envs_subset, 1, dtype=torch.bool, device=self.device),
             ("agents", "terminated"): torch.zeros(num_envs_subset, 1, dtype=torch.bool, device=self.device),
             ("agents", "truncated"): torch.zeros(num_envs_subset, 1, dtype=torch.bool, device=self.device),
         }, batch_size=[num_envs_subset], device=self.device)


         # Debugging print to check observation keys before returning
         print("\n--- Environment _get_state_at output tensordict ---")
         print(f"Output tensordict keys: {state_tensordict.keys(include_nested=True)}") # Added include_nested=True
         # Also print keys of nested observation tensordict
         if ("agents", "observation") in state_tensordict.keys(include_nested=True): # Added include_nested=True
              print(f"  Nested ('agents', 'observation') keys: {state_tensordict.get(('agents', 'observation')).keys(include_nested=True)}") # Added include_nested=True
         print(f"  Nested ('agents', 'observation') shape: {state_tensordict.get(('agents', 'observation')).shape}")
         # Print done keys to verify their location
         if ("done") in state_tensordict.keys():
             print(f"  ('done') value: {state_tensordict.get(('done'))}")
         if ("terminated") in state_tensordict.keys():
             print(f"  ('terminated') value: {state_tensordict.get(('terminated'))}")
         if ("truncated") in state_tensordict.keys():
             print(f"  ('truncated') value: {state_tensordict.get(('truncated'))}")
         if ("agents", "done") in state_tensordict.keys(include_nested=True):
             print(f"  ('agents', 'done') value: {state_tensordict.get(('agents', 'done'))}")
         if ("agents", "terminated") in state_tensordict.keys(include_nested=True):
             print(f"  ('agents', 'terminated') value: {state_tensordict.get(('agents', 'terminated'))}")
         if ("agents", "truncated") in state_tensordict.keys(include_nested=True):
             print(f"  ('agents', 'truncated') value: {state_tensordict.get(('agents', 'truncated'))}")

         print("-------------------------------------------")


         return state_tensordict


    def _set_seed(self, seed: Optional[int] = None) -> int:
        if seed is None:
             seed = torch.randint(0, 1000000, (1,)).item()
        torch.manual_seed(seed)
        np.random.seed(seed)
        return seed

    def _is_terminal(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


    def _batch_reward(self, data_indices: torch.Tensor, tensordict: TensorDict) -> TensorDict: # Changed actions type hint to TensorDict
        # data_indices shape: [num_envs] or [num_envs, num_steps]
        # tensordict is the input tensordict to _step, expected to contain actions under ('agents', 'action')

        # Determine the flat batch size based on data_indices batch size
        original_batch_shape = data_indices.shape
        if len(original_batch_shape) == 2:
            num_envs_current_batch = original_batch_shape[0]
            num_steps_current_batch = original_batch_shape[1]
            flat_batch_size = num_envs_current_batch * num_steps_current_batch
            # Flatten data_indices for combined access
            data_indices_flat = data_indices.view(flat_batch_size)
        elif len(original_batch_shape) == 1:
            num_envs_current_batch = original_batch_shape[0]
            num_steps_current_batch = 1
            flat_batch_size = num_envs_current_batch
            data_indices_flat = data_indices # Already flat
        else:
            raise ValueError(f"Unexpected data_indices batch size dimensions: {len(original_batch_shape)}")


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
            # The input tensordict's batch size will match the batch size of data_indices,
            # which can be [num_envs] for a single step or [num_envs, num_steps] for a rollout.
            # We need to flatten the input tensordict to match the flat_batch_size of data_indices_flat.

            # View the input tensordict to have batch size [flat_batch_size]
            # The total number of elements in the batch dimensions of the input tensordict
            # should match flat_batch_size.
            # We need to figure out the remaining dimensions after flattening the batch.
            # The action should have shape [num_envs, ..., num_agents, num_action_features]
            # When passed to _step (and _batch_reward), the batch shape could be [num_envs] or [num_envs, num_steps].
            # The action tensor inside the input tensordict to _step will have batch shape [num_envs]
            # and action tensor shape [num_agents, num_action_features].
            # The action tensor inside the tensordict passed to _batch_reward (which is the same tensordict from _step)
            # will have batch shape [num_envs] and action tensor shape [num_envs, num_steps, num_agents, num_action_features].

            # Let's check the shape of the action tensor in the input tensordict directly.
            # The input tensordict has batch_size matching the original batch_shape of data_indices.
            # The action key is ('agents', 'action').
            actions_tensor_input = tensordict.get(("agents", "action"))
            print(f"Debug: Input tensordict batch shape: {tensordict.batch_size}")
            print(f"Debug: Action tensor shape in input tensordict: {actions_tensor_input.shape}")

            # Now flatten the actions tensor to match the flat_batch_size of data_indices_flat
            # The flattened shape should be [flat_batch_size, num_agents, num_action_features]
            expected_action_flat_shape = torch.Size([flat_batch_size, num_agents, num_action_features])
            if actions_tensor_input.shape == expected_action_flat_shape:
                 actions_tensor_flat = actions_tensor_input # Already flat (batch size was flat_batch_size)
            elif actions_tensor_input.shape[:len(original_batch_shape)] == original_batch_shape:
                 # The action tensor has the original batch shape and then the agent/action dimensions.
                 # Flatten the batch dimensions.
                 actions_tensor_flat = actions_tensor_input.view(flat_batch_size, num_agents, num_action_features)
            else:
                 print(f"Error: Unexpected action tensor shape in input tensordict: {actions_tensor_input.shape}. Expected batch_shape + [{num_agents}, {num_action_features}]. Using zeros for rewards.")
                 # Keep the corresponding slices in rewards_flat as zeros (initialized)
                 valid_indices_flat = torch.tensor([], dtype=torch.int64, device=self.device) # Treat all as invalid to skip calculation
                 actions_tensor_flat = None # Set to None to prevent use


            if valid_indices_flat.numel() > 0 and actions_tensor_flat is not None:
                try:
                    # Select the relevant slices from the flattened actions tensor using valid_indices_flat
                    actions_tensor_valid_flat = actions_tensor_flat[valid_indices_flat].contiguous() # Shape [num_valid_flat, num_agents, num_action_features]
                    print(f"Debug: Action tensor shape after flattening and slicing: {actions_tensor_valid_flat.shape}")


                    # The action tensor now has shape [num_valid_flat, num_agents, num_action_features].
                    # We need to calculate the reward for each agent based on their specific action features.
                    # The returns_data_valid_flat has shape [num_valid_flat, 13] (returns for 13 agents).
                    # We need to compare the action features for each agent against their corresponding return.

                    # Let's assume for simplicity that the first action feature (index 0) determines the 'down', 'hold', 'up' action.
                    # If your action features have a different meaning, this logic needs adjustment.
                    # Assuming actions_tensor_valid_flat[:, :, 0] is the 'down' (0), 'hold' (1), 'up' (2) action for each agent.
                    # This assumes the MultiCategorical output [num_agents, 13] means each of the 13 values is an independent action (0, 1, or 2) for that agent.
                    # If the 13 values define a single complex action per agent, the reward logic needs to be more complex.

                    # Based on the MultiCategorical definition (nvec with 13 elements, each 3 categories),
                    # it seems each agent has 13 independent categorical actions with 3 choices each.
                    # The reward should then be a function of the returns for each agent and all 13 of their chosen action features.
                    # This requires a more complex reward calculation than the simple 'down/hold/up' based on returns.

                    # Let's revisit the action space definition.
                    # self.num_individual_actions_features = 13
                    # nvec_tensor = torch.tensor([self.num_individual_actions] * self.num_individual_actions_features, ...)
                    # agent_action_spec = MultiCategorical(nvec=nvec_tensor, shape=torch.Size([self.num_individual_actions_features,]), ...)
                    # self.action_spec_unbatched = Composite({("agents","action"): MultiCategorical(nvec=nvec_unbatched, shape=torch.Size([self.num_agents, self.num_individual_actions_features])), ...})
                    # This means for an unbatched env, the action is a TensorDict with key ('agents', 'action') holding a tensor of shape [num_agents, 13] where each element is an integer from 0 to 2.

                    # So, actions_tensor_valid_flat shape should be [num_valid_flat, num_agents, num_action_features] = [num_valid_flat, 13, 13].
                    # The reward calculation needs to use the returns_data_valid_flat [num_valid_flat, 13] and actions_tensor_valid_flat [num_valid_flat, 13, 13].

                    # Let's assume for the reward calculation that each of the 13 action features for an agent corresponds to
                    # whether they are taking a 'down', 'hold', or 'up' stance with respect to one of the 13 return features.
                    # This is a possible interpretation, but the exact reward logic depends on the intended meaning of the 13 action features.

                    # For simplicity, let's calculate a reward for each agent by comparing *each* of their 13 action features
                    # to the corresponding 13 return values for that step.

                    # returns_data_valid_flat shape: [num_valid_flat, 13] (returns for 13 agents)
                    # actions_tensor_valid_flat shape: [num_valid_flat, 13, 13] (actions for 13 agents, each with 13 features)

                    # We need to compare actions_tensor_valid_flat[:, j, i] with returns_data_valid_flat[:, i]
                    # for each agent j (0 to 12) and each action feature/return i (0 to 12).

                    # Reshape returns_data_valid_flat for broadcasting: [num_valid_flat, 1, 13]
                    returns_broadcastable = returns_data_valid_flat.unsqueeze(1)

                    # Create masks based on the actions [num_valid_flat, num_agents, num_action_features]
                    down_mask_valid_flat = (actions_tensor_valid_flat == 0)
                    up_mask_valid_flat = (actions_tensor_valid_flat == 2)
                    hold_mask_valid_flat = (actions_tensor_valid_flat == 1)

                    # Calculate rewards for each action feature comparison [num_valid_flat, num_agents, num_action_features]
                    # If action feature i for agent j is 'down' (0), reward contribution is -returns_data_valid_flat[:, i]
                    # If action feature i for agent j is 'up' (2), reward contribution is +returns_data_valid_flat[:, i]
                    # If action feature i for agent j is 'hold' (1), reward contribution is -0.01 * torch.abs(returns_data_valid_flat[:, i])

                    # We need to broadcast returns_data_valid_flat [num_valid_flat, 13] to [num_valid_flat, num_agents, 13]
                    # by repeating along the agent dimension.
                    returns_repeated = returns_data_valid_flat.unsqueeze(1).repeat(1, num_agents, 1) # Shape [num_valid_flat, num_agents, 13]


                    reward_contributions_down = -returns_repeated * down_mask_valid_flat.float()
                    reward_contributions_up = returns_repeated * up_mask_valid_flat.float()
                    reward_contributions_hold = -0.01 * torch.abs(returns_repeated) * hold_mask_valid_flat.float()

                    # Sum the reward contributions across the action features dimension for each agent
                    agent_rewards_valid_flat = (reward_contributions_down + reward_contributions_up + reward_contributions_hold).sum(dim=-1) # Shape [num_valid_flat, num_agents]


                    # Add the last dimension to match the expected output shape [flat_batch_size, num_agents, 1]
                    agent_rewards_valid_flat = agent_rewards_valid_flat.unsqueeze(-1) # Shape [num_valid_flat, num_agents, 1]

                    # Assign calculated rewards to the selected slice of the rewards_flat tensor
                    rewards_flat[valid_indices_flat] = agent_rewards_valid_flat # Assign to the slice


                except KeyError:
                     print("Error: Action key ('agents', 'action') not found in the input tensordict to _batch_reward.")
                     # Keep the corresponding slices in rewards_flat as zeros (initialized)
                     pass
                except Exception as e:
                     print(f"An error occurred during reward calculation in _batch_reward: {e}")
                     # Keep the corresponding slices in rewards_flat as zeros (initialized)
                     pass


        # Reshape rewards_flat back to the original input batch size [num_envs, num_steps, num_agents, 1] or [num_envs, num_agents, 1]
        # The output TensorDict batch size should match the original batch_shape of data_indices.
        rewards_reshaped = rewards_flat.view(*original_batch_shape, num_agents, 1)


        # Return rewards wrapped in a TensorDict with the expected key and original input batch size
        # The batch size of the output TensorDict should match the batch size of data_indices
        return TensorDict({("agents", "reward"): rewards_reshaped}, batch_size=original_batch_shape, device=self.device) # Use original_batch_shape for batch size
