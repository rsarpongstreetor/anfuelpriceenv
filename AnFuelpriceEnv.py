


import pandas as pd
import os
import networkx as nx
import random
from torch_geometric.nn import GCNConv, global_mean_pool # Import global_mean_pool
from torch_geometric.nn import SplineConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
# Import MultiCategorical specification class from torchrl.data
from torchrl.data import Unbounded, Categorical, Composite, DiscreteTensorSpec, MultiCategorical as MultiCategoricalSpec # Import Composite, DiscreteTensorSpec, and rename MultiCategorical

from typing import Optional
import numpy as np
import torch # Import torch
from tensordict.nn import TensorDictModuleBase # Import TensorDictModuleBase
from torchrl.envs.utils import check_env_specs # Keep check_env_specs

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
            self.obs_min = torch.min(self.combined_data[:, :self.num_nodes_per_graph], dim=0)[0].unsqueeze(-1).to(self.device) # Corrected slicing
            self.obs_max = torch.max(self.combined_data[:, :self.num_nodes_per_graph], dim=0)[0].unsqueeze(-1).to(self.device) # Corrected slicing


        except Exception as e:
             print(f"An error occurred during data loading: {e}")
             self.combined_data = None # Set to None to indicate failure
             raise # Re-raise the exception to stop execution


class AnFuelpriceEnv(EnvBase):
    def __init__(self, num_envs, seed, device, num_agents=13, **kwargs): # Added num_agents argument with default
        self.episode_length = kwargs.get('episode_length', 100)
        self.num_agents = num_agents # Use the provided num_agents
        self.allow_repeat_data = kwargs.get('allow_repeat_data', False)
        self.num_envs = num_envs
        self.current_data_index = torch.zeros(num_envs, dtype=torch.int64, device=device)

        self.graph_generator = FuelpriceenvfeatureGraph()

        self.device = device

        # Pass num_agents to graph_generator BEFORE loading data
        self.graph_generator.num_nodes_per_graph = self.num_agents

        self.graph_generator.device = self.device
        self.graph_generator.allow_repeat_data = self.allow_repeat_data

        self.graph_generator._load_data()
        self.combined_data = self.graph_generator.combined_data

        # num_agents is now initialized from the constructor argument
        # self.num_agents = num_agents # Removed redundant initialization
        self.num_individual_actions = 3 # Defined inside the class
        self.num_individual_actions_features = 13 # Still 13 action features per agent


        # In the new single-graph-per-environment structure, num_nodes_per_graph is num_agents
        self.num_nodes_per_graph = self.num_agents
        # Re-calculate num_edges_per_graph based on a potential graph structure among agents.
        # Assuming a fully connected graph among agents.
        self.num_edges_per_graph = self.num_agents * (self.num_agents - 1) if self.num_agents > 1 else 0
        print(f"AnFuelpriceEnv.__init__: num_edges_per_graph calculated as {self.num_edges_per_graph}")

        # Define a simple, fixed edge index for a single graph (e.g., a ring)
        # This is for debugging the policy's graph processing
        if self.num_agents > 1:
            # Create a ring graph: 0->1, 1->2, ..., 12->0
            sources = torch.arange(self.num_agents)
            targets = (torch.arange(self.num_agents) + 1) % self.num_agents
            self._fixed_edge_index_single = torch.stack([sources, targets], dim=0).to(torch.long)
            self._fixed_num_edges_single = self._fixed_edge_index_single.shape[1]
        else:
            self._fixed_edge_index_single = torch.empty(2, 0, dtype=torch.long)
            self._fixed_num_edges_single = 0


        # Node feature dimension needs to be defined
        # If each node (agent) has a feature vector, define its dimension here.
        # Based on the data loading, the first 13 columns are features.
        # If each agent's node features are these 13 values, then node_feature_dim is 13.
        # Assuming node feature dim is 1, as in the original code, but this might need revisit
        # if each agent node represents one of the 13 features.
        # Let's keep it as 1 for now, assuming each node feature is a single value from the 13 columns.
        # Reverted node_feature_dim to 1
        self.node_feature_dim = 1

        # Define obs_dim before using it
        # obs_dim is the dimension of the observation space for a single agent, or the relevant feature dimensions.
        # Given the structure, the first 13 columns seem to be the node features.
        # obs_dim = 13 # This is the number of node features per graph (which is num_agents)
        # The observation spec should now reflect the variable number of agents.
        # The shape of 'x' should be [num_envs, num_agents, node_feature_dim].

        # Re-calculate observation bounds based on the new node feature slicing
        # The observation bounds should be for the node features, which are the first 13 columns.
        # These bounds are based on the full dataset, assuming the first 13 columns are the relevant features regardless of the number of agents.
        self.obs_min = torch.min(self.combined_data[:, :13], dim=0)[0].unsqueeze(-1).to(self.device) # Still based on 13 features
        self.obs_max = torch.max(self.combined_data[:, :13], dim=0)[0].unsqueeze(-1).to(self.device) # Still based on 13 features

        # Determine the graph attribute dimension based on the data structure
        # The combined_data has shape [num_timesteps, 39].
        # The first 13 columns are assumed to be node features.
        # The remaining columns (39 - 13 = 26) are assumed to be graph attributes.
        # self.graph_attr_dim = self.combined_data.shape[1] - 13 # Calculate graph attribute dimension
        # Removed graph_attr_dim as graph_attributes are being removed

        super().__init__(device=device, batch_size=[num_envs])

        self._make_specs()

    # Add the _set_seed method
    def _set_seed(self, seed: Optional[int] = None):
        # Implement seeding logic here if needed
        # For a simple implementation, you can just store the seed
        if seed is not None:
            self.seed = seed
        else:
            self.seed = torch.seed() # Use torch's current seed if none provided
        # Note: For proper reproducibility, you might need to seed other components
        # like random number generators used in _reset or _step.
        return seed

    # Removed _get_state_at method as its logic will be embedded in _reset and _step


    def _make_specs(self):
        print("Debug: Entering _make_specs...")
        # Define the state_spec to match the structure of the tensordict
        # that will be collected by a collector after a step. This tensordict
        # will contain the current state, action, reward, done, and the *next* state.
        self.state_spec = Composite(
             {
                 # Define the keys for the current state, changed 'observation' to 'data'
                 ("agents", "data"): Composite({ # Nested under "agents"
                     "x": Unbounded( # Node features [num_envs, num_agents, node_feature_dim]
                         shape=torch.Size([self.num_envs, self.num_agents, self.node_feature_dim]),
                         dtype=torch.float32,
                         device=self.device
                     ),
                     # Use the fixed number of edges for the spec shape
                     "edge_index": Unbounded( # Edge indices [num_envs, 2, _fixed_num_edges_single]
                         shape=torch.Size([self.num_envs, 2, self._fixed_num_edges_single]),
                         dtype=torch.int64,
                         device=self.device
                     ),
                     # Removed graph_attributes from observation spec
                      # Add 'batch' key to the observation spec
                     "batch": Unbounded( # Batch tensor [num_envs, num_agents]
                         shape=torch.Size([self.num_envs, self.num_agents]),
                         dtype=torch.int64,
                         device=self.device
                     ),
                     # Add timestamp to observation spec
                     "time": Unbounded( # Timestamp (e.g., data index) [num_envs, 1]
                          shape=torch.Size([self.num_envs, 1]),
                          dtype=torch.int64, # Using int64 for data index
                          device=self.device
                      ),
                 }),
                 # Removed global_reward_in_state from state_spec
                 # Added top-level done, terminated, truncated keys for the current state
                 "done": Categorical(n=2, # Added n=2 back
                      # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                      shape=torch.Size([self.num_envs]), # Corrected shape
                      dtype=torch.bool,
                      device=self.device),
                 "terminated": Categorical(n=2, # Added n=2 back
                      # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                      shape=torch.Size([self.num_envs]), # Corrected shape
                      dtype=torch.bool,
                      device=self.device),
                 "truncated": Categorical(n=2, # Added n=2 back
                      # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                      shape=torch.Size([self.num_envs]), # Corrected shape
                      dtype=torch.bool,
                      device=self.device),
                 # Define the keys for the next state, nested under "next", changed 'observation' to 'data'
                 "next": Composite({
                      ("agents", "data"): Composite({ # Nested under "agents"
                          "x": Unbounded( # Node features [num_envs, num_agents, node_feature_dim]
                              shape=torch.Size([self.num_envs, self.num_agents, self.node_feature_dim]),
                              dtype=torch.float32,
                              device=self.device
                          ),
                          # Use the fixed number of edges for the spec shape
                          "edge_index": Unbounded( # Edge indices [num_envs, 2, _fixed_num_edges_single]
                              shape=torch.Size([self.num_envs, 2, self._fixed_num_edges_single]),
                              dtype=torch.int64,
                              device=self.device
                          ),
                          # Removed graph_attributes from next observation spec
                           # Add 'batch' key to the next observation spec
                          "batch": Unbounded( # Batch tensor [num_envs, num_agents]
                              shape=torch.Size([self.num_envs, self.num_agents]),
                              dtype=torch.int64,
                              device=self.device
                          ),
                          # Add timestamp to next observation spec
                          "time": Unbounded( # Timestamp (e.g., data index) [num_envs, 1]
                               shape=torch.Size([self.num_envs, 1]),
                               dtype=torch.int64, # Using int64 for data index
                               device=self.device
                           ),
                      }),
                      # Removed global_reward_in_state from next state spec
                      # Also include reward key under ('agents',) under 'next'
                      ('agents', 'reward'): Unbounded(shape=torch.Size([self.num_envs, self.num_agents, 1]), dtype=torch.float32, device=self.device),
                       # Removed nested done, terminated, truncated keys from state spec
                       # ("agents", "terminated"): Categorical(n=2, # Added n=2 back
                       #      # shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1] as done/terminated/truncated are per env
                       #      shape=torch.Size([self.num_envs]), # Corrected shape
                       #      dtype=torch.bool,
                       #      device=self.device),
                       # ("agents", "truncated"):  Categorical(n=2, # Added n=2 back
                       #      # shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1]
                       #      shape=torch.Size([self.num_envs]), # Corrected shape
                       #       dtype=torch.bool,
                       #       device=self.device),
                       # ("agents", "done"):  Categorical(n=2, # Added n=2 back
                       #      # shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1]
                       #      shape=torch.Size([self.num_envs]), # Corrected shape
                       #       dtype=torch.bool,
                       #       device=self.device),
                     # Add top-level done, terminated, truncated keys to the "next" composite
                     "done": Categorical(n=2, # Added n=2 back
                          # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                          shape=torch.Size([self.num_envs]), # Corrected shape
                          dtype=torch.bool,
                          device=self.device),
                     "terminated": Categorical(n=2, # Added n=2 back
                          # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                          shape=torch.Size([self.num_envs]), # Corrected shape
                          dtype=torch.bool,
                          device=self.device),
                     "truncated": Categorical(n=2, # Added n=2 back
                          # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                          shape=torch.Size([self.num_envs]), # Corrected shape
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


        # Define the action specification using the MultiCategorical SPEC class from torchrl.data
        # The shape should be [num_agents, num_individual_actions_features] for an unbatched environment
        # The nvec should be [num_agents, num_individual_actions_features]
        nvec_unbatched = nvec_tensor.repeat(self.num_agents).view(self.num_agents, self.num_individual_actions_features)

        self.action_spec_unbatched = Composite(
              {("agents","action"): MultiCategoricalSpec( # Use the MultiCategorical SPEC class
                                                      # The shape here defines the shape of the action tensor for a single environment: [num_agents, num_individual_actions_features]
                                                      shape=torch.Size([self.num_agents, self.num_individual_actions_features]),
                                                      dtype=torch.int64,
                                                      device=self.device,
                                                      nvec=nvec_unbatched # nvec should be defined in the MultiCategorical SPEC
                                                      )},
              batch_size=[], # Unbatched environment has no batch size at the root
              device=self.device
            )

        print("\nUnbatched Multi-Agent Action specification defined using nested Composites and MultiCategoricalSpec.")
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
        # Removed nested done, terminated, truncated keys from done_spec
        self.done_spec = Composite(
            {
                # ("agents", "done"):  Categorical(n=2, # Added n=2 back
                #       # shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1]
                #       shape=torch.Size([self.num_envs]), # Corrected shape
                #       dtype=torch.bool,
                #       device=self.device),

                # ("agents", "terminated"): Categorical(n=2, # Added n=2 back
                #       # shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1]
                #       shape=torch.Size([self.num_envs]), # Corrected shape
                #       dtype=torch.bool,
                #       device=self.device),
                # ("agents", "truncated"):  Categorical(n=2, # Added n=2 back
                #      # shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1]
                #      shape=torch.Size([self.num_envs]), # Corrected shape
                #       dtype=torch.bool,
                #       device=self.device),
                # Add top-level done, terminated, truncated keys to the done_spec
                "done": Categorical(n=2, # Added n=2 back
                     # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                     shape=torch.Size([self.num_envs]), # Corrected shape
                     dtype=torch.bool,
                     device=self.device),
                "terminated": Categorical(n=2, # Added n=2 back
                     # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                     shape=torch.Size([self.num_envs]), # Corrected shape
                     dtype=torch.bool,
                     device=self.device),
                "truncated": Categorical(n=2, # Added n=2 back
                     # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                     shape=torch.Size([self.num_envs]), # Corrected shape
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
        print("Debug: Exiting _make_specs.")


    # Implement the actual _is_terminal method
    def _is_terminal(self) -> torch.Tensor:
        """
        Determines if the current state is a terminal state.

        Returns:
            torch.Tensor: A boolean tensor of shape [num_envs] indicating
                          whether each environment is in a terminal state.
        """
        # --- Implement your actual episode termination logic here ---
        # Example: Terminate if a certain prediction error threshold is crossed,
        #          or if the episode has gone on for too long (handled by truncated).

        # For now, keeping it as always False, allowing episodes to end only by truncation.
        # Replace this with your specific termination conditions.
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Example termination based on a hypothetical error threshold (replace with your logic):
        # if self.current_data_index > 0: # Avoid checking at step 0
        #     # Assuming you have access to predicted and actual prices somewhere
        #     # Example: hypothetical 'predicted_price' and 'actual_price' in your state or data
        #     # (You'll need to adapt this based on your actual data structure and how predictions are made)
        #     predicted_price = ... # Get predicted price based on current state/data and actions
        #     actual_price = ...    # Get actual price from your data at current_data_index + 1 (next step)
        #     error = torch.abs(predicted_price - actual_price)
        #     error_threshold = 10.0 # Define your threshold
        #     terminated = error > error_threshold

        return terminated


    # Implement the actual _batch_reward method
    def _batch_reward(self, data_index: torch.Tensor, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Calculates the reward for a batch of environments.

        Args:
            data_index (torch.Tensor): A tensor of shape [num_envs] indicating the
                                         current data index for each environment.
            tensordict (TensorDictBase): The input tensordict containing the
                                         current state and actions.

        Returns:
            TensorDictBase: A tensordict containing the reward for each environment
                            and agent, with shape [num_envs, num_agents, 1].
        """
        # --- Implement your actual reward calculation logic here ---
        # The reward should reflect how well the agent's actions contribute
        # to the fuel price prediction task goals.

        # Access actions from the input tensordict
        # The shape of actions is [num_envs, num_agents, num_individual_actions_features]
        actions=tensordict.get(('agents', 'action')) # Use .get() for safety

        # Access the returns data from self.combined_data based on the current data_index.
        # The returns data is the first half of the 'Indep' part of combined_data.
        # 'Indep' has shape [num_timesteps, 2 * num_agents].
        # The returns for all agents at data_index are at self.combined_data[data_index, self.num_agents : 2 * self.num_agents].
        # We need to handle the batch of environments, so we index with data_index for each environment.
        # data_index is a tensor of shape [num_envs].
        # We need to gather the data for each environment's current_data_index.

        # Ensure data_index + 1 is within bounds of combined_data for accessing the next step's returns
        # (Rewards are typically given after taking an action and transitioning to the next state,
        # so the reward at step t is based on the transition from state t to state t+1,
        # and thus uses data from time t+1. The returns at index i in combined_data represent
        # the change from time i to i+1. So for reward at step 'data_index', we need returns at 'data_index').

        # Check bounds for accessing combined_data at data_index
        valid_indices = data_index < self.combined_data.shape[0]

        # Initialize reward tensor with shape [num_envs, num_agents, 1] and zeros
        reward = torch.zeros(self.num_envs, self.num_agents, 1, dtype=torch.float32, device=self.device)

        # Calculate reward only for valid environments
        if valid_indices.any():
            # Get the relevant data indices for valid environments
            valid_data_indices = data_index[valid_indices] # Shape: [num_valid_envs]

            # Access the returns data for the valid environments at their respective data_indices
            # The returns for all agents at a given time step are columns self.num_agents to 2*self.num_agents
            # in the combined_data.
            # We need to use advanced indexing to get the returns for each environment at its specific index.
            # combined_data shape: [total_timesteps, total_features]
            # valid_data_indices shape: [num_valid_envs]
            # We want combined_data[valid_data_indices, self.num_agents : 2 * self.num_agents]
            # This will give a tensor of shape [num_valid_envs, num_agents]
            returns_for_valid_envs = self.combined_data[valid_data_indices, self.num_agents : 2 * self.num_agents] # Shape: [num_valid_envs, num_agents]

            # Calculate the absolute value of returns as the reward
            # The reward should be agent-wise, so the shape should be [num_valid_envs, num_agents, 1]
            calculated_reward = torch.abs(returns_for_valid_envs).unsqueeze(-1) # Shape: [num_valid_envs, num_agents, 1]

            # Assign the calculated rewards to the reward tensor for the valid environments
            reward[valid_indices] = calculated_reward


        # Create a tensordict to return the reward, nested under ('agents', 'reward')
        # Ensure the shape matches the reward_spec: [num_envs, num_agents, 1]
        reward_tensordict_output = TensorDict({
            ('agents', 'reward'): reward # Place reward under ('agents', 'reward')
        }, batch_size=[self.num_envs], device=self.device)


        return reward_tensordict_output




    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.current_data_index += 1

        terminated = self._is_terminal()
        truncated = (self.current_data_index >= self.episode_length)

        # The action is now expected to be in the input tensordict passed to step() by the collector
        # It should be at tensordict[('agents', 'action')]
        actions=tensordict.get(('agents', 'action')) # Use .get() here as well
        # Pass the input tensordict directly to _batch_reward
        # _batch_reward will need to extract the action from this tensordict
        # Get the reward tensordict from _batch_reward
        reward_td = self._batch_reward(self.current_data_index, tensordict)


        # Logic previously in _get_state_at for next state
        num_envs = self.current_data_index.shape[0]

        # Get data indices for the next step, handling boundaries
        data_indices = torch.min(self.current_data_index + 1, torch.as_tensor(self.combined_data.shape[0] - 1, device=self.device))
        # Extract the first node_feature_dim columns for each environment
        # If all agents share the same features, extract and then expand
        x_data_time_step = self.combined_data[data_indices, :self.node_feature_dim] # Shape: [num_envs, node_feature_dim]
        # Expand to [num_envs, num_agents, node_feature_dim]
        x_data = x_data_time_step.unsqueeze(1).expand(-1, self.num_agents, -1)

        # Use fixed edge index repeated for the batch
        edge_index_data = self._fixed_edge_index_single.unsqueeze(0).repeat(num_envs, 1, 1).to(self.device)
        print(f"_step: Using fixed edge index. Generated edge_index_data shape = {edge_index_data.shape}")


        next_state_tensordict_data = TensorDict({
             "x": x_data, # Use actual data for node features
             "edge_index": edge_index_data, # Placeholder edge indices
             "batch": torch.arange(num_envs, device=self.device).repeat_interleave(self.num_agents).view(num_envs, self.num_agents), # Placeholder batch tensor
             "time": self.current_data_index.unsqueeze(-1).to(self.device), # Use the provided data_index as timestamp
        }, batch_size=[num_envs], device=self.device)


        # Create the output tensordict containing the next observation, reward, and done flags
        # The next observation and reward should be nested under "next" by the collector.
        # The done flags should be at the root level of the tensordict returned by _step.
        # Structure the output tensordict with reward and done flags at the root level
        output_tensordict = TensorDict({
            # Include the next observation structure under ("agents", "data")
            ("agents", "data"): next_state_tensordict_data,
            # Include the reward at the root level using the key from reward_td
            # The reward_td contains the reward under ('agents', 'reward')
            ('agents', 'reward'): reward_td.get(('agents', 'reward')), # Get reward from reward_td using the correct key
            # Include the done flags at the root level
            "terminated": terminated.unsqueeze(-1), # Ensure shape matches spec [num_envs, 1]
            "truncated": truncated.unsqueeze(-1), # Ensure shape matches spec [num_envs, 1]
            "done": (terminated | truncated).unsqueeze(-1), # Ensure shape matches spec [num_envs, 1]

        }, batch_size=[self.num_envs], device=self.device)


        # Return the output_tensordict
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

        # Logic previously in _get_state_at for initial state
        num_envs = self.current_data_index.shape[0]

        # Get data indices for the initial state
        data_indices = self.current_data_index
        # Extract the first node_feature_dim columns for each environment
        # If all agents share the same features, extract and then expand
        x_data_time_step = self.combined_data[data_indices, :self.node_feature_dim] # Shape: [num_envs, node_feature_dim]
        # Expand to [num_envs, num_agents, node_feature_dim]
        x_data = x_data_time_step.unsqueeze(1).expand(-1, self.num_agents, -1)

        # Use fixed edge index repeated for the batch
        edge_index_data = self._fixed_edge_index_single.unsqueeze(0).repeat(num_envs, 1, 1).to(self.device)
        print(f"_reset: Using fixed edge index. Generated edge_index_data shape = {edge_index_data.shape}")


        initial_state_tensordict_data = TensorDict({
             "x": x_data, # Use actual data for node features
             "edge_index": edge_index_data, # Placeholder edge indices
             "batch": torch.arange(num_envs, device=self.device).repeat_interleave(self.num_agents).view(num_envs, self.num_agents), # Placeholder batch tensor
             "time": self.current_data_index.unsqueeze(-1).to(self.device), # Use the provided data_index as timestamp
        }, batch_size=[num_envs], device=self.device)


        # Create the tensordict to return, containing the initial observation and done flags
        # The initial observation should be nested under ("agents", "data")

        output_tensordict = TensorDict({
            # Include the initial observation structure
            ("agents", "data"): initial_state_tensordict_data,
            # Set initial done, terminated, truncated flags to False at the root level
            # Corrected shapes to [num_envs, 1] to match done_spec
            "terminated": torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
            "truncated": torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
            "done": torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
             # Removed nested done, terminated, truncated keys from reset output
            # ("agents", "terminated"): torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
            # ("agents", "truncated"): torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
            # ("agents", "done"): torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
        }, batch_size=[self.num_envs], device=self.device)


        # Removed all debugging print statements from _reset
        # print("\n--- Environment _reset output tensordict ---")
        # print(f"Output tensordict keys: {output_tensordict.keys(include_nested=True)}") # Added include_nested=True
        # # Also print keys of nested observation tensordict, changed 'observation' to 'data'
        # if ("agents", "data") in output_tensordict.keys(include_nested=True): # Added include_nested=True
        #      nested_obs_td = output_tensordict.get(("agents", "data"))
        #      print(f"  Nested ('agents', 'data') keys: {nested_obs_td.keys(include_nested=True)}") # Added include_nested=True
        #      print(f"  Nested ('agents', 'data') shape: {nested_obs_td.shape}")
        #      if ("x") in nested_obs_td.keys():
        #          print(f"  Nested ('agents', 'data', 'x') shape: {output_tensordict.get(('agents', 'data', 'x')).shape}")
        #          print(f"  Nested ('agents', 'data', 'x') dtype: {output_tensordict.get(('agents', 'data', 'x')).dtype}")
        #      else:
        #           print("  Nested ('agents', 'data', 'x') key NOT found.")
        #      # Check for timestamp key
        #      if "time" in nested_obs_td.keys():
        #           print(f"  Nested ('agents', 'data', 'time') value sample: {output_tensordict.get(('agents', 'data', 'time'))[0][:5]}...")
        #      else:
        #           print("  Nested ('agents', 'data', 'time') key NOT found.")

        # else:
        #      print("  Nested ('agents', 'data') key NOT found in output_tensordict.")
        # # Print done keys to verify their location
        # if ("done") in output_tensordict.keys():
        #      print(f"  ('done') value: {output_tensordict.get(('done')))}")
        # if ("terminated") in output_tensordict.keys():
        #      # Fixed the unmatched parenthesis here
        #      print(f"  ('terminated') value: {output_tensordict.get(('terminated')))}")
        # if ("truncated") in output_tensordict.keys():
        #      # Fixed the unmatched parenthesis here
        #      print(f"  ('truncated') value: {output_tensordict.get(('truncated')))}")
        # if ("agents", "done") in output_tensordict.keys(include_nested=True):
        #      print(f"  ('agents', 'done') value: {output_tensordict.get(('agents', 'done')))}")
        # if ("agents", "terminated") in output_tensordict.keys(include_nested=True):
        #      print(f"  ('agents', 'terminated') value: {output_tensordict.get(('agents', 'terminated')))}")
        # if ("agents", "truncated") in output_tensordict.keys(include_nested=True):
        #      print(f"  ('truncated') value: {output_tensordict.get(('truncated')))}")
        # # Print batch key to verify its location, changed 'observation' to 'data'
        # if ('agents', 'data', 'batch') in output_tensordict.keys(include_nested=True):
        #      print(f"  ('agents', 'data', 'batch') value sample: {output_tensordict.get(('agents', 'data', 'batch'))[0][:5]}...") # Print sample of batch tensor


        # print("-------------------------------------------")


        return output_tensordict


# Assuming AnFuelpriceEnv is now defined above in this cell
# Instantiate the environment
#num_envs = 4 # You can adjust the number of environments
#seed = 0
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#try:
#    env = AnFuelpriceEnv(num_envs=num_envs, seed=seed, device=device, episode_length=10) # Reduced episode_length for a quick test
#    print("\nEnvironment instantiated successfully.")
#except Exception as e:
#    print(f"\nCould not re-instantiate env with correct batch size: {e}")
#    env = None


# Check environment specs (if env is available)
#if env is not None:
#    print("\nChecking environment specs...")
    # check_env_specs(env) # Temporarily comment out check_env_specs
#    print("Skipping check_env_specs to proceed.")


# Instantiate the collector without a policy (This part might be commented out or removed later)
# This collector will collect random actions from the environment's action space
# collector = SyncDataCollector(
#     env,
#     policy=None, # No policy is provided, so it will sample random actions
#     frames_per_batch=env.num_envs, # Collect one step from each environment per batch
#     total_frames=100, # Total frames to collect for the test
#     device=device,
#     storing_device=device,
# )

# print("\nCollector instantiated.")
# print(f"Collector total frames: {collector.total_frames}")
# print(f"Collector frames per batch: {collector.frames_per_batch}")

# Run a small rollout to test (This part might be commented out or removed later)
# print("\nRunning a small test rollout...")
# try:
#     for i, data in enumerate(collector):
#         print(f"\n--- Collected Batch {i+1} ---")
#         print(f"Collected data keys: {data.keys(include_nested=True)}")
#         print(f"Collected data shape: {data.shape}")

#         # Access and print some information from the collected tensordict
#         if "next" in data.keys() and ("agents", "data", "x") in data["next"].keys(include_nested=True):
#             print(f"Shape of next observation 'x': {data['next'][('agents', 'data', 'x')].shape}")
#             print(f"Sample of next observation 'x': {data['next'][('agents', 'data', 'x')][0, 0, :5]}...") # Print sample for first env, first agent, first 5 features
#         else:
#              print("Next observation 'x' not found in collected data.")

#         if "reward" in data["next"].keys(): # Rewards are under "next" at the root level in rollout tensordicts
#              print(f"Shape of next reward: {data['next']['reward'].shape}") # Reward shape should be [batch_size, 1] or [batch_size, num_agents, 1] depending on how it's structured by the env
#              # Adjust indexing based on the reward structure returned by your env's _step and how the collector handles it
#              # If your env returns ('agents', 'reward') with shape [num_envs, num_agents, 1] at the root of the _step output,
#              # the collector will place this under "next" and the shape will be [batch_size, num_envs, num_agents, 1]
#              # If your env returns 'reward' at the root with shape [num_envs, 1] (global reward), the collector places it under "next" with shape [batch_size, num_envs, 1]
#              # Based on your env's _make_specs and _batch_reward, the agent-wise reward ('agents', 'reward') should be what is passed.
#              # The collector will likely put this under next, with the batch_size prepended.
#              if ('agents', 'reward') in data["next"].keys(include_nested=True):
#                   print(f"Shape of next agent-wise reward: {data['next'][('agents', 'reward')].shape}")
#                   print(f"Sample of next agent-wise reward: {data['next'][('agents', 'reward')][0, 0, :5]}...") # Print sample for first batch, first env, first 5 agents
#              else:
#                   print("Next agent-wise reward ('agents', 'reward') not found in collected data.")

#         if "done" in data["next"].keys(): # Done flags are under "next" at the root level
#              print(f"Shape of next done flags: {data['next']['done'].shape}")
#              print(f"Sample of next done flags: {data['next']['done'][:5]}...") # Print sample for first 5 batches
#         else:
#              print("Next done flags not found in collected data.")


#         if i >= 2: # Collect only a few batches for the test
#             break

# except Exception as e:
#     print(f"\nAn error occurred during the test rollout: {e}")

# print("\nTest rollout finished.")

# You can further inspect the 'data' tensordict collected in the loop.
# For example, after the loop, 'data' will hold the last collected batch.
# print("\nLast collected batch:")
# print(data)
