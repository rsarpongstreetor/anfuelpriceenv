import torch
import random
import os
import numpy as np
import pandas as pd
from typing import Dict as TypingDict, Any, Union, List, Optional
from torchrl.envs import EnvBase
from torchrl.data import CompositeSpec, BoundedTensorSpec, DiscreteTensorSpec
from torchrl.envs.utils import step_mdp, ExplorationType, set_exploration_type
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Unbounded, Composite, Bounded, Binary, Categorical
# Instead of importing private functions, use public functions if available
from torchrl.envs.utils import check_env_specs

class DDataenv:
    def __init__(self, data_path: str, data_columns: List[str], data_type: Any = np.float32, allow_repeat: bool = True):
        self.data_path = data_path
        self.data_columns = data_columns
        self.data_type = data_type
        self.data = None
        self.current_index = 0  # Initialize current_index here
        self.allow_repeat = allow_repeat

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        with open(self.data_path, 'rb') as f:
            self.data = torch.load(f, weights_only=False)

        self.data = np.array(self.data)
        if len(self.data.shape) >= 3:
            self.data = self.data.reshape(self.data.shape[1], self.data.shape[2])

        if not isinstance(self.data, pd.DataFrame):
            self.data = pd.DataFrame(self.data, columns=self.data_columns)

        return self.data

    def get_observation(self) -> TypingDict[str, Union[np.ndarray, TypingDict[str, float]]]:
        if self.data is None:
            self.load_data()

        # Sample with replacement if allow_repeat is True
        if self.allow_repeat:
            self.current_index = random.randint(0, len(self.data) - 1)
        else:
            # Reset the index if it exceeds the data size
            if self.current_index >= len(self.data):
                self.current_index = 0

        # Get observation from the data
        observation = self.data.iloc[self.current_index, :].to_numpy().astype(self.data_type)

        # Increment the index for the next observation
        self.current_index += 1

        describe_data = self.data.describe()

        observation_dict = {
            'obsState&Fuel': observation[0:13],
            'Date': observation[-1],
            'rewardState&reward': observation[13:26],
            'actionState&action': observation[26:39],
            'obsState&Fuel_max': describe_data.loc['max'][0:13].values,
            'obsState&Fuel_min': describe_data.loc['min'][0:13].values,
            'Date_max': describe_data['Date'].max(),
            'Date_min': describe_data['Date'].min(),
            'rewardState&reward_max': describe_data.loc['max'][13:26].values,
            'rewardState&reward_min': describe_data.loc['min'][13:26].values,
            'actionState&action_max': describe_data.loc['max'][26:39].values,
            'actionState&action_min': describe_data.loc['min'][26:39].values,
        }
        return observation_dict




def _step(self, tensordict):  # Add 'self' to access instance variables
    """Defines the transition function of the environment.

    This function updates the environment's state based on the given action and returns
    the next state, reward, done flag, and info.
    """

    # Access instance variables using 'self'
    agents_data = {}
    td_params = gen_params(self.batch_size_tuple)  # Use self.batch_size_tuple
    for idx in range(self.n_agents):  # Use self.n_agents

        agentc0_reward_list, agentc0_action_list, agentc0_new_obs_list, agentc0_Dat_list = [], [], [], []
        for _ in range(self.convo_dim[1]):  # Use self.convo_dim
            agentc1_reward_list, agentc1_action_list, agentc1_new_obs_list, agentc1_Dat_list = [], [], [], []
            for _ in range(self.convo_dim[0]):  # Use self.convo_dim
                obs = td_params["params", "obsState&Fuel"].clone().detach()
                rew = td_params["params", "rewardState&reward"].clone().detach()
                act = td_params["params", "actionState&action"].clone().detach()
                Dat = td_params["params", "Date"].clone().detach().reshape(-1, 1)
                new_obs = obs + (act * rew)

                agentc1_reward_list.append(rew)
                agentc1_action_list.append(act)
                agentc1_new_obs_list.append(new_obs)
                agentc1_Dat_list.append(Dat)

            agent_reward = torch.stack(agentc1_reward_list, dim=-1)
            agent_action = torch.stack(agentc1_action_list, dim=-1)
            agent_new_obs = torch.stack(agentc1_new_obs_list, dim=-1)
            agent_Dat = torch.stack(agentc1_Dat_list, dim=-1)

            agentc0_reward_list.append(agent_reward)
            agentc0_action_list.append(agent_action)
            agentc0_new_obs_list.append(agent_new_obs)
            agentc0_Dat_list.append(agent_Dat)

        agent0_reward = torch.stack(agentc0_reward_list, dim=-1)
        agent0_action = torch.stack(agentc0_action_list, dim=-1)
        agent0_new_obs = torch.stack(agentc0_new_obs_list, dim=-1)
        agent0_Dat = torch.stack(agentc0_Dat_list, dim=-1)

        rew1 = agent0_reward.clone().detach().float()

        rew1 = agent0_reward.clone().detach().sum(dim=-3).mean(dim=[-2, -1], )  # Sum across the feature dimension (13)
        episode_reward = rew1.view(*env.batch_size, 1)
        reward = episode_reward.clone().detach().float()

        observat = agent0_new_obs.view(*self.batch_size_tuple, 13, *self.convo_dim)  # Use self.batch_size_tuple, self.convo_dim
        position_key = agent0_Dat.view(*self.batch_size_tuple, *self.convo_dim)  # Use self.batch_size_tuple, self.convo_dim
        agent0_action = agent0_action.view(*self.batch_size_tuple, 13, *self.convo_dim)  # Use self.batch_size_tuple, self.convo_dim

        # Update the observation TensorDict within agent_data
        agent_data = {
            "observation": TensorDict(
                {
                    "observat": observat,
                    "position_key": position_key,
                },
                batch_size=self.batch_size_tuple,
            ),
            "reward": episode_reward,  # Directly store the reward
            "action": agent0_action,
            "done": torch.zeros(self.batch_size_tuple + (1,), dtype=torch.bool),
            "terminated": torch.zeros(self.batch_size_tuple + (1,), dtype=torch.bool),
            "truncated": torch.zeros(self.batch_size_tuple + (1,), dtype=torch.bool),
            "info": {},
        }

        agents_data[f"agent_{idx}"] = TensorDict(agent_data, batch_size=self.batch_size_tuple)
    
    # Simplified "next" structure for rollout compatibility:
    next_obs_data = {
        agent_id: agent_data["observation"]  # Only include observations in "next"
        for agent_id, agent_data in agents_data.items()
    }
    
    # Create the next_tensordict with consistent structure
    next_tensordict = TensorDict(
        {
            "agents": agents_data,
            "next": {  # Simplified "next" structure
                "agents": next_obs_data,  
                "done": torch.zeros(self.batch_size_tuple + (1,), dtype=torch.bool),
                "terminated": torch.zeros(self.batch_size_tuple + (1,), dtype=torch.bool),
                "truncated": torch.zeros(self.batch_size_tuple + (1,), dtype=torch.bool),
            },
            "done": torch.zeros(self.batch_size_tuple + (1,), dtype=torch.bool),
            "terminated": torch.zeros(self.batch_size_tuple + (1,), dtype=torch.bool),
            "truncated": torch.zeros(self.batch_size_tuple + (1,), dtype=torch.bool),
        },
        batch_size=self.batch_size_tuple,
    )
    return next_tensordict



def _make_spec(self, tensordict):
    agent = {f"agent_{i}": {} for i in range(self.n_agents)}
    action_specs = []
    observation_specs = []
    reward_specs = []
    info_specs = []
    done_specs = []
    discount_specs = []
    agent_i_spec = {}

    for i in range(self.n_agents):
        td_agents = self.gen_params(self.batch_size)
        batch_size = self.batch_size
        batch_size_flattened = (batch_size[0], batch_size[1]) if isinstance(batch_size, tuple) and len(
            batch_size) == 2 else batch_size

        obs_max = td_agents['params', 'obsState&Fuel_max'].clone().detach()
        obs_min = td_agents['params', 'obsState&Fuel_min'].clone().detach()
        rew_max = td_agents['params', 'rewardState&reward_max'].clone().detach()
        rew_min = td_agents['params', 'rewardState&reward_min'].clone().detach()
        action_max = td_agents['params', 'actionState&action_max'].clone().detach()
        action_min = td_agents['params', 'actionState&action_min'].clone().detach()
        Date_max = td_agents['params', 'Date_max'].clone().detach()
        Date_min = td_agents['params', 'Date_min'].clone().detach()



        # Updated dimension adjustments for bounds
        obs_min1 = obs_min.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
            tuple((self.batch_size) + tuple(obs_min.shape) + tuple(self.convo_dim)))
        obs_max1 = obs_max.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
            tuple((self.batch_size) + tuple(obs_min.shape) + tuple(self.convo_dim)))
        Date_min1 = Date_min.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
            tuple((self.batch_size) + tuple(Date_min.shape) + tuple(self.convo_dim)))
        Date_max1 = Date_max.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
            tuple((self.batch_size) + tuple(Date_max.shape) + tuple(self.convo_dim)))
        rew_max1 = rew_max.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
            tuple((self.batch_size) + tuple(rew_min.shape) + tuple(self.convo_dim)))
        rew_min1 = rew_min.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
            tuple((self.batch_size) + tuple(rew_min.shape) + tuple(self.convo_dim)))
        action_max1 = action_max.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
            tuple((self.batch_size) + tuple(action_max.shape) + tuple(self.convo_dim)))
        action_min1 = action_min.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
            tuple((self.batch_size) + tuple(action_min.shape) + tuple(self.convo_dim)))


        rewardfuelsum_max = rew_max1.sum(dim=-3,).mean(dim=[-2, -1], )
        rewardfuelsum_min = rew_min1.sum(dim=-3).mean(dim=[-2, -1], )
        rewardfuelsum_max1=rewardfuelsum_max.reshape(self.batch_size)
        rewardfuelsum_min1=rewardfuelsum_min.reshape(self.batch_size)

        agent_i_observation_spec_unbatched = CompositeSpec(
        {
            "observat": Bounded(
                low=obs_min1,
                high=obs_max1,
                shape=obs_max1.shape,
                dtype=torch.float32,
            ),
            "position_key": Bounded(
                low=Date_min1,
                high=Date_max1,
                dtype=torch.float32,
                shape=Date_max1.shape,
            ),
        }, shape=tuple(self.batch_size)
    )

        agent_i_discount_spec_unbatched = Bounded(low=0.0, high=1.0, shape=(), dtype=torch.float32)

        agent_i_action_spec_unbatched = CompositeSpec(
            {
                "action": Bounded(
                    low=action_min1,
                    high=action_max1,
                    shape=action_max1.shape,
                    dtype=torch.float32,
                )
            }
        )


        agent_i_reward_spec_unbatched = Bounded(low=rewardfuelsum_min1,
                                  high=rewardfuelsum_max1,
                                  shape=rewardfuelsum_max1.shape,
                                  dtype=torch.float32)

        agent_i_info_spec_unbatched = CompositeSpec(
            {
                "info": CompositeSpec(
                    {
                        "reset_count": Unbounded(dtype=torch.int64),  # For reset info
                        "initial_state": Unbounded(dtype=torch.float32, shape=(13,)),  # For initial state info
                        "step_count": Unbounded(dtype=torch.int64)  # For step info
                    }
                )
            }, )

        agent_i_dones_spec_unbatched = CompositeSpec(
            {
                "terminated": Binary(shape=self.batch_size + (1,), dtype=torch.bool),  # Changed shape to tuple
                "done": Binary(shape=self.batch_size + (1,), dtype=torch.bool),  # Changed shape to tuple
                "truncated": Binary(shape=self.batch_size + (1,), dtype=torch.bool),  # Changed shape to tuple
            })

        agent_i_observation_spec_batched = agent_i_observation_spec_unbatched.to(self.device)
        agent_i_action_spec_batched = agent_i_action_spec_unbatched.to(self.device)
        agent_i_reward_spec_batched = agent_i_reward_spec_unbatched.to(self.device)
        agent_i_info_spec = agent_i_info_spec_unbatched.to(self.device)
        agent_i_dones_spec_batched = agent_i_dones_spec_unbatched.to(self.device)
        agent_i_discount_spec_batched = agent_i_discount_spec_unbatched.to(self.device)

        agent_i_spec[f"agent_{i}"] = {
            "observation": agent_i_observation_spec_batched,
            "action": agent_i_action_spec_batched,
            "reward": agent_i_reward_spec_batched,
            "done": agent_i_dones_spec_batched["done"],  # Corrected this line
            "terminated": agent_i_dones_spec_batched["terminated"],  # Corrected this line
            "truncated": agent_i_dones_spec_batched["truncated"],  # Corrected this line
            "info": agent_i_info_spec,  # Correct structure
            "discount": agent_i_discount_spec_batched,
        }

        action_specs.append(agent_i_action_spec_batched)
        reward_specs.append(agent_i_reward_spec_batched)
        observation_specs.append(agent_i_observation_spec_batched)
        done_specs.append(agent_i_dones_spec_batched)
        info_specs.append(agent_i_info_spec)
        discount_specs.append(agent_i_discount_spec_batched)

    # Instead of stacking, create a Composite with named sub-specs for each agent
    self.discount_spec = discount_specs[0]

    self.action_spec = CompositeSpec(
        {
            "agents": CompositeSpec(
                {f"agent_{j}": action_specs[j]["action"] for j in range(len(action_specs))},
            ),
        },
        shape=tuple(self.batch_size),  # Shape reflecting the number of agents
        device=self.device,  # Add device here
    )
    self.reward_spec = CompositeSpec(
    {
        "agents": CompositeSpec(
            {f"agent_{j}": reward_specs[j] for j in range(len(reward_specs))},  # Corrected key and loop
        )
    },
    shape=(self.batch_size),
    )


    self.observation_spec = CompositeSpec(
        {
            "agents": CompositeSpec(
                {
                    f"agent_{i}": agent_i_spec[f"agent_{i}"]["observation"]  # Directly use agent_i_spec
                    for i in range(self.n_agents)
                },
              
            ),
        },
        shape=(self.batch_size),
        device=self.device,)

    self.info_spec = CompositeSpec(
        {
            "agents": CompositeSpec(
                {f"agent_{i}": CompositeSpec({}) for i in range(self.n_agents)},
            )
        },
        shape=tuple(self.batch_size),  # Shape reflecting the number of agents
        device=self.device,  # Add device here
    )
    
    self.done_spec = CompositeSpec(
        {
            "terminated": Binary(shape=tuple(self.batch_size) + (1,), dtype=torch.bool),
            "done": Binary(shape=tuple(self.batch_size) + (1,), dtype=torch.bool),
            "truncated": Binary(shape=tuple(self.batch_size) + (1,), dtype=torch.bool),
        },
        shape=tuple(self.batch_size),  # Shape reflecting the number of agents
        device=self.device,  # Add device here
    )

    self.full_reward_spec = self.reward_spec.to(self.device)
    self.full_action_spec = self.action_spec.to(self.device)
    self.full_observation_spec = self.observation_spec.to(self.device)
    self.full_info_spec = self.info_spec.to(self.device)
    self.full_done_spec = self.done_spec.to(self.device)

    return self.full_observation_spec, self.full_action_spec, self.full_reward_spec, self.full_info_spec, self.full_done_spec


def _reset(self, tensordict=None, **kwargs):
    if tensordict is None:
        tensordict = TensorDict({}, batch_size=self.batch_size_tuple,
                                 device=self.device)  # Use self.batch_size_tuple
    agents_data = {}
    #all_agent_observations = []  # Initialize list to store observations

    td_params = self.gen_params(self.batch_size_tuple)  # Use self.batch_size_tuple

    for idx in range(self.n_agents):
        batch_size = self.batch_size_tuple  # Use self.batch_size_tuple
        # Initialize the agent's data dictionary with all expected keys
        #agent_data = {}
        obs = td_params['params', 'obsState&Fuel'].clone().detach()
        obs_max = td_params['params', 'obsState&Fuel_max'].clone().detach()
        obs_min = td_params['params', 'obsState&Fuel_min'].clone().detach()
        Date = td_params['params', 'Date'].clone().detach()
        Date_max = td_params['params', 'Date_max'].clone().detach()
        Date_min = td_params['params', 'Date_min'].clone().detach()

        obs = (obs_max - obs_min) * torch.rand_like(obs) + obs_min  # Generate random observations within bounds
        obs = obs.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
            tuple(self.batch_size_tuple) + obs.shape + tuple(self.convo_dim))
        

        Date = torch.rand((*self.batch_size_tuple, *self.convo_dim),
                          device=self.device) * (Date_max - Date_min) + Date_min  # Use self.batch_size_tuple
        Date = Date.type(torch.float32)  # Ensure correct data type
        obs = obs.type(torch.float32)

        # Expand reward to match batch size
        reward = torch.tensor(0.0, device=self.device).expand(self.batch_size_tuple)

        
        agent_data = TensorDict({
            "action": torch.zeros(self.batch_size_tuple + (13, *self.convo_dim), dtype=torch.float32,
                                  device=self.device),
            "observation": TensorDict({  # Create a nested TensorDict for observation
                "observat": obs,  # Fixed here: add observat to agents_data
                "position_key": Date
            }),
            "reward": reward,
            "done": torch.zeros(self.batch_size_tuple + (1,), dtype=torch.bool, device=self.device),
            "terminated": torch.zeros(self.batch_size_tuple + (1,), dtype=torch.bool, device=self.device),
            "truncated": torch.zeros(self.batch_size_tuple + (1,), dtype=torch.bool, device=self.device),
            "info": {}  # Fixed here: add "info" to agent_data[f"agent_{idx}"].
        }, batch_size=self.batch_size_tuple)
        agents_data[f"agent_{idx}"] = agent_data

    reset_tensordict = TensorDict(
        {
            "agents": agents_data,  # agents_data should already be a dictionary of TensorDicts

            "next": {  # Fixed here: add "next" with agent data to reset_tensordict.
                "agents": agents_data,
                "done": torch.zeros(self.batch_size_tuple + (1,), dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(self.batch_size_tuple + (1,), dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(self.batch_size_tuple + (1,), dtype=torch.bool, device=self.device),
            }
        },
        batch_size=self.batch_size_tuple,
    )
    return reset_tensordict



def gen_params(batch_size=torch.Size()) -> TensorDictBase:
    if batch_size is None or (isinstance(batch_size, list) and len(batch_size) == 0):
        batch_size = torch.Size([])
    data_path = '/content/drive/MyDrive/deep learning codes/EIAAPI_DOWNLOAD/solutions/mergedata/DataDic.pt'
    data_columns = ['Forex', 'WTI', 'Brent', 'OPEC', 'Fuelprice5', 'Fuelprice6', 'Fuelprice7', 'Fuelprice8',
                    'Fuelprice9', 'Fuelprice10', 'Fuelprice11', 'Fuelprice12', 'Fuelprice13',
                    'reward0', 'reward1', 'reward2', 'reward3', 'reward4', 'reward5', 'reward6', 'reward7', 'reward8',
                    'reward9', 'reward10', 'reward11', 'reward12',
                    'action0', 'action1', 'action2', 'action3', 'action4', 'action5', 'action6', 'action7', 'action8',
                    'action9', 'action10', 'action11', 'action12', 'Date']
    envv = DDataenv(data_path, data_columns)  # Assuming DDataenv is your data environment class

    ac = envv.get_observation()

    if batch_size:
        # Change here: Convert batch_size to a tuple if it's not already
        if not isinstance(batch_size, tuple):
            batch_size = (batch_size,)  # Convert to tuple
        ac = {k: torch.tensor(v).expand(*batch_size, *torch.tensor(v).shape) for k, v in ac.items()} # Assuming you want to repeat along the first batch dimension


    td = TensorDict({"params": ac}, batch_size=batch_size, device=torch.device("cpu" if torch.cuda.is_available() else "cpu")) # Removed extra comma, fixed typo
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td



def _set_seed(self, seed: 45):
    self.rng = torch.manual_seed(seed)

def full_info_spec(self):
    # If "step_count" is an integer with batch size
    return CompositeSpec(
        {
            "agents": CompositeSpec(
                {
                    f"agent_{i}": CompositeSpec(
                        {
                            "info": CompositeSpec(
                                {"step_count": Bounded(low=0, high=1000,  # Example values
                                                       shape=(), dtype=torch.int64)}
                            )
                        }
                    )
                    for i in range(self.n_agents)
                }
            )
        },
        batch_size=self.batch_size,
        device=self.device
    )


def group_map(self, env: EnvBase) -> TypingDict[str,List[str]]:
    return {"agents": [agent["name"] for agent in env.agents]}

class AnFuelpriceEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    Scenario = "USDATA_1"
    batch_locked = False

    def __init__(self, td_params=None, seed=None, device="cpu", categorical_actions=True, continuous_actions=True, **kwargs):
        if td_params is None:
            td_params = gen_params()

        _ = kwargs.pop("scenario", None)

        self.n_agents = 2
        self.convo_dim = [7, 9]
        self.batch_size = (10,8)
        self.batch_size_tuple = torch.Size(self.batch_size) if isinstance(self.batch_size, int) else torch.Size(self.batch_size)
        self.device = device
        super().__init__(device=device, batch_size=self.batch_size)

        self._make_spec(td_params)
        self.action_spec = self.full_action_spec

        if seed is None:
            #seed = torch.empty((), dtype=torch.int64).random_().item()
            seed = random.randint(0, 10000)  # Generate a random integer seed
        self.set_seed(seed)




    def set_seed(self, seed):
        self.rng = torch.manual_seed(seed)

    def gen_params(self, batch_size=None):
        return gen_params(batch_size)

    def get_supports_continuous_actions(self):
         # Check if the action spec has a shape and is not a DiscreteTensorSpec
        return hasattr(self.full_action_spec, 'shape') and len(self.full_action_spec.shape) > 0 and not isinstance(self.full_action_spec, DiscreteTensorSpec|Categorical)

    def get_supports_discrete_actions(self):
        return isinstance(self.full_action_spec, DiscreteTensorSpec|Categorical)

    def get_observation_spec(self):
        return self.observation_spec

    def get_full_action_spec(self):
        return self.full_action_spec

    def get_reward_spec(self):
        return self.reward_spec

    def get_done_spec(self):
        return self.done_spec

    def get_group_map(self, env: EnvBase) -> TypingDict[str, List[str]]:
        """
        Returns a mapping of agent group names to agent names.
        This is used to group agents for multi-agent training.
        """
        # If `env.agents` is None, return an empty list to prevent error
        return {"agents": [agent["name"] for agent in env.agents] if env.agents else []}

    def get_full_info_spec(self):
        return CompositeSpec({}, batch_size=self.batch_size, device=self.device)

    def get_discount_spec(self):
        return self.discount_spec

    def get_attr(self, name: str, idx: Optional[int] = None):
        """
        Get an attribute from the environment.

        Args:
            name: Name of the attribute to retrieve.
            idx: Optional index for batched attributes. If None,
                the entire batch is returned.

        Returns:
            The attribute value.
        """
        if hasattr(self, name):  # Check if the attribute exists
            attr = getattr(self, name)

            # Handle batched attributes with indexing
            if idx is not None and isinstance(attr, (torch.Tensor, TensorDict)):
                attr = attr[idx]  # Index into the batched attribute

            return attr
        else:
            raise AttributeError(f"Environment does not have attribute '{name}'")

    @property
    def terminated_spec(self):
        return self.done_spec

    @property
    def truncated_spec(self):
        return self.done_spec

    @property
    def get_env_name(self):
        return "AnFuelpriceEnv"


    _full_observation_spec = None  # Internal variable to store the spec

    @property
    def full_observation_spec(self):
        if self._full_observation_spec is None:  # Initialize if needed
            self._full_observation_spec = self.observation_spec.to(self.device)
        return self._full_observation_spec

    @full_observation_spec.setter
    def full_observation_spec(self, value):
        self._full_observation_spec = value

    @property
    def full_observation_spec_unbatched(self):  # Renamed to full_observation_spec_unbatched
        """Returns the full observation spec with a batch size of 1."""
        from torchrl.envs.transforms.transforms import _apply_to_composite  # Import here

        # Create a copy of the observation spec
        unbatched_spec = self.observation_spec.clone()

        # Set the batch size to 1 for all nested specs
        def _set_batch_size_one(spec):
            if isinstance(spec, (Composite, Bounded, Unbounded, Binary, Categorical)):  # Include Categorical here
                spec.batch_size = torch.Size([1])
            return spec

        unbatched_spec = _apply_to_composite((unbatched_spec, _set_batch_size_one))  # Corrected call

        return unbatched_spec.to(self.device)



    _full_observation_spec_unbatched = None  # Internal variable to store the spec



    @property
    def full_observation_spec(self):
        if self._full_observation_spec is None:  # Initialize if needed
            self._full_observation_spec = self.observation_spec.to(self.device)
        return self._full_observation_spec

    @full_observation_spec.setter
    def full_observation_spec(self, value):
        self._full_observation_spec = value

    @property
    def full_observation_spec_unbatched(self):  # Renamed to full_observation_spec_unbatched
        """Returns the full observation spec with a batch size of 1."""
        from torchrl.envs.transforms.transforms import _apply_to_composite  # Import here

        # Create a copy of the observation spec
        unbatched_spec = self.observation_spec.clone()

        # Set the batch size to 1 for all nested specs
        def _set_batch_size_one(spec):
            if isinstance(spec, (Composite, Bounded, Unbounded, Binary, Categorical)):  # Include Categorical here
                spec.batch_size = torch.Size([1])
            return spec

        unbatched_spec = _apply_to_composite((unbatched_spec, _set_batch_size_one))  # Corrected call

        return unbatched_spec.to(self.device)



    _full_observation_spec_unbatched = None  # Internal variable to store the spec


    @property
    def full_observation_spec_unbatched(self):  # Renamed to full_observation_spec_unbatched
        if self._full_observation_spec_unbatched is None:  # Initialize if needed
            from torchrl.envs.transforms.transforms import _apply_to_composite  # Import here

            # Create a copy of the observation spec
            unbatched_spec = self.observation_spec.clone()

            # Set the batch size to 1 for all nested specs
            def _set_batch_size_one(spec):
                if isinstance(spec, (Composite, Bounded, Unbounded, Binary, Categorical)):  # Include Categorical here
                    spec.batch_size = torch.Size([1])
                return spec

            self._full_observation_spec_unbatched = _apply_to_composite((unbatched_spec, _set_batch_size_one))  # Corrected call
            self._full_observation_spec_unbatched = self._full_observation_spec_unbatched.to(self.device)

        return self._full_observation_spec_unbatched

    _full_observation_spec_batched = None  # Internal variable to store the spec

    @property
    def full_observation_spec_batched(self):
        if self._full_observation_spec_batched is None:
            self._full_observation_spec_batched = self.observation_spec.to(self.device)
        return self._full_observation_spec_batched

    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec
    _reset = _reset
    _step = _step
    _set_seed = _set_seed

    full_info_spec = full_info_spec

env = AnFuelpriceEnv()

print("\n*action_spec:", env.full_action_spec)
print("\n*reward_spec:", env.full_reward_spec)
print("\n*done_spec:", env.full_done_spec)
print("\n*observation_spec:", env.full_observation_spec_batched)  # Use the renamed property

print("\n-action_keys:", env.action_keys)
print("\n-reward_keys:", env.reward_keys)
print("\n-done_keys:", env.done_keys)

print("input_spec:", env.input_spec)
print("reward_spec:", env.reward_spec)
print("action_spec (as defined by input_spec):", env.action_spec)
td = env.reset()
print("reset tensordict",td)
#print("_step tensordict",td)
#action = env.action_spec.rand()

# Use step_mdp for environment transition
#next_td = step_mdp(env, td, action)



env = AnFuelpriceEnv()
check_env_specs(env)
"""
def check_env_specs(env: EnvBase) -> None:
    Checks that the environment specs are correct."
    print("input_spec:", env.input_spec)
    print("reward_spec:", env.reward_spec)
    print("action_spec (as defined by input_spec):", env.action_spec)

    # Check spec types (use isinstance instead of type for more flexibility)
    print("Observation spec type:", type(env.observation_spec))
    print("Observation space type: This property is replaced by `observation_spec` in torchrl environments.")  # Informative message
    print("Action spec type:", type(env.action_spec))
    print("Action space type: This property is replaced by `action_spec` in torchrl environments.")  # Informative message
    print("Reward spec type:", type(env.reward_spec))
    print("Reward space type: This property is replaced by `reward_spec` in torchrl environments.")  # Informative message
    print("Done spec type:", type(env.done_spec))
    print("Done space type: This property is replaced by `done_spec` in torchrl environments.")  # Informative message

    # Check batch size consistency
    print("Batch size:", env.batch_size)
    print("Observation spec batch size:", env.observation_spec.batch_size)
    print("Action spec batch size:", env.action_spec.batch_size)
    print("Reward spec batch size:", env.reward_spec.batch_size)
    print("Done spec batch size:", env.done_spec.batch_size)
    td=env.reset()
    print("reset tensordict",td)
    #print("_step tensordict",td)

check_env_specs(env)
"""
