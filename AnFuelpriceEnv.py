import torch
import numpy as np
import pandas as pd
from typing import Dict as TypingDict, Any, Union, List, Optional
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.data import CompositeSpec, BoundedTensorSpec, DiscreteTensorSpec, UnboundedContinuousTensorSpec
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs
from tensordict import TensorDict, TensorDictBase
import os  # Import os for file existence check
from typing import Dict, List # Import Dict and List here
from torchrl.envs.transforms import RewardSum #changed the location of the import

class DDataenv:
    def __init__(self, data_path: str, data_columns: List[str], data_type: Any = np.float32):
        self.data_path = data_path
        self.data_columns = data_columns
        self.data_type = data_type
        self.data = None

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

        random_row_index = np.random.choice(self.data.shape[0], 1, replace=False)[0]
        observation = self.data.iloc[random_row_index, :].to_numpy().astype(self.data_type)
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




def _step(tensordict):
    td = env.gen_params()
    n_agents = env.n_agents

    # Initialize a dictionary to store agent data
    agent = {f"agent_{i}": {} for i in range(n_agents)}

    for j in range(n_agents):
        # Initialize lists to store data for each agent within the batch
        agent_i_date = []
        agent_i_action = []
        agent_i_rew = []
        agent_i_new_obs = []

        for _ in range(env.batch_size[0]):
            current_td = env.gen_params()
            obs = td['params', 'obsState&Fuel'].clone().detach()
            reward = td['params', 'rewardState&reward'].clone().detach()
            action = td['params', 'actionState&action'].clone().detach()
            Date = td['params', 'Date'].clone().detach()

            new_obs = torch.add(obs, torch.stack([action_i * reward_i for action_i, reward_i in zip(action, reward)]))
            # new_obs = torch.reshape(new_obs, (13,))

            # Append data to agent-specific lists
            agent_i_new_obs.append(new_obs)
            agent_i_rew.append(reward)
            agent_i_date.append(Date)
            agent_i_action.append(action)

        # Stack data for the current agent
        en_Date = torch.stack(agent_i_date, dim=0) if agent_i_date else torch.empty(env.batch_size + (1,), device=env.device)
        en_action = torch.stack(agent_i_action, dim=0) if agent_i_action else torch.empty(env.batch_size + (13,), device=env.device)
        en_reward = torch.stack(agent_i_rew, dim=0) if agent_i_action else torch.empty(env.batch_size + (13,), device=env.device)
        en_new_obs = torch.stack(agent_i_new_obs, dim=0) if agent_i_new_obs else torch.empty(env.batch_size + (13,), device=env.device)

        # Expansion for batch size and convolution
        expanded_agent_new_obs = en_new_obs.reshape(*en_new_obs.shape, 1, 1).expand(tuple(env.batch_size) + (13, *env.convo_dim))
        expanded_agent_reward = en_reward.reshape(*en_reward.shape, 1, 1).expand(tuple(env.batch_size) + (13, *env.convo_dim))
        expanded_agent_action = en_action.reshape(*en_action.shape, 1, 1).expand(tuple(env.batch_size) + (13, *env.convo_dim))
        expanded_agent_Date = en_Date.reshape(*en_Date.shape, 1, 1).expand(tuple(env.batch_size) + (en_Date.shape[-1], *env.convo_dim))
        
        # Calculate the total reward for each agent and reshape to match reward_spec
        expanded_agent_reward = (torch.sum(expanded_agent_reward, dim=2)).reshape((tuple(env.batch_size) + (1,) + tuple(env.convo_dim)))  

        # Store data in agent-specific dictionary
        agent[f"agent_{j}"]["observation"] = {"observat": expanded_agent_new_obs, "position_key": expanded_agent_Date}
        agent[f"agent_{j}"]["action"] = expanded_agent_action  # Assuming you need action here
        agent[f"agent_{j}"]["reward"] = expanded_agent_reward  # Assign the calculated total reward
       
        # Ensure agent reward is not None and has the expected shape
        if agent[f"agent_{j}"]["reward"] is None:
            agent[f"agent_{j}"]["reward"] = torch.zeros(env.batch_size + (1,) + env.convo_dim, device=env.device, dtype=torch.float32)  # Or any appropriate default value
        
         # Reshape reward to match reward_spec
        agent[f"agent_{j}"]["reward"] = agent[f"agent_{j}"]["reward"].view((tuple(env.batch_size) + (1,) + tuple(env.convo_dim))) # Change here

        
    
    dones = torch.zeros((*env.batch_size, 1), dtype=torch.bool)

    next_tensordict = TensorDict(
        {
            "agents": {
                f"agent_{j}": {
                    "observation": agent[f"agent_{j}"]["observation"],
                    "reward": agent[f"agent_{j}"]["reward"],  # Ensure reward is present and under "agents"
                } for j in range(env.n_agents)
            },
            "terminated": dones.clone(),
        },
        batch_size=env.batch_size,
        device=env.device,
    )

    return next_tensordict

def _reset(self, tensordict=None):
    if tensordict is not None and "_reset" not in tensordict:
        tensordict.clear()
    else:
        tensordict = TensorDict({}, batch_size=self.batch_size, device=self.device)

        td = self.gen_params(self.batch_size)

        obs_max = td['params', 'obsState&Fuel_max'].clone().detach()
        obs_min = td['params', 'obsState&Fuel_min'].clone().detach()
        Date = td['params', 'Date_max'].clone().detach()

        n_agents = self.n_agents

        # Create a dictionary to store agent data
        agent_data = {f"agent_{i}": {} for i in range(n_agents)}

        for i in range(n_agents):
            random_numbers = torch.rand(1, device=self.device)
            obs = torch.add(torch.mul(random_numbers, torch.add(obs_max, -obs_min)), obs_min)
            Date=  td['params', 'Date_max'].clone().detach()

            obs = obs.reshape (tuple(self.batch_size) + (13,))
            Date = Date.reshape (tuple(self.batch_size) + (1,))



            # Expand to include batch size
            expanded_obs = obs.unsqueeze(-1).unsqueeze(-1).expand(tuple(self.batch_size) + ( 13, *self.convo_dim))
            expanded_Date = Date.unsqueeze(-1).unsqueeze(-1).expand(tuple(self.batch_size) + (1, *self.convo_dim))

            # Store data in the agent_data dictionary
            agent_data[f"agent_{i}"]["observation"] = {
                "observat": expanded_obs,
                "position_key": expanded_Date,
            }

        # Create the tensordict with agent data
        expanded_agent_observations = TensorDict(
            {"agents": agent_data},
            batch_size=self.batch_size,
            device=self.device
        )

        dones = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)

        # Construct the final tensordict
        return TensorDict(
            {
                **expanded_agent_observations,  # Include agent data
                "terminated": dones.clone()
            },
            batch_size=self.batch_size,
            device=self.device
        )




def make_composite_from_td(td):
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite

def _make_spec(self, td_agents):

    action_spec = []
    observation_spec = []
    reward_spec = []
    info_spec = []
    observat_spec_updated = CompositeSpec()
    Date_spec_updated = CompositeSpec()
    selfobservation_updated = CompositeSpec()
    reward_spec_updated = CompositeSpec()
    done_spec_updated = CompositeSpec()
    info_spec_updated = CompositeSpec()
    action_spec_updated = CompositeSpec()


    td_agents = self.gen_params()  # Or td_agents = self.gen_params() if no batch size is needed
    agent = {f"agent_{i}": {} for i in range(self.n_agents)}

    obs_max = td_agents['params', 'obsState&Fuel_max'].clone().detach()
    obs_min = td_agents['params', 'obsState&Fuel_min'].clone().detach()
    reward_max = td_agents['params', 'rewardState&reward_max'].clone().detach()
    reward_min = td_agents['params', 'rewardState&reward_min'].clone().detach()
    action_max = td_agents['params', 'actionState&action_max'].clone().detach()
    action_min = td_agents['params', 'actionState&action_min'].clone().detach()
    Date_max = td_agents['params', 'Date_max'].clone().detach()
    Date_min = td_agents['params', 'Date_min'].clone().detach()

    # Reshape the tensors to (n_agents, 13, *convo_dim) before the loop
    # Assuming obs_max, obs_min, reward_max, reward_min, action_max, action_min have shape (13,)
    # and Date_max, Date_min have shape (1,)

    obs_max = obs_max.reshape(1, 13, 1, 1).expand(self.n_agents, 13, *self.convo_dim)
    obs_min = obs_min.reshape(1, 13, 1, 1).expand(self.n_agents, 13, *self.convo_dim)
    reward_max = reward_max.reshape(1, 13, 1, 1).expand(self.n_agents, 13, *self.convo_dim)
    reward_min = reward_min.reshape(1, 13, 1, 1).expand(self.n_agents, 13, *self.convo_dim)
    action_max = action_max.reshape(1, 13, 1, 1).expand(self.n_agents, 13, *self.convo_dim)
    action_min = action_min.reshape(1, 13, 1, 1).expand(self.n_agents, 13, *self.convo_dim)
    Date_max = Date_max.reshape(1, 1, 1, 1).expand(self.n_agents, 1, *self.convo_dim)
    Date_min = Date_min.reshape(1, 1, 1, 1).expand(self.n_agents, 1, *self.convo_dim)


    for i in range(self.n_agents):
        # Creating action, reward, observation, and date specs for each agent
        agent[f"agent_{i}"]["action_spec"] = DiscreteTensorSpec(n=3, shape=action_max[i].shape, dtype=torch.float32)  # Use action_max[i].shape directly
        agent[f"agent_{i}"]["reward_spec"] = BoundedTensorSpec(low=reward_min[i], high=reward_max[i], shape=reward_max[i].shape, dtype=torch.float32)  # Use reward_min[i], reward_max[i], and reward_max[i].shape directly
        agent[f"agent_{i}"]["observat_spec"] = BoundedTensorSpec(low=obs_min[i], high=obs_max[i], shape=obs_max[i].shape, dtype=torch.float32)  # Use obs_min[i], obs_max[i], and obs_max[i].shape directly
        agent[f"agent_{i}"]["Date_spec"] = BoundedTensorSpec(low=Date_min[i], high=Date_max[i], shape=Date_max[i].shape, dtype=torch.float32)  # Use Date_min[i], Date_max[i], and Date_max[i].shape directly

        # Calculate the reward value by summing across the convolutional dimensions
        reward_value = torch.sum(agent[f"agent_{i}"]["reward_spec"].low, dim=0).reshape(tuple( (1,) + tuple(self.convo_dim))) # sum the reward.low along all axes other than batch axis.
        # Assuming low and high are the same, otherwise sum reward.high similarly and define BoundedTensorSpec
        agent[f"agent_{i}"]["reward_spec"]= BoundedTensorSpec(low=reward_value, high=reward_value, shape=reward_value.shape, dtype=torch.float32)
    # Creating CompositeSpecs for action, reward, observation, and date
    #action_spec_updated = CompositeSpec({k: v["action_spec"] for k, v in agent.items()})
    action_spec_updated =CompositeSpec({k: v["action_spec"] for k, v in agent.items()})
    reward_spec_updated = CompositeSpec({k: v["reward_spec"] for k, v in agent.items()})
    observat_spec_updated = CompositeSpec({k: v["observat_spec"] for k, v in agent.items()})
    Date_spec_updated = CompositeSpec({k: v["Date_spec"] for k, v in agent.items()})

    # Creating unbatched observation, action, and reward specs
    unbatched_observation_spec = CompositeSpec(
        agents=CompositeSpec(
            observat=observat_spec_updated,
            position_key=Date_spec_updated,
        )
    )
    unbatched_action_spec = CompositeSpec(action_spec_updated)
    unbatched_reward_spec =  CompositeSpec(reward_spec_updated)
    unbatched_done_spec = DiscreteTensorSpec(n=2, shape=torch.Size([1]), dtype=torch.bool).to(self.device)
    
    # Expanding specs to include batch size
   
    self.action_spec = unbatched_action_spec.expand(self.batch_size_tuple).to(self.device)
    self.observation_spec = unbatched_observation_spec.expand(self.batch_size_tuple).to(self.device)
    self.reward_spec = unbatched_reward_spec.expand(self.batch_size_tuple).to(self.device)
    self.done_spec = unbatched_done_spec.expand(self.batch_size_tuple).to(self.device)

    # Creating a CompositeSpec to represent the environment's spec
    return CompositeSpec(
        agents=CompositeSpec(
            observation=CompositeSpec(
                observat=self.observation_spec["agents"]["observat"],
                position_key=self.observation_spec["agents"]["position_key"],
            ),
            reward=self.reward_spec,

           envaction=self.action_spec,

        ),
        terminated=self.done_spec,
    )





def gen_params(batch_size=torch.Size()) -> TensorDictBase:
    if batch_size is None or (isinstance(batch_size, list) and len(batch_size) == 0):
        batch_size = torch.Size([])
    data_path = '/content/drive/MyDrive/deep learning codes/EIAAPI_DOWNLOAD/solutions/mergedata/DataDic.pt'
    data_columns = ['Forex', 'WTI', 'Brent', 'OPEC', 'Fuelprice5', 'Fuelprice6', 'Fuelprice7', 'Fuelprice8', 'Fuelprice9', 'Fuelprice10', 'Fuelprice11', 'Fuelprice12', 'Fuelprice13',
                    'reward0', 'reward1', 'reward2', 'reward3', 'reward4', 'reward5', 'reward6', 'reward7', 'reward8', 'reward9', 'reward10', 'reward11', 'reward12',
                    'action0', 'action1', 'action2', 'action3', 'action4', 'action5', 'action6', 'action7', 'action8', 'action9', 'action10', 'action11', 'action12', 'Date']
    envv = DDataenv(data_path, data_columns)  # Assuming DDataenv is your data environment class

    ac = envv.get_observation()

    if batch_size:
       # Change here: Convert batch_size to a tuple if it's not already
        if not isinstance(batch_size, tuple):
            batch_size = (batch_size,)  # Convert to tuple
        ac = {k: torch.tensor(v).expand(*batch_size, *torch.tensor(v).shape) for k, v in ac.items()}

    td = TensorDict({"params": ac}, batch_size=batch_size, device=torch.device("cpu" if torch.cuda.is_available() else "cpu"))
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td


def _set_seed(self, seed: 45):
    self.rng = torch.manual_seed(seed)


def full_info_spec(self):
    return {}


def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
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
            td_params = self.gen_params()

        _ = kwargs.pop("scenario", None)



        self.n_agents = 1
        self.convo_dim = [9, 9]
        self.batch_size = (10,10) # Corrected batch size to a single-element tuple
        self.batch_size_tuple = torch.Size([self.batch_size]) if isinstance(self.batch_size, int) else torch.Size(self.batch_size)
        



        super().__init__(device=device, batch_size=self.batch_size,)


        self._make_spec(td_params)

        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)



    def get_supports_continuous_actions(self):
        return hasattr(env.full_action_spec, 'shape') and len(env.full_action_spec.shape) > 0

    def get_supports_discrete_actions(self):
        return isinstance(env.full_action_spec, DiscreteTensorSpec)

    def get_observation_spec(self):
        return self.observation_spec

    def get_full_action_spec(self):
        return self.full_action_spec

    def get_full_reward_spec(self):
        return self.full_reward_spec

    def get_done_spec(self):
        return self.done_spec

    def get_group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return {"agents": [agent["name"] for agent in env.agents]}

    def get_full_info_spec(self):
        return {}

    def get_discount_spec(self):
        return self.discount_spec

    @property
    def terminated_spec(self):
        return self.done_spec

    @property
    def truncated_spec(self):
        return self.done_spec

    @property
    def get_env_name(self):
        return "AnFuelpriceEnv"

    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec
    _reset = _reset
    _step = staticmethod(_step)
    _set_seed = _set_seed
    full_info_spec = full_info_spec


env = AnFuelpriceEnv()

print("\n*action_spec:", env.full_action_spec)
print("\n*reward_spec:", env.full_reward_spec)
print("\n*done_spec:", env.full_done_spec)
print("\n*observation_spec:", env.observation_spec)

print("\n-action_keys:", env.action_keys)
print("\n-reward_keys:", env.reward_keys)
print("\n-done_keys:", env.done_keys)

print("input_spec:", env.input_spec)
print("reward_spec:", env.reward_spec)
print("action_spec (as defined by input_spec):", env.action_spec)
td = env.reset()
print("reset tensordict", td)



check_env_specs(env)



