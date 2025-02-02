
from typing import Dict as TypingDict, Any, Union, List, Optional
import torch
import numpy as np
from tensordict import TensorDict, TensorDictBase
from torchrl.modules import MultiAgentConvNet
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    UnsqueezeTransform,
    RewardSum,  # Importing RewardSum
    TransformedEnv,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
# Import DDataenv from DataDic.py

#anfuelpriceenv.py
# Import dependencies
import requests
import torch
import os
from torchrl.envs.transforms.transforms import _apply_to_composite
import pandas as pd

def is_valid_data(data):
    return isinstance(data, dict) or (isinstance(data, list) and all(isinstance(item, dict) for item in data))
from torchrl.envs.utils import check_env_specs # This line imports the missing function
import requests
import torch
from gym import spaces  # Import spaces from gym
from pickle import NONE # Import as pickle.NONE to avoid name conflicts



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



from types import new_class
from types import new_class
def _step(tensordict):
    n_agents = env.n_agents
    agent_new_obs_list=[]
    agent_reward_list=[]
    agent_action_list=[]
    agent_Date_list=[]




    #n_agents Iteration
    for j in range(n_agents):
        td=env.gen_params()
        obs=torch.reshape(td['params','obsState&Fuel'].clone().detach(),(13,))
        reward=torch.reshape(td['params','rewardState&reward'].clone().detach(),(13,))
        action=torch.reshape(td['params','actionState&action'].clone().detach(),(13,))
        Date=torch.tensor(td['params','Date'].clone().detach())
        new_obs = torch.add(obs, torch.stack([action_i * reward_i for action_i, reward_i in zip(action, reward)]))
        new_obs=torch.reshape(new_obs,(13,))


        agent_new_obs_list.append(new_obs)
        agent_reward_list.append(reward)
        agent_Date_list.append(Date)
        agent_action_list.append(action)



    

    agent_new_obs = torch.stack(agent_new_obs_list, dim=0) # shape: [n_agents, 13, 1]
    agent_reward = torch.stack(agent_reward_list, dim=0)
    agent_Date= torch.stack(agent_Date_list, dim=0)
    agent_action = torch.stack(agent_action_list, dim=0)





    #Convolution Expansions
    
    # Now you can safely expand with convo_dim along the middle dimension
   
    expanded_agent_new_obs = agent_new_obs.reshape(1, 13, 1, 1).expand( 1, 13, *env.convo_dim)
    expanded_agent_reward = agent_reward[:,4:].reshape(agent_reward.shape[0], 9, 1, 1).expand(n_agents, 9, *env.convo_dim)
    expanded_agent_action=  agent_action[:,4:].reshape(agent_action.shape[0], 9, 1, 1).expand(n_agents, 9, *env.convo_dim)
    expanded_agent_Date = agent_Date.reshape(agent_Date.shape[0], 1, 1, 1).expand(n_agents, 1, *env.convo_dim) 
    print( expanded_agent_Date.shape)
   


    expanded_agent_Date = agent_Date.expand(*agent_Date.shape,*env.convo_dim)



    #Batch Expansion
    expanded_agent_reward1=expanded_agent_reward.expand(*env.batch_size,*expanded_agent_reward.shape)
    expanded_agent_new_obs1=expanded_agent_new_obs.expand(*env.batch_size, *expanded_agent_new_obs.shape)
   
    expanded_agent_Date = agent_Date.reshape(agent_Date.shape[0], 1, 1, 1).expand(agent_Date.shape[0], 1, *env.convo_dim)
    expanded_agent_Date1 = expanded_agent_Date.expand(*env.batch_size, *expanded_agent_Date.shape)  



    episode_reward=expanded_agent_reward1
    observation=expanded_agent_new_obs1
    Date=expanded_agent_Date1
    reward = expanded_agent_reward1
 # Adjust slicing if necessary





    dones = torch.zeros((*env.batch_size,1), dtype=torch.bool)
    nextt = TensorDict({
        "agents": {
            "observation":{"observat":observation,"position_key": Date},
            "reward": reward,
           # "action": action,
        },
        "terminated": dones.clone(),

    }, batch_size=env.batch_size, device=env.device)
    return nextt

def _reset(self, tensordict=None, **kwargs):
    if tensordict is None:
       batch_size=self.batch_size
       td = self.gen_params()


       obs_max=td['params','obsState&Fuel_max'].clone().detach()
       obs_min=td['params','obsState&Fuel_min'].clone().detach()
       action_max=td['params','actionState&action_max'].clone().detach()
       reward_max=td['params','rewardState&reward_max'].clone().detach()
       Date=td['params','Date_max'].clone().detach()

       n_agents = self.n_agents

          # Initialize agent list here
       batchtentensor=[]

       agents = []
       agent_obs_list = [] # Collect observations in a list first
       agent_tds = []
       agent_obs_tensor=[]
       agent_Date_list=[]
       agent_Date_tensor=[]

          # Iterate over the DataFrame
       low_x=[]
       high_x=[]
       obs=[]
       #unbatch
       random_numbers = torch.rand((1,), generator=self.rng, device=self.device) # Changed random number generation to accommodate batch_size and n_agents

       high_x=obs_max.unsqueeze(0)
       low_x=obs_min.unsqueeze(0)



       for i in range(n_agents):
            obs = torch.add(torch.mul(random_numbers, torch.add(high_x[i,:], -low_x[i,:])), low_x[i,:])  # Added unsqueeze to match dimensions
            agent_obs_list.append(obs)
            Date=Date
            agent_Date_list.append(Date)


       agent_obs_tensor = torch.stack(agent_obs_list, dim=0)
       agent_Date_tensor = torch.stack(agent_Date_list, dim=0).float()

      

       agent_obs_tensor = torch.stack(agent_obs_list, dim=0).float()
       agent_Date_tensor = torch.stack(agent_Date_list, dim=0).float()

      


       agent_obs_tensor = agent_obs_tensor.reshape(agent_obs_tensor.shape[0], agent_obs_tensor.shape[1], 1, 1) # Adding dimensions for the convo_dim
       agent_obs_tensor = agent_obs_tensor.expand(agent_obs_tensor.shape[0], agent_obs_tensor.shape[1], *self.convo_dim) # Expand to include convo_dim
       agent_Date_tensor =agent_Date_tensor.reshape(agent_Date_tensor.shape[0], agent_Date_tensor.shape[-1], 1, 1) # Adding dimensions for the convo_dim
       agent_Date_tensor = agent_Date_tensor.expand(agent_Date_tensor.shape[0], 1, *self.convo_dim) # Expand to include convo_dim

       
       # Adjust the expansion for expanded_agent_Date_tensor# Reshape and expand agent_Date_tensor to match expected shape
       expanded_agent_Date_tensor = agent_Date_tensor.expand(*self.batch_size,*agent_Date_tensor.shape) # Reshape to (1, 1) and expand
       expanded_agent_obs_tensor = agent_obs_tensor.expand(*self.batch_size, *agent_obs_tensor.shape) # expand obs to match the batch size
       print(expanded_agent_Date_tensor.shape)



    # Corrected the shape of dones to match the batch size
    dones =torch.zeros((*self.batch_size,1), dtype=torch.bool)

    resett = TensorDict(
        {
            "agents": {
                "observation": {"observat": expanded_agent_obs_tensor, "position_key": expanded_agent_Date_tensor},

            },
            "terminated": dones.clone(),
        },
        batch_size=self.batch_size,
        device=self.device,
    )
    return resett



# @title Default title text
def _make_spec(self, td_agents):
    agent =[{}]*self.n_agents
    action_specs = []
    observation_specs = []
    reward_specs = []
    Date_specs=[]



    # Initialize result lists outside the loop
    obs_max = torch.reshape(td_agents['params', 'obsState&Fuel_max'].clone().detach(), (13,))
    obs_min = torch.reshape(td_agents['params', 'obsState&Fuel_min'].clone().detach(), (13,))
    reward_max = torch.reshape(td_agents['params', 'rewardState&reward_max'].clone().detach(), (13,))
    reward_min = torch.reshape(td_agents['params', 'rewardState&reward_min'].clone().detach(), (13,))
    action_max = torch.reshape(td_agents['params', 'actionState&action_max'].clone().detach(), (13,))
    action_min = torch.reshape(td_agents['params', 'actionState&action_min'].clone().detach(), (13,))
    Date_max=td_agents['params','Date_max'].clone().detach()
    Date_min=td_agents['params','Date_min'].clone().detach()



    for i in range(self.n_agents):
        agent[i]["action_spec"] =  DiscreteTensorSpec( n=3,
                                                     shape= tuple([dim for dim in self.batch_size] if self.batch_size else [1]) + (self.n_agents)+(13,1),
                                                     dtype=torch.float32),

        agent[i]["reward_spec"] =  BoundedTensorSpec(low = reward_min[i],
                                                     high = reward_max[i],
                                                     shape= tuple([dim for dim in self.batch_size] if self.batch_size else [1]) + (self.n_agents)+(13,1),
                                                     dtype=torch.float32),


        agent[i]["observat_spec"]  = BoundedTensorSpec(low = obs_min[i],
                                                          high =obs_max[i],
                                                          shape= tuple([dim for dim in self.batch_size] if self.batch_size else [1]) + (self.n_agents)+(13,1),
                                                          dtype=torch.float32),

        agent[i]["Date_spec"]  = BoundedTensorSpec(low = Date_min,
                                                          high =Date_max,
                                                          shape= tuple([dim for dim in self.batch_size] if self.batch_size else [1]) + (self.n_agents,1),
                                                          dtype=torch.float32),

        action_specs.append(agent[i]["action_spec"])
        reward_specs.append(agent[i]["reward_spec"])
        observation_specs.append(agent[i]["observation_spec"])
        Date_specs.append(agent[i]["Date_spec"])



# Construct CompositeSpec objects with the correct nesting and batch size
def _make_spec_updated(self, td_agents):
    agent = [{}] * self.n_agents
    action_specs = []
    observation_specs = []
    reward_specs = []

    # Initialize result lists outside the loop
    obs_max = torch.reshape(td_agents['params', 'obsState&Fuel_max'].clone().detach(), (13,))
    obs_min = torch.reshape(td_agents['params', 'obsState&Fuel_min'].clone().detach(), (13,))
    reward_max = torch.reshape(td_agents['params', 'rewardState&reward_max'].clone().detach(), (13,))
    reward_min = torch.reshape(td_agents['params', 'rewardState&reward_min'].clone().detach(), (13,))
    reward_max=reward_max[4:,]
    reward_min=reward_min[4:,]
    action_max = torch.reshape(td_agents['params', 'actionState&action_max'].clone().detach(), (13,))
    action_min = torch.reshape(td_agents['params', 'actionState&action_min'].clone().detach(), (13,))
    action_max=action_max[4:,]
    action_min=action_min[4:,]
    Date_max = td_agents['params', 'Date_max'].clone().detach()
    Date_min = td_agents['params', 'Date_min'].clone().detach()

    # Initialize result lists with the correct shape
    result77 = [Date_max for _ in range(self.n_agents)]  # Use reshaped and squeezed Date_max
    result66 = [Date_min for _ in range(self.n_agents)]  # Use reshaped and squeezed Date_min
    result55 = [action_min for _ in range(self.n_agents)]
    result44 = [action_max for _ in range(self.n_agents)]
    result33 = [reward_min for _ in range(self.n_agents)]
    result22 = [reward_max for _ in range(self.n_agents)]
    result11 = [obs_min for _ in range(self.n_agents)]
    result00 = [obs_max for _ in range(self.n_agents)]

    expand_shape = (self.n_agents, 13, *self.convo_dim)  # Change here
    expand_shape1 = (self.n_agents, 1,*self.convo_dim)  # Change here




    result555 = action_min.reshape(self.n_agents, 9, 1, 1).expand(self.n_agents, 9, *self.convo_dim)
    result444 = action_max.reshape(self.n_agents, 9, 1, 1).expand(self.n_agents, 9, *self.convo_dim)
    # For obs/rewards that may have 13 features:
    result333 = reward_min.reshape(self.n_agents, 9, 1, 1).expand(self.n_agents, 9, *self.convo_dim)
    result222 = reward_max.reshape(self.n_agents, 9, 1, 1).expand(self.n_agents, 9, *self.convo_dim)
     # Reshape to (1, 13) and then expand to (n_agents, 13, *convo_dim)
    result111 = obs_min.reshape(self.n_agents, 13, 1, 1).expand(self.n_agents, 13, *self.convo_dim)
    result000 = obs_max.reshape(self.n_agents, 13, 1, 1).expand(self.n_agents, 13, *self.convo_dim)



     #Reshape tensors before expanding with the correct number of features
    result777 = Date_max.reshape(self.n_agents, 1, 1, 1).expand(*expand_shape1)  # Use expand_shape for Date
    result666 = Date_min.reshape(self.n_agents, 1, 1, 1).expand(*expand_shape1)  # Use expand_shape for Date





    self.unbatched_action_spec = CompositeSpec(
        {"agents": {"action": DiscreteTensorSpec( n=3,
                shape=result555.shape,
                dtype=torch.float32,
            )}}
    )

    self.unbatched_reward_spec = CompositeSpec(
         {"agents": {"reward": BoundedTensorSpec(
                low=result333,
                high=result222,
                shape=result333.shape,
                dtype=torch.float32,
            )}}
    )


    self.unbatched_observation_spec = CompositeSpec(
        {"agents": {"observation": { # Add a nested dictionary for observation
                "observat": BoundedTensorSpec(
                    low=result111,
                    high=result000,
                    shape=result111.shape,
                    dtype=torch.float32
                ),
                "position_key": BoundedTensorSpec(  # Include "Date" within "observation"
                    low=result666,
                    high=result777,
                    shape=result666.shape,
                    dtype=torch.float32
                )
            }
        }}
    )





    # Now you can expand the specs
    self.unbatched_done_spec = DiscreteTensorSpec(
        n=2, shape=torch.Size((1,)), dtype=torch.bool  # Change shape to (1,)
    ).to(self.device)

     # Expanded batch size should match env.batch_size
    expanded_batch_size = tuple([dim for dim in self.batch_size] if self.batch_size else [1])
    self.action_spec = self.unbatched_action_spec.expand(
        *expanded_batch_size  # Remove *self.unbatched_action_spec.shape
    ).to(self.device)
    self.observation_spec = self.unbatched_observation_spec.expand(
        *expanded_batch_size  # Remove *self.unbatched_observation_spec.shape
    ).to(self.device)
    self.reward_spec = self.unbatched_reward_spec.expand(
        *expanded_batch_size  # Use expanded_batch_size for reward_spec as well
    ).to(self.device)
    self.done_spec = self.unbatched_done_spec.expand(
        *expanded_batch_size  # Use expanded_batch_size for done_spec as well
    ).to(self.device)
    return self.action_spec, self.observation_spec, self.reward_spec, self.done_spec
def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
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


def gen_params(batch_size=torch.Size()) -> TensorDictBase:
    """Returns a ``tensordict`` containing the input tensors."""
    if batch_size is None:
      batch_size = []
     #Instantiate the environment with your data
    data_path = '/content/drive/MyDrive/deep learning codes/EIAAPI_DOWNLOAD/solutions/mergedata/DataDic.pt'  # Replace with your actual data path
    data_columns = ['Forex','WTI','Brent','OPEC','Fuelprice5','Fuelprice6','Fuelprice7','Fuelprice8','Fuelprice9','Fuelprice10','Fuelprice11','Fuelprice12','Fuelprice13',
                          'reward0','reward1','reward2','reward3','reward4','reward5','reward6','reward7','reward8','reward9','reward10','reward11','reward12',
                          'action0','action1','action2','action3','action4','action5','action6','action7','action8','action9','action10','action11','action12','Date']  # Add all your column names
    envv = DDataenv(data_path, data_columns) # Create an instance of the class

    # Get an observation
    ac = envv.get_observation()

    if batch_size:
        # Assuming 'ac' is a dictionary of tensors, expand each tensor
      ac = {k: torch.tensor(v).expand(*batch_size, *torch.tensor(v).shape)  for k, v in ac.items()} # Convert lists to tensors before expanding


    td = TensorDict({

          "params": ac,
          }
        ,
        batch_size=batch_size,
        device=torch.device("cpu" if torch.cuda.is_available() else "cpu"),
    )
    if batch_size:
      td = td.expand(batch_size).contiguous()
    return td



def _set_seed(self, seed:45):
    rng = torch.manual_seed(seed)
    self.rng = rng
    

class AnFuelpriceEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False
    def __init__(self,td_params=None, seed=None, device="cpu"):
        if td_params is None:
           td_params = self.gen_params()


        # Extract the variables needed in _make_spec
        self.n_agents = 1
        self.convo_dim = [9, 9]
        self.batch_size = [10, 10]
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.unbatched_observation_spec = None
        self.unbatched_reward_spec = None




        self.agent_tds = []
        self.agents = [{} for _ in range(self.n_agents)]






        super().__init__(device=device, batch_size=[10,10])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # Helpers: _make_step and gen_params
    gen_params =staticmethod(gen_params)
    _make_spec = _make_spec_updated  # w

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = staticmethod(_step)
    _set_seed = _set_seed
    def __getattr__(self, name):
        if name == 'supports_continuous_actions':
                # Check if the action space is continuous:
            if isinstance(self.action_space, spaces.Box) and np.issubdtype(self.action_space.dtype, np.floating):
                return True  
            else:
                return False
        elif:
            # If the attribute is not found, raise an AttributeError
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        else:
            pass

    #Define action_spec
    @property
    def action_spec(self):
        if isinstance(self.action_space, spaces.Box):
              # Assuming continuous action space, adjust bounds as needed
            return BoundedTensorSpec(
                low=torch.tensor(self.action_space.low),
                high=torch.tensor(self.action_space.high),
                dtype=torch.float32,
                shape=self.action_space.shape
            )
        elif isinstance(self.action_space, spaces.Discrete):
              # Assuming discrete action space
            return DiscreteTensorSpec(n_actions=self.action_space.n)
        else:
            raise NotImplementedError(f"Unsupported action space type: {type(self.action_space)}")    __getattr__=__getattr__

env = AnFuelpriceEnv()
print("\n*action_spec:", env.full_action_spec)
print("\n*reward_spec:", env.full_reward_spec)
print("\n*done_spec:", env.full_done_spec)
print("\n*observation_spec:", env.observation_spec)

print("\n-action_keys:", env.action_keys)
print("\n-reward_keys:", env.reward_keys)
print("\n-done_keys:", env.done_keys)

print("input_spec:", env.input_spec)
print("action_spec (as defined by input_spec):", env.action_spec)
print("reward_spec:", env.reward_spec)
td = env.reset()
print("reset tensordict", td)
check_env_specs(env)

"""Vectorization& Abstracting features from VMAS   Using VMAS WRAPPING"""
##############################################################################################################################################################
from typing import Union, Optional
import torch
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec

def make_env(
    scenario: Union[str, "AnFuelpriceEnv"],
    num_envs: int = 1,
    device: Union[torch.device, str, int] = "cpu",
    continuous_actions: bool = False,
    wrapper: Optional["vmas.simulator.environment.Wrapper"] = True,
    max_steps: Optional[int] = 20,
    seed: Optional[int] = None,
    dict_spaces: bool = False,
    multidiscrete_actions: bool = True,
    clamp_actions: bool = False,
    grad_enabled: bool = False,
    **kwargs  # Environment specific kwargs
):
    """
    Create a vectorized environment with the specified configuration.

    # ... (rest of the docstring)
    """
    # ... (rest of your code)
    # Assuming observation_spec is defined in AnFuelpriceEnv
    if isinstance(scenario, str):
        scenario = AnFuelpriceEnv()  # Instantiate env if scenario is a string

    # Assign observation_spec to the function
    make_env.observation_spec = scenario.observation_spec

    return scenario  # or some other appropriate environment

# Call the function before accessing the attribute
# Make sure to pass an appropriate 'scenario'
env = make_env(scenario='your_scenario')  # Replace 'your_scenario' with an actual scenario




