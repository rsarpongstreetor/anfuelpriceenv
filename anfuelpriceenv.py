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

import requests
import torch



data_path="https://drive.google.com/uc?id=1K7OBG-qZnVC4Sm7-zwLqIXTmNRLYe02e" #fixed the link

response = requests.get(data_path)
response.raise_for_status() # Raise an exception for bad status codes

with open("temp_file.pt", 'wb') as f:
  f.write(response.content)

with open("temp_file.pt", 'rb') as f:
  DataDic = torch.load(f, map_location=torch.device('cpu'))  # Load with map_location
  # Assuming DDataenv is at index 0 in the list
  DDataenv = DataDic[0] if isinstance(DataDic, list) and len(DataDic) > 0 else None
  # Add a check to handle cases where DDataenv might not be present
  if DDataenv is None:
    print("Warning: DDataenv not found in the loaded data.")
import os
os.remove("temp_file.pt")



# Access DDataenv from the loaded DataDic (assuming DDataenv is a key in DataDic)
DDataenv =np.array(DDataenv)
DDataenv=DataDic(DDataenv)

# ... (rest of the code)


from types import new_class
def _step(tensordict):
    n_agents = env.n_agents
    agent_new_obs_list=[]
    agent_reward_list=[]
    agent_action_list=[]
    agent_Date_list=[]
    #expanded_agent_Date
    #expanded_agent_new_obs1
    #expanded_agent_reward1



    #n_agents Iteration
    for j in range(n_agents):
        td=env.gen_params()
        obs=torch.reshape(td['params','obsState&Fuel'].clone().detach(),(13,))
        reward=torch.reshape(td['params','rewardState&reward'].clone().detach(),(13,))
        action=torch.reshape(td['params','actionState&action'].clone().detach(),(13,))
        Date=torch.reshape(td['params','Date'].clone().detach(),(1,))
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


    #agent_reward = agent_reward.reshape(*agent_reward.shape)
    # Now you can safely expand with convo_dim along the middle dimension


    #Convolution Expansions
    expanded_agent_new_obs = agent_new_obs.reshape(1, 13, 1, 1).expand( 1, 13, *env.convo_dim)
    expanded_agent_reward = agent_reward[:,4:].reshape(agent_reward.shape[0], 9, 1, 1).expand(n_agents, 9, *env.convo_dim)
    expanded_agent_action=  agent_action[:,4:].reshape(agent_action.shape[0], 9, 1, 1).expand(n_agents, 9, *env.convo_dim)
    expanded_agent_Date = agent_Date[:,:].reshape(agent_Date.shape[0], 1, 1, 1).expand(n_agents, 1, *env.convo_dim)  # Reshape to (1, 1, 1, 1)




    #Batch Expansion

    expanded_agent_new_obs1=expanded_agent_new_obs.expand(*env.batch_size, *expanded_agent_new_obs.shape)
    expanded_agent_reward1=expanded_agent_reward.expand(*env.batch_size,*expanded_agent_reward.shape)
    expanded_agent_action1=expanded_agent_action.expand(*env.batch_size,*expanded_agent_action.shape)
    expanded_agent_Date1 = expanded_agent_Date.expand(*env.batch_size, *expanded_agent_Date.shape)


    observation=expanded_agent_new_obs1
    Date=expanded_agent_Date1
    reward = expanded_agent_reward1
    action=expanded_agent_action1

 # Adjust slicing if necessary
   





    dones = torch.zeros((*env.batch_size,1), dtype=torch.bool)
    nextt = TensorDict({
        "agents": {
            "observation":{"observat":observation,"position_key": Date},
            "reward": reward,
          #  "action": action,
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

       agent_obs_tensor=agent_obs_tensor.float() # Added unsqueeze to add a new dimension


       agent_obs_tensor = agent_obs_tensor.reshape(agent_obs_tensor.shape[0], agent_obs_tensor.shape[1], 1, 1) # Adding dimensions for the convo_dim
       agent_obs_tensor = agent_obs_tensor.expand(agent_obs_tensor.shape[0], agent_obs_tensor.shape[1], *self.convo_dim) # Expand to include convo_dim
       agent_Date_tensor = agent_Date_tensor.expand(agent_Date_tensor.shape[0], 1, *self.convo_dim) # Expand to include convo_dim
       
       expanded_agent_obs_tensor = agent_obs_tensor.expand(*self.batch_size, *agent_obs_tensor.shape) # expand obs to match the batch size
       # Adjust the expansion for expanded_agent_Date_tensor# Reshape and expand agent_Date_tensor to match expected shape
       expanded_agent_Date_tensor = agent_Date_tensor.expand(*self.batch_size,*agent_Date_tensor.shape) # Reshape to (1, 1) and expand






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
        agent[i]["action_spec"] =  DiscreteTensorSpec(n=3,
                                                     shape= tuple([dim for dim in self.batch_size] if self.batch_size else [1]) + (self.n_agents)+(13,1),
                                                     dtype=torch.float32),

        agent[i]["reward_spec"] =  BoundedTensorSpec(low = reward_min[i],
                                                     high = reward_max[i],
                                                     shape= tuple([dim for dim in self.batch_size] if self.batch_size else [1]) + (self.n_agents)+(13,1),
                                                     dtype=torch.float32),


        agent[i]["observation_spec"]  = BoundedTensorSpec(low = obs_min[i],
                                                          high =obs_max[i],
                                                          shape= tuple([dim for dim in self.batch_size] if self.batch_size else [1]) + (self.n_agents)+(13,1),
                                                          dtype=torch.float32),

       #
        action_specs.append(agent[i]["action_spec"])
        reward_specs.append(agent[i]["reward_spec"])
        observation_specs.append(agent[i]["observation_spec"])




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
    expand_shape1 = (self.n_agents,1, *self.convo_dim)  # Change here




    result555 = action_min.reshape(9).expand(self.n_agents, 9, *self.convo_dim)
    result444 = action_max.reshape(9).expand(self.n_agents, 9, *self.convo_dim)
    # For obs/rewards that may have 13 features:
    result333 = reward_min.reshape(9).expand(self.n_agents, 9, *self.convo_dim)
    result222 = reward_max.reshape(9).expand(self.n_agents, 9, *self.convo_dim)
     # Reshape to (1, 13) and then expand to (n_agents, 13, *convo_dim)
    result111 = obs_min.reshape(1, 13, 1, 1).expand(self.n_agents, 13, *self.convo_dim)
    result000 = obs_max.reshape(1, 13, 1, 1).expand(self.n_agents, 13, *self.convo_dim)



     #Reshape tensors before expanding with the correct number of features
    result777 = Date_max.expand(*expand_shape1)  # Use expand_shape for Date
    result666 = Date_min.expand(*expand_shape1)  # Use expand_shape for Date



    self.unbatched_action_spec = CompositeSpec(
        {"agents": {"action": DiscreteTensorSpec(
            n=3,
                shape=result555.shape,  # Changed to 9 actions,
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
       ac = {k: v.clone().detach().expand(*batch_size, *v.shape)  for k, v in ac.items()} # Convert lists to tensors before expanding

    td = TensorDict({

        "params": ac,
          },
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


