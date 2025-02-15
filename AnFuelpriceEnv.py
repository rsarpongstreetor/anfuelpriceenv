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
import google.colab
from typing import Dict as TypingDict, Any, Union, List, Optional
# Mount Google Drive
google.colab.drive.mount('/content/drive')

def is_valid_data(data):
    return isinstance(data, dict) or (isinstance(data, list) and all(isinstance(item, dict) for item in data))
from torchrl.envs.utils import check_env_specs # This line imports the missing function
import requests
import torch
from gym import spaces  # Import spaces from gym
from pickle import NONE # Import as pickle.NONE to avoid name conflict



class DDataenv:
    def __init__(self, data_path: str, data_columns: List[str], data_type: Any = np.float32):
        self.data_path = data_path
        self.data_columns = data_columns
        self.data_type = data_type
        self.data = None

    def load_data(self) -> pd.DataFrame:
        with open(self.data_path, 'rb') as f:
            self.data = torch.load(f,weights_only=False )
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
   ####################################################################################
        """ with open(self.data_path, 'rb') as f:
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
            self.data = pd.DataFrame(self.data, columns=self.data_columns)"""
        ######################################################################
        """Loads data from the specified path."""
        # Download the file if it doesn't exist locally
        """ if not os.path.exists(self.data_path.split('/')[-1]): # Check for file existence locally
            import requests
            print(f"Downloading data from {self.data_path}...") # Indicate download start
            response = requests.get(self.data_path, stream=True)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            with open(self.data_path.split('/')[-1], 'wb') as f: # Save with filename from URL
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.") # Indicate download end

        # Now load the data
        with open(self.data_path.split('/')[-1], 'rb') as f: # Load local file
            self.data = torch.load(f, weights_only=False) # Assuming it's a PyTorch file
            self.data = np.array(self.data) # Convert to NumPy array
        return self.data"""

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
def _step(tensordict):
    td=env.gen_params()
    n_agents = env.n_agents
    
    agent_new_obs_list=[]
    agent_reward_list=[]
    agent_Date_list=[]
    agent_action_list=[]

    #n_agents Iteration
    for j in range(n_agents):
        # Initialize lists to store data for each agent within the batch
        agent_i_Date = []
        agent_i_action = []
        agent_i_rew = []
        agent_i_new_obs = []


        for convo_dim in range(env.convo_dim[0]): #Assuming batch_size is a tuple or list, use the first element
           # Generate parameters for the current batch index:
          current_td = env.gen_params(batch_size=[1])  #Batch size of 1 for each iteration

          obs=torch.reshape(td['params','obsState&Fuel'].clone().detach(),(13,))
          reward=torch.reshape(td['params','rewardState&reward'].clone().detach(),(13,))
          action=torch.reshape(td['params','actionState&action'].clone().detach(),(13,))
          Date=torch.reshape(td['params','Date'].clone().detach(),(1,))

          new_obs = torch.add(obs, torch.stack([action_i * reward_i for action_i, reward_i in zip(action, reward)]))
          new_obs=torch.reshape(new_obs,(13,))

           # Append updated data to lists:
          agent_new_obs_list.append(new_obs)
          agent_reward_list.append(reward)
          agent_Date_list.append(Date)
          agent_action_list.append(action)

       # Prepare data for next step:
        agent_new_obs = torch.stack(agent_new_obs_list, dim=-1)  # shape: [13,convo_dim[0]]
        agent_reward = torch.stack(agent_reward_list, dim=-1)
        agent_Date = torch.stack(agent_Date_list, dim=-1)
        agent_action = torch.stack(agent_action_list, dim=-1)

        #Agentiteration
        agent_i_Date.append(agent_Date)   # shape: [ n_agent, 13, convo_dim[0]]
        agent_i_action.append(agent_action)
        agent_i_rew.append(agent_reward)
        agent_i_new_obs.append(agent_new_obs)
    en_Date=torch.stack(agent_i_Date, dim=0)
    en_action=torch.stack(agent_i_action, dim=0)
    en_reward=torch.stack(agent_i_rew, dim=0)
    en_new_obs=torch.stack(agent_i_new_obs, dim=0)





    #Convolution Expansions

    # Now you can safely expand with convo_dim[1] along the -1 dimension

    expanded_agent_new_obs = en_new_obs.reshape(env.n_agents, 13, env.convo_dim[0], 1).expand(env.n_agents, 13, *env.convo_dim)
    expanded_agent_reward = en_reward.reshape(env.n_agents, 13, env.convo_dim[0], 1).expand(env.n_agents, 13, *env.convo_dim) 
    expanded_agent_action=  en_action.reshape( env.n_agents, 13, env.convo_dim[0], 1).expand(env.n_agents, 13, *env.convo_dim) 
    expanded_agent_Date = en_Date.reshape(env.n_agents, 1, env.convo_dim[0], 1).expand(env.n_agents, 1, *env.convo_dim)


           
       

    #batch_size Expansion
    expanded_agent_reward1 = expanded_agent_reward.expand(tuple(env.batch_size) + expanded_agent_reward.shape)  # Change to tuple for batch_size
    expanded_agent_new_obs1 = expanded_agent_new_obs.expand(tuple(env.batch_size) + expanded_agent_new_obs.shape)  # Change to tuple for batch_size
    expanded_agent_action1 = expanded_agent_action.expand(tuple(env.batch_size) + expanded_agent_action.shape)  # Change to tuple for batch_size
    expanded_agent_Date1 = expanded_agent_Date.expand(tuple(env.batch_size) + expanded_agent_Date.shape)  # Change to tuple for batch_size
    
    
   
  
    observation=expanded_agent_new_obs1
    Date=expanded_agent_Date1
    reward = expanded_agent_reward1
    action = TensorDict({"agents": {"action": expanded_agent_action1}}, env.batch_size)
  



    dones =torch.zeros(*env.batch_size, env.n_agents, dtype=torch.bool, device=env.device)

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




    # Corrected the shape of dones to match the batch size

    dones =torch.zeros(*self.batch_size, self.n_agents,dtype=torch.bool,device=self.device)

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


         # Define single_observation_space using torchrl's TensorSpec
        # Assuming your observation space is continuous and unbounded:
        # self.single_observation_space = UnboundedContinuousTensorSpec(
        #     shape=self.observation_space['agents']['observation']['observation'].space.shape,
        #     dtype=torch.float32,
        #     device=self.device
        # )
    
    
    self.single_observation_space = BoundedTensorSpec(
            low=obs_min,
            high=obs_max,
            shape=obs_min.shape,  # Assuming low and high have the same shape
            dtype=torch.float32,
            device=self.device )
        

    for i in range(self.n_agents):
        agent[i]["action_spec"] =  DiscreteTensorSpec( n=3,
                                                    shape= (13, *self.convo_dim),
                                                     dtype=torch.float32),

        agent[i]["reward_spec"] =  BoundedTensorSpec(low = reward_min[i],
                                                     high = reward_max[i],
                                                     shape= (13, *self.convo_dim),
                                                     dtype=torch.float32),


        agent[i]["observat_spec"]  = BoundedTensorSpec(low = obs_min[i],
                                                          high =obs_max[i],
                                                          shape= (13, *self.convo_dim),
                                                          dtype=torch.float32),

        agent[i]["Date_spec"]  = BoundedTensorSpec(low = Date_min,
                                                          high =Date_max,
                                                          shape=(1, *self.convo_dim),
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

    action_max = torch.reshape(td_agents['params', 'actionState&action_max'].clone().detach(), (13,))
    action_min = torch.reshape(td_agents['params', 'actionState&action_min'].clone().detach(), (13,))

    Date_max = td_agents['params', 'Date_max'].clone().detach()
    Date_min = td_agents['params', 'Date_min'].clone().detach()

   

    

    # Ensure result variables have the correct shape
    result555 = action_min.reshape(self.n_agents, 13,1,1).expand(self.n_agents, 13,*self.convo_dim)  # Modified to (1, 13)
    result444 = action_max.reshape(self.n_agents, 13,1,1).expand(self.n_agents, 13,*self.convo_dim)  # Modified to (1, 13)
    result333 = reward_min.reshape(self.n_agents, 13,1,1).expand(self.n_agents, 13,*self.convo_dim)  # Modified to (1, 13)
    result222 = reward_max.reshape(self.n_agents, 13,1,1).expand(self.n_agents, 13,*self.convo_dim)  # Modified to (1, 13)
    result111 = obs_min.reshape(self.n_agents, 13,1,1).expand(self.n_agents, 13,*self.convo_dim)  # Modified to (1, 13)
    result000 = obs_max.reshape(self.n_agents, 13,1,1).expand(self.n_agents, 13,*self.convo_dim)  # Modified to (1, 13)
    result777 = Date_max.reshape(self.n_agents,1,1,1).expand(self.n_agents, 1,*self.convo_dim) # Modified to (1, 1)
    result666 = Date_min.reshape(self.n_agents,1,1,1).expand(self.n_agents, 1,*self.convo_dim) # Modified to (1, 1)






    self.unbatched_action_spec = CompositeSpec(
        {"agents": {"action": DiscreteTensorSpec(
            n=3,
            shape=torch.Size([13, *self.convo_dim]), 
            dtype=torch.float32,
            )}} ,
    )

    self.unbatched_reward_spec = CompositeSpec(
         {"agents": {"reward": BoundedTensorSpec(
                low=result333,
                high=result222,
                shape=result222.shape,
                dtype=torch.float32,
            )}},
    )


    self.unbatched_observation_spec = CompositeSpec(
        {"agents": {"observation": { # Add a nested dictionary for observation
                "observat": BoundedTensorSpec(
                    low=result111,
                    high=result000,
                    shape=result000.shape,
                    dtype=torch.float32
                ),
                "position_key": BoundedTensorSpec(  # Include "Date" within "observation"
                    low=result666,
                    high=result777,
                    shape=result666.shape,
                    dtype=torch.float32
                )
            }
        }},
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

def full_info_spec(self):
        # Implementation of your full_info_spec logic here
        # For example, if it should always return an empty dictionary:
    return {}

from typing import Dict, List # Import Dict and List here
def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        # The group map mapping group names to agent names
        # The data in the tensordict will have to be presented this way
    return {"agents": [agent.name for agent in env.agents]}











class AnFuelpriceEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    # You can use the scenario argument here if needed
    Scenario="USDATA_1"
    batch_locked = False
    def __init__(self,td_params=None, seed=None, device="cpu", categorical_actions=True,continuous_actions=True,**kwargs):

        if td_params is None:
           td_params = self.gen_params()


        
        
  
       
        _ = kwargs.pop("scenario", None)
        # Extract the variables needed in _make_spec
       
        self.n_agents = 1
        self.convo_dim = [9, 9]
        self.batch_size = [10, 10]
       



        self.unbatched_observation_spec = None
        self.unbatched_reward_spec = None
        self.agent_tds = []
        #self.agents = [{"name": f"agent_{i}"} for i in range(self.n_agents)]
        self.agents = [{"name": "USDATA"}] # Change here: Make agents a list of dictionaries with a "name" key


        







       # Convert batch_size to tuple of integers, NOT torch.Size:
        self.batch_size_tuple = tuple(self.batch_size)
 

        super().__init__(device=device, batch_size=self.batch_size)
        
        self._make_spec(td_params)
        self.single_observation_space = self.observation_spec["agents"]["observation"]["observat"] # Assuming "observat" is the correct key for your single observation space

        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        
         

    def get_supports_continuous_actions(self):
        from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
          # Check if your environment supports continuous actions
          # and return True or False accordingly
          # For example, if your environment uses Box action spaces:
        return hasattr(env.full_action_spec, 'shape') and len(env.full_action_spec.shape) > 0

    def get_supports_discrete_actions(self):
        from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
          # Check if your environment supports continuous actions
          # and return True or False accordingly
          # For example, if your environment uses Box action spaces:
        return isinstance(env.full_action_spec, DiscreteTensorSpec) #fixed indentation here by ensuring it aligns with the 'return' statement


    @property
    def get_env_name():
        return "AnFuelpriceEnv"

    def get_observation_spec(self):
        return self.observation_spec

    def get_full_action_spec(self):
        return self.full_action_spec

    def get_full_reward_spec(self):
        return self.full_reward_spec

    def get_done_spec(self):
        return self.done_spe
    def get_group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        # The group map mapping group names to agent names
        # The data in the tensordict will have to be presented this way
       return {"agents": [agent["name"] for agent in env.agents]} #Access 'name' using dictionary key

    @property
    def terminated_spec(self):
        return self.done_spec
    @property
    def truncated_spec(self):
        return self.done_spec

    def get_full_info_spec(self):
        return {}

    def get_discount_spec(self):
        return self.discount_spec

    # Helpers: _make_step and gen_params
    gen_params =staticmethod(gen_params)
    _make_spec = _make_spec_updated  #



    # Mandatory methods: _step, _reset and _set_seed
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
print("action_spec (as defined by input_spec):", env.action_spec)
print("reward_spec:", env.reward_spec)
td = env.reset()
print("reset tensordict", td)
check_env_specs(env)









