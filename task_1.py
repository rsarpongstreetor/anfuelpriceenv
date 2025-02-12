
from dataclasses import dataclass, MISSING 


@dataclass
class TaskConfig:
    max_steps: int =10
    supports_continuous_actions:bool=True
    supports_discrete_actions: bool=True
    seed: int=5
    num_envs: int=1
    
    

     
