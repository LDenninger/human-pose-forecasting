import json
import numpy as np


#####===== Config Data Structures =====#####
# TODO: The config object itself is broken right now. The get/set functions do not work properly.
#   --> For now we simply use a dictionary as config
##== Transformer Configs ==##

class TransformerConfigBase(object):
    TYPE: str = None

class SPLTransformerConfig(TransformerConfigBase):
    SPATIAL_HEADS: int = None
    TEMPORAL_HEADS: int = None
    SPATIAL_DROPOUT: float = None
    TEMPORAL_DROPOUT: float = None
    FF_DROPOUT: float = None

class VanillaTransformerConfig(TransformerConfigBase):
    HEADS: int = None
    ATTN_DROPOUT: float = None
    FF_DROPOUT: float = None

##== General Configs ==##
# The object is filled with default values that are not intended to work.
class Config(object):

    _self = None
    def __new__(cls, *args, **kwargs):
        if cls._self is None:
            cls._self = super(Config, cls).__new__(cls, *args, **kwargs)
        return cls._self

    def __init__(self):
        super().__init__()
        # Meta Parameters
        self.RANDOM_SEED: int = 0
        self.DEBUG: bool = None
        self.LOG: bool = None
        # Model Parameters
        self.NUM_BLOCKS: int = None
        self.INPUT_DROPOUT: float = None
        self.EMBEDDING_DIM: int = None
        self.POSITIONAL_ENCODING_TYPE: str = None
        self.TRANSFORMER_TYPE: str = None
        self.TRANSFORMER_CONFIG: TransformerConfigBase = None
        # Training Parameters
        self.BATCH_SIZE: int = None
        # Optimizer Parameters
        self.OPTIMIZER_TYPE: str = None
        self.LEARNING_RATE: float = None
        self.OPTIMIZER_ADD_PARAMS: dict = None
        # Learning Rate Scheduler Parameters
        self.LR_SCHEDULER_TYPE: str = None
        self.LR_SCHEDULER_DECAY_RATE: float = None
        self.LR_SCHEDULER_ADD_PARAMS: dict = None
        # Data Parameters
        self.SEQ_LENGTH: int = None
        self.NUM_JOINTS: int = None
        self.JOINT_DIM: int = None
        self.TRAINING_SIZE: int = None
        self.VALIDATION_SIZE: int = None
        self.TEST_SIZE: int = None
        return
    
    def __getattribute__(self, name):
        if hasattr(self, name):
            value = object.__getattribute__(self, name)
            if value is None:
                print(f"Config was not defined: {name}")
                return None
            return value
        elif hasattr(self.TRANSFORMER_CONFIG, name):
            value = object.__getattribute__(self.TRANSFORMER_CONFIG, name)
            if value is None:
                print(f"Config was not defined: {name}")
                return None
            return value
        else:
            print(f"Config does not exists for: {name}")
            return None 
        
    
    def __setattr__(self, name, value):
        import ipdb; ipdb.set_trace()
        if hasattr(self, name):
            return object.__setattr__(self, name, value)
        if self.TRANSFORMER_CONFIG is None:
            if self.TRANSFORMER_TYPE is None:
                print(f'Can not set transformer config without known type')
                return
            elif self.TRANSFORMER_TYPE == 'spl':
                self.TRANSFORMER_CONFIG = SPLTransformerConfig()
            elif self.TRANSFORMER_TYPE == 'vanilla':
                self.TRANSFORMER_CONFIG = VanillaTransformerConfig()
        if hasattr(self.TRANSFORMER_CONFIG, name):
            return object.__setattr__(self.TRANSFORMER_CONFIG, name, value)

    


#####===== Config Functions =====#####

def getConfig(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        return json.load(f)

def getConfigFromDict(config_dict: dict) -> Config:
    import ipdb; ipdb.set_trace()
    configObj = Config()
    for key, value in config_dict.items():
        setattr(configObj, key, value)

def saveConfig(config: Config, config_path: str):
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=4)

