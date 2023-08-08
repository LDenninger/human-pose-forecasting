import json
import numpy as np


#####===== Config Data Structures =====#####

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

    def __init__(self):
        super().__init__()
        return
    
    def __getattribute__(self, name):
        if name in Config.__dict__:
            value = object.__getattribute__(self, name)
            if value is None:
                print(f"Config was not defined: {name}")
                return None
            return value
        elif name in dir(Config.TRANSFORMER_CONFIG):
            value = object.__getattribute__(Config.TRANSFORMER_CONFIG, name)
            if value is None:
                print(f"Config was not defined: {name}")
                return None
            return value
        else:
            print(f"Config does not exists for: {name}")
            return None 
        
    
    def __setattr__(self, name, value):
        if name in Config.__dict__:
            return object.__setattr__(self, name, value)
        if self.TRANSFORMER_CONFIG is None:
            if self.TRANSFORMER_TYPE is None:
                print(f'Can not set transformer config without known type')
                return
            elif self.TRANSFORMER_TYPE == 'spl':
                self.TRANSFORMER_CONFIG = SPLTransformerConfig()
            elif self.TRANSFORMER_TYPE == 'vanilla':
                self.TRANSFORMER_CONFIG = VanillaTransformerConfig()
        if name in dir(self.TRANSFORMER_CONFIG):
            return object.__setattr__(self.TRANSFORMER_CONFIG, name, value)

    
    # Meta Parameters
    RANDOM_SEED: int = None
    DEBUG: bool = None
    LOG: bool = None

    # Model Parameters
    NUM_BLOCKS: int = None
    INPUT_DROPOUT: float = None
    EMBEDDING_DIM: int = None
    TRANSFORMER_TYPE: str = None
    TRANSFORMER_CONFIG: TransformerConfigBase = None

    # Training Parameters
    BATCH_SIZE: int = None

    # Optimizer Parameters
    OPTIMIZER_TYPE: str = None
    LEARNING_RATE: float = None
    OPTIMIZER_ADD_PARAMS: dict = None

    # Learning Rate Scheduler Parameters
    LR_SCHEDULER_TYPE: str = None
    LR_SCHEDULER_DECAY_RATE: float = None
    LR_SCHEDULER_ADD_PARAMS: dict = None

    # Data Parameters
    SEQ_LENGTH: int = None
    NUM_JOINTS: int = None
    JOINT_DIM: int = None
    TRAINING_SIZE: int = None
    VALIDATION_SIZE: int = None
    TEST_SIZE: int = None


#####===== Config Functions =====#####

def getConfig(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        return json.load(f)

def getConfigFromDict(config_dict: dict) -> Config:
    config = Config()
    for key, value in config_dict.items():
        setattr(config, key, value)

def saveConfig(config: Config, config_path: str):
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=4)

