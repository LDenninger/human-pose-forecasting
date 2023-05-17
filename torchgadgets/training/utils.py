import torch

###--- Initialization ---###
def initialize_optimizer(model: torch.nn.Module, config: dict):
    """
        Initialize optimizer.

        Arguments:
            model (torch.nn.Module): Model to be optimized.
            config (dict): Optimizer configuration dictionary.
          
    """
    if config['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer']['type'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['optimizer']['momentum'], weight_decay=config['optimizer']['weight_decay'])
        
    return optimizer

def initialize_loss( config: dict):
    """
        Initialize criterion.

        Arguments:
            config (dict): Criterion configuration dictionary.
                Format:
                     
    """
    criterion = torch.nn.CrossEntropyLoss()

    return criterion

def get_num_iterations(config: dict):
    num_iterations = config['dataset']['train_size'] // config['batch_size']
    num_iterations += 0 if config['dataset']['drop_last'] else 1 
    return num_iterations