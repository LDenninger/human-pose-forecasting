import optuna
from ..models import *
from ..data import get_train_loaders, ImageDataAugmentor
from ..evaluation import *
from ..logging import *
from .utils import *
from .basic_training import *
from .scheduler import SchedulerManager

###--- Model Specific Training Sciprts ---###

def trainNN(config, 
                model: torch.nn.Module = None,
                    train_loader = None,
                        test_loader = None,
                            criterion = None,
                                logger: Logger = None,
                                    optimizer = None,
                                        data_augmentor = None,
                                            scheduler = None,
            return_all=True):
    """
        Script for a basic trainig of the neural network. 
        All relevant parts for the training can either be provided as an argument.
        If these are not provided as an argument, the modules will be initialized using the config.

        Arguments:
            model (torch.nn.Module): Model to train. If None, a model according to the config is initialized.
    """
    ###-- Initialization ---###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config['num_iterations'] == 0:
        config['num_iterations'] = config['dataset']['train_size'] // config['batch_size']
        config['num_iterations'] += 0 if config['dataset']['drop_last'] else 1 

    # Build Model from config
    if model is None:
        model = NeuralNetwork(config['layers'])
    model = model.to(device)

    # Initializae the data loaders
    if train_loader is None or test_loader is None:
        t_loader, te_loader = get_train_loaders(config)
        train_loader = t_loader if train_loader is None else train_loader
        test_loader = te_loader if test_loader is None else test_loader

    # Define criterion for the loss
    if criterion is None:
        criterion = initialize_loss(config)

    # Define optimizer
    if optimizer is None:
        optimizer = initialize_optimizer(model, config)

    # Initialize logger to save all data for later evaluations
    if logger is None:
        logger = Logger(model_config=config, save_external=False, save_internal=True)

    if scheduler is None:
        if config['scheduler'] is not None:
            scheduler = SchedulerManager(optimizer, config)
    if data_augmentor is None:
        data_augmentor = ImageDataAugmentor(config=config['pre_processing'])


    train_model(model=model, 
                    config=config, 
                        train_loader=train_loader, 
                            val_loader=test_loader, 
                                optimizer=optimizer, 
                                    data_augmentor=data_augmentor, 
                                        criterion=criterion, 
                                            scheduler=scheduler,
                                                logger=logger)
    
    if return_all:
        return model, train_loader, test_loader, data_augmentor, logger
    else:
        return logger
    
###--- Model Specific Optimization Scripts ---###
def optimizeNN(config, trial, train_loader=None, test_loader= None, scheduler=None, score_metric='accuracy'):

    def __train():
        ###--- Hyperparameters ---###
        EPOCHS = config['num_epochs']

        evaluation_config = config['evaluation']

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        eval_scores = []

        ###--- Training ---###
        
        # Train for EPOCHES epochs and evaluate the model according to the pre-defined frequency
        for epoch in range(EPOCHS):
            model.train()
            outputs = []
            targets = []
            training_metrics = []
            ###--- Training Epoch ---###
            for i, (img, label) in enumerate(train_loader):
                img = img.to(DEVICE)
                label = label.to(DEVICE)
                # Apply data augmentation and pre-processing
                img, label = data_augmentor((img, label))
                # Zero gradients
                optimizer.zero_grad()
                # Compute output of the model
                output = model(img)
                # Compute loss
                loss = criterion(output, label)
                # Backward pass to compute the gradients wrt to the loss
                loss.backward()
                # Update weights
                optimizer.step()
                outputs.append(output.cpu().detach())
                targets.append(label.cpu().detach())
                
                if scheduler is not None:
                    # Learning rate scheduler takes a step
                    scheduler.step(i+1)
                
            ###--- Evaluation Epoch ---###
            if epoch % evaluation_config['frequency'] == 0:
                evaluation_metrics, eval_loss = run_evaluation(model,data_augmentor,test_loader,config, criterion, suppress_output=True)
                eval_score = evaluation_metrics[score_metric][0]
                eval_scores.append(eval_score)
                trial.report(eval_score, step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
        return eval_scores
            
            

    ###-- Initialization ---###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build Model from config
    model = NeuralNetwork(config['layers'])
    model = model.to(device)

    if train_loader is None or test_loader is None:
        t1, t2 = get_train_loaders(config)
        if train_loader is None:
            train_loader = t1
        if test_loader is None:
            test_loader = t2

    # Define criterion for the loss
    criterion = initialize_loss(config)



    # Define optimizer
    optimizer = initialize_optimizer(model, config)

    # Scheduler
    if scheduler is None:
        if config['scheduler'] is not None:
            scheduler = SchedulerManager(optimizer, config)

    data_augmentor = ImageDataAugmentor(config=config['pre_processing'])

    if config['num_iterations'] == 0:
        config['num_iterations'] = config['dataset']['train_size'] // config['batch_size']
        config['num_iterations'] += 0 if config['dataset']['drop_last'] else 1

    scores = __train()

    return max(scores)


