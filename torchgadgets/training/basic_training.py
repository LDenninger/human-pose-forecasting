import torch
from tqdm import tqdm

from ..evaluation import *

###--- Training Scripts ---###
def train_model(model, config, train_loader, val_loader, optimizer, criterion,  data_augmentor, scheduler=None, logger=None):

    ###--- Hyperparameters ---###

    EPOCHS = config['num_epochs']
    ITERATIONS = config['num_iterations']

    evaluation_config = config['evaluation']

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ###--- Initial Evaluation ---###
    # Evaluate the untrained model to have some kind of base line for the training progress
    print('Initial Evluation')
    evaluation_metrics, eval_loss = run_evaluation(model,data_augmentor,val_loader,config, criterion, suppress_output=False)

    # Log evaluation data
    if logger is not None:
        logger.log_data(epoch=0, data=evaluation_metrics)
        logger.log_data(epoch=0, data={'eval_loss': eval_loss})
    if logger is not None and logger.save_internal:
        logs = logger.get_last_log()
        print("".join([(f' {key}: {value},') for key, value in logs.items()]))


    ###--- Training ---###
    
    # Train for EPOCHES epochs and evaluate the model according to the pre-defined frequency
    for epoch in range(EPOCHS):
        print('\nEpoch {}/{}'.format(epoch + 1, EPOCHS))
        model.train()

        outputs = []
        targets = []
        #training_metrics = []

        ###--- Training Epoch ---###
        progress_bar = tqdm(enumerate(train_loader), total=config['num_iterations'])
        for i, (img, label) in progress_bar:
            
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
            # Log training data
            if logger is not None:
                logger.log_data(data={'train_loss': loss.cpu().float().item()}, epoch=epoch+1, iteration=i+1, model = model, optimizer = optimizer)
            #tr_metric = eval_resolve(output, label, config)['accuracy'][0]
            #raining_metrics.append(tr_metric)
            outputs.append(output.cpu().detach())
            targets.append(label.cpu().detach())
            
            progress_bar.set_description(f'Loss: {loss.cpu().item():.4f}')
            
            if scheduler is not None:
                # Learning rate scheduler takes a step
                scheduler.step(i+1)
        
        
        ###--- Evaluation Epoch ---###
        if epoch % evaluation_config['frequency'] == 0:
            evaluation_metrics, eval_loss = run_evaluation(model,data_augmentor,val_loader,config, criterion, suppress_output=False)

        # Log evaluation data
        if logger is not None:
            #logger.log_data(epoch=epoch+1, data={'train_accuracy': training_metrics})
            logger.log_data(epoch=epoch+1, data=evaluation_metrics)
            logger.log_data(epoch=epoch+1, data={'eval_loss': eval_loss})
        
        # If the logger is activated and saves the data internally we can print out the data after each epoch
        if logger is not None and logger.save_internal:
            logs = logger.get_last_log()
            print(", ".join([(f'{key}: {value}') for key, value in logs.items()]))
        

    
        
    