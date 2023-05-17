import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .plotting_utils import smooth_curve

from ..training import initialize_optimizer, SchedulerManager
from ..models import NeuralNetwork

def visualize_training_results(logger, config: dict):

    iteration_corrected = [np.sum(np.arange(i)*config['num_iterations']) for i in range(config['num_epochs']+1)]

    # Load the data from the MLP training
    data = logger.get_log()

    train_loss = data['train_loss']
    train_acc = data['train_accuracy']
    eval_loss = data['eval_loss']
    eval_acc = data['accuracy']


    fig = plt.figure(figsize=(14,10))

    smooth_loss = smooth_curve(train_loss, 31)
    ax1 = fig.add_subplot(221)
    ax1.plot(train_loss, c="blue", label="Training Loss", linewidth=3, alpha=0.5)
    ax1.plot(smooth_loss, c="red", label="Smoothed Loss", linewidth=3)
    ax1.set_xlabel("# Iteration")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.set_title('Training Loss')

    ax2 = fig.add_subplot(222)
    ax2.plot(iteration_corrected, eval_loss, c="red", label="Evaluation Loss", linewidth=3)
    ax2.set_xlabel("# Iteration")
    ax2.set_ylabel("CrossEntropy Loss")
    ax2.set_title('Evaluation Loss')


    smooth_acc = smooth_curve(train_acc, 31)
    ax3 = fig.add_subplot(223)
    ax3.plot(train_acc, c="blue", label="Training Accuracy", linewidth=3, alpha=0.5)
    ax3.plot(smooth_acc, c="red", label="Smoothed Accuracy", linewidth=3)
    ax3.set_ylim(0.0,1.0)
    ax3.set_yticks(np.arange(0.0, 1.1, 0.1).tolist())
    ax3.set_xlabel("# Iteration")
    ax3.set_ylabel("Accuracy")
    ax3.set_title('Training Accuracy')

    ax4 = fig.add_subplot(224)
    ax4.plot(iteration_corrected, eval_acc, c="red", label='Evaluation Accuracy', linewidth=3)
    ax4.set_xlabel("# Iteration")
    ax4.set_ylabel("Accuracy")
    ax4.set_ylim(0.0,1.0)
    ax4.yticks(np.arange(0.0, 1.1, 0.1).tolist())
    ax4.set_title('Evaluation Accuracy')

    return fig

def compare_training_results(logger_list: list, model_names: list, config: dict):
    assert len(logger_list)==len(model_names), f'Length of logger list {len(logger_list)} does not correspond to the length of the mode list {len(model_names)}'
    #iteration_corrected = [np.sum(np.arange(i)*config['num_iterations']) for i in range(config['num_epochs']+1)]
    xticks = range(0, config['num_epochs']*config['num_iterations'], config['num_iterations'])
    plt.style.use('seaborn-v0_8')
    # Load the data from the MLP training
    data = []
    train_loss = []
    eval_loss = []
    eval_acc = []
    for logger in logger_list:
        data = logger.get_log()
        train_loss.append(data['train_loss'])
        eval_loss.append(data['eval_loss'])
        eval_acc.append(data['accuracy'])
    fig = plt.figure(figsize=(12,15))

    ax1 = fig.add_subplot(311)
    for i in range(len(logger_list)):
        smooth_loss = smooth_curve(train_loss[i], 31)
        ax1.plot(smooth_loss, label=model_names[i], linewidth=2)

    ax1.set_xlabel("# Epochs")
    ax1.set_xticks(xticks, np.arange(1, config['num_epochs']+1))
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.set_title('Training Loss')
    ax1.legend()

    ax2 = fig.add_subplot(312)
    for i in range(len(logger_list)):
        ax2.plot(eval_loss[i], label=model_names[i], linewidth=2)

    ax2.set_xlabel("# Epochs")
    ax2.set_ylabel("CrossEntropy Loss")
    ax2.set_title('Evaluation Loss')
    ax2.legend()

    ax3 = fig.add_subplot(313)
    for i in range(len(logger_list)):
        ax3.plot(eval_acc[i], label=model_names[i], linewidth=2)

    ax3.set_ylim(0.0,1.0)
    ax3.set_yticks(np.arange(0.0, 1.1, 0.1).tolist())
    ax3.set_xlabel("# Iteration")
    ax3.set_ylabel("Accuracy")
    ax3.set_title(' Evaluation Accuracy')
    ax3.legend()


def visualize_convolution(image, kernels, outputs, layer_names, layer_types=None, random=True, num_kernels=5):
    """
        Visualization of the kernels and the output of the convolution layer for the given kernel. 
        Row-wise different kernels for the given image are visualized in random order or in order of the kernels within the model.
        If there are multiple input channels in a layer a subset of the kernels for each input dimension are visualized.

        Note: The visualization is not complete yet. It does not included biases and has spacing issues.
            But for now it is sufficient to reason about specific kernels, especially in the first layer, and the extracted features
    
    """
    assert num_kernels <= kernels[0].shape[0]

    # Spacing parameters
    image_size = 5.0
    output_size = 3.0
    kernel_layout = (6,3)
    num_layers = len(kernels)
    num_col = 3*num_layers+1
    width_ratio = np.ones(num_col).tolist()
    height_ratio = (10*np.ones(num_kernels+1)).tolist()
    height_ratio[0] = 2
    kernel_annot = (2,-10)

    image = image.numpy()

    # Define figure and grid to hold the visualizations of each grid
    fig = plt.figure(figsize=(30,15))
    figure_grid = GridSpec(num_kernels+1, num_col, figure=fig, width_ratios=width_ratio, height_ratios=height_ratio)

    # Define random indices for the choice of the kernels to visualize
    if random:
        random_inds = []
        for i in range(num_layers):
            random_inds.append(np.arange(kernels[i].shape[0]))
            np.random.shuffle(random_inds[-1])
    # Visualize the image
    ax = fig.add_subplot(figure_grid[:,0])
    im = ax.imshow(image, cmap='gray')
    ax.axis('off')
    ax.set_title('Input Image')
    # Set the colorbar
    cbar_ax = fig.add_axes([0.02, 0, 0.12, 0.01])
    cbar = fig.colorbar(im,cax=cbar_ax, orientation='horizontal')
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1)
    # Set the titles for each layer column
    for i in range(num_layers):
        module_plot = fig.add_subplot(figure_grid[0,(1+i):(i+2)])
        module_plot.axis('off')
        in_channel = len(kernels[i][0].shape) if len(kernels[i][0].shape)==3 else 1
        out_channel = kernels[i].shape[0]
        label = f'Type: {layer_names[i]}\n' if layer_names is not None else ''
        label += f"In Channels: {in_channel}\nOut Channels: {out_channel}"
        module_plot.set_title(f'Layer {layer_names[i][-1]}')
        module_plot.annotate(label, (0.1,0.5), xycoords='axes fraction', va='center')

    # Create the visualizations of the kernels of each layer and stack them into the grid
    print_rows = np.arange(0,num_kernels)        
    for i in print_rows:
        for k in range(1,num_layers+1):
            # Get the kernel
            kernel_ind = random_inds[k-1][i] if random else i
            kernel = kernels[k-1][kernel_ind].numpy()
            # Create plot to hold the visualization of the kernel and output
            layer_plot = figure_grid[i+1,k].subgridspec(1,2, wspace=0.02)
            # If we have multiple input channels, visualize multiple kernels for the input dimension
            if len(kernel.shape)==3:
                # Create a grid of size kernel_layout
                kernel_layer_plot = layer_plot[0,0].subgridspec(kernel_layout[0],kernel_layout[1], wspace=0.1, hspace=0.0)
                num_kernels = kernel.shape[0]
                k_ind = 0
                # Fill each grid cell with a kernel
                for i_k in range(kernel_layout[0]):
                    for j_k in range(kernel_layout[1]):
                        if k_ind == num_kernels:
                            break
                        # Visualize a kernel
                        ax1 = fig.add_subplot(kernel_layer_plot[i_k,j_k])
                        # Normalize to [0,1] 
                        kernel_mat = (kernel[k_ind] - kernel[k_ind] .min()) / (kernel[k_ind] .max() - kernel[k_ind] .min())
                        ax1.imshow(kernel_mat,  cmap='gray')
                        ax1.axis('off')
                        k_ind += 1
                    if k_ind == num_kernels:
                            break
                ax1 = fig.add_subplot(kernel_layer_plot[:,:])
                ax1.axis('off')
                label = f'kernel index: {kernel_ind}'
                ax1.annotate(label, kernel_annot, xycoords='axes pixels', fontsize=8)

            else:
                # Just visualize a single kernel
                ax1 = fig.add_subplot(layer_plot[0,0])
                kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
                ax1.imshow(kernel,  cmap='gray')
                ax1.set_xticks([])
                ax1.set_yticks([])   
                ax1.axis('off')
                label = f'kernel index: {kernel_ind}'
                ax1.annotate(label, kernel_annot, xycoords='axes pixels', fontsize=8)             

            # Visualize the output
            ax2 = fig.add_subplot(layer_plot[0,1])
            output = outputs[k-1][kernel_ind].numpy()
            # Normalize to [0,1] 
            output = (output - output.min()) / (output.max() - output.min())
            ax2.imshow(output, cmap='gray')
            ax2.axis('off')

            if i==0:
                # Set the titles for the kernels and output columns of each layer
                ax = fig.add_subplot(layer_plot[0,0])
                ax.axis('off')
                ax.set_title('Kernels')
                ax = fig.add_subplot(layer_plot[0,1])
                ax.axis('off')
                ax.set_title('Kernels')
                ax.set_title('Output')
                #ax3 = fig.add_subplot(figure_grid[:,k])
                #ax3.set_title(f'Layer {layer_names[k-2]}')
    figure_grid.tight_layout(fig)

    return fig

def visualize_scheduler(config):

    xticks = range(0, config['num_epochs']*config['num_iterations'], config['num_iterations'])

    model = NeuralNetwork(config['layers'])
    optimizer = initialize_optimizer(model, config)
    scheduler = SchedulerManager(optimizer, config)
    learning_rates = []

    for e in range(config['num_epochs']):
        for i in range(config['num_iterations']):
            scheduler.step(i)
            learning_rates.append(scheduler.get_last_lr())
    if type(learning_rates[0]) == list:
        learning_rates = [l[0] for l in learning_rates]
    smooth_lr = tg.visualization.smooth_curve(learning_rates, 20)
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.plot(smooth_lr, c='red')
    ax.set_xlabel('# Epochs')
    ax.set_xticks(xticks, np.arange(1, config['num_epochs']+1))
    ax.set_ylabel('Learning rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True)
    ax.set_yscale('linear')
    ax.set_xscale('linear')

    return fig