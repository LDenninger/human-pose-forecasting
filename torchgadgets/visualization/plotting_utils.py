import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 


def smooth_curve(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f



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