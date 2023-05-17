from fvcore.nn import FlopCountAnalysis, flop_count_str, ActivationCountAnalysis

def count_model_params(model, verbose=False):
    """Get the layers with trainable parameters and the number of trainable parameters."""
    layer_parameter_number = {}
    for p in model.named_parameters():
        if p.requires_grad:
            layer_parameter_number[p[0]] = p[1].numel()
    layer_parameter_number['total'] = sum(layer_parameter_number.values())
    return layer_parameter_number

def compute_flops(model, dummy_input, verbose=True, detailed=False):
    """ Computing the number of activations and flops in a forward pass """

    # benchmarking
    fca = FlopCountAnalysis(model, dummy_input)
    act = ActivationCountAnalysis(model, dummy_input)
    
    total_flops = fca.total()
    total_act = act.total()

    return total_flops, total_act

def check_configuration(layer_config, ind=None, padding=0, input_shape=(1,28,28), suppress_output=False, fix=False):
    """
        Simple function to inspect the output dimension of each layer of a CNN or MLP given a layer configuration.
        If the layer_configuration is not possible, whether output and input shape do not fit or the output dimensions are non-sense,
        the function returns the output dimension of each layer of to the point of failure and returns a False flag indicating a non-possible architecture.

        Arguments:
            layer_config (list): Model architecture configuration according to our convention.
            ind (int): Index of the layer to get the output dimension from.
            padding (int): Amount of padding to add to the input shape.
            input_shape (tuple): Shape of the input image.
            suppress_output (bool): Whether to suppress the print output for further inspection.

        Returns: List of output dimensions, Bool indicating a possible architecture
    
    """


    def _print_dims(message=None):
        if len(shapes)==len(layer_config)+1:
            print(f'--SUCCESS')
            print(message)
        else:
            print(f'--FAILURE: Layer {len(shapes)}:\t'+message)
        print(f'\n---Model Architecture---')
        print(f'Input Shape: \t{shapes[0]}')
        print('\n'.join(['Layer \t{}: \tType: \t{}:    \tShape: \t{}'.format(k-1,layer_config[k-1]['type'], shapes[k]) for k in range(1, len(shapes))]))
        print(f'Output Shape: \t{shapes[-1]}')

    shapes = [input_shape]
    for i, layer in enumerate(layer_config):
        shape = shapes[-1]
        if any(dim <=0 for dim in (shape if type(shape)==tuple else [shape])):
            if not suppress_output:
                _print_dims(f'Architecture is not possible with shape {shape} after layer {i-1}')
            return shapes, False

        if ind is not None and i-1 == ind:
            return shape, True
        
        if layer['type'] == 'conv2d':
            if len(shape) == 2:
                shape = (1, shape[0], shape[1])
            if shape[1]+padding<layer['kernel_size'][0] or shape[2]+padding<layer['kernel_size'][1]:
                if not suppress_output:
                    _print_dims(f'Kernel of size {layer["kernel_size"]} is larger than input shape {shape}')
                return shapes, False
            if shape[0]!=layer['in_channels']:
                if fix:
                    if not suppress_output:
                        print(f'Layer {i} input channel {layer["in_channels"]} != Input channel {shape}. Fixing...')
                    layer['in_channels'] = shape[0]
                    return check_configuration(layer_config, ind, padding, input_shape, suppress_output, fix)
                else:
                    if not suppress_output:
                        _print_dims(f'Channel size of input {shape[0]} differs from layer input channel size {layer["in_channels"]}')
                    return shapes, False
                
            channel = layer['out_channels']
            height = int(np.floor(float(shape[1] - layer['kernel_size'][0] + 2 * padding) / float(layer['stride'][0]))) + 1
            width = int(np.floor(float(shape[2] - layer['kernel_size'][1] + 2 * padding) / float(layer['stride'][1]))) + 1
            shape = (channel, height, width)

        elif layer['type'] in ['avgpool2d', 'maxpool2d']:
            if len(shape) == 2:
                shape = (1, shape[0], shape[1])
            if shape[1]+padding<layer['kernel_size'][0] or shape[2]+padding<layer['kernel_size'][1]:
                if not suppress_output:
                    _print_dims(f'Pool size {layer["kernel_size"]} is larger than input shape {shape}')
                return shapes, False

            height = int((shape[1] - layer['kernel_size'][0]) / layer['stride'][0] + 1)
            width = int((shape[2] - layer['kernel_size'][1]) / layer['stride'][1] + 1)
            shape = (shape[0], height, width)

        elif layer['type']=='flatten':
            if type(shape)==tuple:
                f = shape[0]
                for k in range(1, len(shape)):
                    f *= shape[k]
                shape = f
        
        elif layer['type'] == 'linear':
            if type(shape) == tuple:
                if not suppress_output:
                    _print_dims(f'Linear layer received input of dimension {shape}')
                return shapes, False
            if shape!= layer['in_features']:
                if fix:
                    if not suppress_output:
                        print(f'Layer {i} input features {layer["in_features"]} != Input features {shape}. Fixing...')
                    layer['in_features'] = shape
                    return check_configuration(layer_config, ind, padding, input_shape, suppress_output, fix)
                if not suppress_output:
                    _print_dims('Linear layer received input of size {} instead of {}'.format(shape, layer['in_features']))
                return shapes, False
            shape = layer['out_features']
    
        shapes.append(shape)
    if not suppress_output:
        _print_dims()

    if fix:
        return (layer_config, shapes), True
    return shapes, True
