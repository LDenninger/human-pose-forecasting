import torch
import ipdb

import inspect

class InspectorGadgets():
    """
        Module that encapsulates different functions to investigate everything going on withing a Pytorch module.
        Each logging function creates a hook on the forward or backward pass of the given layers.
        Currently multiple hooks are potentially acquired for a single layer pass which might cause an overhead.

    """
    def __init__(self):
        
        self._layer_output = {}
        self._gradient_input = {}
        self._gradient_output = {}
        self._weights = {}
        self._bias = {}

        self._layers_to_inspect = {}

        self._hooks = {}
        self._active_hooks = {}
        self._inspect_once = False
    
    def inspect_once(self):
        """
            The inspector only hooks onto the model once and then removes all hooks
        """
        self._inspect_once = True
        self._activate_all()

    def inspect(self):
        self._inspect_once = False
        self._activate_all()

    
    def log_outputs(self, layers: list, names: list):
        """
            Log the output of a layer in each forward pass.
            Arguments:
                layers (list[torch.nn.Module]): The layers to log the output of.
                names (list[str]): The names of the layers to log the output of.
        """
        assert len(layers)==len(names), f'Number of layers ({len(layers)}) and names ({len(names)}) must be the same...'


        for i,name in enumerate(names):
            self._layer_output[name] = []

        self._setup_hook(layers, names, 'output_hook')

    def log_weights(self, layers:list, names: list, log_once=False):
        """
            Log the weights of a layer in each forward pass.
            Arguments:
                layers (list[torch.nn.Module]): The layers to log the output of.
                names (list[str]): The names of the layers to log the output of.
        """
        assert len(layers)==len(names), f'Number of layers ({len(layers)}) and names ({len(names)}) must be the same...'

        self.log_weights_once = log_once
        for i,name in enumerate(names):
            self._weights[name] = []
            self._bias[name] = []


        self._setup_hook(layers, names, 'weight_hook')

    def log_gradients(self, layers: list, names: list):
        """
            Log the gradients in the backward pass of the model.
            Arguments:
                layers (list[torch.nn.Module]): The layers to log the output of.
                names (list[str]): The names of the layers to log the output of.
        """

        assert len(layers)==len(names), f'Number of layers ({len(layers)}) and names ({len(names)}) must be the same...'

        for i,name in enumerate(names):

            self._gradient_output[name] = []

        self._setup_hook(layers, names, 'gradient_hook', forward=False, backward=True)


    def set_breakpoint(self, layers: list, names: list, forward=True, backward=False):
        """
            Set a breakpoint using the ipdb debugger after a specific layer in the forward and/or backward pass.
            Arguments:
                layers (list[torch.nn.Module]): The layers to set breakpoints after.
                names (list[str]): The names of the layers.
                forward (bool): Whether to put breakpoints in the forward pass.
                backward (bool): Whether to put breakpoints in the backward pass.
            
        """
        assert len(layers)==len(names), f'Number of layers ({len(layers)}) and names ({len(names)}) must be the same...'

        self._setup_hook(layers, names, 'break_hook', forward=forward, backward=backward)

    def get_weights(self, name=None):
        if name is None:
            return self._weights, self._bias
        if name in self._weights.keys():
            return self._weights[name], self._bias[name]
        else:
            print(f'Weights not logged for layer {name}')
    
    def get_gradients(self, name=None):
        if name is None:
            return self._gradient_output, self._bias
        if name in self._gradient_output.keys():
            return self._gradient_output[name], self._bias[name]
        else:
            print(f'Gradients not logged for layer {name}')

    def get_output(self, name=None):
        if name is None:
            return self._layer_output
        if name in self._layer_output.keys():
            return self._layer_output[name]
        else:
            print(f'Output not logged for layer {name}')
    
    def _setup_hook(self, layers, names, hook: str, forward=True, backward=False):
        """
            Setup a forward and/or backward hook in the model given the layers and their names from the model.
        
        """
        
        for (i, name) in enumerate(names):

            if not name in self._layers_to_inspect.keys():
                self._layers_to_inspect[name] = layers[i]

            if not name in self._hooks.keys():
                self._hooks[name] = {}
            
            if forward:
                hook_key = ('' if hook!='break_hook' else 'forward_')+ hook
                hook_func = '_' + hook_key
                self._hooks[name][hook_key] = layers[i].register_forward_hook(self._hook_wrapper(name=name, hook=hook_func))
            if backward:
                hook_key = ('' if hook!='break_hook' else 'backward_')+ hook
                hook_func = '_' + hook_key
                self._hooks[name][hook_key] = layers[i].register_full_backward_hook(self._hook_wrapper(name=name, hook=hook_func))

    
    def _remove_hook(self, name, hook):
        """
            Remove a hook by giving the layer name and the hook to remove.
        """
        if not name in self._hooks.keys():
            print(f'Layer {hook} not found for any actions')
            return
        if not hook in self._hooks[name].keys():
            print(f'No hook {hook} set up for layer {name}')
            return

        self._hooks[name][hook].remove()
        self._hooks[name][hook] = None

    def _activate_all(self):
        for layer_name, hooks in self._hooks.items():
            for hook_name, hook in hooks.items():
                if hook is None:
                    self._activate_hook(layer_name, hook_name)

    def _activate_hook(self, name, hook):

        if not name in self._hooks.keys():
            print(f'Layer {hook} not found for any actions')
            return
        if not hook in self._hooks[name].keys():
            print(f'No existing hook {hook} set up for layer {name}')
            return
        
        if self._hooks[name][hook] is None:
            layer = self._layers_to_inspect[name]
            self._attach(name, layer, hook)

    def _hook_wrapper(self, name, hook):
        """
            Wrapper function for the different hooks that can be set in the model. 
            The function mainly serves the purpose to determine which layer of the model triggered the hook function.
        """

        def _gradient_hook(module, grad_input, grad_output) -> None:
            if grad_input[0] is not None:
                self._gradient_input[name].append(grad_input[0].cpu().detach())
            self._gradient_output[name].append(grad_output[0].cpu().detach())
            if self._inspect_once:
                self._remove_hook([name], 'gradient_hook')


        def _forward_break_hook(module, input, output) -> None:
            print(f'Module {name}: '+module.__class__.__name__)
            print(f'Input Shape: {input[0].shape}')
            print(f'Output Shape: {output.shape}')
            ipdb.set_trace()
            if self._inspect_once:
                self._remove_hook([name], 'forward_break_hook')
        
        def _backward_break_hook(module,  grad_input, grad_output) -> None:
            ipdb.set_trace()
            print(f'Grad Input Shape: {grad_input[0].shape}')
            print(f'Grad Output Shape: {grad_output[0].shape}')
            print(f'Grad Module {name}:\n'+module)
            ipdb.set_trace()
            if self._inspect_once:
                self._remove_hook([name], 'backward_break_hook')

        def _output_hook(module, input, output) -> None:
            self._layer_output[name].append(output.cpu().detach())

            if self._inspect_once:
                self._remove_hook([name], 'output_hook')

        def _weight_hook(module, input, output) -> None:
            try:
                self._weights[name].append(module.weight.cpu().detach())
                self._bias[name].append(module.bias.cpu().detach())
            except:
                print(f'Layer {name} has no weights...  ')

            if self._inspect_once:
                self._remove_hook([name], 'weight_hook')
        local_vars = inspect.currentframe().f_locals
        return local_vars[hook]

    def _attach(self, name, layer, hook):
        if hook in ['gradient_hook','backward_break_hook']:
            self._hooks[name][hook] = layer.register_full_backward_hook(self._hook_wrapper(name=name, hook=('_'+hook)))
        else:
            self._hooks[name][hook] = layer.register_forward_hook(self._hook_wrapper(name=name, hook=('_'+hook)))
