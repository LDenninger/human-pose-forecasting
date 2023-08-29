import torch
import numpy as np
import os
from typing import Optional, Dict, Any, List, Union

from ..utils import print_
from .metrics import axis_angle_diff, euler_diff, positional_mse

METRICS_IMPLEMENTED = {
    'axis_angle_diff': axis_angle_diff,
    'positional_mse': positional_mse,
    'euler_diff': euler_diff
}
VISUALIZATION_IMPLEMENTED = {
    '3dpose': angle_diff # placeholder
}

class EvaluationEngine:

    def __init__(self, 
                  metric_names: Optional[List[str]] = None,
                   visualization_names: Optional[List[str]] = None,
                    keep_log: Optional[bool] = False) -> None:

        self.metric_names = metric_names
        self.visualization_names = visualization_names
        self.keep_log = keep_log
        self.output_log = None
        self.target_log = None


    def log(self, 
             output: Union[torch.Tensor, List[torch.Tensor]],
              target: Union[torch.Tensor, List[torch.Tensor]],
               single_iteration: Optional[bool] = True,
                is_batched: Optional[bool] = True) -> None:
        """
            Enter the output and target data.


            Arguments:
                output (Union[torch.Tensor, List[torch.Tensor]): Output of the model.
                target (Union[torch.Tensor, List[torch.Tensor]): Target of the model.
                single_iteration (Optional[bool], optional): Whether the output and target are for a single iteration. Defaults to True.
                is_batched (Optional[bool], optional): Whether the output and target are batched. Defaults to True.

            Data Format:

            
        """
        import ipdb; ipdb.set_trace()
        # Parse output and target torch torch tensors
        if not torch.is_tensor(output):
            output = torch.stack(output)
        if not torch.is_tensor(target):
            target = torch.stack(target)
        # Flatten the batch and iteration dimensions 
        if is_batched and not single_iteration:
            output = torch.flatten(output, start_dim=1, end_dim=2)
            target = torch.flatten(target, start_dim=1, end_dim=2)
        # If wanted, the logs are saved internally to use for computation later
        if self.output_log is None or not self.keep_log:
            self.output_log = output.cpu()
        else:
            self.output_log = torch.cat([self.output_log, output.cpu()], dim=0)
        if self.target_log is None or not self.keep_log:
            self.target_log = target.cpu()
        else:
            self.target_log = torch.cat([self.target_log, target.cpu()], dim=0)


    def compute(self, metric_names: Optional[List[str]] = None,) -> Dict[str, float]:
        """
            Compute the quantitative metrics for the model.
        """
        import ipdb; ipdb.set_trace()

        if metric_names is None and self.metric_names is None:
            print_("No metric provided to compute for the EvaluationEngine.")
            return
        if metric_names is None:
            metric_names = self.metric_names

        results = {}
        
        for metric_name in metric_names:
            if metric_name not in METRICS_IMPLEMENTED.keys():
                print_(f"Metric {metric_name} not implemented.")
                continue
            metric = METRICS_IMPLEMENTED[metric_name]
            metric_value = metric(self.output_log, self.target_log)
            results[metric_name] = metric_value
            print_(f"Metric {metric_name}: {metric_value}")

        return results
    
    def visualize(self, visualization_name: Optional[List[str]] = None) -> Dict[str, Any]:
        """
            Compute the qualitative metric for the model.
        """
        if visualization_name is None and self.visualization_names is None:
            print_("No visualization provided to compute for the EvaluationEngine.")
            return
        if visualization_name is None:
            visualization_name = self.visualization_names

        results = {}
        
        for visualization_name in visualization_name:
            if visualization_name not in VISUALIZATION_IMPLEMENTED.keys():
                print_(f"Visualization {visualization_name} not implemented.")
                continue
            visualization = VISUALIZATION_IMPLEMENTED[visualization_name]
            visualization_value = visualization(self.output_log, self.target_log)
            results[visualization_name] = visualization_value

        return results
    
    def clear_log(self) -> None:
        """
            Clear the internal logs of the output and target data.
        """
        self.output_log = None
        self.target_log = None
        return