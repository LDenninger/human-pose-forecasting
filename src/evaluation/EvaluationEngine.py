import torch
import math
import numpy as np
from tqdm import tqdm
import os
from prettytable import PrettyTable
from typing import Optional, Dict, Any, List, Union, Literal

from ..utils import print_, log_function
from ..data_utils import H36M_DATASET_ACTIONS, H36M_STEP_SIZE_MS, H36MDataset, SkeletonModel32, h36m_forward_kinematics
from .metrics import evaluate_distance_metrics, geodesic_distance, positional_mse, euler_angle_error, accuracy_under_curve


METRICS_IMPLEMENTED = {
    'geodesic_distance': geodesic_distance,
    'positional_mse': positional_mse,
    'euler_error': euler_angle_error,
    'auc': accuracy_under_curve
}
VISUALIZATION_IMPLEMENTED = {
    '3dpose': geodesic_distance # placeholder
}


class EvaluationEngineActive:
    """
        Active evaluation engine that directly computes the model outputs.
        Doing so we are able to compute more high-level metrics on the H3.6M dataset.

        Here we compute all defined metrics, instead of a small subset.
    """

    def __init__(self, device: str = 'cpu'):
        """
            Initialize the active evaluation engine.
        """
        self.device = torch.device(device)

    @log_function
    def initialize_evaluation(self, 
                               batches_per_action: int,
                                seed_length: int,
                                 prediction_lengths: List[int],
                                  down_sampling_factor: int = 1,
                                    actions: Optional[List[str]] = H36M_DATASET_ACTIONS,
                                     skeleton_model: Optional[Literal['s26', None]] = None,
                                      rot_representation: Optional[Literal['axis', 'mat', 'quat', '6d', None]] = None,
                                       batch_size: Optional[int] = 32,) -> None:
        ##== Data Structures ==##
        self.datasets = {}
        self.evaluation_results = {}
        for a in (actions+['overall']):
            self.evaluation_results[a] = {}
            for prediction_length in prediction_lengths:
                self.evaluation_results[a][prediction_length] = {}
                for metric_names in METRICS_IMPLEMENTED.keys():
                    self.evaluation_results[a][prediction_length][metric_names] = []
        ##== Evaluation Parameters ==##
        self.prediction_lengths = prediction_lengths
        self.batches_per_action = batches_per_action
        self.evaluation_finished = False

        ##== Dataset Parameters ==##
        self.seed_length = seed_length
        self.target_frames = np.ceil(prediction_lengths / (H36M_STEP_SIZE_MS * down_sampling_factor))
        self.last_target_frame = max(self.target_frames)
        self.target_length = math.ceil(max(prediction_lengths) / (H36M_STEP_SIZE_MS * down_sampling_factor))
        self.skeleton_model = skeleton_model

        self.rot_representation = rot_representation
        self.down_sampling_factor = down_sampling_factor
        self.actions = actions
        self.batch_size = batch_size

        ##== Load Action dataset ==##
        for a in self.actions:
            self.datasets[a] = H36MDataset(
                actions=self.actions,
                seed_length=self.seed_length,
                target_length=self.target_length,
                down_sampling_factor=self.down_sampling_factor,
                skeleton_model=skeleton_model,

            )

    @log_function
    def evaluate(self, model: torch.nn.Module, mode: Optional[Literal['standard', 'long_prediction']] = 'standard'):
        """
            Evaluate the provided model on the H3.6M dataset.
            For this the evaluation needs to be initialized.
        """
        self.evaluation_results = {}
        
        model.eval()
        for action, dataset in self.datasets.items():
            print_(f"Evaluating model on action {action}...")
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
            if mode == 'standard':
                self.evaluation_loop(action, model, data_loader)
        self._compute_overall_means()
        print_(f"Evaluation finished!")

    
    @torch.no_grad()
    def evaluation_loop(self, action: str, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader) -> None:
        """
            A single evaluation loop for an action.
        
        """
        self.model.eval()
        # Initialize progress bar
        progress_bar = tqdm(enumerate(self.test_loader), total=self.batches_per_action)
        progress_bar.set_description('Evaluation: ')

        for batch_idx, data in progress_bar:
            if batch_idx==self.batches_per_action:
                break
            # Load data
            data = data.to(self.device)
            # Set input for the model
            cur_input = data[:, :self.seed_length]
            targets = []
            outputs = []
            # Predict future frames in an auto-regressive manner
            for i in range(1, self.last_target_frame+1):
                # Compute the output
                output = model(cur_input)
                # Take the predicted timestamp from the last position
                outputs.append(output[:,-1])
                targets.append(data[:, self.seed_length + i])
                # Check if we want to compute metrics for this timestep
                if i in self.target_frames:
                    # Compute the implemented metrics
                    self.evaluation_results[action][i] = evaluate_distance_metrics(outputs, targets, reduction='mean', representation=self.representation)
                # Update model input for auto-regressive prediction
                cur_input = output
        
        self._reduce_action_metrics(action)
            
    def _reduce_action_metrics(self, action: str) -> None:
        """
            Compute the mean over the metrics logged for several iterations
        """
        for pred_length in self.evaluation_results[action].keys():
            for metric_name in  self.evaluation_results[action][pred_length].keys():
                self.evaluation_results[action][pred_length][metric_name] = np.mean(self.evaluation_results[action][pred_length][metric_name]) 

    def _compute_overall_means(self):
        """
            Compute the mean over all actions.
        """
        for pred_length in self.evaluation_results['overall'].keys():
            for metric_name in  self.evaluation_results['overall'][pred_length].keys():
                self.evaluation_results['overall'][pred_length][metric_name] = np.mean(self.evaluation_results[:][pred_length][metric_name])

    def _print_results(self) -> None:
        """
            Print the results into the console using the PrettyTable library.
        """

        if not self.evaluation_finished:
            return
        for a in self.evaluation_results.keys():
            if a == 'overall':
                print(f'Average over all actions:')
            else:
                print_(f'Evaluation results for action {a}:')
            table = PrettyTable()
            table.field_names = METRICS_IMPLEMENTED.keys()
            for pred_length in self.evaluation_results[a].keys():
                table.add_row([pred_length] + self.evaluation_results[a][pred_length].values())
            print_(table)
        

class EvaluationEnginePassive:
    """
        A passive evaluation engine meaning that it only receives output and targets and computes the metrics and visualizations.
        This only enables to compute simple metrics during training.
    """

    def __init__(self, 
                  metric_names: Optional[List[str]] = None,
                   visualization_names: Optional[List[str]] = None,
                    skeleton_model: Optional[Literal['s26', None]] = None,
                     representation: Optional[Literal['axis','mat', 'quat', '6d']] = 'mat',
                     keep_log: Optional[bool] = False,
                      device: str = 'cpu') -> None:

        self.metric_names = metric_names
        self.visualization_names = visualization_names
        self.representation = representation
        self.skeleton_model = skeleton_model
        self.keep_log = keep_log
        self.output_log = None
        self.target_log = None
        self.device = torch.device(device)


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
        output = output.cpu().detach()
        target = target.cpu().detach()
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
            self.output_log = output
        else:
            self.output_log = torch.cat([self.output_log, output], dim=0)
        if self.target_log is None or not self.keep_log:
            self.target_log = target
        else:
            self.target_log = torch.cat([self.target_log, target], dim=0)

    @log_function
    def compute(self, metric_names: Optional[List[str]] = None,) -> Dict[str, float]:
        """
            Compute the quantitative metrics for the model.
        """
        if metric_names is None and self.metric_names is None:
            print_("No metric provided to compute for the EvaluationEngine.")
            return
        if metric_names is None:
            metric_names = self.metric_names

        results = evaluate_distance_metrics(self.output_log, self.target_log, metric_names, reduction='mean', representation=self.representation)
        return results
    @log_function
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