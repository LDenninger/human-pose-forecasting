"""
    This file contains the evaluation engine that is used to compute the evaluation metrics and produce visualizations.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""

import torch
from torch.utils.data import DataLoader
import math
import numpy as np
from tqdm import tqdm
import os
from prettytable import PrettyTable
from typing import Optional, Dict, Any, List, Union, Literal
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path as P

from ..utils import print_, log_function, LOGGER
from ..data_utils import (
    H36M_DATASET_ACTIONS,
    H36M_STEP_SIZE_MS,
    H36M_SKELETON_STRUCTURE,
    H36M_NON_REDUNDANT_SKELETON_STRUCTURE,
    H36M_SKELETON_PARENTS,
    SH_SKELETON_PARENTS,
    VLP_PARENTS,
    H36M_NON_REDUNDANT_PARENT_IDS,
    SH_SKELETON_STRUCTURE,
    H36MDataset,
    AISDataset,
    SkeletonModel32,
    h36m_forward_kinematics,
    DataAugmentor,
    get_data_augmentor
)
from .metrics import (
    evaluate_distance_metrics,
    evaluate_distribution_metrics,
    geodesic_distance,
    positional_mse,
    euler_angle_error,
    accuracy_under_curve,
    power_spectrum,
    ps_entropy,
    ps_kld,
    npss
)

from .visualization import(
    compare_sequences_plotly,
)

from ..visualization import compare_skeleton, animate_pose_matplotlib, visualize_attention

METRICS_IMPLEMENTED = {
    "geodesic_distance": geodesic_distance,
    "positional_mse": positional_mse,
    "euler_error": euler_angle_error,
    "auc": accuracy_under_curve,
}
DISTRIBUTION_METRICS_IMPLEMENTED = {
    "ps_entropy": ps_entropy,
    "ps_kld": ps_kld,
    "npss": npss,
}
VISUALIZATION_IMPLEMENTED = {"3dpose": geodesic_distance}  # placeholder


class EvaluationEngine:
    """
        The evaluation engine computes different quantitative metrics and visualizations.
        All our evaluation results are produced using this module.
        First, one needs to load the data using the load_data() function for the evaluation.
        Next, we can choose from different evaluation methods:
            - distance metrics: Compute the distances to the ground truth data
            - distribution metrics: Compute the distribution metrics over long prediction horizons
            - visualization 2d: Produces a visualization showing a complete sequence
            - visualization 3d: Produces a 3D visualization showing the sequence. 
                This can be done interactively in a window or be saved as video files.
    """

    def __init__(self, device: str = "cpu"):
        """
            Initialize the evaluation engine.

            Simply initializes empty data structures required for the evaluation.
        """
        ##== Evaluation Parameters ==##
        self.visualization_2d_active = False
        self.visualization_3d_active = False
        self.long_predictions_active = False
        self.distance_metric_active = False
        self.visualization_attn_active = False
        self.data_loaded = False

        self.prediction_timesteps = {
            "visualization_2d": None,
            "visualization_3d": None,
            "long_predictions": None,
            "distance_metric": None,
            "visualization_attn": None
        }
        self.target_frames = {
            "visualization_2d": None,
            "visualization_3d": None,
            "long_predictions": None,
            "distance_metric": None,
            "visualization_attn": None
        }

        self.num_iterations = {
            "visualization_2d": None,
            "visualization_3d": None,
            "long_predictions": None,
            "distance_metric": None,
            "visualization_attn": None
        }
        self.step_sizes = {
            "h36m": None,
            "ais": None,
        }
        self.evaluation_results = {
            "long_predictions": {},
            "distance_metric": {}
        }
        self.vis3d_max_length = None
        self.distance_metrics = []
        self.distribution_metrics = []
        self.device = torch.device(device)

    def initialize_long_prediction_evaluation(self,
                                           iterations: int,
                                           metric_names: List[str],
                                           skeleton_model: str,
                                           prediction_timesteps: List[int],
                                           variable_window: bool = False,
                                           distr_pred_sec: int = 15,
                                           ):
        """
            Initialize the evaluation for long predictions using distribution metrics

            Arguments:
                iterations (int): Number of iterations for the evaluation
                metric_names (List[str]): List of the metrics to compute
                prediction_timesteps (List[int]): List of the prediction timesteps to compute the metrics for
        """
        # Load distribution metrics of dataset
        self.skeleton_model = skeleton_model
        try:
            entropy = torch.load(f'configurations/distribution_values/entropy_{self.skeleton_model}.pt')
            kld = torch.load(f'configurations/distribution_values/kld_{self.skeleton_model}.pt')
            test_ps = torch.load(f'configurations/distribution_values/test_ps_{self.skeleton_model}.pt')
        except IOError as e:
            print_(f"Could not load distribution metrics from disk: {e}", "error")
            return
        # Store them in dictionary
        self.distribution_values = {
            'entropy': entropy,
            'kld': kld,
            'test_ps': test_ps
        }
        self.num_iterations['long_predictions'] = iterations
        self.distr_pred_sec = distr_pred_sec
        self.prediction_timesteps['long_predictions'] = [40 * i for i in range(1, self.distr_pred_sec * 25 + 1)]
        prediction_timesteps = self.prediction_timesteps['long_predictions']
        target_frames =  np.ceil(np.array(prediction_timesteps) / self.step_size).astype(int)
        prediction_steps_real = (target_frames * self.step_size).tolist()
        if prediction_steps_real != prediction_timesteps:
            print_(f"Goal prediction timesteps: {prediction_timesteps} not reachable using timesteps: {prediction_steps_real}","warn")
        self.target_frames['long_predictions'] = target_frames.tolist()
        self.prediction_timesteps['long_predictions'] = prediction_steps_real
        self.distribution_metrics = metric_names if metric_names is not None else list(DISTRIBUTION_METRICS_IMPLEMENTED.keys())
        self.long_predictions_active = True
        self.variable_window = variable_window



    def initialize_distance_evaluation(self,
                                       iterations: int,
                                       metric_names: List[str],
                                       prediction_timesteps: List[int],
                                       variable_window: bool = False):
        """
            Initialize the evaluation for short predictions using distance metrics

            Arguments:
                iterations (int): Number of iterations for the evaluation
                metric_names (List[str]): List of the metrics to compute
                prediction_timesteps (List[int]): List of the prediction timesteps to compute the metrics for
        """
        self.num_iterations['distance_metric'] = iterations
        target_frames =  np.ceil(np.array(prediction_timesteps) / self.step_size).astype(int)
        prediction_steps_real = (target_frames * self.step_size).tolist()
        if prediction_steps_real != prediction_timesteps:
            print_(f"Goal prediction timesteps: {prediction_timesteps} not reachable using timesteps: {prediction_steps_real}","warn")
        self.target_frames['distance_metric'] = target_frames.tolist()
        self.prediction_timesteps['distance_metric'] = prediction_steps_real
        self.distance_metrics = metric_names
        self.distance_metric_active = True
        self.variable_window = variable_window
        for t in prediction_timesteps:
            self.evaluation_results['distance_metric'][t] = {}
            for m in metric_names:
                self.evaluation_results['distance_metric'][t][m] = []
    
    def initialize_visualization_2d(self,
                                    prediction_timesteps: List[int],
                                    variable_window: bool = False):
        """
            Initialize the 2d visualization

            Arguments:
                prediction_timesteps (List[int]): List of the prediction timesteps to include in the visualization.
                variable_window (Optional[bool]): Whether to visualize the model uses a variable temporal window.
        """
        target_frames =  np.ceil(np.array(prediction_timesteps) / self.step_size).astype(int)
        prediction_steps_real = (target_frames * self.step_size).tolist()
        if prediction_steps_real != prediction_timesteps:
            print_(f"Goal prediction timesteps: {prediction_timesteps} not reachable using timesteps: {prediction_steps_real}","warn")
        self.target_frames['visualization_2d'] = target_frames.tolist()
        self.prediction_timesteps['visualization_2d'] = prediction_steps_real
        self.variable_window = variable_window
        self.visualization_2d_active = True
    
    def initialize_visualization_3d(self,
                                    interactive: bool,
                                    max_length: int,
                                    overlay: Optional[bool]=False,
                                    variable_window: Optional[bool] = False,
                                    ):
        """
            Initialize the 2d visualization

            Arguments:
                interactive (bool): Whether to visualize the sequence in an interactive matplotlib window
                max_length (int): The maximum length of the sequence to visualize. Determines the prediction horizon.
                overlay (Optional[bool]): Whether to overlay the sequence predicted and ground truth sequence.
        """
        target_frame = np.ceil(max_length / self.step_size).astype(int)
        prediction_steps_real = (target_frame * self.step_size).tolist()
        if prediction_steps_real!= max_length:
            print_(f"Goal prediction timesteps: {max_length} not reachable using timesteps: {prediction_steps_real}","warn")
        self.prediction_timesteps['visualization_3d'] = prediction_steps_real
        self.target_frames['visualization_3d'] = target_frame
        self.visualization_3d_active = True
        self.interactive_visualization = interactive
        self.overlay_visualization = overlay 
        self.variable_window = variable_window
    
    def initialize_visualization_attention(self,
                                    prediction_timesteps: List[int],
                                    vanilla: Optional[bool] = False,
                                    variable_window: bool = False):
        """
            Initialize the visualization of the attention weights. This requires the model being initialized to give a full return.

            Arguments:
                prediction_timesteps (List[int]): List of the prediction timesteps to include in the visualization.
                variable_window (Optional[bool]): Whether to visualize the model uses a variable temporal window.
        """
        target_frames =  np.ceil(np.array(prediction_timesteps) / self.step_size).astype(int)
        prediction_steps_real = (target_frames * self.step_size).tolist()
        if prediction_steps_real != prediction_timesteps:
            print_(f"Goal prediction timesteps: {prediction_timesteps} not reachable using timesteps: {prediction_steps_real}","warn")
        self.target_frames['visualization_attn'] = target_frames.tolist()
        self.prediction_timesteps['visualization_attn'] = prediction_steps_real
        self.variable_window = variable_window
        self.visualization_attn_active = True
        self.vanilla_attention = vanilla


    def load_data(self,
            dataset: Literal['h36m', 'ais'],
            seed_length: int,
            target_length: Optional[int] = None,
            prediction_length: Optional[int] = None,
            down_sampling_factor: int = 1,
            sequence_spacing: Optional[int] = 1,
            actions: Optional[List[str]] = H36M_DATASET_ACTIONS,
            split_actions: Optional[bool] = False,
            representation: Optional[Literal["axis", "mat", "quat", "6d", "pos", None]] = None,
            absolute_positions: Optional[bool] = False,
            skeleton_representation: Optional[Literal['s26','s21','s16']] = 's26',
            batch_size: Optional[int] = 32,
            normalize: Optional[bool] = False,
            normalize_orientation: Optional[bool] = False,
    ):
        """
            Load the evaluation data.

            Arguments:
                dataset (Literal['h36m', 'ais']): Which dataset to load
                seed_length (int): The length of the seed sequence
                target_length (Optional[int]): The length of the target sequence
                prediction_length (Optional[int]): The length of the prediction sequence
                down_sampling_factor (int): The down sampling factor
                sequence_spacing (Optional[int]): The spacing between the frames
                actions (Optional[List[str]]): List of the actions to include in the dataset
                split_actions (Optional[bool]): Whether to split the actions into different datasets
                representation (Optional[Literal["axis", "mat", "quat", "6d", "pos", None]]): Which joint representation to use
                absolute_positions (Optional[bool]): Whether to use absolute positions
                skeleton_representation (Optional[Literal['s26','s21','s16']]): Which skeleton representation to use
                batch_size (Optional[int]): The batch size
                normalize (Optional[bool]): Whether to normalize the joints
                normalize_orientation (Optional[bool]): Whether to normalize the joint orientations


        """
        
        # Set the target frames and reachable timesteps given the time intervals between frames
        ##== Dataset Parameters ==##
        self.seed_length = seed_length
        self.target_length = target_length
        self.normalize = normalize
        self.skeleton_representation = skeleton_representation
        self.split_actions = split_actions
        self.representation = representation
        self.down_sampling_factor = down_sampling_factor

        self.actions = actions
        self.batch_size = batch_size
        print_(f"Load the evaluation data for each action")

        ##== Load Action dataset ==##
        if dataset == 'h36m':
            self.h36m_evaluation = True
            self.step_size = (H36M_STEP_SIZE_MS * down_sampling_factor)
            if self.target_length is None:
                if prediction_length is None:
                    print_("Please provide either a prediction length or a target length for the Dataset", "error")
                    return
                self.target_length = np.ceil(prediction_length / self.step_size).astype(int)
            if split_actions:
                self.split_actions = split_actions
                self.datasets = {}
                for a in self.actions:
                    self.datasets[a] = H36MDataset(
                        actions=[a],
                        seed_length=self.seed_length,
                        target_length=self.target_length,
                        sequence_spacing=sequence_spacing,
                        absolute_position=absolute_positions,
                        normalize_orientation=normalize_orientation,
                        down_sampling_factor=self.down_sampling_factor,
                        stacked_hourglass=True if skeleton_representation == 's16' else False,
                        rot_representation=representation,
                        is_train=False
                    )
            else:
                self.split_actions = split_actions
                self.datasets = {}
                self.datasets['overall'] =  H36MDataset(
                        seed_length=self.seed_length,
                        target_length=self.target_length,
                        sequence_spacing=sequence_spacing,
                        absolute_position=absolute_positions,
                        normalize_orientation=normalize_orientation,
                        down_sampling_factor=self.down_sampling_factor,
                        stacked_hourglass=True if skeleton_representation == 's16' else False,
                        rot_representation=representation,
                        is_train=False
                    )
            
        elif dataset == 'ais':
            self.h36m_evaluation = False
            self.step_size = 33
            if self.target_length is None:
                if prediction_length is None:
                    print_("Please provide either a prediction length or a target length for the Dataset", "error")
                    return
                self.target_length = np.ceil(prediction_length / self.step_size).astype(int)
            self.datasets = {}
            self.datasets['overall'] =  AISDataset(
                seed_length=self.seed_length,
                target_length=self.target_length,
                sequence_spacing=sequence_spacing,
                normalize_orientation=normalize_orientation,
                absolute_position=absolute_positions
            )


        self.data_loaded = True        


    def reset(self) -> None:
        """
        Reset the evaluation engine.
        """
        self.evaluation_results = {
            "long_predictions": {},
            "distance_metric": {}
        }
        for eval_type in self.evaluation_results.keys():
            if eval_type == "distance_metric":
                metric_names = self.distance_metrics
                if len(metric_names)==0:
                    continue
            else:
                metric_names = self.distribution_metrics
                if len(metric_names)==0:
                    continue
            if self.split_actions:
                for a in self.actions + ["overall"]:
                    self.evaluation_results[eval_type][a] = {}
                    for timestep in self.prediction_timesteps[eval_type]:
                        self.evaluation_results[eval_type][a][timestep] = {}
                        for m_name in metric_names:
                            self.evaluation_results[eval_type][a][timestep][m_name] = []
            else:
                self.evaluation_results[eval_type]['overall'] = {}
                for timestep in self.prediction_timesteps[eval_type]:
                    self.evaluation_results[eval_type]['overall'][timestep] = {}
                    for m_name in metric_names:
                        self.evaluation_results[eval_type]['overall'][timestep][m_name] = []
        self.evaluation_finished = False

    def print(self, file_name: str = None) -> None:
        """
        Print the evaluation results in a table to the console.
        """
        return self._print_results(file_name)

    def log_results(self, step: int) -> None:
        """
        Log the evaluation results using the global logger

        Arguments:
            step (int): The current step of the training.
        """
        logger = LOGGER
        if logger is None:
            print_(f"No logger defined, cannot log evaluation results")
        data_dir = {}
        actions = self.actions + ["overall"] if self.split_actions else ['overall']
        for eval_type in self.evaluation_results.keys():
            if eval_type == "long_predictions":
                continue
            if len(self.evaluation_results[eval_type]) == 0:
                continue
            for a in actions:
                for p in self.prediction_timesteps[eval_type]:
                    for m in self.evaluation_results[eval_type][a][p].keys():
                        data_dir[f"{a}/{p}/{m}"] = self.evaluation_results[eval_type][a][p][m]
        logger.log(data_dir, step)

    def set_normalization(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        """
        Set the mean and variance of the train dataset.
        """
        self.norm_mean = mean
        self.norm_var = var

    #####===== Getter Functions =====#####
    def get_results(self):
        """
        Returns the complete directory holding the evaluation results.

        Format:
        {
            action:
            {
                prediction_length:
                {
                    metric_name: [metric_value]
                }
            }
        }

        """
        return self.evaluation_results

    #####===== Evaluation Functions =====#####
    @log_function
    def evaluate(self, model: torch.nn.Module, data_augmentor: Optional[DataAugmentor] = None) -> None:
        """
        Evaluate the provided model on the H3.6M dataset.
        For this the evaluation needs to be initialized.
        """
        self.reset()
        model.eval()
        if self.h36m_evaluation:
            print_(f"Start evaluation on H3.6M dataset using actions: {self.actions}")
        else:
            print_(f"Start evaluation on AIS dataset")
        for action, dataset in self.datasets.items():
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
            )
            self.data_augmentor = data_augmentor
            if self.normalize:
                self.data_augmentor.set_mean_var(self.norm_mean.to(self.device), self.norm_var.to(self.device))
            if self.distance_metric_active:
                self.evaluation_loop_distance(action, model, data_loader)
            if self.long_predictions_active:
                self.evaluation_loop_distribution(action, model, data_loader)
        if self.split_actions:
            self._compute_overall_means()
        self.evaluation_finished = True
        print_(f"Evaluation finished!")

    @torch.no_grad()
    def evaluation_loop_distance(
        self,
        action: str,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
    ) -> None:
        """
        A single evaluation loop for an action.
        """
        model.eval()
        # Initialize progress bar
        dataset = self.datasets[action]
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        progress_bar = tqdm(enumerate(data_loader), total=self.num_iterations['distance_metric'])
        progress_bar.set_description(f"Evaluation {action}")

        predictions = {}
        targets = {}
        for i in self.target_frames['distance_metric']:
            predictions[i] = []
            targets[i] = []
        for batch_idx, data in progress_bar:
            if batch_idx == self.num_iterations['distance_metric']:
                break
            # Load data
            data = data.to(self.device)
            # Set input for the model
            cur_input = self.data_augmentor(data[:, : self.seed_length], is_train=False)
            # Predict future frames in an auto-regressive manner
            for i in range(1,max(self.target_frames['distance_metric']) + 1):
                # Compute the output
                output = model(cur_input)
                # Check if we want to compute metrics for this timestep
                if i in self.target_frames['distance_metric']:
                    # Compute the implemented metrics
                    if self.normalize:
                        pred = self.data_augmentor.reverse_normalization(
                            output[:, -1].detach().cpu()
                        )
                    else:
                        pred = output[:, -1].detach().cpu()
                    predictions[i].append(pred)
                    targets[i].append(data[:, self.seed_length + i -1].detach().cpu())
                # Update model input for auto-regressive prediction
                if not self.variable_window:
                    cur_input = torch.concatenate([cur_input[:,1:], output[:, -1].unsqueeze(1)], dim=1)
                else:
                    cur_input = torch.concatenate([cur_input, output[:, -1].unsqueeze(1)], dim=1)

        # Compute the distance metrics for each timestep
        for i, frame in enumerate(self.target_frames['distance_metric']):
            timestep = self.prediction_timesteps['distance_metric'][i]
            timestep_prediction = torch.stack(predictions[frame])
            timestep_target = torch.stack(targets[frame])
            timestep_prediction = torch.flatten(
                timestep_prediction, start_dim=0, end_dim=1
            )
            timestep_target = torch.flatten(
                timestep_target, start_dim=0, end_dim=1
            )

            self.evaluation_results['distance_metric'][action][timestep].update(evaluate_distance_metrics(
                timestep_prediction,
                timestep_target,
                reduction="mean",
                metrics=self.distance_metrics,
                representation=self.representation,
                s16_mask=True if self.skeleton_representation=="s26" else False,
            ))

    @torch.no_grad()
    def evaluation_loop_distribution(
        self,
        action: str,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader) -> None:
        """
        A single evaluation loop for an action.

        TODO: 
                - Calculate entropy for each predicted frame wrt the frames predicted beforehand.
                - Create 1-second bins from the entire predicted sequence and calculate the 
                  symmetric KL-divergence between the bins and randomly sampled 1-second sequences 
                  from the test dataset.
        """

        model.eval()
        # Initialize progress bar
        dataset = self.datasets[action]
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        progress_bar = tqdm(enumerate(data_loader), total=self.num_iterations['long_predictions'])
        progress_bar.set_description(f"Evaluation {action}")
        
        # Get distribution values of the dataset which are loaded from disk by the Session class
        entropy = self.distribution_values['entropy']
        kld = self.distribution_values['kld']
        test_ps = self.distribution_values['test_ps']

        sequences =  [[] for _ in range(self.num_iterations['long_predictions'])]
        for j, (batch_idx, data) in enumerate(progress_bar):
            if batch_idx == self.num_iterations['long_predictions']:
                break
            # Load data
            data = data.to(self.device)
            # Set input for the model
            cur_input = self.data_augmentor(data[:, : self.seed_length], is_train=False)
            # Predict future frames in an auto-regressive manner
            for i in tqdm(range(1,max(self.target_frames['long_predictions']) + 1), total=max(self.target_frames['long_predictions'])):
                # Compute the output
                output = model(cur_input)
                # Check if we want to compute metrics for this timestep
                if i in self.target_frames['long_predictions']:
                    # Compute the implemented metrics
                    if self.normalize:
                        pred = self.data_augmentor.reverse_normalization(
                            output[:, -1].detach().cpu()
                        )
                    else:
                        pred = output[:, -1].detach().cpu()
                    # Change representation to joint position if needed
                    # This is because the reference entropy on the dataset is also calculated using joint positions
                    if self.representation != 'pos':
                        pred, _ = h36m_forward_kinematics(pred, self.representation)
                    # Add predicted frame to predicted_frames
                    sequences[j].append(pred.squeeze())
                    
                # Update model input for auto-regressive prediction
                if not self.variable_window:
                    cur_input = torch.concatenate([cur_input[:,1:], output[:, -1].unsqueeze(1)], dim=1)
                else:
                    cur_input = torch.concatenate([cur_input, output[:, -1].unsqueeze(1)], dim=1)
        # Compute distribution metrics
        sequences_tensor = torch.stack([torch.stack(sequence) for sequence in sequences])
        ent_per_seq_len = []
        # Compute entropy
        for j in range(sequences_tensor.shape[0]):
            cur_sequence = sequences_tensor[j]
            cur_ent = []
            for i in range(1, sequences_tensor.shape[1] + 1):
                # Calculate the power spectrum of the predicted sequences
                ps = power_spectrum(cur_sequence[:i].unsqueeze(0))
                # Calculate the entropy of the predicted sequences
                entropy_pred = ps_entropy(ps)
                cur_ent.append(torch.mean(entropy_pred))
            ent_per_seq_len.append(torch.stack(cur_ent))
        ent_per_seq_len = torch.stack(ent_per_seq_len)
        # Get mean over all sequences
        ent_per_seq_len = torch.mean(ent_per_seq_len, dim=0)
        # Draw plot of entropy per sequence length
        # Comparing it to the reference entropy
        entropies = [entropy.item() for _ in range(len(ent_per_seq_len))]


        # Create 25 frame bins for each sequence in sequences_tensor
        # Calculate the power spectrum of each bin
        # Calculate the KL-divergence between the bins and the test dataset
        bins = [[sequences_tensor[i, j*25:j*25+25] for j in range(int(sequences_tensor.shape[1] / 25))] for i in range(sequences_tensor.shape[0])]
        klds = []
        for i in range(len(bins)):
            cur_klds = []
            for j in range(len(bins[i])):
                # Calculate the power spectrum of the predicted sequences
                ps = power_spectrum(bins[i][j].unsqueeze(0))
                # Calculate the KL-divergence between the bins and the test dataset
                cur_klds.append(torch.mean(ps_kld(ps, test_ps)))

            klds.append((torch.stack(cur_klds)))
        klds = torch.stack(klds)
        # Get mean over all sequences
        klds = torch.mean(klds, dim=0)
        # Draw plot of kld per bin
        # Comparing it to the reference kld
        klds_ref = [kld.item() for _ in range(len(klds))]

        # Reset evaluation results of long predictions
        self.evaluation_results['long_predictions'] = {}
        self.evaluation_results['long_predictions'][action] = {}
        # Add results to evaluation results
        self.evaluation_results['long_predictions'][action]['entropy'] = ent_per_seq_len.tolist()
        self.evaluation_results['long_predictions'][action]['entropy_baseline'] = entropies
        self.evaluation_results['long_predictions'][action]['kld'] = klds.tolist()
        self.evaluation_results['long_predictions'][action]['kld_baseline'] = klds_ref
        pass

    ###=== Visualization Functions ===###
    def visualize(self, model: torch.nn.Module, num_visualizations: int = 1, data_augmentor: DataAugmentor = None) -> None:
        model.eval()
        if self.h36m_evaluation:
            print_(f"Start visualization on H3.6M dataset using actions: {self.actions}")
        else:
            print_(f"Start visualization on the AIS dataset")
        self.data_augmentor = data_augmentor
        for action, dataset in self.datasets.items():
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
            )
            if self.visualization_2d_active:
                self.visualization_2d_loop(model, action, num_visualizations, data_loader)
            if self.visualization_3d_active:
                self.visualization_3d_loop(model, action, num_visualizations, data_loader)
        self.evaluation_finished = True
        print_(f"Evaluation finished!")

    def visualization_2d_loop(
        self,
        model: torch.nn.Module,
        action: str,
        num: int,
        data_loader: torch.utils.data.DataLoader,
    ):
        if num == 0:
            return

        model.eval()
        # Initialize progress bar
        dataset = self.datasets[action]
        data_loader = DataLoader(dataset, batch_size=num, shuffle=True)

        predictions = []
        targets = []

        data = next(iter(data_loader))
        # data = data.reshape([1, *data.shape])
        # Load data
        data = data.to(self.device)
        # Set input for the model
        cur_input = self.data_augmentor(data[:, : self.seed_length])
        # Predict future frames in an auto-regressive manner
        for i in range(1, max(self.target_frames['visualization_2d']) + 1):
            # Compute the output
            output = model(cur_input)
            # Check if we want to compute metrics for this timestep
            if i in self.target_frames['visualization_2d']:
                # Compute the implemented metrics
                if self.normalize:
                    pred = self.data_augmentor.reverse_normalization(
                        output[:, -1].detach().cpu()
                    )
                else:
                    pred = output[:, -1].detach().cpu()
                predictions.append(pred)
                targets.append(data[:, self.seed_length + i - 1].detach().cpu())
            # Update model input for auto-regressive prediction
            if not self.variable_window:
                    cur_input = torch.concatenate([cur_input[:,1:], output[:, -1].unsqueeze(1)], dim=1)
            else:
                cur_input = torch.concatenate([cur_input, output[:, -1].unsqueeze(1)], dim=1)
        # Create visualizations
        logger = LOGGER
        
        predictions = torch.stack(predictions)  # [seq_len, batch_size, num_joints, joint_dim]
        predictions = torch.transpose(predictions, 0, 1)  # [batch_size, seq_len, num_joints, joint_dim]
        # Get as many batches as specified in self.visualizations_per_batch
        targets = torch.stack(targets)  # [seq_len, batch_size, num_joints, joint_dim]
        targets = torch.transpose(targets, 0, 1)  # [batch_size, seq_len, num_joints, joint_dim]
        
        # Add seed data in front of predictions and targets
        # seed_data is of shape [batch_size, seq_len, num_joints, joint_dim]
        # seed_data should be added in seq_len dimension in front of the rest of the data
        seed_data = data[:, : self.seed_length].detach().cpu()
        
        predictions = torch.cat((seed_data, predictions), 1)
        targets = torch.cat((seed_data, targets), 1)

        # Create millisecond timesteps backward for the seed data
        seed_timesteps = [i * -40 for i in range(self.seed_length)]
        seed_timesteps.reverse()
        # Insert seed timestepsbefore the prediction timesteps
        seed_timesteps.extend(self.prediction_timesteps['visualization_2d'])
        # Add constant to each timestep to make them positive
        seed_timesteps = [i + 40 * (self.seed_length -1) for i in seed_timesteps]
        time_steps_ms = seed_timesteps

        # Get joint positions from forward kinematics for visualization
        if self.representation!= 'pos':
            targets, _ = h36m_forward_kinematics(targets, self.representation)
            targets /= 1000
            predictions, _ = h36m_forward_kinematics(predictions, self.representation)
            predictions /= 1000
            seed_data, _ = h36m_forward_kinematics(seed_data, self.representation)
            seed_data /= 1000
        # Get the skeleton model for the visualization
        skeleton_structure = self._get_skeleton_model()
        # Get parents for drawing
        parent_ids = self._get_skeleton_parents()
        # Create visualizations
        self.vis2d_figures = []
        for i in range(num):
            comparison_img = compare_sequences_plotly(
                sequence_names=["ground truth", "prediction"],
                sequences=[targets[i], predictions[i]],
                time_steps_ms=time_steps_ms,
                skeleton_structure=skeleton_structure,
                parent_ids=parent_ids,
                prediction_positions=[None, self.seed_length]
            )
            # Log comparison image
            if logger is not None:
                logger.log_image(name=f"vis_{action}_i", image=comparison_img)
            # Show image
            save_dir = logger.get_path('visualization')
            fname = f"h36m_{action}" if self.h36m_evaluation else f"ais_{action}"
            save_to = os.path.join(save_dir, "sequences", fname + "_skeleton")
            # Create save_to directory if it does not exist
            if not os.path.exists(save_to):
                os.makedirs(save_to)
            # Store image in that directory
            pil_image = Image.fromarray(comparison_img)
            self.vis2d_figures.append(pil_image)
            pil_image.save(os.path.join(save_to, f"sequence_{i:0>4}.png"))
            # Image.fromarray(comparison_img).show()
    
    def visualization_3d_loop(
        self,
        model: torch.nn.Module,
        action: str,
        num: int,
        data_loader: torch.utils.data.DataLoader):
        if num == 0:
            return
        model.eval()
        # Initialize progress bar
        dataset = self.datasets[action]
        data_loader = DataLoader(dataset, batch_size=num, shuffle=True)

        predictions = {}
        targets = {}
        predictions = []
        targets = []

        max_length = self.target_frames['visualization_3d']
        # Get a single batch from the data loader
        data = next(iter(data_loader))

        predictions = []
        targets = []
        # Load data
        data = data.to(self.device)
        # Set input for the model
        seed_data = data[:, : self.seed_length]
        cur_input = self.data_augmentor(seed_data, is_train=False)
        # Predict future frames in an auto-regressive manner
        for i in range(max_length):
            # Compute the output
            output = model(cur_input)
            # Check if we want to compute metrics for this timestep
            # Compute the implemented metrics
            if self.normalize:
                pred = self.data_augmentor.reverse_normalization(
                    output[:, -1].detach().cpu()
                )
            else:
                pred = output[:, -1].detach().cpu()
            predictions.append(pred)
            targets.append(data[:, self.seed_length + i].detach().cpu())
            # Update model input for auto-regressive prediction
            if not self.variable_window:
                    cur_input = torch.concatenate([cur_input[:,1:], output[:, -1].unsqueeze(1)], dim=1)
            else:
                cur_input = torch.concatenate([cur_input, output[:, -1].unsqueeze(1)], dim=1)


        predictions = torch.stack(predictions)
        predictions = torch.transpose(predictions, 0, 1) 
            
        targets = torch.stack(targets)
        targets = torch.transpose(targets, 0, 1)
        if self.representation!= 'pos':
            targets, _ = h36m_forward_kinematics(targets, self.representation)
            targets /= 1000
            predictions, _ = h36m_forward_kinematics(predictions, self.representation)
            predictions /= 1000
            seed_data, _ = h36m_forward_kinematics(seed_data, self.representation)
            seed_data /= 1000

        parent_ids = self._get_skeleton_parents()
        logger = LOGGER
        adjust_dim = [0,1,2]
        seed_data = seed_data[...,adjust_dim]
        # Detach seed_data
        seed_data = seed_data.detach().cpu()
        for i in range(num):
            cur_pred = predictions[i,...,adjust_dim].numpy()
            cur_pred = np.concatenate([seed_data[i].numpy(), cur_pred], axis=0)
            cur_target = targets[i,...,adjust_dim].numpy()
            cur_target = np.concatenate([seed_data[i].numpy(), cur_target], axis=0)
            if not self.interactive_visualization:
                save_dir = logger.get_path('visualization')
                fname = f"h36m_{action}_{i}" if self.h36m_evaluation else f"ais_{i}"
            else:
                save_dir = None
                fname = None
            animate_pose_matplotlib(
                positions = (cur_pred, cur_target),
                titles = ('Predicted', 'Ground Truth'),
                fig_title = 'Model Evaluation',
                colors = ('g', 'g'),
                parents = parent_ids,
                change_color_after_frame=(self.seed_length, None),
                out_dir=save_dir,
                step_size=self.step_size,
                fname = fname,
                color_after_change='r',
                overlay=self.overlay_visualization,
                fps=5,
                show_axis=True,
                constant_limits=True
            )

    def visualize_attention_loop(
        self,
        model: torch.nn.Module,
        action: str,
        num: int,
        data_loader: torch.utils.data.DataLoader):
        """
            Produce a visualization of the attention weights.
        """
        def _save_attn(output, num_block):
            if self.vanilla_attention:
                attn_weights[num_block]["temporal"].append(output[1].detach().cpu().numpy())
            else:
                attn_weights[num_block]["temporal"].append(output[1].detach().cpu().numpy())
                attn_weights[num_block]["spatial"].append(output[2].detach().cpu().numpy())
        def _attn_hook_1(module, input, output):
            _save_attn(output, 1)
        def _attn_hook_2(module, input, output):
            _save_attn(output, 2)
        def _attn_hook_4(module, input, output):
            _save_attn(output, 4)
        def _attn_hook_8(module, input, output):
            _save_attn(output, 8)

        if num == 0:
            return

        model.eval()
        import ipdb; ipdb.set_trace()
        # Initialize progress bar
        dataset = self.datasets[action]
        data_loader = DataLoader(dataset, batch_size=num, shuffle=True)

        attn_weights = {
            "1": {
                "spatial": [],
                "temporal": []
            },
            "2": {
                "spatial": [],
                "temporal": []
            },
            "4": {
                "spatial": [],
                "temporal": []
            },
            "8": {
                "spatial": [],
                "temporal": []
            }
        }


        data = next(iter(data_loader))
        # data = data.reshape([1, *data.shape])
        # Load data
        data = data.to(self.device)
        # Set input for the model
        cur_input = self.data_augmentor(data[:, : self.seed_length])
        # Register the forward hooks on the model to retrieve the attention outputs
        model.attnBlocks[1].register_forward_hook(_attn_hook_1)
        model.attnBlocks[2].register_forward_hook(_attn_hook_2)
        model.attnBlocks[4].register_forward_hook(_attn_hook_4)
        model.attnBlocks[8].register_forward_hook(_attn_hook_8)
        # Predict the future steps and save attention weights
        for i in range(1, max(self.target_frames['visualization_attn']) + 1):
            # Compute the output
            output = model(cur_input)
            # Check if we want to compute metrics for this timestep
            if i in self.target_frames['visualization_attn']:
                # Compute the implemented metrics
                if self.normalize:
                    pred = self.data_augmentor.reverse_normalization(
                        output[:, -1].detach().cpu()
                    )
                else:
                    pred = output[:, -1].detach().cpu()
            # Update model input for auto-regressive prediction
            if not self.variable_window:
                    cur_input = torch.concatenate([cur_input[:,1:], output[:, -1].unsqueeze(1)], dim=1)
            else:
                cur_input = torch.concatenate([cur_input, output[:, -1].unsqueeze(1)], dim=1)

        logger = LOGGER
        save_dir = logger.get_path('visualization')
        parents = self._get_skeleton_parents()
        for i in range(num):
            attn_temporal = attn_weights["1"]["temporal"]
            attn_spatial = attn_weights["1"]["spatial"]
            fig = visualize_attention(
                temporal_attention=attn_temporal,
                spatial_attention=attn_spatial,
                skeleton_parents=parents,
                timesteps=self.prediction_timesteps['visualization_attn']
            )
            fname = P(save_dir) / f"attn_vis_{i}"
            fig.savefig(fname)
            

    #####===== Utility Functions =====#####
    def _reduce_action_metrics(self, sub_type: str) -> None:
        """
        Compute the mean over the metrics logged for several iterations
        """
        for eval_type in self.evaluation_results.keys():
            if len(self.evaluation_results[eval_type])==0 or sub_type not in self.evaluation_results[eval_type].keys():
                continue
            for pred_length in self.evaluation_results[eval_type][sub_type].keys():
                for metric_name in self.evaluation_results[eval_type][sub_type][pred_length].keys():
                    self.evaluation_results[eval_type][sub_type][pred_length][metric_name] = np.mean(
                        self.evaluation_results[eval_type][sub_type][pred_length][metric_name]
                    )

    def _compute_overall_means(self):
        """
        Compute the mean over all actions.
        """
        for eval_type in self.evaluation_results.keys():
            if len(self.evaluation_results[eval_type])==0:
                continue
            for pred_length in self.evaluation_results[eval_type]["overall"].keys():
                for metric_name in self.evaluation_results[eval_type]["overall"][pred_length].keys():
                    metric_data = [
                        self.evaluation_results[eval_type][a][pred_length][metric_name]
                        for a in self.actions
                    ]
                    self.evaluation_results[eval_type]["overall"][pred_length][metric_name] = np.mean(
                        metric_data
                    )

    def _print_results(self, file_name: str = None) -> None:
        """
        Print the results into the console using the PrettyTable library.
        """
        for eval_type in self.evaluation_results.keys():
            if eval_type == 'long_predictions':
                continue
            if len(self.evaluation_results[eval_type])==0:
                continue
            for a in self.evaluation_results[eval_type].keys():
                if a == "overall":
                    print_(f"Average over all actions:", "info", file_name, "log")
                else:
                    print_(f"Evaluation results for action {a}:", "info", file_name, "log")
                table = PrettyTable()
                if eval_type == 'distance_metric':
                    table.field_names = ["Pred. length"] + list(self.distance_metrics)
                for pred_length in self.evaluation_results[eval_type][a].keys():
                    table.add_row(
                        [pred_length]
                        + list(self.evaluation_results[eval_type][a][pred_length].values())
                    )
                print_(table, "info", file_name, "log")
    
    def _get_skeleton_model(self) -> dict:
        if self.skeleton_representation == "s26":
            return H36M_SKELETON_STRUCTURE
        elif self.skeleton_representation == "s21":
            return H36M_NON_REDUNDANT_SKELETON_STRUCTURE
        elif self.skeleton_representation == "s16":
            return SH_SKELETON_STRUCTURE
        else:
            raise ValueError(f"Unknown skeleton model: {self.skeleton_representation}")
        
    def _get_skeleton_parents(self) -> list:
        if self.skeleton_representation == "s26":
            return H36M_SKELETON_PARENTS
        elif self.skeleton_representation == "s21":
            return H36M_NON_REDUNDANT_PARENT_IDS.values()
        elif self.skeleton_representation == "s16":
            return SH_SKELETON_PARENTS
        else:
            raise ValueError(f"Unknown skeleton model: {self.skeleton_representation}")


