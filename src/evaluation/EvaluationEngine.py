import torch
from torch.utils.data import DataLoader
import math
import numpy as np
from tqdm import tqdm
import os
from prettytable import PrettyTable
from typing import Optional, Dict, Any, List, Union, Literal

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
)
from .metrics import (
    evaluate_distance_metrics,
    evaluate_distribution_metrics,
    geodesic_distance,
    positional_mse,
    euler_angle_error,
    accuracy_under_curve,
)

from .visualization import(
    compare_sequences_plotly,
)

from ..visualization import compare_skeleton, animate_pose_matplotlib

METRICS_IMPLEMENTED = {
    "geodesic_distance": geodesic_distance,
    "positional_mse": positional_mse,
    "euler_error": euler_angle_error,
    "auc": accuracy_under_curve,
}
VISUALIZATION_IMPLEMENTED = {"3dpose": geodesic_distance}  # placeholder


class EvaluationEngine:
    """
    Active evaluation engine that directly computes the model outputs.
    Doing so we are able to compute more high-level metrics on the H3.6M dataset.

    Here we compute all defined metrics, instead of a small subset.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize the active evaluation engine.
        """
        ##== Evaluation Parameters ==##
        self.visualization_2d_active = False
        self.visualization_3d_active = False
        self.long_predictions_active = False
        self.distance_metric_active = False
        self.data_loaded = False

        self.prediction_timesteps = {
            "visualization_2d": None,
            "visualization_3d": None,
            "long_predictions": None,
            "distance_metric": None
        }
        self.target_frames = {
            "visualization_2d": None,
            "visualization_3d": None,
            "long_predictions": None,
            "distance_metric": None
        }

        self.num_iterations = {
            "visualization_2d": None,
            "visualization_3d": None,
            "long_predictions": None,
            "distance_metric": None
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
                                           prediction_timesteps: List[int]):
        """
            Initialize the evaluation for long predictions using distribution metrics
        """
        self.num_iterations['long_predictions'] = iterations
        self.prediction_timesteps['long_predictions'] = prediction_timesteps
        target_frames =  np.ceil(np.array(prediction_timesteps) / self.step_size).astype(int)
        prediction_steps_real = (target_frames * self.step_size).tolist()
        if prediction_steps_real != prediction_timesteps:
            print_(f"Goal prediction timesteps: {prediction_timesteps} not reachable using timesteps: {prediction_steps_real}","warn")
        self.prediction_timesteps['visualization_2d'] = prediction_steps_real
        self.distribution_metrics = metric_names
        self.long_predictions_active = True
        for t in prediction_timesteps:
            self.evaluation_results['long_predictions'][t] = {}
            for m in metric_names:
                self.evaluation_results['long_predictions'][t][m] = []


    def initialize_distance_evaluation(self,
                                       iterations: int,
                                       metric_names: List[str],
                                       prediction_timesteps: List[int]):
        """
            Initialize the evaluation for long predictions using distribution metrics
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
        for t in prediction_timesteps:
            self.evaluation_results['distance_metric'][t] = {}
            for m in metric_names:
                self.evaluation_results['distance_metric'][t][m] = []
    
    def initialize_visualization_2d(self,
                                    prediction_timesteps: List[int]):
        """
            Initialize the 2d visualization
        """
        target_frames =  np.ceil(np.array(prediction_timesteps) / self.step_size).astype(int)
        prediction_steps_real = (target_frames * self.step_size).tolist()
        if prediction_steps_real != prediction_timesteps:
            print_(f"Goal prediction timesteps: {prediction_timesteps} not reachable using timesteps: {prediction_steps_real}","warn")
        self.target_frames['visualization_2d'] = target_frames.tolist()
        self.prediction_timesteps['visualization_2d'] = prediction_steps_real
        self.visualization_2d_active = True
    
    def initialize_visualization_3d(self,
                                    interactive: bool,
                                    max_length: int,
                                    overlay: Optional[bool]=False,
                                    ):
        """
            Initialize the 2d visualization
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
    ):
        
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
                        down_sampling_factor=self.down_sampling_factor,
                        stacked_hourglass=True if skeleton_representation == 's16' else False,
                        rot_representation=representation,
                        is_train=False
                    )
            
        elif dataset == 'ais':
            self.h36m_evaluation = False
            self.datasets = {}
            self.datasets['overall'] =  AISDataset(
                seed_length=self.seed_length,
                target_length=self.target_length,
                sequence_spacing=sequence_spacing,
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

    def print(self) -> None:
        """
        Print the evaluation results in a table to the console.
        """
        return self._print_results()

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
    def evaluate(self, model: torch.nn.Module):
        """
        Evaluate the provided model on the H3.6M dataset.
        For this the evaluation needs to be initialized.
        """
        self.reset()
        model.eval()
        if self.h36m_evaluation:
            print_(f"Start evaluation on H3.6M dataset using actions: {self.actions}")
            for action, dataset in self.datasets.items():
                data_loader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
                )
                self.data_augmentor = DataAugmentor(normalize=self.normalize)
                if self.normalize:
                    self.data_augmentor.set_mean_var(self.norm_mean.to(self.device), self.norm_var.to(self.device))
                if self.distance_metric_active:
                    self.evaluation_loop_distance(action, model, data_loader)
                if self.long_predictions_active:
                    self.evaluation_loop_distribution(action, model, data_loader)
            if self.split_actions:
                self._compute_overall_means()
        else:
            return
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
            cur_input = self.data_augmentor(data[:, : self.seed_length])
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
                cur_input = torch.concatenate([cur_input[:,1:], output[:, -1].unsqueeze(1)], dim=1)
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
            ))

    @torch.no_grad()
    def evaluation_loop_distribution(
        self,
        action: str,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader) -> None:
        """
        A single evaluation loop for an action.
        """

        model.eval()
        # Initialize progress bar
        dataset = self.datasets[action]
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        progress_bar = tqdm(enumerate(data_loader), total=self.num_iterations['long_predictions'])
        progress_bar.set_description(f"Evaluation {action}")

        predictions = {}
        targets = {}
        for i in self.target_frames['long_predictions']:
            predictions[i] = []
            targets[i] = []

        for batch_idx, data in progress_bar:
            if batch_idx == self.num_iterations['long_predictions']:
                break
            # Load data
            data = data.to(self.device)
            # Set input for the model
            cur_input = self.data_augmentor(data[:, : self.seed_length])
            # Predict future frames in an auto-regressive manner
            for i in range(1,max(self.target_frames['long_predictions']) + 1):
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
                    predictions[i].append(pred)
                    targets[i].append(data[:, self.seed_length + i -1].detach().cpu())
                # Update model input for auto-regressive prediction
                cur_input = torch.concatenate([cur_input[:,1:], output[:, -1].unsqueeze(1)], dim=1)

        # Compute the distance metrics for each timestep
        for i, frame in enumerate(self.target_frames['long_predictions']):
            timestep = self.prediction_timesteps['long_predictions']
            timestep_prediction = torch.stack(predictions[frame])
            timestep_target = torch.stack(targets[frame])
            # Needs stacked tensor instead of flattened one
            self.evaluation_results[action][timestep].update(evaluate_distribution_metrics(
                timestep_prediction,
                timestep_target,
                reduction="mean",
                metrics=self.distribution_metric_names,
            ))
            
    ###=== Visualization Functions ===###
    def visualize(self, model: torch.nn.Module, num_visualizations: int = 1) -> None:
        model.eval()
        if self.h36m_evaluation:
            print_(f"Start evaluation on H3.6M dataset using actions: {self.actions}")
            for action, dataset in self.datasets.items():
                data_loader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
                )
                self.data_augmentor = DataAugmentor(normalize=self.normalize)
                if self.normalize:
                    mean, var = dataset.get_mean_variance()
                    self.data_augmentor.set_mean_var(mean.to(self.device), var.to(self.device))
                if self.visualization_2d_active:
                    self.visualization_2d_loop(model, action, num_visualizations, data_loader)
                if self.visualization_3d_active:
                    self.visualization_3d_loop(model, action, num_visualizations, data_loader)
            if self.split_actions:
                self._compute_overall_means()
        else:
            return
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
                targets.append(data[:, self.seed_length + i].detach().cpu())
            # Update model input for auto-regressive prediction
            cur_input = torch.concatenate([cur_input[:,1:], output[:, -1].unsqueeze(1)], dim=1)
        # Create visualizations
        logger = LOGGER
        predictions = torch.stack(predictions)  # [seq_len, batch_size, num_joints, joint_dim]
        predictions = torch.transpose(predictions, 0, 1)  # [batch_size, seq_len, num_joints, joint_dim]
        # Get as many batches as specified in self.visualizations_per_batch
        targets = torch.stack(targets)  # [seq_len, batch_size, num_joints, joint_dim]
        targets = torch.transpose(targets, 0, 1)  # [batch_size, seq_len, num_joints, joint_dim]
        # Get the skeleton model for the visualization
        skeleton_structure = self._get_skeleton_model()
        # Create visualizations
        for i in range(num):
            import ipdb; ipdb.set_trace()

            comparison_img = compare_sequences_plotly(
                sequence_names=["ground truth", "prediction"],
                sequences=[targets[i], predictions[i]],
                time_steps_ms=self.prediction_timesteps['visualization_2d'],
                skeleton_structure=skeleton_structure
            )
            # Log comparison image
            if logger is not None:
                logger.log_image(name=f"vis_{action}_i", image=comparison_img)
    
    def visualization_3d_loop(
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
            cur_input = torch.concatenate([cur_input[:,1:], output[:, -1].unsqueeze(1)], dim=1)


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
        adjust_dim = [2,0,1]
        seed_data = seed_data[...,adjust_dim]

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

    #####===== Utility Functions =====#####
    def _reduce_action_metrics(self, sub_type: str) -> None:
        """
        Compute the mean over the metrics logged for several iterations
        """
        for eval_type in self.evaluation_results.keys:
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
        for eval_type in self.evaluation_results.keys:
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

    def _print_results(self) -> None:
        """
        Print the results into the console using the PrettyTable library.
        """
        for eval_type in self.evaluation_results.keys():
            if len(self.evaluation_results[eval_type])==0:
                continue
            for a in self.evaluation_results[eval_type].keys():
                if a == "overall":
                    print(f"Average over all actions:")
                else:
                    print_(f"Evaluation results for action {a}:")
                table = PrettyTable()
                if eval_type == 'long_predictions':
                    table.field_names = ["Pred. length"] + list(self.distribution_metrics)
                elif eval_type == 'distance_metric':
                    table.field_names = ["Pred. length"] + list(self.distance_metrics)
                for pred_length in self.evaluation_results[eval_type][a].keys():
                    table.add_row(
                        [pred_length]
                        + list(self.evaluation_results[eval_type][a][pred_length].values())
                    )
                print_(table)
    
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


