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
    H36MDataset,
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

from .visualizer import(
    compare_sequences_plotly
)

METRICS_IMPLEMENTED = {
    "geodesic_distance": geodesic_distance,
    "positional_mse": positional_mse,
    "euler_error": euler_angle_error,
    "auc": accuracy_under_curve,
}
VISUALIZATION_IMPLEMENTED = {"3dpose": geodesic_distance}  # placeholder


class EvaluationEngineActive:
    """
    Active evaluation engine that directly computes the model outputs.
    Doing so we are able to compute more high-level metrics on the H3.6M dataset.

    Here we compute all defined metrics, instead of a small subset.
    """

    def __init__(self,
                iterations: int,
                prediction_timesteps: List[int],
                metric_names: List[str],
                distribution_metric_names: List[str] = ["ps_entropy", "ps_kld"],
                visualizations_per_batch: int = 0,
                calculate_distribution_metrics: bool = False, 
                device: str = "cpu"):
        """
        Initialize the active evaluation engine.
        """
        ##== Evaluation Parameters ==##
        self.prediction_timesteps = prediction_timesteps
        self.iterations = iterations
        self.evaluation_finished = False
        self.metric_names = metric_names
        self.distribution_metric_names = distribution_metric_names
        self.visualizations_per_batch = visualizations_per_batch
        self.calculate_distribution_metrics = calculate_distribution_metrics
        self.device = torch.device(device)
        self.initialized = False


    def load_data(self,
            dataset: Literal['h36m', 'ais'],
            seed_length: int,
            down_sampling_factor: int = 1,
            actions: Optional[List[str]] = H36M_DATASET_ACTIONS,
            split_actions: Optional[bool] = False,
            representation: Optional[Literal["axis", "mat", "quat", "6d", "pos", None]] = None,
            stacked_hourglass: Optional[bool] = False,
            batch_size: Optional[int] = 32,
            normalize: Optional[bool] = False,
    ):
        
        # Set the target frames and reachable timesteps given the time intervals between frames
        self.target_frames = np.ceil(
            np.array(self.prediction_timesteps) / (H36M_STEP_SIZE_MS * down_sampling_factor)
        ).astype(int)
        self.target_timesteps = (
            self.target_frames * (H36M_STEP_SIZE_MS * down_sampling_factor)
        ).tolist()
        self.target_frames = self.target_frames.tolist()
        if self.prediction_timesteps != self.target_timesteps:
            print_(
                f"Goal prediction timesteps: {self.prediction_timesteps} not reachable using timesteps: {self.target_timesteps}",
                "warn",
            )
        self.last_target_frame = max(self.target_frames)
        self.target_length = math.ceil(
            max(self.prediction_timesteps) / (H36M_STEP_SIZE_MS * down_sampling_factor)
        )
        ##== Dataset Parameters ==##
        self.seed_length = seed_length
        self.normalize = normalize
        self.split_actions = split_actions
        self.representation = representation
        self.down_sampling_factor = down_sampling_factor
        self.actions = actions
        self.batch_size = batch_size
        print_(f"Load the evaluation data for each action")

        self.total_iterations = self.iterations

        ##== Load Action dataset ==##
        if dataset == 'h36m':
            if split_actions:
                self.datasets = {}
                for a in self.actions:
                    self.datasets[a] = H36MDataset(
                        actions=[a],
                        seed_length=self.seed_length,
                        target_length=self.target_length,
                        down_sampling_factor=self.down_sampling_factor,
                        stacked_hourglass=stacked_hourglass,
                        rot_representation=representation,
                        is_train=False
                    )
                self.evaluation_results = {}
                for a in actions + ["overall"]:
                    self.evaluation_results[a] = {}
                    for timestep in self.target_timesteps:
                        self.evaluation_results[a][timestep] = {}
                        for metric_names in self.metric_names:
                            self.evaluation_results[a][timestep][metric_names] = []
            else:
                self.datasets = {}
                self.datasets['overall'] =  H36MDataset(
                        actions=actions,
                        seed_length=self.seed_length,
                        target_length=self.target_length,
                        down_sampling_factor=self.down_sampling_factor,
                        stacked_hourglass=stacked_hourglass,
                        rot_representation=representation,
                        is_train=False
                    )
                self.evaluation_results = {}
                self.evaluation_results['overall'] = {}
                for timestep in self.target_timesteps:
                    self.evaluation_results['overall'][timestep] = {}
                    for metric_names in self.metric_names:
                        self.evaluation_results['overall'][timestep][metric_names] = []
        elif dataset == 'ais':
            return


    def reset(self) -> None:
        """
        Reset the evaluation engine.
        """
        if self.split_actions:
            self.evaluation_results = {}
            for a in self.actions + ["overall"]:
                self.evaluation_results[a] = {}
                for timestep in self.target_timesteps:
                    self.evaluation_results[a][timestep] = {}
                    for metric_names in self.metric_names:
                        self.evaluation_results[a][timestep][metric_names] = []
        else:
            self.evaluation_results = {}
            self.evaluation_results['overall'] = {}
            for timestep in self.target_timesteps:
                self.evaluation_results['overall'][timestep] = {}
                for metric_names in self.metric_names:
                    self.evaluation_results['overall'][timestep][metric_names] = []
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
        for a in actions:
            for p in self.target_timesteps:
                for m in self.metric_names:
                    data_dir[f"{a}/{p}/{m}"] = self.evaluation_results[a][p][m]
        logger.log(data_dir, step)

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
    def evaluate(
        self,
        model: torch.nn.Module,
        mode: Optional[Literal["standard", "long_prediction"]] = "standard",
    ):
        """
        Evaluate the provided model on the H3.6M dataset.
        For this the evaluation needs to be initialized.
        """
        self.reset()

        model.eval()
        print_(f"Start evaluation on actions: {self.actions}")
        for action, dataset in self.datasets.items():
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
            )
            self.data_augmentor = DataAugmentor(normalize=self.normalize)
            if self.normalize:
                mean, var = dataset.get_mean_variance()
                self.data_augmentor.set_mean_var(mean, var)
            self.evaluation_loop(action, model, data_loader)
        if self.split_actions:
            self._compute_overall_means()
        self.evaluation_finished = True
        print_(f"Evaluation finished!")

    @torch.no_grad()
    def evaluation_loop(
        self,
        action: str,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        mode: Optional[Literal["standard", "long_prediction"]] = "standard",
    ) -> None:
        """
        A single evaluation loop for an action.
        """

        model.eval()
        # Initialize progress bar
        dataset = self.datasets[action]
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        progress_bar = tqdm(enumerate(data_loader), total=self.iterations)
        progress_bar.set_description(f"Evaluation {action}")

        predictions = {}
        targets = {}
        for i in self.target_frames:
            predictions[i] = []
            targets[i] = []

        for batch_idx, data in progress_bar:
            if batch_idx == self.iterations:
                break
            # Load data
            data = data.to(self.device)
            # Set input for the model
            cur_input = self.data_augmentor(data[:, : self.seed_length])
            # Predict future frames in an auto-regressive manner
            for i in range(1, self.last_target_frame + 1):
                # Compute the output
                output = model(cur_input)
                # Check if we want to compute metrics for this timestep
                if i in self.target_frames:
                    # Compute the implemented metrics
                    if self.normalize:
                        pred = self.data_augmentor.reverse_normalization(
                            output[:, -1].detach().cpu()
                        )
                    else:
                        pred = output[:, -1].detach().cpu()
                    predictions[i].append(pred)
                    targets[i].append(data[:, self.seed_length + i - 2].detach().cpu())
                # Update model input for auto-regressive prediction
                cur_input = output
        # Compute the distance metrics for each timestep
        for frame in self.target_frames:
            timestep = frame * (H36M_STEP_SIZE_MS * self.down_sampling_factor)
            timestep_prediction = torch.flatten(
                torch.stack(predictions[frame]), start_dim=0, end_dim=1
            )
            timestep_target = torch.flatten(
                torch.stack(targets[frame]), start_dim=0, end_dim=1
            )
            self.evaluation_results[action][timestep]["distance_metrics"] = evaluate_distance_metrics(
                timestep_prediction,
                timestep_target,
                reduction="mean",
                metrics=self.metric_names,
                representation=self.representation,
            )
        if self.visualizations_per_batch > 0:
            # Create visualizations
            for frame in self.target_frames:
                predictions = torch.stack(predictions[frame], start_dim=0, end_dim=1)  # [seq_len, batch_size, num_joints, joint_dim]
                predictions = torch.transpose(predictions, 0, 1)  # [batch_size, seq_len, num_joints, joint_dim]
                # Get as many batches as specified in self.visualizations_per_batch
                predictions = predictions[:self.visualizations_per_batch]
                targets = torch.stack(targets[frame], start_dim=0, end_dim=1)  # [seq_len, batch_size, num_joints, joint_dim]
                targets = torch.transpose(targets, 0, 1)  # [batch_size, seq_len, num_joints, joint_dim]
                # Get first 4 batches
                targets = targets[:self.visualizations_per_batch]
                # Create visualizations
                for i in range(self.visualizations_per_batch):
                    comparison_img = compare_sequences_plotly(
                        sequence_names=["ground truth", "prediction"],
                        sequences=[targets[i], predictions[i]],
                        time_steps_ms=self.target_timesteps,
                    )
                    # Log comparison image
                    logger = LOGGER
                    if logger is not None:
                        logger.log_image(name=f"{action}_{frame}_{i}", image=comparison_img)
        if self.calculate_distribution_metrics:
            for frame in self.target_frames:
                timestep = frame * (H36M_STEP_SIZE_MS * self.down_sampling_factor)
                timestep_prediction = torch.stack(predictions[frame])
                timestep_target = torch.stack(targets[frame])
                self.evaluation_results[action][timestep]["distribution_metrics"] = evaluate_distribution_metrics(
                    timestep_prediction,
                    timestep_target,
                    reduction="mean",
                    metrics=self.distribution_metric_names,
                )



    #####===== Utility Functions =====#####

    def _reduce_action_metrics(self, action: str) -> None:
        """
        Compute the mean over the metrics logged for several iterations
        """
        for pred_length in self.evaluation_results[action].keys():
            for metric_name in self.evaluation_results[action][pred_length].keys():
                self.evaluation_results[action][pred_length][metric_name] = np.mean(
                    self.evaluation_results[action][pred_length][metric_name]
                )

    def _compute_overall_means(self):
        """
        Compute the mean over all actions.
        """
        for pred_length in self.evaluation_results["overall"].keys():
            for metric_name in self.evaluation_results["overall"][pred_length].keys():
                metric_data = [
                    self.evaluation_results[a][pred_length][metric_name]
                    for a in self.actions
                ]
                self.evaluation_results["overall"][pred_length][metric_name] = np.mean(
                    metric_data
                )

    def _print_results(self) -> None:
        """
        Print the results into the console using the PrettyTable library.
        """
        if not self.evaluation_finished:
            return
        for a in self.evaluation_results.keys():
            if a == "overall":
                print(f"Average over all actions:")
            else:
                print_(f"Evaluation results for action {a}:")
            table = PrettyTable()
            table.field_names = ["Pred. length"] + list(self.metric_names)
            for pred_length in self.evaluation_results[a].keys():
                table.add_row(
                    [pred_length]
                    + list(self.evaluation_results[a][pred_length].values())
                )
            print_(table)


class EvaluationEnginePassive:
    """
    A passive evaluation engine meaning that it only receives output and targets and computes the metrics and visualizations.
    This only enables to compute simple metrics during training.
    """

    def __init__(
        self,
        metric_names: Optional[List[str]] = None,
        visualization_names: Optional[List[str]] = None,
        skeleton_model: Optional[Literal["s26", None]] = None,
        representation: Optional[Literal["axis", "mat", "quat", "6d"]] = "mat",
        keep_log: Optional[bool] = False,
        device: str = "cpu",
    ) -> None:
        self.metric_names = metric_names
        self.visualization_names = visualization_names
        self.representation = representation
        self.skeleton_model = skeleton_model
        self.keep_log = keep_log
        self.output_log = None
        self.target_log = None
        self.device = torch.device(device)

    def log(
        self,
        output: Union[torch.Tensor, List[torch.Tensor]],
        target: Union[torch.Tensor, List[torch.Tensor]],
        single_iteration: Optional[bool] = True,
        is_batched: Optional[bool] = True,
    ) -> None:
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
    def compute(
        self,
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute the quantitative metrics for the model.
        """
        if metric_names is None and self.metric_names is None:
            print_("No metric provided to compute for the EvaluationEngine.")
            return
        if metric_names is None:
            metric_names = self.metric_names

        results = evaluate_distance_metrics(
            self.output_log,
            self.target_log,
            metric_names,
            reduction="mean",
            representation=self.representation,
        )
        return results

    @log_function
    def visualize(
        self, visualization_name: Optional[List[str]] = None
    ) -> Dict[str, Any]:
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
