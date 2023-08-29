"""
Metrics for quantitative evaluation of the results ported to PyTorch and adapted.
Following https://github.com/eth-ait/motion-transformer/blob/master/metrics/motion_metrics.py.
@Author: Leon Herbrik
"""

import numpy as np
import torch
from typing import Optional, Literal, Union

from ..utils import print_
from ..data_utils import matrix_to_euler_angles, matrix_to_axis_angle

def pck(
    predictions: torch.tensor, targets: torch.tensor, thresh: float
) -> torch.tensor:
    """
    Percentage of correct keypoints.
    Args:
        predictions: torch tensor of predicted 3D joint positions in format (..., n_joints, 3)
        targets: torch tensor of same shape as `predictions`
        thresh: radius within which a predicted joint has to lie.

    Returns:
        Percentage of correct keypoints at the given threshold level, stored in a torch tensor of shape (..., len(threshs))
    """
    dist = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=-1))
    pck = torch.mean((dist <= thresh).to(dtype=torch.float32), dim=-1)
    return pck


def geodesic_distance(predictions: torch.tensor,
                      targets: torch.tensor,
                       reduction: Optional[Literal['mean','sum','mse',None]] = None) -> Union[torch.tensor,float]:
    """
    Computes the angular distance between the target and predicted rotations. We define this as the angle that is
    required to rotate one rotation into the other. This essentially computes || log(R_diff) || where R_diff is the
    difference rotation between prediction and target.

    Args:
        predictions: torch tensor of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3) or (..., n_joints, 9)
        targets: torch tensor of same shape as `predictions`

    Returns:
        The geodesic distance for each joint as a torch tensor of shape (..., n_joints)
    """
    import ipdb; ipdb.set_trace()

    preds, _ = _fix_dimensions(predictions)
    targs, orig_shape = _fix_dimensions(targets)
    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = torch.matmul(preds, targs.transpose(-2,-1))
    angles = matrix_to_axis_angle(r)
    angles = torch.linalg.vector_norm(angles, dim=-1)

    return _reduce(angles.view(*orig_shape), reduction)


def positional_mse(predictions: torch.tensor,
                    targets: torch.tensor,
                     reduction: Optional[Literal['mean','sum','mse',None]] = None) -> Union[torch.tensor,float]:
    """
    Computes the Euclidean distance between joints in 3D space.
    Args:
        predictions: torch tensor of predicted 3D joint positions in format (..., n_joints, 3)
        targets: torch tensor of same shape as `predictions`

    Returns:
        The Euclidean distance for each joint as a torch tensor of shape (..., n_joints)
    """
    import ipdb; ipdb.set_trace()

    return _reduce(torch.sqrt(torch.sum((predictions - targets) ** 2, dim=-1)), reduction)


def euler_angle_error(predictions: torch.tensor,
                      targets: torch.tensor,
                       reduction: Optional[Literal['mean','sum','mse',None]] = None) -> torch.tensor:
    """
    Computes the Euler angle error using pytorch3d.
    Args:
        predictions: torch tensor of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
        targets: torch tensor of same shape as `predictions`

    Returns:
        The Euler angle error as a torch tensor of shape (..., )
    """
    import ipdb; ipdb.set_trace()

    n_joints = predictions.shape[-3]

    preds, _ = _fix_dimensions(predictions)
    targs, orig_shape = _fix_dimensions(targets)

    # Convert rotation matrices to Euler angles using pytorch3d
    euler_preds = matrix_to_euler_angles(preds, "ZYX")  # (N, 3)
    euler_targs = matrix_to_euler_angles(targs, "ZYX")  # (N, 3)

    # reshape to (-1, n_joints*3) to be consistent with previous work
    euler_preds = euler_preds.view(-1, n_joints * 3)
    euler_targs = euler_targs.view(-1, n_joints * 3)

    # l2 error on euler angles
    idx_to_use = torch.where(torch.std(euler_targs, dim=0) > 1e-4)[0]
    euc_error = torch.pow(euler_targs[:, idx_to_use] - euler_preds[:, idx_to_use], 2)
    euc_error = torch.sqrt(torch.sum(euc_error, dim=1))  # (-1, ...)

    # reshape to original
    return _reduce(euc_error.view(*orig_shape), reduction)

#####===== Helper Functions =====#####

def _reduce(input, reduction: Literal['mean','sum','mse', None] = None):
    """
        Reduce the output of the quantitative metrics to a scalar.
    """
    if reduction is None:
        return input
    if reduction == 'mean':
        return torch.mean(input)
    elif reduction =='sum':
        return torch.sum(input)
    elif 'mse':
        return torch.mean(torch.sqrt(torch.sum((input) ** 2, dim=-1)))
    else:
        raise NotImplementedError

def _fix_dimensions(input: torch.Tensor,):
    """
        Fix input dimensions by flattening the leading dimensions and eventually stacking a flattened rotation matrix.
    """
    shape = input.shape
    if shape[-1] == 9:
        return torch.reshape(input, (-1,3,3)), input.shape[:-1]
    elif shape[-1] == 3 and shape[-2] == 3:
        return torch.view(-1,3,3), input.shape[:-2]
    else:
        print_(f'Evaluation functions received invalid input shape: {shape}')