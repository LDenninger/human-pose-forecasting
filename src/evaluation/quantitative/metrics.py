"""
Metrics for quantitative evaluation of the results ported to PyTorch and adapted.
Following https://github.com/eth-ait/motion-transformer/blob/master/metrics/motion_metrics.py.
@Author: Leon Herbrik
"""


import numpy as np
import cv2
import quaternion
import torch
import copy
import pytorch3d.transforms


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


def angle_diff(predictions: torch.tensor, targets: torch.tensor) -> torch.tensor:
    """
    Computes the angular distance between the target and predicted rotations. We define this as the angle that is
    required to rotate one rotation into the other. This essentially computes || log(R_diff) || where R_diff is the
    difference rotation between prediction and target.

    Args:
        predictions: torch tensor of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
        targets: torch tensor of same shape as `predictions`

    Returns:
        The geodesic distance for each joint as a torch tensor of shape (..., n_joints)
    """
    assert predictions.shape[-1] == predictions.shape[-2] == 3
    assert targets.shape[-1] == targets.shape[-2] == 3

    ori_shape = predictions.shape[:-2]
    preds = predictions.view(-1, 3, 3)
    targs = targets.view(-1, 3, 3)

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = torch.matmul(preds, targs.transpose(1, 2))

    angles = []
    for i in range(r.shape[0]):
        aa, _ = cv2.Rodrigues(
            r[i].numpy()
        )  # Convert to numpy array for use with OpenCV
        angles.append(torch.norm(torch.from_numpy(aa)))
    angles = torch.tensor(angles)

    return angles.view(ori_shape)


def positional(predictions: torch.tensor, targets: torch.tensor) -> torch.tensor:
    """
    Computes the Euclidean distance between joints in 3D space.
    Args:
        predictions: torch tensor of predicted 3D joint positions in format (..., n_joints, 3)
        targets: torch tensor of same shape as `predictions`

    Returns:
        The Euclidean distance for each joint as a torch tensor of shape (..., n_joints)
    """
    return torch.sqrt(torch.sum((predictions - targets) ** 2, dim=-1))


def euler_diff(predictions: torch.tensor, targets: torch.tensor) -> torch.tensor:
    """
    Computes the Euler angle error using pytorch3d.
    Args:
        predictions: torch tensor of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
        targets: torch tensor of same shape as `predictions`

    Returns:
        The Euler angle error as a torch tensor of shape (..., )
    """
    assert predictions.shape[-1] == 3 and predictions.shape[-2] == 3
    assert targets.shape[-1] == 3 and targets.shape[-2] == 3
    n_joints = predictions.shape[-3]

    ori_shape = predictions.shape[:-3]
    preds = predictions.view(-1, 3, 3)
    targs = targets.view(-1, 3, 3)

    # Convert rotation matrices to Euler angles using pytorch3d
    euler_preds = pytorch3d.transforms.rotation_matrix_to_euler_angles(preds)  # (N, 3)
    euler_targs = pytorch3d.transforms.rotation_matrix_to_euler_angles(targs)  # (N, 3)

    # reshape to (-1, n_joints*3) to be consistent with previous work
    euler_preds = euler_preds.view(-1, n_joints * 3)
    euler_targs = euler_targs.view(-1, n_joints * 3)

    # l2 error on euler angles
    idx_to_use = torch.where(torch.std(euler_targs, dim=0) > 1e-4)[0]
    euc_error = torch.pow(euler_targs[:, idx_to_use] - euler_preds[:, idx_to_use], 2)
    euc_error = torch.sqrt(torch.sum(euc_error, dim=1))  # (-1, ...)

    # reshape to original
    return euc_error.view(ori_shape)
