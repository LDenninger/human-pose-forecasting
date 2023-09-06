# Third party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Local application imports
from ..data_utils import *
from ..evaluation import *


def test_metrics():
    # Create 26 x 3 x 3 tensor
    frame = torch.zeros((26, 3, 3))

    # Generate random rotation matrix for each joint using R.random().as_matrix()
    for i in range(26):
        frame[i] = torch.from_numpy(R.random().as_matrix())

    # TODO: Test euler diff
    # Test 1 - Test if the euler diff is zero for the same input
    euler_same = euler_angle_error(frame, frame)
    assert torch.all(euler_same == 0)
    # Test 2 - Test if the euler diff is slightly larger than zero for a slightly different input
    slightly_different_frame = frame.clone()
    # Add small random rotation to each joint
    for i in range(26):
        slightly_different_frame[i] = (
            slightly_different_frame[i]
            + torch.from_numpy(R.random().as_matrix()) * 0.01
        )
    euler_slightly_different = euler_angle_error(frame, slightly_different_frame)
    # Test 3 - Test if the euler diff is larger for a 90 degree rotation
    frame_90_degrees = frame.clone()
    # Add 90 degree rotation to each joint
    for i in range(26):
        frame_90_degrees[i] = frame_90_degrees[i] + torch.from_numpy(
            R.from_euler("x", 90, degrees=True).as_matrix()
        )
    euler_90_degrees = euler_angle_error(frame, frame_90_degrees)

    # TODO: Test angle diff
    # Test 1 - Test if the angle diff is zero for the same input
    angle_same = geodesic_distance(frame, frame)
    assert torch.all(angle_same == 0)

    # Test 2 - Test if the angle diff is slightly larger than zero for a slightly different input
    angle_slightly_different = geodesic_distance(frame, slightly_different_frame)

    # Test 3 - Test the angle diff is larger for a 90 degree rotation
    angle_90_degrees = geodesic_distance(frame, frame_90_degrees)

    pass


def test_distribution_metrics():
    # Create 64x41x99 tensor with random values
    batch = torch.rand((64, 41, 99))

    results = {}

    # Test power spectrum
    # Test 1 - Test if calculating something works at all
    results["power_spectrum"] = power_spectrum(batch)


if __name__ == "__main__":
    test_metrics()
