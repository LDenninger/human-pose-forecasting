"""
    Functions to transform 3D data.
    Most functions are adapted from PyTorch3d: https://github.com/facebookresearch/pytorch3d

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal

#####===== Outside Helper Functions =====#####

def get_conv_from_axis_angle(representation=Literal['axis', 'mat', 'quat', '6d', 'euler']):
    """
        Get conversion function from rotation matrix to an arbitrary 3D rotation representations.
    """
    if representation == 'axis':
        return blank_processing
    elif representation =='mat':
        return axis_angle_to_matrix_direct
    elif representation == 'quat':
        return axis_angle_to_quaternion
    elif representation == '6d':
        return axis_angle_to_rotation_6d
    elif representation == 'euler':
        return axis_angle_to_euler_angles
    else:
        raise ValueError(f'Unknown representation {representation}')
    
def get_conv_from_rotation_matrix(representation=Literal['axis', 'mat', 'quat', '6d', 'euler']):
    """
        Get conversion function from rotation matrix to an arbitrary 3D rotation representations.
    """
    if representation == 'axis':
        return matrix_to_axis_angle
    elif representation =='mat':
        return unflatten_rotation_matrix
    elif representation == 'quat':
        return matrix_to_quaternion
    elif representation == '6d':
        return matrix_to_rotation_6d
    elif representation == 'euler':
        return matrix_to_euler_angles
    else:
        raise ValueError(f'Unknown representation {representation}')
    
def get_conv_to_rotation_matrix(representation=Literal['axis','mat', 'quat', '6d', 'euler']):
    """
        Get conversion function from an arbitrary 3D rotation representations to rotation matrix.
    """
    if representation == 'axis':
        return axis_angle_to_matrix
    elif representation =='mat':
        return unflatten_rotation_matrix
    elif representation == 'quat':
        return quaternion_to_matrix
    elif representation == '6d':
        return rotation_6d_to_matrix
    elif representation == 'euler':
        return euler_angles_to_matrix
    else:
        raise ValueError(f'Unknown representation {representation}')

    
def get_conv_from_vectors(representation=Literal['axis','mat', 'quat', '6d', 'euler', 'pos']):
    """
        Get conversion function to compute a 3D rotation between two vectors.
    """
    if representation == 'pos':
        return blank_processing
    elif representation =='axis':
        return vectors_to_axis_angle
    elif representation =='mat':
        return vectors_to_rotation_matrix
    elif representation == 'quat':
        return vectors_to_quaternion
    elif representation == '6d':
        return vectors_to_rotation_6d
    elif representation == 'euler':
        return vectors_to_euler_angles
    else:
        raise ValueError(f'Unknown representation {representation}')
    
def correct_rotation_matrix(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
        Correct a predicted rotation matrix using a singular value decomposition.
    """
    shape = rotation_matrix.shape
    if len(rotation_matrix.shape) > 3:
        rotation_matrix = rotation_matrix.view(-1, 3, 3)

    U, S, V = torch.linalg.svd(rotation_matrix)
    rotation_matrix = torch.bmm(U, V)
    rotation_matrix = torch.reshape(rotation_matrix, shape)
    return rotation_matrix

#####===== Additional Conversion Functions =====#####
# These are some conversions not provided by PyTorch3d


def axis_angle_to_matrix_direct(angle: torch.Tensor) -> torch.Tensor:
    """
        Converts a 3D axis angle to a 3x3 rotation matrix using the Rodrigues formula.
        This is used as an alternative to the PyTorch3d implementation which converts the angles to quaternions as an intermediate step.

        This function gives the same results as the implementation of expmap2rotmat()-function but different than the PyTorch3d implementation.
        Arguments:
            angle (torch.Tensor): The 3D axis angle to be converted. shape: [batch_size, 3]
                --> The magnitude of rotation is determined by the norm of the axis angle.
    """
    if len(angle.shape) == 2:
        bs = angle.shape[0]
    elif len(angle.shape) == 1:
        bs = 1
        angle = angle.unsqueeze(0)
    else:
        raise ValueError("The input tensor must be either 2D or 1D.")
    theta = torch.linalg.vector_norm(angle, dim=-1)
    r_norm = torch.divide(angle, theta.unsqueeze(-1) + torch.finfo(angle.dtype).eps)
    S =  torch.zeros((bs, 3, 3)).to(angle.device)
    S[:, 0, 1] = - r_norm[:, 2]
    S[:, 0, 2] = r_norm[:, 1]
    S[:, 1, 2] = - r_norm[:, 0]
    S = S - torch.transpose(S, -2, -1)
    rot_mat = torch.repeat_interleave(torch.eye(3).unsqueeze(0), bs, dim=0) + torch.sin(theta).unsqueeze(-1).unsqueeze(-1)* S + (1-torch.cos(theta)).unsqueeze(-1).unsqueeze(-1) * (S@S)
    return rot_mat.squeeze()

def vectors_to_axis_angle(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    """
        Computes the axis angle between the provided vectors.
        Method was taken from: https://www.euclideanspace.com/maths/algebra/vectors/angleBetween/
    """
    # Deal with additional leading dimensions
    if len(vector1.shape)>2:
        vector1 = torch.flatten(vector1, start_dim=0, end_dim=-2)
    if len(vector2.shape)>2:
        vector2 = torch.flatten(vector2, start_dim=0, end_dim=-2)
    if vector1.shape!= vector2.shape:
        raise ValueError("The input vectors must have the same shape.")
    # Normalize the vectors
    vector1_norm = vector1 / torch.linalg.norm(vector1, dim=-1).unsqueeze(-1)
    vector2_norm = vector2 / torch.linalg.norm(vector2, dim=-1).unsqueeze(-1)
    # compute the dot and cross products
    dot_prod = torch.einsum("bi,bi->b", vector1_norm, vector2_norm)
    cross_prod = torch.cross(vector1_norm, vector2_norm)
    # prevent numerical instabilities in the arccos when the vectors are equal.
    eps = torch.finfo(dot_prod.dtype).eps
    dot_prod = torch.clamp(dot_prod, min=-1+eps, max=1-eps)
    # compute the axis rotated around and the magnitude of the rotation
    angle = torch.acos(dot_prod)
    axis = cross_prod / torch.linalg.norm(cross_prod, dim=-1).unsqueeze(-1)
    # Combine to axis angle representation
    return axis * angle.unsqueeze(-1)

def vectors_to_rotation_matrix(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    """
        Compute the rotation between two vectors using the rotation matrix representation.
        The method first computes the axis angles and then calculates the rotation matrix from that.
    """
    return axis_angle_to_matrix(vectors_to_axis_angle(vector1, vector2))

def vectors_to_quaternion(vector1: torch.Tensor, vector2: torch.Tensor):
    """
        Compute the quaternion between two vectors using the quaternion representation.
        The method first computes the axis angles and then calculates the quaternion from that.
    """
    return axis_angle_to_quaternion(vectors_to_axis_angle(vector1, vector2))

def vectors_to_euler_angles(vector1: torch.Tensor, vector2: torch.Tensor):
    """
        Compute the Euler angles between two vectors using the Euler angles representation.
        The method first computes the axis angles and then calculates the Euler angles from that.
    """
    return axis_angle_to_euler_angles(vectors_to_axis_angle(vector1, vector2))

def vectors_to_rotation_6d(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    """
        Compute the rotation between two vectors using the rotation matrix representation.
        The method first computes the axis angles and then calculates the rotation matrix from that.
    """
    return axis_angle_to_rotation_6d(vectors_to_axis_angle(vector1, vector2))

def axis_angle_to_euler_angles(axis_angle: torch.Tensor, convention: str) -> torch.Tensor:
    """
        Convert the axis angle representation to the Euler angles representation.
        The method first computes the axis angles and then calculates the Euler angles from that.
    """
    return matrix_to_euler_angles(axis_angle_to_matrix(axis_angle), convention)

def unflatten_rotation_matrix(rotation_matrix: torch.Tensor) -> torch.Tensor:
    if rotation_matrix.shape[-1]== 9:
        return torch.reshape(rotation_matrix, (*rotation_matrix.shape[:-1], 3, 3))
    return rotation_matrix

#####===== Computation Helper Functions =====#####

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

#####===== Conversions from PyTorch3D =====#####

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))



def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))



def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions



def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles



def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)



def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def axis_angle_to_rotation_6d(angle: torch.Tensor) -> torch.Tensor:
    """
        Transform axis angles to 6D rotation representation by Zhou et al. [1].
    """
    return matrix_to_rotation_6d(axis_angle_to_matrix(angle))

def rotation_6d_to_axis_angle(d6: torch.Tensor) -> torch.Tensor:
    """
        Transform 6D rotation representation by Zhou et al. [1] to axis angles.
    """
    return matrix_to_axis_angle(rotation_6d_to_matrix(d6))


def blank_processing(input: torch.Tensor):
    """
        A blank function to use as a placeholder.
    """
    return input
