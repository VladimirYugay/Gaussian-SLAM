import numpy as np
import torch
from scipy.spatial.transform import Rotation
from typing import Union
from src.utils.utils import np2torch


def multiply_quaternions(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Performs batch-wise quaternion multiplication.

    Given two quaternions, this function computes their product. The operation is
    vectorized and can be performed on batches of quaternions.

    Args:
        q: A tensor representing the first quaternion or a batch of quaternions. 
           Expected shape is (... , 4), where the last dimension contains quaternion components (w, x, y, z).
        r: A tensor representing the second quaternion or a batch of quaternions with the same shape as q.
    Returns:
        A tensor of the same shape as the input tensors, representing the product of the input quaternions.
    """
    w0, x0, y0, z0 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w1, x1, y1, z1 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]

    w = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
    x = x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0
    y = -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0
    z = x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
    return torch.stack((w, x, y, z), dim=-1)


def transformation_to_quaternion(RT: Union[torch.Tensor, np.ndarray]):
    """ Converts a rotation-translation matrix to a tensor representing quaternion and translation.

    This function takes a 3x4 transformation matrix (rotation and translation) and converts it
    into a tensor that combines the quaternion representation of the rotation and the translation vector.

    Args:
        RT: A 3x4 matrix representing the rotation and translation. This can be a NumPy array
            or a torch.Tensor. If it's a torch.Tensor and resides on a GPU, it will be moved to CPU.

    Returns:
        A tensor combining the quaternion (in w, x, y, z order) and translation vector. The tensor
        will be moved to the original device if the input was a GPU tensor.
    """
    gpu_id = -1
    if isinstance(RT, torch.Tensor):
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    R, T = RT[:3, :3], RT[:3, 3]

    rot = Rotation.from_matrix(R)
    quad = rot.as_quat(canonical=True)
    quad = np.roll(quad, 1)
    tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor


def interpolate_poses(poses: np.ndarray) -> np.ndarray:
    """ Generates an interpolated pose based on the first two poses in the given array.
    Args:
        poses: An array of poses, where each pose is represented by a 4x4 transformation matrix.
    Returns:
        A 4x4 numpy ndarray representing the interpolated transformation matrix.
    """
    quat_poses = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    init_rot = quat_poses[1] + (quat_poses[1] - quat_poses[0])
    init_trans = poses[1, :3, 3] + (poses[1, :3, 3] - poses[0, :3, 3])
    init_transformation = np.eye(4)
    init_transformation[:3, :3] = Rotation.from_quat(init_rot).as_matrix()
    init_transformation[:3, 3] = init_trans
    return init_transformation


def compute_camera_opt_params(estimate_rel_w2c: np.ndarray) -> tuple:
    """ Computes the camera's rotation and translation parameters from an world-to-camera transformation matrix.
    This function extracts the rotation component of the transformation matrix, converts it to a quaternion,
    and reorders it to match a specific convention. Both rotation and translation parameters are converted
    to torch Parameters and intended to be optimized in a PyTorch model.
    Args:
        estimate_rel_w2c: A 4x4 numpy ndarray representing the estimated world-to-camera transformation matrix.
    Returns:
        A tuple containing two torch.nn.Parameters: camera's rotation and camera's translation.
    """
    quaternion = Rotation.from_matrix(estimate_rel_w2c[:3, :3]).as_quat(canonical=True)
    quaternion = quaternion[[3, 0, 1, 2]]
    opt_cam_rot = torch.nn.Parameter(np2torch(quaternion, "cuda"))
    opt_cam_trans = torch.nn.Parameter(np2torch(estimate_rel_w2c[:3, 3], "cuda"))
    return opt_cam_rot, opt_cam_trans
