""" This module includes the Odometer class, which is allows for fast pose estimation from RGBD neighbor frames  """
import numpy as np
import open3d as o3d
import open3d.core as o3c


class VisualOdometer(object):

    def __init__(self, intrinsics: np.ndarray, method_name="hybrid", device="cuda"):
        """ Initializes the visual odometry system with specified intrinsics, method, and device.
        Args:
            intrinsics: Camera intrinsic parameters.
            method_name: The name of the odometry computation method to use ('hybrid' or 'point_to_plane').
            device: The computation device ('cuda' or 'cpu').
        """
        device = "CUDA:0" if device == "cuda" else "CPU:0"
        self.device = o3c.Device(device)
        self.intrinsics = o3d.core.Tensor(intrinsics, o3d.core.Dtype.Float64)
        self.last_abs_pose = None
        self.last_frame = None
        self.criteria_list = [
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(500),
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(500),
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(500)]
        self.setup_method(method_name)
        self.max_depth = 10.0
        self.depth_scale = 1.0
        self.last_rgbd = None

    def setup_method(self, method_name: str) -> None:
        """ Sets up the odometry computation method based on the provided method name.
        Args:
            method_name: The name of the odometry method to use ('hybrid' or 'point_to_plane').
        """
        if method_name == "hybrid":
            self.method = o3d.t.pipelines.odometry.Method.Hybrid
        elif method_name == "point_to_plane":
            self.method = o3d.t.pipelines.odometry.Method.PointToPlane
        else:
            raise ValueError("Odometry method does not exist!")

    def update_last_rgbd(self, image: np.ndarray, depth: np.ndarray) -> None:
        """ Updates the last RGB-D frame stored in the system with a new RGB-D frame constructed from provided image and depth.
        Args:
            image: The new RGB image as a numpy ndarray.
            depth: The new depth image as a numpy ndarray.
        """
        self.last_rgbd = o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(np.ascontiguousarray(
                image).astype(np.float32)).to(self.device),
            o3d.t.geometry.Image(np.ascontiguousarray(depth).astype(np.float32)).to(self.device))

    def estimate_rel_pose(self, image: np.ndarray, depth: np.ndarray, init_transform=np.eye(4)):
        """ Estimates the relative pose of the current frame with respect to the last frame using RGB-D odometry.
        Args:
            image: The current RGB image as a numpy ndarray.
            depth: The current depth image as a numpy ndarray.
            init_transform: An initial transformation guess as a numpy ndarray. Defaults to the identity matrix.
        Returns:
            The relative transformation matrix as a numpy ndarray.
        """
        rgbd = o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(np.ascontiguousarray(image).astype(np.float32)).to(self.device),
            o3d.t.geometry.Image(np.ascontiguousarray(depth).astype(np.float32)).to(self.device))
        rel_transform = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
            self.last_rgbd, rgbd, self.intrinsics, o3c.Tensor(init_transform),
            self.depth_scale, self.max_depth, self.criteria_list, self.method)
        self.last_rgbd = rgbd.clone()

        # Adjust for the coordinate system difference
        rel_transform = rel_transform.transformation.cpu().numpy()
        rel_transform[0, [1, 2, 3]] *= -1
        rel_transform[1, [0, 2, 3]] *= -1
        rel_transform[2, [0, 1, 3]] *= -1

        return rel_transform
