from collections import OrderedDict
from copy import deepcopy
from typing import List, Union

import numpy as np
import open3d as o3d
from matplotlib import colors

COLORS_ANSI = OrderedDict({
    "blue": "\033[94m",
    "orange": "\033[93m",
    "green": "\033[92m",
    "red": "\033[91m",
    "purple": "\033[95m",
    "brown": "\033[93m",  # No exact match, using yellow
    "pink": "\033[95m",
    "gray": "\033[90m",
    "olive": "\033[93m",  # No exact match, using yellow
    "cyan": "\033[96m",
    "end": "\033[0m",  # Reset color
})


COLORS_MATPLOTLIB = OrderedDict({
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'yellow-green': '#bcbd22',
    'cyan': '#17becf'
})


COLORS_MATPLOTLIB_RGB = OrderedDict({
    'blue': np.array([31, 119, 180]) / 255.0,
    'orange': np.array([255, 127,  14]) / 255.0,
    'green': np.array([44, 160,  44]) / 255.0,
    'red': np.array([214,  39,  40]) / 255.0,
    'purple': np.array([148, 103, 189]) / 255.0,
    'brown': np.array([140,  86,  75]) / 255.0,
    'pink': np.array([227, 119, 194]) / 255.0,
    'gray': np.array([127, 127, 127]) / 255.0,
    'yellow-green': np.array([188, 189,  34]) / 255.0,
    'cyan': np.array([23, 190, 207]) / 255.0
})


def get_color(color_name: str):
    """ Returns the RGB values of a given color name as a normalized numpy array.
    Args:
        color_name: The name of the color. Can be any color name from CSS4_COLORS.
    Returns:
        A numpy array representing the RGB values of the specified color, normalized to the range [0, 1].
    """
    if color_name == "custom_yellow":
        return np.asarray([255.0, 204.0, 102.0]) / 255.0
    if color_name == "custom_blue":
        return np.asarray([102.0, 153.0, 255.0]) / 255.0
    assert color_name in colors.CSS4_COLORS
    return np.asarray(colors.to_rgb(colors.CSS4_COLORS[color_name]))


def plot_ptcloud(point_clouds: Union[List, o3d.geometry.PointCloud], show_frame: bool = True):
    """ Visualizes one or more point clouds, optionally showing the coordinate frame.
    Args:
        point_clouds: A single point cloud or a list of point clouds to be visualized.
        show_frame: If True, displays the coordinate frame in the visualization. Defaults to True.
    """
    # rotate down up
    if not isinstance(point_clouds, list):
        point_clouds = [point_clouds]
    if show_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        point_clouds = point_clouds + [mesh_frame]
    o3d.visualization.draw_geometries(point_clouds)


def draw_registration_result_original_color(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                                            transformation: np.ndarray):
    """ Visualizes the result of a point cloud registration, keeping the original color of the source point cloud.
    Args:
        source: The source point cloud.
        target: The target point cloud.
        transformation: The transformation matrix applied to the source point cloud.
    """
    source_temp = deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])


def draw_registration_result(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                             transformation: np.ndarray, source_color: str = "blue", target_color: str = "orange"):
    """ Visualizes the result of a point cloud registration, coloring the source and target point clouds.
    Args:
        source: The source point cloud.
        target: The target point cloud.
        transformation: The transformation matrix applied to the source point cloud.
        source_color: The color to apply to the source point cloud. Defaults to "blue".
        target_color: The color to apply to the target point cloud. Defaults to "orange".
    """
    source_temp = deepcopy(source)
    source_temp.paint_uniform_color(COLORS_MATPLOTLIB_RGB[source_color])

    target_temp = deepcopy(target)
    target_temp.paint_uniform_color(COLORS_MATPLOTLIB_RGB[target_color])

    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
