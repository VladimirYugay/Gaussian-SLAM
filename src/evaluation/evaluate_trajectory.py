import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return JSONEncoder.default(self, obj)


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,
                                         column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1) - rot * model.mean(1)

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(
        np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def align_trajectories(t_pred: np.ndarray, t_gt: np.ndarray):
    """
    Args:
        t_pred: (n, 3) translations
        t_gt: (n, 3) translations
    Returns:
        t_align: (n, 3) aligned translations
    """
    t_align = np.matrix(t_pred).transpose()
    R, t, _ = align(t_align, np.matrix(t_gt).transpose())
    t_align = R * t_align + t
    t_align = np.asarray(t_align).T
    return t_align


def pose_error(t_pred: np.ndarray, t_gt: np.ndarray, align=False):
    """
    Args:
        t_pred: (n, 3) translations
        t_gt: (n, 3) translations
    Returns:
        dict: error dict
    """
    n = t_pred.shape[0]
    trans_error = np.linalg.norm(t_pred - t_gt, axis=1)
    return {
        "compared_pose_pairs": n,
        "rmse": np.sqrt(np.dot(trans_error, trans_error) / n),
        "mean": np.mean(trans_error),
        "median": np.median(trans_error),
        "std": np.std(trans_error),
        "min": np.min(trans_error),
        "max": np.max(trans_error)
    }


def plot_2d(pts, ax=None, color="green", label="None", title="3D Trajectory in 2D"):
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(pts[:, 0], pts[:, 1], color=color, label=label, s=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    return ax


def evaluate_trajectory(estimated_poses: np.ndarray, gt_poses: np.ndarray, output_path: Path):
    output_path.mkdir(exist_ok=True, parents=True)
    # Truncate the ground truth trajectory if needed
    if gt_poses.shape[0] > estimated_poses.shape[0]:
        gt_poses = gt_poses[:estimated_poses.shape[0]]
    valid = ~np.any(np.isnan(gt_poses) |
                    np.isinf(gt_poses), axis=(1, 2))
    gt_poses = gt_poses[valid]
    estimated_poses = estimated_poses[valid]

    gt_t = gt_poses[:, :3, 3]
    estimated_t = estimated_poses[:, :3, 3]
    estimated_t_aligned = align_trajectories(estimated_t, gt_t)
    ate = pose_error(estimated_t, gt_t)
    ate_aligned = pose_error(estimated_t_aligned, gt_t)

    with open(str(output_path / "ate.json"), "w") as f:
        f.write(json.dumps(ate, cls=NumpyFloatValuesEncoder))

    with open(str(output_path / "ate_aligned.json"), "w") as f:
        f.write(json.dumps(ate_aligned, cls=NumpyFloatValuesEncoder))

    ate_rmse, ate_rmse_aligned = ate["rmse"], ate_aligned["rmse"]
    ax = plot_2d(
        estimated_t, label=f"ate-rmse: {round(ate_rmse * 100, 2)} cm", color="orange")
    ax = plot_2d(estimated_t_aligned, ax,
                 label=f"ate-rsme (aligned): {round(ate_rmse_aligned * 100, 2)} cm", color="lightskyblue")
    ax = plot_2d(gt_t, ax, label="GT", color="green")
    ax.legend()
    plt.savefig(str(output_path / "eval_trajectory.png"), dpi=300)
    plt.close()
    print(
        f"ATE-RMSE: {ate_rmse * 100:.2f} cm, ATE-RMSE (aligned): {ate_rmse_aligned * 100:.2f} cm")
