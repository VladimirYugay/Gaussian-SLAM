import json
import random
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
from evaluate_3d_reconstruction import run_evaluation
from tqdm import tqdm


def normalize(x):
    return x / np.linalg.norm(x)


def get_align_transformation(rec_meshfile, gt_meshfile):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    o3d_rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc,
        o3d_gt_pc,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    transformation = reg_p2p.transformation
    return transformation


def check_proj(points, W, H, fx, fy, cx, cy, c2w):
    """
    Check if points can be projected into the camera view.

    Returns:
        bool: True if there are points can be projected

    """
    c2w = c2w.copy()
    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0
    points = torch.from_numpy(points).cuda().clone()
    w2c = np.linalg.inv(c2w)
    w2c = torch.from_numpy(w2c).cuda().float()
    K = torch.from_numpy(
        np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]).reshape(3, 3)
    ).cuda()
    ones = torch.ones_like(points[:, 0]).reshape(-1, 1).cuda()
    homo_points = (
        torch.cat([points, ones], dim=1).reshape(-1, 4, 1).cuda().float()
    )  # (N, 4)
    cam_cord_homo = w2c @ homo_points  # (N, 4, 1)=(4,4)*(N, 4, 1)
    cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
    cam_cord[:, 0] *= -1
    uv = K.float() @ cam_cord.float()
    z = uv[:, -1:] + 1e-5
    uv = uv[:, :2] / z
    uv = uv.float().squeeze(-1).cpu().numpy()
    edge = 0
    mask = (
        (0 <= -z[:, 0, 0].cpu().numpy())
        & (uv[:, 0] < W - edge)
        & (uv[:, 0] > edge)
        & (uv[:, 1] < H - edge)
        & (uv[:, 1] > edge)
    )
    return mask.sum() > 0


def get_cam_position(gt_meshfile):
    mesh_gt = trimesh.load(gt_meshfile)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh_gt)
    extents[2] *= 0.7
    extents[1] *= 0.7
    extents[0] *= 0.3
    transform = np.linalg.inv(to_origin)
    transform[2, 3] += 0.4
    return extents, transform


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def calc_2d_metric(
    rec_meshfile, gt_meshfile, unseen_gt_pointcloud_file, align=True, n_imgs=1000
):
    """
    2D reconstruction metric, depth L1 loss.

    """
    H = 500
    W = 500
    focal = 300
    fx = focal
    fy = focal
    cx = H / 2.0 - 0.5
    cy = W / 2.0 - 0.5

    gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    pc_unseen = np.load(unseen_gt_pointcloud_file)
    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        rec_mesh = rec_mesh.transform(transformation)

    # get vacant area inside the room
    extents, transform = get_cam_position(gt_meshfile)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H, visible=False)
    vis.get_render_option().mesh_show_back_face = True
    errors = []
    for i in tqdm(range(n_imgs)):
        while True:
            # sample view, and check if unseen region is not inside the camera view
            # if inside, then needs to resample
            up = [0, 0, -1]
            origin = trimesh.sample.volume_rectangular(
                extents, 1, transform=transform)
            origin = origin.reshape(-1)
            tx = round(random.uniform(-10000, +10000), 2)
            ty = round(random.uniform(-10000, +10000), 2)
            tz = round(random.uniform(-10000, +10000), 2)
            # will be normalized, so sample from range [0.0,1.0]
            target = [tx, ty, tz]
            target = np.array(target) - np.array(origin)
            c2w = viewmatrix(target, up, origin)
            tmp = np.eye(4)
            tmp[:3, :] = c2w  # sample translations
            c2w = tmp
            # if unseen points are projected into current view (c2w)
            seen = check_proj(pc_unseen, W, H, fx, fy, cx, cy, c2w)
            if ~seen:
                break

        param = o3d.camera.PinholeCameraParameters()
        param.extrinsic = np.linalg.inv(c2w)  # 4x4 numpy array

        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            W, H, fx, fy, cx, cy)

        ctr = vis.get_view_control()
        ctr.set_constant_z_far(20)
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.add_geometry(
            gt_mesh,
            reset_bounding_box=True,
        )
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        gt_depth = vis.capture_depth_float_buffer(True)
        gt_depth = np.asarray(gt_depth)
        vis.remove_geometry(
            gt_mesh,
            reset_bounding_box=True,
        )

        vis.add_geometry(
            rec_mesh,
            reset_bounding_box=True,
        )
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        ours_depth = vis.capture_depth_float_buffer(True)
        ours_depth = np.asarray(ours_depth)
        vis.remove_geometry(
            rec_mesh,
            reset_bounding_box=True,
        )

        # filter missing surfaces where depth is 0
        if (ours_depth > 0).sum() > 0:
            errors += [
                np.abs(gt_depth[ours_depth > 0] -
                       ours_depth[ours_depth > 0]).mean()
            ]
        else:
            continue

    errors = np.array(errors)
    return {"depth l1": errors.mean() * 100}


def clean_mesh(mesh):
    mesh_tri = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=np.asarray(mesh.vertex_colors),
    )
    components = trimesh.graph.connected_components(
        edges=mesh_tri.edges_sorted)

    min_len = 200
    components_to_keep = [c for c in components if len(c) >= min_len]

    new_vertices = []
    new_faces = []
    new_colors = []
    vertex_count = 0
    for component in components_to_keep:
        vertices = mesh_tri.vertices[component]
        colors = mesh_tri.visual.vertex_colors[component]

        # Create a mapping from old vertex indices to new vertex indices
        index_mapping = {
            old_idx: vertex_count + new_idx for new_idx, old_idx in enumerate(component)
        }
        vertex_count += len(vertices)

        # Select faces that are part of the current connected component and update vertex indices
        faces_in_component = mesh_tri.faces[
            np.any(np.isin(mesh_tri.faces, component), axis=1)
        ]
        reindexed_faces = np.vectorize(index_mapping.get)(faces_in_component)

        new_vertices.extend(vertices)
        new_faces.extend(reindexed_faces)
        new_colors.extend(colors)

    cleaned_mesh_tri = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    cleaned_mesh_tri.visual.vertex_colors = np.array(new_colors)

    cleaned_mesh_tri.update_faces(cleaned_mesh_tri.nondegenerate_faces())
    cleaned_mesh_tri.update_faces(cleaned_mesh_tri.unique_faces())
    print(
        f"Mesh cleaning (before/after), vertices: {len(mesh_tri.vertices)}/{len(cleaned_mesh_tri.vertices)}, faces: {len(mesh_tri.faces)}/{len(cleaned_mesh_tri.faces)}")

    cleaned_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(cleaned_mesh_tri.vertices),
        o3d.utility.Vector3iVector(cleaned_mesh_tri.faces),
    )
    vertex_colors = np.asarray(cleaned_mesh_tri.visual.vertex_colors)[
        :, :3] / 255.0
    cleaned_mesh.vertex_colors = o3d.utility.Vector3dVector(
        vertex_colors.astype(np.float64)
    )

    return cleaned_mesh


def evaluate_reconstruction(
    mesh_path: Path,
    gt_mesh_path: Path,
    unseen_pc_path: Path,
    output_path: Path,
    to_clean=True,
):
    if to_clean:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        print(mesh)
        cleaned_mesh = clean_mesh(mesh)
        cleaned_mesh_path = output_path / "mesh" / "cleaned_mesh.ply"
        o3d.io.write_triangle_mesh(str(cleaned_mesh_path), cleaned_mesh)
        mesh_path = cleaned_mesh_path

    result_3d = run_evaluation(
        str(mesh_path.parts[-1]),
        str(mesh_path.parent),
        str(gt_mesh_path).split("/")[-1].split(".")[0],
        distance_thresh=0.01,
        full_path_to_gt_ply=gt_mesh_path,
        icp_align=True,
    )
    
    try:
        result_2d = calc_2d_metric(str(mesh_path), str(gt_mesh_path), str(unseen_pc_path), align=True, n_imgs=1000)
    except Exception as e:
        print(e)
        result_2d = {"depth l1": None}
    
    result = {**result_3d, **result_2d}
    with open(str(output_path / "reconstruction_metrics.json"), "w") as f:
        json.dump(result, f)
