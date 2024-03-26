import argparse
import os
import time
import uuid

import wandb

from src.entities.gaussian_slam import GaussianSLAM
from src.evaluation.evaluator import Evaluator
from src.utils.io_utils import load_config, log_metrics_to_wandb
from src.utils.utils import setup_seed


def get_args():
    parser = argparse.ArgumentParser(
        description='Arguments to compute the mesh')
    parser.add_argument('config_path', type=str,
                        help='Path to the configuration yaml file')
    parser.add_argument('--input_path', default="")
    parser.add_argument('--output_path', default="")
    parser.add_argument('--track_w_color_loss', type=float)
    parser.add_argument('--track_alpha_thre', type=float)
    parser.add_argument('--track_iters', type=int)
    parser.add_argument('--track_filter_alpha', action='store_true')
    parser.add_argument('--track_filter_outlier', action='store_true')
    parser.add_argument('--track_wo_filter_alpha', action='store_true')
    parser.add_argument("--track_wo_filter_outlier", action="store_true")
    parser.add_argument("--track_cam_trans_lr", type=float)
    parser.add_argument('--alpha_seeding_thre', type=float)
    parser.add_argument('--map_every', type=int)
    parser.add_argument("--map_iters", type=int)
    parser.add_argument('--new_submap_every', type=int)
    parser.add_argument('--project_name', type=str)
    parser.add_argument('--group_name', type=str)
    parser.add_argument('--gt_camera', action='store_true')
    parser.add_argument('--help_camera_initialization', action='store_true')
    parser.add_argument('--soft_alpha', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--submap_using_motion_heuristic', action='store_true')
    parser.add_argument('--new_submap_points_num', type=int)
    return parser.parse_args()


def update_config_with_args(config, args):
    if args.input_path:
        config["data"]["input_path"] = args.input_path
    if args.output_path:
        config["data"]["output_path"] = args.output_path
    if args.track_w_color_loss is not None:
        config["tracking"]["w_color_loss"] = args.track_w_color_loss
    if args.track_iters is not None:
        config["tracking"]["iterations"] = args.track_iters
    if args.track_filter_alpha:
        config["tracking"]["filter_alpha"] = True
    if args.track_wo_filter_alpha:
        config["tracking"]["filter_alpha"] = False
    if args.track_filter_outlier:
        config["tracking"]["filter_outlier_depth"] = True
    if args.track_wo_filter_outlier:
        config["tracking"]["filter_outlier_depth"] = False
    if args.track_alpha_thre is not None:
        config["tracking"]["alpha_thre"] = args.track_alpha_thre
    if args.map_every:
        config["mapping"]["map_every"] = args.map_every
    if args.map_iters:
        config["mapping"]["iterations"] = args.map_iters
    if args.new_submap_every:
        config["mapping"]["new_submap_every"] = args.new_submap_every
    if args.project_name:
        config["project_name"] = args.project_name
    if args.alpha_seeding_thre is not None:
        config["mapping"]["alpha_thre"] = args.alpha_seeding_thre
    if args.seed:
        config["seed"] = args.seed
    if args.help_camera_initialization:
        config["tracking"]["help_camera_initialization"] = True
    if args.soft_alpha:
        config["tracking"]["soft_alpha"] = True
    if args.submap_using_motion_heuristic:
        config["mapping"]["submap_using_motion_heuristic"] = True
    if args.new_submap_points_num:
        config["mapping"]["new_submap_points_num"] = args.new_submap_points_num
    if args.track_cam_trans_lr:
        config["tracking"]["cam_trans_lr"] = args.track_cam_trans_lr
    return config


if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config_path)
    config = update_config_with_args(config, args)

    if os.getenv('DISABLE_WANDB') == 'true':
        config["use_wandb"] = False
    if config["use_wandb"]:
        wandb.init(
            project=config["project_name"],
            config=config,
            dir="/home/yli3/scratch/outputs/slam/wandb",
            group=config["data"]["scene_name"]
            if not args.group_name
            else args.group_name,
            name=f'{config["data"]["scene_name"]}_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}_{str(uuid.uuid4())[:5]}',
        )
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

    setup_seed(config["seed"])
    gslam = GaussianSLAM(config)
    gslam.run()

    evaluator = Evaluator(gslam.output_path, gslam.output_path / "config.yaml")
    evaluator.run()
    if config["use_wandb"]:
        evals = ["rendering_metrics.json",
                 "reconstruction_metrics.json", "ate_aligned.json"]
        log_metrics_to_wandb(evals, gslam.output_path, "Evaluation")
        wandb.finish()
    print("All done.✨")
