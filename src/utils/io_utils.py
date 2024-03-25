import json
import os
from pathlib import Path
from typing import Union

import open3d as o3d
import torch
import wandb
import yaml


def mkdir_decorator(func):
    """A decorator that creates the directory specified in the function's 'directory' keyword
       argument before calling the function.
    Args:
        func: The function to be decorated.
    Returns:
        The wrapper function.
    """
    def wrapper(*args, **kwargs):
        output_path = Path(kwargs["directory"])
        output_path.mkdir(parents=True, exist_ok=True)
        return func(*args, **kwargs)
    return wrapper


@mkdir_decorator
def save_clouds(clouds: list, cloud_names: list, *, directory: Union[str, Path]) -> None:
    """ Saves a list of point clouds to the specified directory, creating the directory if it does not exist.
    Args:
        clouds: A list of point cloud objects to be saved.
        cloud_names: A list of filenames for the point clouds, corresponding by index to the clouds.
        directory: The directory where the point clouds will be saved.
    """
    for cld_name, cloud in zip(cloud_names, clouds):
        o3d.io.write_point_cloud(str(directory / cld_name), cloud)


@mkdir_decorator
def save_dict_to_ckpt(dictionary, file_name: str, *, directory: Union[str, Path]) -> None:
    """ Saves a dictionary to a checkpoint file in the specified directory, creating the directory if it does not exist.
    Args:
        dictionary: The dictionary to be saved.
        file_name: The name of the checkpoint file.
        directory: The directory where the checkpoint file will be saved.
    """
    torch.save(dictionary, directory / file_name,
               _use_new_zipfile_serialization=False)


@mkdir_decorator
def save_dict_to_yaml(dictionary, file_name: str, *, directory: Union[str, Path]) -> None:
    """ Saves a dictionary to a YAML file in the specified directory, creating the directory if it does not exist.
    Args:
        dictionary: The dictionary to be saved.
        file_name: The name of the YAML file.
        directory: The directory where the YAML file will be saved.
    """
    with open(directory / file_name, "w") as f:
        yaml.dump(dictionary, f)


@mkdir_decorator
def save_dict_to_json(dictionary, file_name: str, *, directory: Union[str, Path]) -> None:
    """ Saves a dictionary to a JSON file in the specified directory, creating the directory if it does not exist.
    Args:
        dictionary: The dictionary to be saved.
        file_name: The name of the JSON file.
        directory: The directory where the JSON file will be saved.
    """
    with open(directory / file_name, "w") as f:
        json.dump(dictionary, f)


def load_config(path: str, default_path: str = None) -> dict:
    """
    Loads a configuration file and optionally merges it with a default configuration file.

    This function loads a configuration from the given path. If the configuration specifies an inheritance
    path (`inherit_from`), or if a `default_path` is provided, it loads the base configuration and updates it
    with the specific configuration.

    Args:
        path: The path to the specific configuration file.
        default_path: An optional path to a default configuration file that is loaded if the specific configuration
                      does not specify an inheritance or as a base for the inheritance.

    Returns:
        A dictionary containing the merged configuration.
    """
    # load configuration from per scene/dataset cfg.
    with open(path, 'r') as f:
        cfg_special = yaml.full_load(f)
    inherit_from = cfg_special.get('inherit_from')
    cfg = dict()
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.full_load(f)
    update_recursive(cfg, cfg_special)
    return cfg


def update_recursive(dict1: dict, dict2: dict) -> None:
    """ Recursively updates the first dictionary with the contents of the second dictionary.

    This function iterates through `dict2` and updates `dict1` with its contents. If a key from `dict2`
    exists in `dict1` and its value is also a dictionary, the function updates the value recursively.
    Otherwise, it overwrites the value in `dict1` with the value from `dict2`.

    Args:
        dict1: The dictionary to be updated.
        dict2: The dictionary whose entries are used to update `dict1`.

    Returns:
        None: The function modifies `dict1` in place.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def log_metrics_to_wandb(json_files: list, output_path: str, section: str = "Evaluation") -> None:
    """ Logs metrics from JSON files to Weights & Biases under a specified section.

    This function reads metrics from a list of JSON files and logs them to Weights & Biases (wandb).
    Each metric is prefixed with a specified section name for organized logging.

    Args:
        json_files: A list of filenames for JSON files containing metrics to be logged.
        output_path: The directory path where the JSON files are located.
        section: The section under which to log the metrics in wandb. Defaults to "Evaluation".

    Returns:
        None: Metrics are logged to wandb and the function does not return a value.
    """
    for json_file in json_files:
        file_path = os.path.join(output_path, json_file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                metrics = json.load(file)
            prefixed_metrics = {
                f"{section}/{key}": value for key, value in metrics.items()}
            wandb.log(prefixed_metrics)
