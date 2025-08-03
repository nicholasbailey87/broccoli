import importlib.resources
import torch


def get_weights(name: str) -> torch.Tensor:
    resource_path = importlib.resources.files("broccoli.assets") / name
    with importlib.resources.as_file(resource_path) as path_to_weights:
        weights = torch.load(path_to_weights)
        return weights
