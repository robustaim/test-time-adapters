from torch import nn, load, save, hub
from os import path, makedirs
from typing import Union, Any, Optional

from ..datasets import BaseDataset


class BaseModel(nn.Module):
    model_name = "BaseModel"

    def __init__(self, dataset: Union[BaseDataset, str] = ""):
        super().__init__()
        self.dataset_name = dataset if isinstance(dataset, str) else dataset.dataset_name

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError("The forward method must be implemented in subclasses.")

    def save_to(self, save_path: str = path.join(".", "weights"), version: str = "", silence: bool = False):
        if not path.isdir(save_path):
            makedirs(save_path)
        if version:
            version = f"_{version}"
        model_id = f"{self.model_name}_{self.dataset_name}{version}"
        file_name = path.join(save_path, f"{model_id}.pt")
        save(self.state_dict(), file_name)
        if not silence: print(f"INFO: Model saved to {file_name}")

    def load_from(
        self,
        version: Union[str, Any] = "",
        weight_path: str = path.join(".", "weights"),
        weight_key: Optional[str] = None,
    ):
        if isinstance(version, str):
            if version:
                version = f"_{version}"
            model_id = f"{self.model_name}_{self.dataset_name}{version}"
            if "http" in weight_path:
                state = hub.load_state_dict_from_url(weight_path, map_location="cpu")
            else:
                file_name = path.join(weight_path, f"{model_id}.pt")
                state = load(file_name, map_location="cpu")
        elif callable(version):
            state = version()
        else:
            state = version
        if weight_key:
            state = state[weight_key]
        return self.load_state_dict(state)
