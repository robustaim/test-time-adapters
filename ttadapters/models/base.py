from typing import Union, Any, Optional
from dataclasses import dataclass
from os import path, makedirs
from enum import Enum
import gc

from torch import nn, load, save, hub, cuda

from ..datasets import BaseDataset, DataPreparation


class ModelProvider(Enum):
    Detectron2 = "Detectron2"
    HuggingFace = "HuggingFace"
    Ultralytics = "Ultralytics"


@dataclass
class WeightsInfo:
    weight_path: str = ""
    version: str = ""
    weight_key: Optional[str] = None
    exclude_keys: Optional[list[str]] = None


class BaseModel(nn.Module):
    model_name: str = "BaseModel"
    model_provider: ModelProvider = ModelProvider.Detectron2  # Default provider
    DataPreparation = DataPreparation
    class Trainer:
        pass

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
        weight_path: str = path.join(".", "weights"),
        version: str = "",
        weight_key: Optional[str] = None,
        exclude_keys: Optional[list[str]] = None,
        strict: bool = True
    ):
        """Load model weights from a file or URL.

        Args:
            weight_path (str, optional): Path to the weights directory or file. 
                Can be a local path, detectron2 model path, or URL. Defaults to path.join(".", "weights").
            version (str, optional): Version suffix to append to the model filename. Defaults to "".
            weight_key (Optional[str], optional): Key to extract weights from a nested state dict. Defaults to None.
            exclude_keys (Optional[list[str]], optional): List of state dict keys to exclude from loading. Defaults to None.
            strict (bool, optional): Whether to strictly enforce that the keys in state_dict match 
                the keys returned by this module's state_dict() function. Defaults to True.
                Note: This parameter is automatically set to False when loading from detectron2 checkpoints.

        Returns:
            dict: Missing and unexpected keys from loading the state dict, or checkpoint metadata for detectron2 models.
        """
        if "detectron2://" in weight_path:
            from detectron2.checkpoint import DetectionCheckpointer
            checkpointer = DetectionCheckpointer(self)
            checkpointables = [weight_key] if weight_key else None
            exclude_states = {k: v for k, v in self.state_dict().items() if exclude_keys and k in exclude_keys}
            result = checkpointer.load(weight_path, checkpointables=checkpointables)
            if exclude_states:  # re-apply excluded states
                self.load_state_dict(exclude_states, strict=False)
            if result and exclude_keys:
                return {k: v for k, v in result.items() if k not in exclude_keys}
            return result
        elif "http://" in weight_path or "https://" in weight_path:
            state = hub.load_state_dict_from_url(weight_path, map_location="cpu")
        else:
            if self.model_provider == ModelProvider.HuggingFace:
                revision = version if version else "main"
                reference_model = self.from_pretrained(
                    weight_path, revision=revision, config=self.config, dtype=self.dtype, ignore_mismatched_sizes=True
                )  # to initialize the model architecture
                state = reference_model.state_dict()
                if hasattr(reference_model, "generation_config"):
                    self.generation_config = reference_model.generation_config
                del reference_model
                cuda.empty_cache()
                gc.collect()
            else:
                model_id = f"{self.model_name}_{self.dataset_name}{'_'+version if version else ''}"
                file_name = path.join(weight_path, f"{model_id}.pt")
                state = load(file_name, map_location="cpu")

        if weight_key:
            state = state[weight_key]
        if hasattr(state, "state_dict"):  # some checkpoints have nested state_dict
            state = state.state_dict()

        if exclude_keys and len(exclude_keys) > 0:  # exclude specified keys
            state = {k: v for k, v in state.items() if k not in exclude_keys}

        model_state = self.state_dict()  # exclude size-mismatched keys
        new_state = {}
        for k, v in state.items():
            if k in model_state and v.size() == model_state[k].size():
                new_state[k] = v
            else:
                if k not in model_state:
                    print(f"NOTE: Key '{k}' found in checkpoint but not in current model. Skipping.")
                else:
                    print(f"NOTE: Size mismatch for {k}: copying a param with shape {v.size()} from checkpoint, the shape in current model is {model_state[k].size()}")

        result = self.load_state_dict(new_state, strict=strict)
        if self.model_provider == ModelProvider.HuggingFace:
            self.tie_weights()  # for HF models with tied weights
            if hasattr(self, "post_init"):
                self.post_init()

        return result
