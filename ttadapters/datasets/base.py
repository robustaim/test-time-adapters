from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Union


class BaseDataset(Dataset):
    def __len__(self):
        raise NotImplementedError("This method should be overridden by subclasses.")


@dataclass
class DatasetHolder:
    train: Union[BaseDataset, Dataset] = None
    valid: Union[BaseDataset, Dataset] = None
    test: Union[BaseDataset, Dataset] = None

    def __post_init__(self):
        print(f"INFO: Dataset loaded successfully. Number of samples - ", end='')
        if self.train:
            print(f"Train: {len(self.train)}", end='')
        if self.valid:
            if self.train: print(', ', end='')
            print(f"Valid: {len(self.valid)}", end='')
        if self.test:
            if self.train: print(', ', end='')
            print(f"Test: {len(self.test)}", end='')
        print('\n')


@dataclass
class DataLoaderHolder:
    train: object = None
    valid: object = None
    test: object = None
