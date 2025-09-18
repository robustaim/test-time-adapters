from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional
import math


class BaseDataset(Dataset):
    dataset_name = "BaseDataset"

    def __len__(self):
        raise NotImplementedError("This method should be overridden by subclasses.")


@dataclass
class DatasetHolder:
    train: Optional[Dataset] = None
    valid: Optional[Dataset] = None
    test: Optional[Dataset] = None

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
    train: Optional[DataLoader] = None
    train_count: Optional[int] = 0  # total number of samples
    train_len: Optional[int] = 0  # iteration length
    valid: Optional[DataLoader] = None
    valid_count: Optional[int] = 0
    valid_len: Optional[int] = 0
    test: Optional[DataLoader] = None
    test_count: Optional[int] = 0
    test_len: Optional[int] = 0

    def __post_init__(self):
        print(f"INFO: Loader length - ", end='')
        if self.train:
            if self.train_len == 0:
                self.train_len = math.ceil(self.train_count/self.train.batch_size)
            print(f"Train: {self.train_len}", end='')
        if self.valid:
            if self.valid_len == 0:
                self.valid_len = math.ceil(self.valid_count/self.valid.batch_size)
            if self.train: print(', ', end='')
            print(f"Valid: {self.valid_len}", end='')
        if self.test:
            if self.test_len == 0:
                self.test_len = math.ceil(self.test_count/self.test.batch_size)
            if self.train: print(', ', end='')
            print(f"Test: {self.test_len}", end='')
        print('\n')
