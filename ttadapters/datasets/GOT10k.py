"""
### GOT-10k Dataset for Next-frame Prediction Task (Default Pretraining Process)
http://got-10k.aitestunion.com/downloads

#### Data File Structure
The downloaded and extracted full dataset should follow the file structure:
```
    |-- GOT-10k/
        |-- train/
        |  |-- GOT-10k_Train_000001/
        |  |   ......
        |  |-- GOT-10k_Train_009335/
        |  |-- list.txt
        |-- val/
        |  |-- GOT-10k_Val_000001/
        |  |   ......
        |  |-- GOT-10k_Val_000180/
        |  |-- list.txt
        |-- test/
        |  |-- GOT-10k_Test_000001/
        |  |   ......
        |  |-- GOT-10k_Test_000180/
        |  |-- list.txt
```

#### Annotation Description
Each sequence folder contains 4 annotation files and 1 meta file. A brief description of these files follows (let N denotes sequence length):

* groundtruth.txt -- An N×4 matrix with each line representing object location [xmin, ymin, width, height] in one frame.
* cover.label -- An N×1 array representing object visible ratios, with levels ranging from 0~8.
* absense.label -- An binary N×1 array indicating whether an object is absent or present in each frame.
* cut_by_image.label -- An binary N×1 array indicating whether an object is cut by image in each frame.
* meta_info.ini -- Meta information about the sequence, including object and motion classes, video URL and more.
* Values 0~8 in file cover.label correspond to ranges of object visible ratios: 0%, (0%, 15%], (15%~30%], (30%, 45%], (45%, 60%], (60%, 75%], (75%, 90%], (90%, 100%) and 100% respectively.
"""
from typing import Callable, Optional, Union
from collections import defaultdict
from dataclasses import dataclass
from copy import copy
from os import path
import random

import torch
from torch.utils.data import Dataset
from torchvision import datasets

from gdown.exceptions import FileURLRetrievalError

import numpy as np
import pandas as pd

from .base import BaseDataset


class GOT10kDataset(datasets.ImageFolder, BaseDataset):
    download_method = datasets.utils.download_and_extract_archive
    download_url = "https://drive.google.com/file/d/1b75MBq7MbDQUc682IoECIekoRim_Ydk1/view?usp=sharing"
    alternate_url = "https://drive.google.com/file/d/1LnWzvO6ymr5MA1ITYjni3-FOjoyZx6uS/view?usp=sharing"
    dataset_name = "GOT10k"
    file_name = "full_data.zip"
    extract_method = datasets.utils.extract_archive

    def __init__(
        self, root: str, force_download: bool = False,
        train: bool = True, valid: bool = False,
        transform: Optional[Callable] = None, target_transform: Optional[Callable] = None
    ):
        self.root = path.join(root, self.dataset_name)
        self.download(self.root, force=force_download)
        self.train, self.valid = train, valid

        if train:
            self.root = path.join(self.root, "val") if valid else path.join(self.root, "train")
        else:
            self.root = path.join(self.root, "test")

        super().__init__(root=self.root, transform=transform, target_transform=target_transform)
        self._load_data()

    def _load_data(self):
        sequences = defaultdict(list)
        for idx, (img_path, _) in enumerate(self.samples):
            seq_name = path.dirname(img_path)
            sequences[seq_name].append((idx, img_path))

        for seq_name in sequences:
            sequences[seq_name].sort(key=lambda x: x[1])

        targets = []
        for seq_name, frames in sequences.items():
            gt_path = path.join(seq_name, 'groundtruth.txt')
            if path.exists(gt_path):
                targets.extend(np.loadtxt(gt_path, delimiter=','))
        if self.train:
            self.samples = [(s[0], t) for s, t in zip(self.samples, targets)]
        else:
            start = (self.samples[0], targets)
            follows = [(s[0], [0, 0, 0, 0]) for s in self.samples[1:]]
            self.samples = [start] + follows

    @classmethod
    def download(cls, root: str, force: bool = False):
        print(f"INFO: Downloading '{cls.dataset_name}' from google drive to {root}...")
        downloaded = path.isfile(path.join(root, cls.file_name))
        extracted = not any(not path.isdir(path.join(root, target)) for target in ("train", "val", "test"))
        if force or not (downloaded or extracted):
            try:
                cls.download_method(cls.download_url, download_root=root, extract_root=root, filename=cls.file_name)
            except FileURLRetrievalError:
                print("INFO: Trying to download from the alternate link...")
                cls.download_method(cls.alternate_url, download_root=root, extract_root=root, filename=cls.file_name)
            print("INFO: Dataset archive downloaded and extracted.")
        else:
            print("INFO: Dataset archive found in the root directory. Skipping download.")
            if not extracted:
                cls.extract_method(from_path=path.join(root, cls.file_name), to_path=root)

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(dict(path=[d[0] for d in self.samples], label=[self.classes[lb] for lb in self.targets]))


@dataclass
class PairedGOT10kSample:
    prev_idx: int
    curr_idx: int


class PairedGOT10kDataset(Dataset):
    def __init__(self, base_dataset: GOT10kDataset, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.pairs = self._create_pairs(base_dataset)
        self.base_dataset = base_dataset
        self.use_teacher_forcing = False

    def _create_pairs(self, base) -> list[PairedGOT10kSample]:
        seq_id = -1
        pairs_by_seq: list[PairedGOT10kSample] = []

        for i, _id in enumerate(base.targets):
            if _id != seq_id:
                seq_id = _id  # do not include the first frame of the sequence
            else:
                pairs_by_seq.append(PairedGOT10kSample(prev_idx=i-1, curr_idx=i))

        return pairs_by_seq

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        curr_img = self.transform(self.base_dataset[pair.curr_idx][0])
        prev_gt = self.target_transform(self.base_dataset[pair.prev_idx][1])
        curr_gt = self.target_transform(self.base_dataset[pair.curr_idx][1])

        if self.use_teacher_forcing:
            return curr_img, prev_gt, curr_gt
        else:
            prev_img = self.transform(self.base_dataset[pair.prev_idx][0])
            return prev_img, curr_img, prev_gt, curr_gt

    def extract_valid(self, train_ratio=0.9, seed=42) -> 'PairedGOT10kDataset':
        # Set random seed and sequences
        sequences = list(range(len(self)))
        random.seed(seed)
        random.shuffle(sequences)
        split_idx = int(len(self) * train_ratio)
        indices = set(sequences[:split_idx]), set(sequences[split_idx:])

        pairs, self.pairs = self.pairs, []
        valid_set = copy(self)
        self.pairs = [pairs[i] for i in indices[0]]
        valid_set.pairs = [pairs[i] for i in indices[1]]
        return valid_set


GOT10kDatasetForObjectTracking = GOT10kDataset
