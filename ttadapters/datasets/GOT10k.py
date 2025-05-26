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
from typing import Callable, Optional
from collections import defaultdict
from os import path
import random

from torch.utils.data import Dataset
from torchvision import datasets
import torch

import numpy as np
import pandas as pd

from PIL import Image


class GOT10kDataset(datasets.ImageFolder):
    download_method = datasets.utils.download_and_extract_archive
    download_url = "https://drive.google.com/file/d/1b75MBq7MbDQUc682IoECIekoRim_Ydk1/view?usp=sharing"
    dataset_name = "GOT10k"
    file_name = "full_data.zip"
    extract_method = datasets.utils.extract_archive

    def __init__(self, root: str, force_download: bool = True, train: bool = True, valid: bool = False, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        self.root = path.join(root, self.dataset_name)
        self.download(self.root, force=force_download)

        if train:
            self.root = path.join(self.root, "val") if valid else path.join(self.root, "train")
        else:
            self.root = path.join(self.root, "test")

        super().__init__(root=self.root, transform=transform, target_transform=target_transform)

    @classmethod
    def download(cls, root: str, force: bool = False):
        print(f"INFO: Downloading '{cls.dataset_name}' from google drive to {root}...")
        if force or not path.isfile(path.join(root, cls.file_name)):
            cls.download_method(cls.download_url, download_root=root, extract_root=root, filename=cls.file_name)
            print("INFO: Dataset archive downloaded and extracted.")
        else:
            print("INFO: Dataset archive found in the root directory. Skipping download.")
            if not path.isdir(path.join(root, "train")) \
                    or not path.isdir(path.join(root, "val")) or not path.isdir(path.join(root, "test")) \
                    :
                cls.extract_method(from_path=path.join(root, cls.file_name), to_path=root)

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(dict(path=[d[0] for d in self.samples], label=[self.classes[lb] for lb in self.targets]))


class PairedGOT10kDataset(Dataset):
    def __init__(self, base_dataset: GOT10kDataset):
        super().__init__()
        self.base_dataset = base_dataset
        self.pairs = self._create_pairs()
        self.use_teacher_forcing = False

    def _create_pairs(self):
        sequences = defaultdict(list)
        for idx, (img_path, _) in enumerate(self.base_dataset.samples):
            seq_name = path.dirname(img_path)
            sequences[seq_name].append((idx, img_path))

        for seq_name in sequences:
            sequences[seq_name].sort(key=lambda x: x[1])

        pairs = []
        for seq_name, frames in sequences.items():
            gt_path = path.join(seq_name, 'groundtruth.txt')
            if path.exists(gt_path):
                groundtruth = np.loadtxt(gt_path, delimiter=',')

                # Get original image dimensions for normalization
                img_path = frames[0][1]  # Use first frame to get dimensions
                with Image.open(img_path) as img:
                    orig_w, orig_h = img.size

                # Normalize groundtruth coordinates
                for i in range(len(frames) - 1):
                    # Original format: [x_min, y_min, width, height]
                    # Convert to normalized coordinates
                    # Result format: [x_center, y_center, width, height]
                    gt_curr = groundtruth[i + 1].copy()
                    gt_prev = groundtruth[i].copy()

                    # Normalize coordinates
                    gt_prev[0] = (gt_prev[0] + gt_prev[2]/2) / orig_w  # x_center
                    gt_prev[1] = (gt_prev[1] + gt_prev[3]/2) / orig_h  # y_center
                    gt_prev[2] /= orig_w  # width
                    gt_prev[3] /= orig_h  # height

                    gt_curr[0] = (gt_curr[0] + gt_curr[2]/2) / orig_w  # x_center
                    gt_curr[1] = (gt_curr[1] + gt_curr[3]/2) / orig_h  # y_center
                    gt_curr[2] /= orig_w  # width
                    gt_curr[3] /= orig_h  # height

                    pairs.append({
                        'prev_idx': frames[i][0],
                        'curr_idx': frames[i + 1][0],
                        'prev_gt': gt_prev,
                        'curr_gt': gt_curr
                    })

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        prev_img = None
        if not self.use_teacher_forcing:
            prev_img, _ = self.base_dataset[pair['prev_idx']]
        curr_img, _ = self.base_dataset[pair['curr_idx']]

        prev_gt = torch.FloatTensor(pair['prev_gt'])
        curr_gt = torch.FloatTensor(pair['curr_gt'])

        if prev_img is None:
            return curr_img, curr_gt, curr_gt
        return prev_img, curr_img, prev_gt, curr_gt

    @classmethod
    def create_train_val_split(cls, base_dataset: GOT10kDataset, train_ratio=0.9, seed=42):
        # Get unique sequence paths efficiently using dict.fromkeys()
        sequences = list(dict.fromkeys(path.dirname(img_path) for img_path, _ in base_dataset.samples))

        # Set random seed and shuffle sequences
        random.seed(seed)
        random.shuffle(sequences)
        split_idx = int(len(sequences) * train_ratio)

        # Create sequence sets for faster lookups
        train_sequences = set(sequences[:split_idx])
        val_sequences = set(sequences[split_idx:])

        # Create train and val datasets
        data_root = path.dirname(path.dirname(base_dataset.root))
        train_dataset = GOT10kDataset(root=data_root, force_download=False, train=True, transform=base_dataset.transform)
        val_dataset = GOT10kDataset(root=data_root, force_download=False, train=True, transform=base_dataset.transform)

        # Split samples and targets in one pass
        train_samples = []
        train_targets = []
        val_samples = []
        val_targets = []

        for i, (sample, target) in enumerate(zip(base_dataset.samples, base_dataset.targets)):
            seq_dir = path.dirname(sample[0])
            if seq_dir in train_sequences:
                train_samples.append(sample)
                train_targets.append(target)
            else:
                val_samples.append(sample)
                val_targets.append(target)

        train_dataset.samples = train_samples
        train_dataset.targets = train_targets
        val_dataset.samples = val_samples
        val_dataset.targets = val_targets

        return cls(train_dataset), cls(val_dataset)


def get_GOT10k_dataset():
    DATA_ROOT = path.join(".", "data", "food11")

    train_dataset = GOT10kDataset(root=DATA_ROOT, force_download=False, train=True, transform=None)
    valid_dataset = GOT10kDataset(root=DATA_ROOT, force_download=False, valid=True, transform=None)
    test_dataset = GOT10kDataset(root=DATA_ROOT, force_download=False, train=False, transform=None)

    print(f"INFO: Dataset loaded successfully. Number of samples - Train({len(train_dataset)}), Valid({len(valid_dataset)}), Test({len(test_dataset)})")
