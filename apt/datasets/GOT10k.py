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
from os import path

from torchvision import datasets
import pandas as pd


class GOT10kDataset(datasets.ImageFolder):
    download_method = datasets.utils.download_and_extract_archive
    download_url = "https://www.kaggle.com/api/v1/datasets/download/trolukovich/food11-image-dataset"

    def __init__(self, root: str, force_download: bool = True, train: bool = True, valid: bool = False, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        self.download(root, force=force_download)

        if train:
            if valid:
                root = path.join(root, "validation")
            else:
                root = path.join(root, "training")
        else:
            root = path.join(root, "evaluation")

        super().__init__(root=root, transform=transform, target_transform=target_transform)

    @classmethod
    def download(cls, root: str, force: bool = False):
        if force or not path.isfile(path.join(root, "archive.zip")):
            cls.download_method(cls.download_url, download_root=root, extract_root=root, filename="archive.zip")
            print("INFO: Dataset archive downloaded and extracted.")
        else:
            print("INFO: Dataset archive found in the root directory. Skipping download.")

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(dict(path=[d[0] for d in self.samples], label=[self.classes[lb] for lb in self.targets]))


def get_GOT10k_dataset():
    DATA_ROOT = path.join(".", "data", "food11")

    train_dataset = GOT10kDataset(root=DATA_ROOT, force_download=False, train=True, transform=None)
    valid_dataset = GOT10kDataset(root=DATA_ROOT, force_download=False, valid=True, transform=None)
    test_dataset = GOT10kDataset(root=DATA_ROOT, force_download=False, train=False, transform=None)

    print(f"INFO: Dataset loaded successfully. Number of samples - Train({len(train_dataset)}), Valid({len(valid_dataset)}), Test({len(test_dataset)})")
