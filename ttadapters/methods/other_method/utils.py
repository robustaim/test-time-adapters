import os
import io
import math
import copy
import logging
from datetime import datetime
from collections import defaultdict
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import box_convert
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from transformers.models.rt_detr.modeling_rt_detr import RTDetrFrozenBatchNorm2d

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from supervision.metrics.mean_average_precision import MeanAveragePrecision
from supervision.detection.core import Detections

from ttadapters.datasets import (
    SHIFTClearDatasetForObjectDetection,
    SHIFTDiscreteSubsetForObjectDetection,
    BaseDataset,
)

# ActMAD-specific function
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out.clone())

    def clear(self):
        self.outputs = []

    def get_out_mean(self):
        out = torch.vstack(self.outputs)
        out = torch.mean(out, dim=[0, 2, 3])
        return out

    def get_out_var(self):
        out = torch.vstack(self.outputs)
        out = torch.var(out, dim=[0, 2, 3], correction=0)
        return out

class SaveOutputRTDETR:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out.clone())

    def clear(self):
        self.outputs = []

    def get_out_mean(self):
        out = torch.vstack(self.outputs)
        out = torch.mean(out, dim=0)
        return out

    def get_out_var(self):
        out = torch.vstack(self.outputs)
        out = torch.var(out, dim=0, correction=0)
        return out

        

    