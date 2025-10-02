from .got10k import GOT10kDatasetForObjectTracking, PairedGOT10kDataset
from .shift import (
    SHIFTDataset, patch_fast_download_for_object_detection,
    SHIFTDiscreteDatasetForObjectDetection, SHIFTDiscreteSubsetForObjectDetection,
    SHIFTClearDatasetForObjectDetection, SHIFTCorruptedDatasetForObjectDetection,
    SHIFTContinuousDatasetForObjectDetection, SHIFTContinuous10DatasetForObjectDetection,
    SHIFTContinuous100DatasetForObjectDetection, SHIFTContinuousSubsetForObjectDetection,
    SHIFTContinuous10SubsetForObjectDetection, SHIFTContinuous100SubsetForObjectDetection
)

from .base import BaseDataset, DatasetHolder, DataLoaderHolder
from .transform import ResizeShortestEdge, MaskedImageList, ConvertRGBtoBGR

from torchvision.transforms import v2 as T


default_image_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

detectron_image_transform = T.Compose([
    ConvertRGBtoBGR()
])

default_train_transforms = T.Compose([
    ResizeShortestEdge([640, 672, 704, 736, 768, 800], max_size=1333, box_key='boxes2d'),  # Detectron2 Faster R-CNN default training transform
    T.RandomHorizontalFlip(p=0.5)  # Random horizontal flip with 50% probability
])

default_valid_transforms = T.Compose([
    ResizeShortestEdge(800, max_size=1333, box_key='boxes2d')  # Detectron2 Faster R-CNN default validation transform
])
