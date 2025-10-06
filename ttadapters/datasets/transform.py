import torch
from torch.nn import functional
from torchvision import tv_tensors
import torchvision.transforms.v2.functional as F

from typing import Tuple, List, Dict, Union, Optional
import random


class ConvertRGBtoBGR:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            result = img[[2, 1, 0], ...]  # RGB -> BGR for tensor
            if isinstance(img, tv_tensors.Image):
                result = tv_tensors.Image(result)
            return result
        else:  # PIL Image
            import numpy as np
            from PIL import Image
            return Image.fromarray(np.array(img)[:, :, [2, 1, 0]])


class ResizeShortestEdge:
    """ Detectron2-style ResizeShortestEdge

    Features:
        - Random size selection for each call
        - Automatic BBox scaling
        - tv_tensors support
    """

    def __init__(
        self,
        sizes: Union[int, List[int]],
        max_size: int = 1333,
        box_key: str = 'boxes'
    ):
        if isinstance(sizes, int):
            self.sizes = [sizes]
        else:
            self.sizes = list(sizes)
        self.max_size = max_size
        self.box_key = box_key

    def __call__(self, image, target=None):
        """ Resize image and scale bounding boxes.

        Args:
            image: tv_tensors.Image or torch.Tensor (C, H, W)
            target: dict with 'boxes' key (optional)
        """
        # Original size
        _, h, w = image.shape

        # Calculate new size
        size = random.choice(self.sizes)
        scale = size / min(h, w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        # Apply max_size constraint
        if max(h, w) * scale > self.max_size:
            scale = self.max_size / max(h, w)  # 1333/1920 = 0.694
            new_h = int(h * scale)  # 1080 * 0.694 = 749
            new_w = int(w * scale)  # 1920 * 0.694 = 1333

        # Resize image
        resized_image = F.resize(image, size=[new_h, new_w], antialias=True)

        # Process BBox
        if target is not None and self.box_key in target:
            # Scale BBox
            if isinstance(target[self.box_key], tv_tensors.BoundingBoxes):
                # Handle tv_tensors.BoundingBoxes
                boxes = target[self.box_key].clone()
                boxes[:, [0, 2]] *= scale  # x1, x2
                boxes[:, [1, 3]] *= scale  # y1, y2

                # Create new BoundingBoxes
                target[self.box_key] = tv_tensors.BoundingBoxes(
                    boxes,
                    format=target[self.box_key].format,
                    canvas_size=(new_h, new_w)
                )
            elif isinstance(target[self.box_key], torch.Tensor):
                # Handle regular tensor
                target[self.box_key] = target[self.box_key].clone()
                target[self.box_key][:, [0, 2]] *= scale  # x1, x2
                target[self.box_key][:, [1, 3]] *= scale  # y1, y2

        if target is not None:
            return resized_image, target
        else:
            return resized_image


# The following code is copied and modified from
# Detectron2/structures/image_list.py.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
class MaskedImageList:
    """ Extended Detectron2 ImageList """

    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]], pixel_mask: torch.Tensor = None):
        self.tensor = tensor
        self.image_sizes = image_sizes
        self.pixel_mask = pixel_mask

    def __len__(self) -> int:
        # Scripting doesn't support len on list of tensors
        if torch.jit.is_scripting() or not torch.jit.is_tracing():
            return len(self.image_sizes)
        # In tracing, image_sizes is a list of tensors
        return self.tensor.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]], self.pixel_mask[idx, : size[0], : size[1]]

    @property
    def device(self):
        return self.tensor.device

    @staticmethod
    def from_tensors(
        tensors: List[torch.Tensor],
        size_divisibility: int = 0,
        pad_value: float = 0.0,
        padding_constraints: Optional[Dict[str, int]] = None,
    ) -> 'MaskedImageList':
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
        image_sizes_tensor = [
            torch.as_tensor(im.shape[-2:], device=im.device) for im in tensors
        ]
        max_size = torch.stack(image_sizes_tensor).max(0).values

        if padding_constraints is not None:
            square_size = padding_constraints.get("square_size", 0)
            if square_size > 0:
                # pad to square.
                max_size[0] = max_size[1] = square_size
            if "size_divisibility" in padding_constraints:
                size_divisibility = padding_constraints["size_divisibility"]
        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride

        # handle weirdness of scripting and tracing ...
        if torch.jit.is_scripting():
            max_size: List[int] = max_size.to(dtype=torch.long).tolist()
        else:
            if torch.jit.is_tracing():
                image_sizes = image_sizes_tensor

        if len(tensors) == 1:
            # This seems slightly (2%) faster.
            # TODO: check whether it's faster for multiple images as well
            image_size = image_sizes[0]
            u0 = max_size[-1] - image_size[1]
            u1 = max_size[-2] - image_size[0]
            padding_size = [0, u0, 0, u1]
            batched_imgs = functional.pad(tensors[0], padding_size, value=pad_value).unsqueeze_(0)
        else:
            # max_size can be a tensor in tracing mode, therefore convert to list
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
            device = (
                None if torch.jit.is_scripting() else ("cpu" if torch.jit.is_tracing() else None)
            )
            batched_imgs = tensors[0].new_full(batch_shape, pad_value, device=device)
            for i, img in enumerate(tensors):
                # Use `batched_imgs` directly instead of `img, pad_img = zip(tensors, batched_imgs)`
                # Tracing mode cannot capture `copy_()` of temporary locals
                batched_imgs[i, ..., : img.shape[-2], : img.shape[-1]].copy_(img)

        # batched_imgs shape: (B, C, H, W) or (B, H, W)
        if batched_imgs.dim() == 4:
            padded_h, padded_w = batched_imgs.shape[-2:]
        else:
            padded_h, padded_w = batched_imgs.shape[-2:]
        pixel_mask = torch.zeros(
            len(tensors), padded_h, padded_w,
            dtype=torch.bool, device=batched_imgs.device
        )
        for i, shape in enumerate(image_sizes):
            if torch.jit.is_tracing():
                # In tracing, shape is a tensor
                h, w = shape[0], shape[1]
            else:
                # In eager/scripting, shape is a tuple of ints
                h, w = shape[0], shape[1]
            pixel_mask[i, :h, :w] = True

        return MaskedImageList(batched_imgs.contiguous(), image_sizes, pixel_mask)

    def to(self, *args, **kwargs):
        cast_tensor = self.tensor.to(*args, **kwargs)
        cast_mask = None
        if self.pixel_mask is not None:
            cast_mask = self.pixel_mask.to(*args, **kwargs)

        # In tracing, image_sizes could be a list of tensors, which also need to be moved.
        cast_image_sizes = self.image_sizes
        if torch.jit.is_tracing():
            # Create a new list to avoid modifying the original one
            new_sizes = []
            for size in self.image_sizes:
                new_sizes.append(size.to(*args, **kwargs))
            cast_image_sizes = new_sizes

        return MaskedImageList(cast_tensor, cast_image_sizes, cast_mask)
