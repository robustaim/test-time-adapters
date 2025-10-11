from detectron2.data.transforms import Transform, Augmentation
import numpy as np


class PermuteChannelsTransform(Transform):
    """
    Permute image channels from (C, H, W) to (H, W, C).
    Converts Tensor to Numpy if needed.
    """
    def __init__(self, reverse=False):
        self.reverse = reverse

    def apply_image(self, img):
        """
        Args:
            img: Tensor of shape (C, H, W) or ndarray of shape (H, W, C)

        Returns:
            ndarray: shape (H, W, C)
        """
        if self.reverse:
            return np.transpose(img, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        else:
            return np.transpose(img, (1, 2, 0))  # (C,H,W) -> (H,W,C)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """Permuting channels doesn't affect coordinates."""
        return coords

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """Permuting channels doesn't affect boxes."""
        return box

    def apply_polygons(self, polygons: list) -> list:
        """Permuting channels doesn't affect polygons."""
        return polygons

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Permuting channels doesn't affect segmentation masks."""
        return segmentation

    def inverse(self):
        """Inverse not needed for this use case."""
        raise NotImplementedError("Inverse transform not implemented")


class PermuteChannels(Augmentation):
    """
    Augmentation that permutes image channels from (C, H, W) to (H, W, C).
    Also handles Tensor to Numpy conversion.
    """
    def __init__(self, reverse=False):
        self.reverse = reverse

    def get_transform(self, image) -> Transform:
        """
        Args:
            image: Tensor (C, H, W) or ndarray (H, W, C)

        Returns:
            Transform: PermuteChannelsTransform instance
        """
        return PermuteChannelsTransform(reverse=self.reverse)


class ConvertRGBtoBGRTransform(Transform):
    """
    Convert RGB to BGR or vice versa (symmetric operation).
    """
    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: ndarray of shape (H, W, C)

        Returns:
            ndarray: image with RGB and BGR channels swapped
        """
        return img[:, :, [2, 1, 0]]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """Color conversion doesn't affect coordinates."""
        return coords

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """Color conversion doesn't affect boxes."""
        return box

    def apply_polygons(self, polygons: list) -> list:
        """Color conversion doesn't affect polygons."""
        return polygons

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Color conversion doesn't affect segmentation masks."""
        return segmentation

    def inverse(self) -> Transform:
        """RGB<->BGR is symmetric, so inverse is itself."""
        return ConvertRGBtoBGRTransform()


class ConvertRGBtoBGR(Augmentation):
    """
    Augmentation that converts RGB to BGR or vice versa.
    """
    def get_transform(self, image: np.ndarray) -> Transform:
        """
        Args:
            image: ndarray in RGB or BGR format

        Returns:
            Transform: ConvertRGBtoBGRTransform instance
        """
        return ConvertRGBtoBGRTransform()
