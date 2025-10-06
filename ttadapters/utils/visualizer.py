from random import randint, random
import time

import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from ipywidgets import Output

from torchvision.transforms.v2.functional import convert_bounding_box_format
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat


def visualize_bbox_frame(
    dataset, idx: int | None = None,
    bbox_key: str = "boxes2d", bbox_class_key: str | None = "boxes2d_classes",
    figsize: tuple[int, int] = (6, 6), edgecolors: list | None = []  # keep color context
):
    """
    Visualize a single frame from a dataset with its bounding boxes.

    Args:
        dataset (Dataset): The dataset containing images and annotations. Each item should return (image, annotation).
        idx (int | None, optional): Index of the frame to visualize. If None, a random index is selected. Defaults to None.
        bbox_key (str, optional): The key in the annotation dict for bounding boxes. Defaults to "boxes2d".
        bbox_class_key (str | None, optional): The key in the annotation dict for class labels. If None, all boxes are shown with the same color. Defaults to "boxes2d_classes".
        figsize (tuple[int, int], optional): Figure size for matplotlib. Defaults to (6, 6).
        edgecolors (list | None, optional): List of RGB colors for bounding box edges. If None and bbox_class_key is not None, random colors will be generated for each class.

    Returns:
        None. Displays the image with bounding boxes using matplotlib.
    """
    if idx is None:
        idx = randint(0, len(dataset)-1)

    rgb_image, annotation = dataset[idx]
    bbox: BoundingBoxes = annotation[bbox_key]
    if bbox_class_key is None:
        cls_labels = [0] * len(bbox)
        edgecolors = [(random(), random(), random())] if edgecolors is None else edgecolors
    else:
        cls_labels = annotation[bbox_class_key]
        num_classes = max(cls_labels) + 1 if len(cls_labels) > 0 else 1
        if edgecolors is None:
            edgecolors = [(random(), random(), random()) for _ in range(num_classes)]
        elif len(edgecolors) != num_classes:
            edgecolors.extend([(random(), random(), random()) for _ in range(num_classes-len(edgecolors))])

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    axes.imshow(rgb_image.permute(1, 2, 0) / 255.0)
    axes.set_title(f"Frame {idx}")
    axes.axis('off')

    if bbox.format != BoundingBoxFormat.XYWH:
        bbox = convert_bounding_box_format(bbox, new_format=BoundingBoxFormat.XYWH)

    for box, cls in zip(bbox, cls_labels):
        x1, y1, w, h = box
        rect = plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=edgecolors[cls], linewidth=1)
        axes.add_patch(rect)

    plt.tight_layout()
    plt.show()


def visualize_bbox_frames(
    dataset, start_idx: int | None = None, period: int = 60, fps: int = 30,
    bbox_key: str = "boxes2d", bbox_class_key: str | None = "boxes2d_classes", figsize: tuple[int, int] = (6, 6)
):
    """
    Visualize a sequence of frames from a dataset with their bounding boxes as an animation.

    Args:
        dataset (Dataset): The dataset containing images and annotations. Each item should return (image, annotation).
        start_idx (int | None, optional): The starting index of the frame sequence. If None, a random start index is selected. Defaults to None.
        period (int, optional): Number of consecutive frames to visualize. Defaults to 60.
        fps (int, optional): Frames per second for the animation. Defaults to 30.
        bbox_key (str, optional): The key in the annotation dict for bounding boxes. Defaults to "boxes2d".
        bbox_class_key (str | None, optional): The key in the annotation dict for class labels. If None, all boxes are shown with the same color. Defaults to "boxes2d_classes".
        figsize (tuple[int, int], optional): Figure size for matplotlib. Defaults to (6, 6).

    Returns:
        None. Displays the animation of frames with bounding boxes using matplotlib and ipywidgets.
    """
    output = Output()
    display(output)

    if start_idx is None:
        start_idx = randint(0, len(dataset)-1-period)
    with output:
        for idx in range(start_idx, start_idx+period):
            if idx >= len(dataset):
                break

            output.clear_output(wait=True)
            visualize_bbox_frame(dataset, idx, bbox_key, bbox_class_key, figsize)
            time.sleep(1 / fps)


def visualize_bbox_frame_pair(
    dataset, idx: int | None = None,
    bbox_key: str = "boxes2d", bbox_class_key: str | None = "boxes2d_classes", figsize: tuple[int, int] = (7, 5)
):
    """
    Visualize a frame pair from a dataset with its bounding boxes.

    Args:
        dataset (Dataset): The dataset containing images and annotations. Each item should return (image, annotation).
        idx (int | None, optional): Index of the frame to visualize. If None, a random index is selected. Defaults to None.
        bbox_key (str, optional): The key in the annotation dict for bounding boxes. Defaults to "boxes2d".
        bbox_class_key (str | None, optional): The key in the annotation dict for class labels. If None, all boxes are shown with the same color. Defaults to "boxes2d_classes".
        figsize (tuple[int, int], optional): Figure size for matplotlib. Defaults to (7, 5).

    Returns:
        None. Displays the image with bounding boxes using matplotlib.
    """
    if idx is None:
        idx = randint(0, len(dataset)-1)

    rgb_image1, annotation1, rgb_image2, annotation2 = dataset[idx]
    bbox1: BoundingBoxes = annotation1[bbox_key]
    bbox2: BoundingBoxes = annotation2[bbox_key]
    if bbox_class_key is None:
        cls_labels1, cls_labels2 = [0] * len(bbox1), [0] * len(bbox2)
        edgecolors = [(random(), random(), random())]
    else:
        cls_labels1, cls_labels2 = annotation1[bbox_class_key], annotation2[bbox_class_key]
        num_classes = max(*cls_labels1, *cls_labels2) + 1 if len(cls_labels1)+len(cls_labels2) > 0 else 1
        edgecolors = [(random(), random(), random()) for _ in range(num_classes)]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, img, bbox, cls_labels in zip(axes, (rgb_image1, rgb_image2), (annotation1, annotation2), (cls_labels1, cls_labels2)):
        ax.imshow(img.permute(1, 2, 0) / 255.0)
        ax.set_title(f"Frame {idx}")
        ax.axis('off')

        if bbox.format != BoundingBoxFormat.XYWH:
            bbox = convert_bounding_box_format(bbox, new_format=BoundingBoxFormat.XYWH)

        for box, cls in zip(bbox, cls_labels):
            x1, y1, w, h = box
            rect = plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=edgecolors[cls], linewidth=1)
            ax.add_patch(rect)

    plt.tight_layout()
    plt.show()


def visualize_metrics(operations, cols=("mAP@0.50:0.95", "fps", "fwd", "bwd"), exclusive_cols=("fps", "fwd", "bwd")):
    """
    Visualize key performance metrics from an operations iterator in real time.

    Args:
        operations (iterator): An iterator that yields (results, ids) tuples at each step. 'results' is a list of dicts containing metric values, and 'ids' is a list of identifiers for each experiment.
        cols (tuple, optional): Tuple of metric column names to visualize. Defaults to ("mAP@0.50:0.95", "fps", "fwd", "bwd").
        exclusive_cols (tuple, optional): Tuple of column names that should be displayed exclusively (only once per row). Defaults to ("fps", "fwd", "bwd").

    Returns:
        list: The 'results' from the last iteration. (Mainly for visualization; the return value is secondary.)
    """
    output = Output()
    display(output)

    while True:
        try:
            results, ids = next(operations)
            visuals = []
            for res in results:
                data = {}
                for k, v in res.items():
                    for col, val in v.items():
                        if col in cols:
                            if col in exclusive_cols:
                                if col in data:
                                    del data[col]
                                data[col] = val
                            else:
                                key = k if isinstance(k, str) else k.value.replace("_daytime", "").replace("clear_", "")
                                data[f"{key} {col.split('@')[0]}"] = val
                visuals.append(data)
            with output:
                output.clear_output(wait=True)
                display(pd.DataFrame(visuals, ids))
        except StopIteration:
            return results
