from random import randint, random
import os
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
    bbox_key: str | None = "boxes2d", bbox_class_key: str | None = "boxes2d_classes", figsize: tuple[int, int] = (7, 5)
):
    """
    Visualize a frame pair from a dataset with its bounding boxes.

    Args:
        dataset (Dataset): The dataset containing images and annotations. Each item should return (image, annotation).
        idx (int | None, optional): Index of the frame to visualize. If None, a random index is selected. Defaults to None.
        bbox_key (str | None, optional): The key in the annotation dict for bounding boxes. If None, whole annotation object is treated as bounding boxes itself. Defaults to "boxes2d".
        bbox_class_key (str | None, optional): The key in the annotation dict for class labels. If None, all boxes are shown with the same color. Defaults to "boxes2d_classes".
        figsize (tuple[int, int], optional): Figure size for matplotlib. Defaults to (7, 5).

    Returns:
        None. Displays the image with bounding boxes using matplotlib.
    """
    if idx is None:
        idx = randint(0, len(dataset)-1)

    rgb_image1, rgb_image2, annotation1, annotation2 = dataset[idx]
    bbox1: BoundingBoxes = annotation1[bbox_key] if bbox_key else annotation1
    bbox2: BoundingBoxes = annotation2[bbox_key] if bbox_key else annotation2

    if bbox1.format != BoundingBoxFormat.XYWH:
        bbox1 = convert_bounding_box_format(bbox1, new_format=BoundingBoxFormat.XYWH)
    if bbox2.format != BoundingBoxFormat.XYWH:
        bbox2 = convert_bounding_box_format(bbox2, new_format=BoundingBoxFormat.XYWH)

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


def visualize_detectron2_sample(
    input_data: dict,
    output: dict,
    classes: list[str],
    save_path: str,
    combined: bool = True,
    gt_suffix: str = "_gt",
    pred_suffix: str = "_pred",
    font_scale: float = 2.0,
    box_thickness: int = 4,
    save_scale: float = 0.5,
):
    """
    Render GT and predicted bounding boxes on a single Detectron2 sample and save the result.

    Both GT and prediction panels are always rendered and saved. The ``combined`` flag controls
    whether they are written to a single file or to separate files.

    Args:
        input_data (dict): A single sample from a Detectron2-formatted batch. Expected keys:
            - ``"image"``     : (C, H, W) uint8 Tensor — the raw input image.
            - ``"instances"`` : Detectron2 ``Instances`` containing ``gt_boxes`` and ``gt_classes``.
            - ``"height"``    : original image height before any resizing.
            - ``"width"``     : original image width before any resizing.
        output (dict): Raw model output for the same sample (before ``post_process``). Expected key:
            - ``"instances"`` : Detectron2 ``Instances`` containing ``pred_boxes``, ``pred_classes``,
                                and ``scores``.
        classes (list[str]): List of class names indexed by class ID.
        save_path (str): Destination file path (e.g. ``"vis/sample_0.jpg"``).
            - If ``combined=True``  → saved as-is (single file).
            - If ``combined=False`` → saved as ``<stem><gt_suffix><ext>`` and ``<stem><pred_suffix><ext>``.
            Parent directories are created automatically if they do not exist.
        combined (bool): If ``True``, GT (left) and prediction (right) are concatenated
            horizontally into one image. If ``False``, each panel is saved as a separate file.
            Defaults to ``True``.
        gt_suffix (str): Suffix appended to the stem for the GT file and its panel label
            when ``combined=False``. Defaults to ``"_gt"``.
        pred_suffix (str): Suffix appended to the stem for the prediction file and its panel
            label when ``combined=False``. Defaults to ``"_pred"``.
        font_scale (float): Font scale for the panel label drawn at the bottom of the image.
            Increase for publication-quality figures. Defaults to ``2.0``.
        box_thickness (int): Line thickness (in pixels) for the panel label text.
            Also passed to Detectron2's ``Visualizer`` as the bounding-box line width.
            Defaults to ``4``.
        save_scale (float): Scaling factor applied to the final image before writing to disk.
            ``0.5`` saves at half resolution. ``1.0`` keeps the original size.
            Defaults to ``0.5``.

    Returns:
        None
    """
    import cv2
    import torch
    from detectron2.data import MetadataCatalog
    from detectron2.modeling.postprocessing import detector_postprocess
    from detectron2.structures import Instances
    from detectron2.utils.visualizer import ColorMode, Visualizer

    # --- Prepare the image as an RGB numpy array (H, W, C) ---
    # Detectron2 stores images as (C, H, W) uint8 tensors; Visualizer expects (H, W, C) RGB.
    image_chw: torch.Tensor = input_data['image']              # (C, H, W) uint8
    image_hwc_bgr = image_chw.permute(1, 2, 0).cpu().numpy()  # (H, W, C) BGR
    image_rgb = image_hwc_bgr[:, :, ::-1].copy()               # BGR → RGB

    # --- Register class names to a temporary MetadataCatalog entry ---
    _META_NAME = "__visualizer_temp__"
    MetadataCatalog.get(_META_NAME).thing_classes = classes
    metadata = MetadataCatalog.get(_META_NAME)

    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_thickness

    # --- Render GT panel ---
    # Visualizer.draw_instance_predictions() reads pred_boxes / pred_classes / scores,
    # so we copy GT fields into those slots before calling it.
    gt_instances = Instances(image_size=input_data['instances'].image_size)
    gt_instances.pred_boxes   = input_data['instances'].gt_boxes
    gt_instances.pred_classes = input_data['instances'].gt_classes
    gt_instances.scores = torch.ones(len(gt_instances), dtype=torch.float32)  # dummy confidence

    # Rescale boxes from the model's internal resolution to the original (height × width).
    gt_instances = detector_postprocess(gt_instances, input_data['height'], input_data['width'])

    vis_gt = Visualizer(image_rgb, metadata=metadata, instance_mode=ColorMode.IMAGE, scale=box_thickness / 2)
    gt_bgr = cv2.cvtColor(
        vis_gt.draw_instance_predictions(gt_instances.to("cpu")).get_image(),
        cv2.COLOR_RGB2BGR,
    )

    # --- Render prediction panel ---
    pred_instances = output['instances'].to("cpu")
    vis_pred = Visualizer(image_rgb, metadata=metadata, instance_mode=ColorMode.IMAGE, scale=box_thickness / 2)
    pred_bgr = cv2.cvtColor(
        vis_pred.draw_instance_predictions(pred_instances).get_image(),
        cv2.COLOR_RGB2BGR,
    )

    # --- Resize helper: scale down before saving ---
    def _save(path: str, img):
        if save_scale != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * save_scale), int(h * save_scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(path, img)

    # --- Save to disk ---
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    if combined:
        # Concatenate GT (left) and prediction (right) into a single image.
        canvas = cv2.hconcat([gt_bgr, pred_bgr])
        h, half_w = canvas.shape[0], canvas.shape[1] // 2
        cv2.putText(canvas, "GT",   (10, h - 10),           font, scale, (0, 255, 0),   thickness)
        cv2.putText(canvas, "Pred", (half_w + 10, h - 10),  font, scale, (0,   0, 255), thickness)
        _save(save_path, canvas)
    else:
        # Derive sibling file paths using the caller-supplied suffixes.
        stem, ext = os.path.splitext(save_path)
        cv2.putText(gt_bgr,   gt_suffix,   (10, gt_bgr.shape[0]   - 10), font, scale, (0, 255, 0),   thickness)
        cv2.putText(pred_bgr, pred_suffix, (10, pred_bgr.shape[0] - 10), font, scale, (0,   0, 255), thickness)
        _save(f"{stem}{gt_suffix}{ext}",   gt_bgr)
        _save(f"{stem}{pred_suffix}{ext}", pred_bgr)

