import sys
import os
from os import path
from argparse import ArgumentParser
from pathlib import Path
import copy
import warnings
from types import ModuleType
import math
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.layers import FrozenBatchNorm2d

# Import tensor-based Kalman Filter
from ttadapters.methods.regularizers.plugin.apt.filter_detr import (
    AdaptiveKalmanFilterCXCYWH,
    initialize_state_cxcywh
)


# Bbox conversion functions (tensor-based)
def bbox_xyxy_to_cxcywh(bbox: torch.Tensor) -> torch.Tensor:
    """
    Convert bbox from [x1, y1, x2, y2] to [cx, cy, w, h]

    Args:
        bbox: [..., 4] tensor
    Returns:
        cxcywh: [..., 4] tensor
    """
    x1, y1, x2, y2 = bbox.unbind(-1)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return torch.stack([cx, cy, w, h], dim=-1)


def bbox_cxcywh_to_xyxy(bbox: torch.Tensor) -> torch.Tensor:
    """
    Convert bbox from [cx, cy, w, h] to [x1, y1, x2, y2]

    Args:
        bbox: [..., 4] tensor
    Returns:
        xyxy: [..., 4] tensor
    """
    cx, cy, w, h = bbox.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def encode_bbox_delta(bbox: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Encode bbox as delta (offset) relative to reference box.
    This is similar to Faster R-CNN's bbox regression target encoding.

    Args:
        bbox: [..., 4] target box [x1, y1, x2, y2]
        reference: [..., 4] reference box [x1, y1, x2, y2]
    Returns:
        delta: [..., 4] encoded as [tx, ty, tw, th]
    """
    # Reference box
    rx1, ry1, rx2, ry2 = reference.unbind(-1)
    rw = (rx2 - rx1).clamp(min=1e-6)
    rh = (ry2 - ry1).clamp(min=1e-6)
    rcx = rx1 + rw * 0.5
    rcy = ry1 + rh * 0.5

    # Target box
    x1, y1, x2, y2 = bbox.unbind(-1)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5

    # Encode
    tx = (cx - rcx) / rw
    ty = (cy - rcy) / rh
    tw = torch.log(w / rw)
    th = torch.log(h / rh)

    return torch.stack([tx, ty, tw, th], dim=-1)


class Track:
    """
    Represents a single tracked object with tensor-based Kalman filter.
    """

    def __init__(self, bbox: torch.Tensor, class_id: int, track_id: int,
                 kalman_filter: AdaptiveKalmanFilterCXCYWH, confidence: torch.Tensor):
        """
        Initialize a track.

        Parameters
        ----------
        bbox : torch.Tensor (4,)
            Initial bounding box [x1, y1, x2, y2]
        class_id : int
            Class ID of the object
        track_id : int
            Unique track ID
        kalman_filter : AdaptiveKalmanFilterCXCYWH
            Kalman filter instance
        confidence : torch.Tensor
            Detection confidence score
        """
        self.track_id = track_id
        self.class_id = class_id
        self.kf = kalman_filter
        self.device = bbox.device

        # Convert bbox to cxcywh format
        cxcywh = bbox_xyxy_to_cxcywh(bbox)

        # Initialize Kalman filter state
        self.mean, self.covariance = initialize_state_cxcywh(cxcywh)

        self.age = 0
        self.hits = 1
        self.time_since_update = 0

    def predict(self) -> Tuple[torch.Tensor, float]:
        """
        Propagate the state distribution to the current time step using Kalman filter prediction.

        Returns
        -------
        predicted_bbox : torch.Tensor (4,)
            Predicted bounding box [x1, y1, x2, y2]
        uncertainty : float
            Prediction uncertainty (trace of bbox covariance)
        """
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

        # Convert predicted state to bbox
        predicted_cxcywh = self.mean[:4, 0]  # [cx, cy, w, h]
        predicted_bbox = bbox_cxcywh_to_xyxy(predicted_cxcywh)

        # Compute uncertainty (trace of bbox covariance)
        # Covariance is [8, 8], we only need bbox part [cx, cy, w, h]
        bbox_covariance = self.covariance[:4, :4]
        uncertainty = torch.trace(bbox_covariance).item()

        # Store for later use
        self.uncertainty = uncertainty

        return predicted_bbox, uncertainty

    def update(self, bbox: torch.Tensor, confidence: torch.Tensor):
        """
        Update the track with a matched detection.

        Parameters
        ----------
        bbox : torch.Tensor (4,)
            Detected bounding box [x1, y1, x2, y2]
        confidence : torch.Tensor
            Detection confidence score
        """
        # Convert to cxcywh and prepare measurement
        cxcywh = bbox_xyxy_to_cxcywh(bbox)
        measurement = cxcywh.unsqueeze(-1)  # [4, 1]

        # Update Kalman filter
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, measurement, confidence)

        self.hits += 1
        self.time_since_update = 0

    def get_state(self) -> torch.Tensor:
        """
        Return the current bounding box estimate.

        Returns
        -------
        bbox : torch.Tensor (4,)
            Current bounding box [x1, y1, x2, y2]
        """
        cxcywh = self.mean[:4, 0]
        bbox = bbox_cxcywh_to_xyxy(cxcywh)
        return bbox


@dataclass
class OursConfig:
    model_type: str = "rcnn"  # "swinrcnn", "yolo11", "rtdetr"

    data_root: str = './datasets'
    device: torch.device = torch.device("cuda")
    batch_size: int = 1  # Must be 1 for tracking

    lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    optimizer_option: str = "SGD"  # AdamW

    # Adaptation parameters
    adapt_bn: bool = True  # Adapt BatchNorm/LayerNorm layers
    adapt_conv: bool = False  # Adapt Conv2d layers
    adapt_linear: bool = False  # Adapt Linear layers

    # Tracking parameters
    iou_threshold: float = 0.3  # IoU threshold for matching
    min_confidence: float = 0.5  # Minimum detection confidence
    max_age: int = 30  # Maximum frames to keep track without update

    # Loss weights
    bbox_loss_weight: float = 1.0
    smooth_l1_beta: float = 1.0
    use_delta_loss: bool = True  # Use delta encoding (tx,ty,tw,th) instead of absolute coords

    # Innovation weighting parameters (inverse weighting)
    use_covariance_weighting: bool = False  # Use covariance-based confidence (DISABLED - not helpful)
    use_innovation_weighting: bool = True  # Use innovation-based weighting (match quality)
    max_innovation: float = 100.0  # Innovation threshold for capping (beyond this = tracking failure)
    min_innovation_weight: float = 0.2  # Minimum weight when innovation is 0 (already aligned)

    # Detection confidence gating (exponential penalty for low-confidence detections)
    confidence_penalty_exponent: float = 2.0  # Apply detection_confidence ** exponent (2.0 or 3.0)

    # Kalman filter update strategy
    use_model_for_kalman_update: bool = True  # CRITICAL: Use detection as measurement (not prediction!)
    kalman_detection_blend: float = 1.0  # Blend ratio: 0=pure Kalman, 1=pure detection

    # Batch aggregation for stable learning
    batch_accumulation_steps: int = 1  # Accumulate gradients over N frames before update (1=no accumulation)

    # Scene change detection and tracker reset
    shift_detection_window: int = 10  # Number of frames to track for average matches
    shift_detection_threshold: float = 0.3  # Reset if matches < threshold * avg_matches
    shift_detection_min_matches: int = 2  # Or reset if matches <= this absolute value
    reset_tracker_on_shift: bool = True  # Reset tracker when scene change detected

    # Quality-based loss filtering (adaptive learning control)
    enable_quality_filter: bool = True  # Enable loss quality filtering
    min_matches: int = 4  # Minimum number of matches required for loss computation (4-8 recommended)
    min_track_hits: float = 3.0  # Minimum average track maturity
    min_match_iou: float = 0.4  # Minimum average IoU for reliable matching
    max_innovation_cv: float = 0.8  # Maximum CV for consistency
    min_avg_innovation: float = 5.0  # Skip if already adapted
    max_avg_innovation: float = 80.0  # Skip if too large (likely failure)
    max_outlier_ratio: float = 3.0  # Max innovation / Avg innovation threshold


class Ours(nn.Module):
    """
    Kalman Filter-based Tracking TTA method.

    Pipeline:
    1. At frame t: detect objects and initialize/update tracks
    2. Predict frame t+1 bboxes using Kalman filter
    3. At frame t+1: detect objects with the model
    4. Match Kalman predictions with model detections
    5. Compute loss between matched predictions
    6. Update the model
    """

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.cfg = config

        self.data_root = config.data_root
        self.device = config.device
        self.batch_size = config.batch_size

        # Tracking components (tensor-based Kalman filter)
        self.kf = AdaptiveKalmanFilterCXCYWH(learnable_uncertainty=False).to(self.device)
        self.tracks: List[Track] = []
        self.next_track_id = 0
        self.frame_count = 0

        # Store predictions from Kalman filter for next frame
        self.kalman_predictions: Dict[int, torch.Tensor] = {}  # track_id -> bbox (tensor)
        self.kalman_uncertainties: Dict[int, float] = {}  # track_id -> uncertainty

        # Scene change detection
        self.current_scene = None  # Track current scene/video
        self.prev_innovations = []  # Store recent innovations (for stats)
        self.prev_matches = []  # Store recent match counts (for stats)
        self.matches_history = []  # Rolling window for shift detection

        # Adaptation statistics tracking
        self.total_frames = 0  # Total frames processed
        self.adapted_frames = 0  # Frames where adaptation occurred
        self.scene_total_frames = 0  # Frames in current scene
        self.scene_adapted_frames = 0  # Adapted frames in current scene

        # Batch accumulation for gradient smoothing
        self.accumulated_loss = None  # Accumulated loss over multiple frames
        self.accumulation_count = 0  # Number of frames accumulated

        self._setup()

    def _setup(self):
        self.model = copy.deepcopy(self.model)
        self.model.to(self.device)

        # Collect backbone parameters for adaptation
        params = []

        if self.cfg.model_type in ("rcnn", "swinrcnn"):
            # Adapt backbone only (similar to MeanTeacher approach)
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'bottom_up'):
                for m_name, m in self.model.backbone.bottom_up.named_modules():
                    # BatchNorm and LayerNorm layers
                    if self.cfg.adapt_bn and isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, FrozenBatchNorm2d)):
                        if hasattr(m, 'weight') and m.weight is not None:
                            m.weight.requires_grad = True
                            params.append(m.weight)
                        if hasattr(m, 'bias') and m.bias is not None:
                            m.bias.requires_grad = True
                            params.append(m.bias)

                    # Conv2d layers
                    if self.cfg.adapt_conv and isinstance(m, nn.Conv2d):
                        if hasattr(m, 'weight') and m.weight is not None:
                            m.weight.requires_grad = True
                            params.append(m.weight)
                        if hasattr(m, 'bias') and m.bias is not None:
                            m.bias.requires_grad = True
                            params.append(m.bias)

                    # Linear layers
                    if self.cfg.adapt_linear and isinstance(m, nn.Linear):
                        if hasattr(m, 'weight') and m.weight is not None:
                            m.weight.requires_grad = True
                            params.append(m.weight)
                        if hasattr(m, 'bias') and m.bias is not None:
                            m.bias.requires_grad = True
                            params.append(m.bias)

        # Count total number of parameters (elements)
        total_params = sum(p.numel() for p in params)
        print(f"Ours: Adapting {len(params)} parameter tensors ({total_params:,} total elements)")
        print(f"  BN/LayerNorm: {self.cfg.adapt_bn}, Conv2d: {self.cfg.adapt_conv}, Linear: {self.cfg.adapt_linear}")

        if self.cfg.optimizer_option == "SGD":
            self.optimizer = optim.SGD(
                params,
                lr=self.cfg.lr,
                momentum=self.cfg.momentum,
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer_option == "AdamW":
            self.optimizer = optim.AdamW(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer_option == "Adam":
            self.optimizer = optim.Adam(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay
            )
        else:
            warnings.warn("Unknown optimizer_option.")

    def _set_bn_train_mode(self):
        """
        Set BN/LN layers to train mode while keeping other modules in eval mode.
        This ensures gradient flows through BN parameters during TTA.
        """
        self.model.train()
        for module in self.model.modules():
            # Keep only BN/LN in train mode
            if not isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, FrozenBatchNorm2d)):
                module.eval()

    def _get_detections_from_output(self, outputs):
        """
        Extract detections from model output (keep as tensors for gradient flow).

        Returns
        -------
        detections : List[Dict]
            List of detections, each dict contains:
            - 'bbox': torch.Tensor (4,) [x1, y1, x2, y2] with gradient
            - 'score': torch.Tensor with gradient
            - 'class_id': int
            - 'index': int (index in original predictions)
        """
        detections = []

        for output in outputs:
            instances = output['instances']

            # Filter by confidence
            valid_mask = instances.scores >= self.cfg.min_confidence
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            boxes = instances.pred_boxes.tensor[valid_mask]  # Keep as tensor!
            scores = instances.scores[valid_mask]  # Keep as tensor!
            classes = instances.pred_classes[valid_mask]

            for idx, (bbox, score, class_id) in enumerate(zip(boxes, scores, classes)):
                detections.append({
                    'bbox': bbox,  # torch.Tensor with gradient
                    'score': score,  # torch.Tensor with gradient
                    'class_id': int(class_id.item()),
                    'index': int(valid_indices[idx].item())  # Original index for loss computation
                })

        return detections

    def _match_tracks_to_detections(self, detections):
        """
        Match Kalman predictions to current detections using IoU (tensor-based).

        Parameters
        ----------
        detections : List[Dict]
            Current frame detections (with tensor bboxes)

        Returns
        -------
        matches : List[Tuple[int, int, float]]
            List of (track_idx, detection_idx, iou) tuples
        unmatched_tracks : List[int]
            Indices of unmatched tracks
        unmatched_detections : List[int]
            Indices of unmatched detections
        """
        if len(self.kalman_predictions) == 0 or len(detections) == 0:
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_detections = list(range(len(detections)))
            return [], unmatched_tracks, unmatched_detections

        # Build cost matrix (1 - IoU)
        track_indices = []
        kalman_bboxes = []

        for i, track in enumerate(self.tracks):
            if track.track_id in self.kalman_predictions:
                track_indices.append(i)
                kalman_bboxes.append(self.kalman_predictions[track.track_id])

        if len(kalman_bboxes) == 0:
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_detections = list(range(len(detections)))
            return [], unmatched_tracks, unmatched_detections

        # Stack kalman bboxes (already tensors)
        kalman_bboxes_tensor = torch.stack(kalman_bboxes)  # [N, 4]

        # Stack detection bboxes (already tensors)
        detection_bboxes_tensor = torch.stack([d['bbox'] for d in detections])  # [M, 4]

        # Compute IoU matrix (detach for matching - no gradient needed)
        with torch.no_grad():
            iou_matrix = pairwise_iou(
                Boxes(kalman_bboxes_tensor),
                Boxes(detection_bboxes_tensor)
            ).cpu().numpy()

        # Greedy matching: for each track, find best detection
        matches = []
        unmatched_tracks = list(range(len(track_indices)))
        unmatched_detections = list(range(len(detections)))

        for track_local_idx in range(len(track_indices)):
            if len(unmatched_detections) == 0:
                break

            best_iou = self.cfg.iou_threshold
            best_det_idx = -1

            for det_idx in unmatched_detections:
                iou = iou_matrix[track_local_idx, det_idx]
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_det_idx != -1:
                track_idx = track_indices[track_local_idx]
                matches.append((track_idx, best_det_idx, best_iou))
                unmatched_tracks.remove(track_local_idx)
                unmatched_detections.remove(best_det_idx)

        # Convert unmatched_tracks from local indices to global indices
        unmatched_tracks = [track_indices[i] for i in unmatched_tracks]

        return matches, unmatched_tracks, unmatched_detections

    def _update_tracks(self, detections, matches, unmatched_tracks, unmatched_detections):
        """
        Update tracks based on matches and create new tracks for unmatched detections.

        Parameters
        ----------
        detections : List[Dict]
            Current frame detections (with tensor bboxes)
        matches : List[Tuple[int, int, float]]
            Matched (track_idx, detection_idx, iou) tuples
        unmatched_tracks : List[int]
            Unmatched track indices
        unmatched_detections : List[int]
            Unmatched detection indices
        """
        # Update matched tracks (detach for tracking - no gradient needed)
        for track_idx, det_idx, iou in matches:
            track = self.tracks[track_idx]
            det = detections[det_idx]

            # CRITICAL: Always use detection as measurement (Kalman filter principle)
            # Kalman filter requires NEW measurement to update belief
            det_bbox = det['bbox'].detach()

            if self.cfg.use_model_for_kalman_update and self.cfg.kalman_detection_blend < 1.0:
                # Optional: Blend with Kalman prediction for robustness
                kalman_bbox = self.kalman_predictions[track.track_id].detach()
                blend_ratio = self.cfg.kalman_detection_blend
                update_bbox = blend_ratio * det_bbox + (1 - blend_ratio) * kalman_bbox
            else:
                # Default: Use pure detection (standard Kalman update)
                update_bbox = det_bbox

            track.update(
                update_bbox,
                det['score'].detach()
            )

        # Remove old tracks
        self.tracks = [t for i, t in enumerate(self.tracks)
                      if i not in unmatched_tracks or t.time_since_update < self.cfg.max_age]

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_track = Track(
                bbox=det['bbox'].detach(),  # Detach for initialization
                class_id=det['class_id'],
                track_id=self.next_track_id,
                kalman_filter=self.kf,
                confidence=det['score'].detach()
            )
            self.tracks.append(new_track)
            self.next_track_id += 1

    def _predict_next_frame(self):
        """
        Use Kalman filter to predict bboxes for next frame.
        Also stores uncertainty for each prediction.
        """
        self.kalman_predictions = {}
        self.kalman_uncertainties = {}
        for track in self.tracks:
            predicted_bbox, uncertainty = track.predict()
            self.kalman_predictions[track.track_id] = predicted_bbox
            self.kalman_uncertainties[track.track_id] = uncertainty

    def _compute_tracking_loss(self, matches, detections, outputs):
        """
        Compute loss between Kalman predictions and model predictions for matched tracks.
        Uses Covariance-based confidence and Innovation-based weighting.
        CRITICAL: Model bbox must retain gradient for backprop!

        Parameters
        ----------
        matches : List[Tuple[int, int]]
            Matched (track_idx, detection_idx) pairs
        detections : List[Dict]
            Current frame detections from model (with gradient)
        outputs : List[Dict]
            Raw model outputs (unused, kept for compatibility)

        Returns
        -------
        loss : torch.Tensor
            Tracking loss with gradient
        adapted : bool
            Whether adaptation occurred (loss > 0)
        """
        # Skip loss computation if too few matches (unreliable for adaptation)
        if len(matches) < self.cfg.min_matches:
            return torch.tensor(0.0, device=self.device, requires_grad=True), False

        losses = []
        weights = []
        innovations_list = []  # Track all innovations for logging
        ious_list = []  # Track IoU for each match
        track_hits_list = []  # Track maturity
        conf_list = []  # Detection confidence

        for track_idx, det_idx, iou in matches:
            track = self.tracks[track_idx]

            # Collect statistics
            ious_list.append(iou)
            track_hits_list.append(track.hits)
            conf_list.append(detections[det_idx]['score'].item())

            # Get Kalman prediction (target - detach, no gradient needed)
            kalman_bbox = self.kalman_predictions[track.track_id].detach()
            kalman_bbox_cxcywh = bbox_xyxy_to_cxcywh(kalman_bbox)

            # Get model prediction (KEEP GRADIENT!)
            model_bbox = detections[det_idx]['bbox']  # torch.Tensor with gradient!

            # === Component 1: Covariance-based Confidence (Adaptive) ===
            covariance_confidence = 1.0  # Default (no weighting)
            base_confidence = 1.0  # For debugging
            uncertainty_clipped = 0.0  # For debugging

            if self.cfg.use_covariance_weighting:
                uncertainty = self.kalman_uncertainties[track.track_id]

                # Clip uncertainty to reasonable range
                # Max uncertainty = 20 → base min confidence = 0.048
                uncertainty_clipped = min(uncertainty, 20.0)

                # Base confidence from uncertainty
                base_confidence = 1.0 / (1.0 + uncertainty_clipped)

                # Adaptive weighting based on track maturity (hits count)
                track_hits = track.hits

                if track_hits < 3:
                    # New track (1-2 matches): needs learning but don't trust too much
                    min_weight = 0.3
                    max_weight = 0.5
                    covariance_confidence = max(min_weight, min(base_confidence, max_weight))

                elif track_hits < 10:
                    # Intermediate track (3-9 matches): progressive trust
                    min_weight = 0.1
                    covariance_confidence = max(min_weight, base_confidence)

                else:
                    # Mature track (10+ matches): full uncertainty-based weighting
                    covariance_confidence = base_confidence

            # === Component 2: Innovation-based Weight (Match Quality) ===
            innovation_weight = 1.0  # Default
            innovation_norm = 0.0  # For debugging

            # Compute innovation (prediction error)
            measured_bbox_cxcywh = bbox_xyxy_to_cxcywh(detections[det_idx]['bbox'].detach())
            innovation = measured_bbox_cxcywh - kalman_bbox_cxcywh
            innovation_norm = torch.norm(innovation).item()
            innovations_list.append(innovation_norm)  # Log for output

            if self.cfg.use_innovation_weighting:
                # INVERSE weighting: Higher innovation = Higher weight (need more adaptation)
                # Map innovation to weight based on quality filter range
                # [min_avg_innovation, max_avg_innovation] → [min_innovation_weight, 1.0]
                #
                # Example with default settings:
                #   innovation=5   → weight=0.2 (minimum, already adapted)
                #   innovation=42.5→ weight=0.6 (medium adaptation)
                #   innovation=80  → weight=1.0 (maximum, strong adaptation)
                #   innovation>80  → weight=0.1 (tracking failure, almost ignore)

                if innovation_norm > self.cfg.max_avg_innovation:
                    # Beyond max threshold = tracking failure or unreliable
                    innovation_weight = 0.1  # Almost ignore (will be filtered anyway)
                elif innovation_norm < self.cfg.min_avg_innovation:
                    # Below min threshold = already adapted
                    innovation_weight = self.cfg.min_innovation_weight  # Minimum (will be filtered anyway)
                else:
                    # Linear interpolation within valid learning range
                    min_w = self.cfg.min_innovation_weight
                    valid_range = self.cfg.max_avg_innovation - self.cfg.min_avg_innovation
                    normalized = (innovation_norm - self.cfg.min_avg_innovation) / valid_range
                    innovation_weight = min_w + (1.0 - min_w) * normalized

            # === Component 3: Detection Confidence with Exponential Penalty ===
            detection_confidence_raw = detections[det_idx]['score'].item()

            # Apply exponential penalty to heavily penalize low-confidence detections
            # This prevents learning from unreliable detections (especially in night/challenging domains)
            detection_confidence_gated = detection_confidence_raw ** self.cfg.confidence_penalty_exponent

            # === Combined Weight ===
            # covariance_confidence is always 1.0 (disabled), so effectively:
            # total_weight = innovation_weight * (detection_confidence ** exponent)
            total_weight = covariance_confidence * innovation_weight * detection_confidence_gated

            # === Compute Loss ===
            if self.cfg.use_delta_loss:
                # Delta-based loss (better for BN adaptation)
                # Use Kalman prediction as reference for both
                reference = kalman_bbox.detach()
                delta_model = encode_bbox_delta(model_bbox, reference)
                delta_target = encode_bbox_delta(kalman_bbox, reference)
                loss = F.smooth_l1_loss(
                    delta_model,
                    delta_target,
                    beta=self.cfg.smooth_l1_beta
                )
            else:
                # Absolute coordinate loss (original)
                loss = F.smooth_l1_loss(
                    model_bbox,
                    kalman_bbox,
                    beta=self.cfg.smooth_l1_beta
                )

            # Store weighted loss
            losses.append(loss)
            weights.append(total_weight)

        # === Calculate Innovation Statistics ===
        if len(innovations_list) > 0:
            avg_innovation = sum(innovations_list) / len(innovations_list)
            min_innovation = min(innovations_list)
            max_innovation = max(innovations_list)
            std_innovation = (sum([(x - avg_innovation)**2 for x in innovations_list]) / len(innovations_list)) ** 0.5
            cv_innovation = std_innovation / avg_innovation if avg_innovation > 0 else 0.0

            # Calculate additional statistics
            avg_iou = sum(ious_list) / len(ious_list)
            avg_hits = sum(track_hits_list) / len(track_hits_list)
            avg_conf = sum(conf_list) / len(conf_list)

            # === Quality-based Loss Filtering ===
            if self.cfg.enable_quality_filter:
                skip_reasons = []

                # Criterion 1: Track maturity
                if avg_hits < self.cfg.min_track_hits:
                    skip_reasons.append(f"Immature tracks (avg_hits={avg_hits:.1f} < {self.cfg.min_track_hits})")

                # Criterion 2: Match quality (IoU)
                if avg_iou < self.cfg.min_match_iou:
                    skip_reasons.append(f"Poor matching (avg_IoU={avg_iou:.3f} < {self.cfg.min_match_iou})")

                # Criterion 3: Consistency (CV)
                if cv_innovation > self.cfg.max_innovation_cv:
                    skip_reasons.append(f"Inconsistent (CV={cv_innovation:.2f} > {self.cfg.max_innovation_cv})")

                # Criterion 4: Already adapted (too small)
                if avg_innovation < self.cfg.min_avg_innovation:
                    skip_reasons.append(f"Already adapted (avg={avg_innovation:.2f} < {self.cfg.min_avg_innovation})")

                # Criterion 5: Too large (tracking failure)
                if avg_innovation > self.cfg.max_avg_innovation:
                    skip_reasons.append(f"Too large (avg={avg_innovation:.2f} > {self.cfg.max_avg_innovation})")

                # Criterion 6: Extreme outliers
                if max_innovation > self.cfg.max_outlier_ratio * avg_innovation:
                    skip_reasons.append(f"Outlier detected (max={max_innovation:.2f} > {self.cfg.max_outlier_ratio}*avg)")

                # If any criterion fails, skip loss
                if len(skip_reasons) > 0:
                    return torch.tensor(0.0, device=self.device, requires_grad=True), False

            # Store for scene statistics
            self.prev_innovations.append(avg_innovation)
            self.prev_matches.append(len(matches))
        else:
            self.prev_matches.append(len(matches))

        if len(losses) > 0:
            # Convert weights to tensor
            weights_tensor = torch.tensor(weights, device=self.device, dtype=torch.float32)

            # Weighted average of losses
            losses_tensor = torch.stack(losses)
            weighted_loss = (losses_tensor * weights_tensor).sum() / (weights_tensor.sum() + 1e-8)

            return self.cfg.bbox_loss_weight * weighted_loss, True
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True), False

    def forward(self, x):
        """
        Forward pass with tracking-based adaptation.

        Parameters
        ----------
        x : List[Dict]
            Batch of images (batch_size must be 1)

        Returns
        -------
        outputs : List[Dict]
            Model predictions
        """
        assert len(x) == 1, "Batch size must be 1 for tracking"

        # Detect scene change (based on videoName or file_name)
        scene_name = None
        if 'videoName' in x[0]:
            scene_name = x[0]['videoName']
        elif 'file_name' in x[0]:
            # Extract scene from file path (e.g., "cloudy", "overcast", etc.)
            file_name = x[0]['file_name']
            for corruption in ['clear', 'cloudy', 'overcast', 'rainy', 'foggy', 'night']:
                if corruption in file_name.lower():
                    scene_name = corruption
                    break

        scene_changed = (self.current_scene is not None and scene_name != self.current_scene)

        if scene_changed:
            # Print previous scene statistics before reset
            if self.scene_total_frames > 0:
                scene_adapt_ratio = (self.scene_adapted_frames / self.scene_total_frames) * 100
                print(f"\n{'='*80}")
                print(f"Scene [{self.current_scene}] Adaptation Statistics:")
                print(f"  Total frames: {self.scene_total_frames}")
                print(f"  Adapted frames: {self.scene_adapted_frames}")
                print(f"  Adaptation ratio: {scene_adapt_ratio:.2f}%")
                print(f"{'='*80}\n")

            # Reset statistics for new scene
            self.prev_innovations = []
            self.prev_matches = []
            self.scene_total_frames = 0
            self.scene_adapted_frames = 0

            # Reset batch accumulation (discard incomplete batch)
            self.accumulated_loss = None
            self.accumulation_count = 0

        self.current_scene = scene_name
        self.frame_count += 1
        self.total_frames += 1
        self.scene_total_frames += 1

        # First frame: just detect and initialize tracks
        if self.frame_count == 1:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)

            detections = self._get_detections_from_output(outputs)

            # Initialize tracks (detach since no gradient needed for first frame)
            for det in detections:
                new_track = Track(
                    bbox=det['bbox'].detach(),
                    class_id=det['class_id'],
                    track_id=self.next_track_id,
                    kalman_filter=self.kf,
                    confidence=det['score'].detach()
                )
                self.tracks.append(new_track)
                self.next_track_id += 1

            # Predict for next frame
            self._predict_next_frame()

            return outputs

        # Subsequent frames: use tracking for adaptation
        # CRITICAL: Set BN to train mode for gradient flow
        self._set_bn_train_mode()

        outputs = self.model(x)
        detections = self._get_detections_from_output(outputs)

        # Match Kalman predictions with model detections
        matches, unmatched_tracks, unmatched_detections = \
            self._match_tracks_to_detections(detections)

        # Compute tracking loss and check if adaptation occurred
        tracking_loss, adapted = self._compute_tracking_loss(matches, detections, outputs)

        # === Batch Accumulation for Gradient Smoothing ===
        if tracking_loss > 0:
            # Clear gradients only at the start of accumulation batch
            if self.accumulation_count == 0:
                self.optimizer.zero_grad(set_to_none=True)

            # Compute and accumulate gradients
            (tracking_loss / self.cfg.batch_accumulation_steps).backward()
            self.accumulation_count += 1

            # Update model when accumulation is complete
            if self.accumulation_count >= self.cfg.batch_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)  # Clear for next batch
                self.accumulation_count = 0

        # Track adaptation statistics
        if adapted:
            self.adapted_frames += 1
            self.scene_adapted_frames += 1

        # Update tracks
        self._update_tracks(detections, matches, unmatched_tracks, unmatched_detections)

        # Predict for next frame
        self._predict_next_frame()

        # Print periodic adaptation statistics
        if self.frame_count % 100 == 0:
            overall_ratio = (self.adapted_frames / self.total_frames) * 100 if self.total_frames > 0 else 0
            scene_ratio = (self.scene_adapted_frames / self.scene_total_frames) * 100 if self.scene_total_frames > 0 else 0
            print(f"\n[Frame {self.frame_count}] Adaptation Statistics:")
            print(f"  Overall: {self.adapted_frames}/{self.total_frames} ({overall_ratio:.2f}%)")
            print(f"  Current scene [{self.current_scene}]: {self.scene_adapted_frames}/{self.scene_total_frames} ({scene_ratio:.2f}%)\n")

        # Return predictions from the first forward pass (before adaptation)
        return outputs

    def get_adaptation_stats(self):
        """
        Get overall adaptation statistics.

        Returns
        -------
        stats : dict
            Dictionary containing adaptation statistics
        """
        overall_ratio = (self.adapted_frames / self.total_frames) * 100 if self.total_frames > 0 else 0
        scene_ratio = (self.scene_adapted_frames / self.scene_total_frames) * 100 if self.scene_total_frames > 0 else 0

        return {
            'total_frames': self.total_frames,
            'adapted_frames': self.adapted_frames,
            'overall_ratio': overall_ratio,
            'current_scene': self.current_scene,
            'scene_total_frames': self.scene_total_frames,
            'scene_adapted_frames': self.scene_adapted_frames,
            'scene_ratio': scene_ratio
        }

    def print_final_stats(self):
        """Print final adaptation statistics."""
        # Print current scene stats
        if self.scene_total_frames > 0:
            scene_adapt_ratio = (self.scene_adapted_frames / self.scene_total_frames) * 100
            print(f"\n{'='*80}")
            print(f"Scene [{self.current_scene}] Final Adaptation Statistics:")
            print(f"  Total frames: {self.scene_total_frames}")
            print(f"  Adapted frames: {self.scene_adapted_frames}")
            print(f"  Adaptation ratio: {scene_adapt_ratio:.2f}%")
            print(f"{'='*80}\n")

        # Print overall stats
        if self.total_frames > 0:
            overall_ratio = (self.adapted_frames / self.total_frames) * 100
            print(f"\n{'='*80}")
            print(f"Overall Adaptation Statistics:")
            print(f"  Total frames: {self.total_frames}")
            print(f"  Adapted frames: {self.adapted_frames}")
            print(f"  Adaptation ratio: {overall_ratio:.2f}%")
            print(f"{'='*80}\n")