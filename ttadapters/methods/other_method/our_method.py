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

    # Tracking parameters
    iou_threshold: float = 0.3  # IoU threshold for matching
    min_confidence: float = 0.5  # Minimum detection confidence
    max_age: int = 30  # Maximum frames to keep track without update

    # Loss weights
    bbox_loss_weight: float = 1.0
    smooth_l1_beta: float = 1.0

    # Innovation weighting parameters (inverse weighting)
    use_covariance_weighting: bool = False  # Use covariance-based confidence (DISABLED - not helpful)
    use_innovation_weighting: bool = True  # Use innovation-based weighting (match quality)
    max_innovation: float = 100.0  # Innovation threshold for capping (beyond this = tracking failure)
    min_innovation_weight: float = 0.2  # Minimum weight when innovation is 0 (already aligned)

    # Detection confidence gating (exponential penalty for low-confidence detections)
    confidence_penalty_exponent: float = 2.0  # Apply detection_confidence ** exponent (2.0 or 3.0)


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
                    if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, FrozenBatchNorm2d)):
                        if hasattr(m, 'weight') and m.weight is not None:
                            m.weight.requires_grad = True
                            params.append(m.weight)
                        if hasattr(m, 'bias') and m.bias is not None:
                            m.bias.requires_grad = True
                            params.append(m.bias)

                    # Conv2d and Linear layers
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        if hasattr(m, 'weight') and m.weight is not None:
                            m.weight.requires_grad = True
                            params.append(m.weight)
                        if hasattr(m, 'bias') and m.bias is not None:
                            m.bias.requires_grad = True
                            params.append(m.bias)

        print(f"Ours: Adapting {len(params)} backbone parameters")

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
        else:
            warnings.warn("Unknown optimizer_option.")

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
        matches : List[Tuple[int, int]]
            List of (track_idx, detection_idx) pairs
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
                matches.append((track_idx, best_det_idx))
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
        matches : List[Tuple[int, int]]
            Matched (track_idx, detection_idx) pairs
        unmatched_tracks : List[int]
            Unmatched track indices
        unmatched_detections : List[int]
            Unmatched detection indices
        """
        # Update matched tracks (detach for tracking - no gradient needed)
        for track_idx, det_idx in matches:
            det = detections[det_idx]
            self.tracks[track_idx].update(
                det['bbox'].detach(),  # Detach for tracking update
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
        """
        if len(matches) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        losses = []
        weights = []

        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]

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

            if self.cfg.use_innovation_weighting:
                # INVERSE weighting: Higher innovation = Higher weight (need more adaptation)
                # Rationale:
                #   - Small innovation (0~20) = already aligned → minimal learning
                #   - Medium innovation (20~80) = adaptation needed → learn!
                #   - Large innovation (80~100) = strong adaptation needed → learn aggressively!
                #   - Very large (>100) = tracking failure → ignore

                if innovation_norm > self.cfg.max_innovation:
                    # Beyond threshold = likely tracking failure or outlier
                    innovation_weight = 0.1  # Almost ignore
                else:
                    # Linear increase: 0 → min_weight, 100 → 1.0
                    # innovation=0   → weight=0.2 (already aligned)
                    # innovation=50  → weight=0.6 (moderate adaptation)
                    # innovation=100 → weight=1.0 (strong adaptation)
                    min_w = self.cfg.min_innovation_weight
                    innovation_weight = min_w + (1.0 - min_w) * (innovation_norm / self.cfg.max_innovation)

            # === Component 3: Detection Confidence with Exponential Penalty ===
            detection_confidence_raw = detections[det_idx]['score'].item()

            # Apply exponential penalty to heavily penalize low-confidence detections
            # This prevents learning from unreliable detections (especially in night/challenging domains)
            detection_confidence_gated = detection_confidence_raw ** self.cfg.confidence_penalty_exponent

            # === Combined Weight ===
            # covariance_confidence is always 1.0 (disabled), so effectively:
            # total_weight = innovation_weight * (detection_confidence ** exponent)
            total_weight = covariance_confidence * innovation_weight * detection_confidence_gated

            # === Debug: Print weight distribution ===
            if self.frame_count % 10 == 0 and track_idx == matches[0][0]:  # Print every 10 frames, first match only
                innov_status = "CAPPED" if innovation_norm > self.cfg.max_innovation else "OK"
                print(f"[Frame {self.frame_count}] Weight breakdown (INVERSE weighting + Confidence Gating):")
                print(f"  Track: hits={track.hits}, age={track.age}")
                print(f"  Innovation: norm={innovation_norm:.1f} ({innov_status}) → weight={innovation_weight:.4f}")
                print(f"    Logic: innovation ↑ = weight ↑ (higher mismatch = more adaptation needed)")
                print(f"  Detection conf: {detection_confidence_raw:.4f} → gated={detection_confidence_gated:.4f} (exp={self.cfg.confidence_penalty_exponent})")
                print(f"    Penalty: {detection_confidence_raw:.4f}^{self.cfg.confidence_penalty_exponent} = {detection_confidence_gated:.4f} ({(detection_confidence_gated/detection_confidence_raw - 1)*100:+.1f}%)")
                print(f"  → Total weight: {total_weight:.6f}")

            # Smooth L1 loss between bboxes
            # model_bbox has gradient -> loss will have gradient -> backprop works!
            loss = F.smooth_l1_loss(
                model_bbox,  # Source with gradient
                kalman_bbox,  # Target without gradient
                beta=self.cfg.smooth_l1_beta
            )

            # Store weighted loss
            losses.append(loss)
            weights.append(total_weight)

        if len(losses) > 0:
            # Convert weights to tensor
            weights_tensor = torch.tensor(weights, device=self.device, dtype=torch.float32)

            # Weighted average of losses
            losses_tensor = torch.stack(losses)
            weighted_loss = (losses_tensor * weights_tensor).sum() / (weights_tensor.sum() + 1e-8)

            return self.cfg.bbox_loss_weight * weighted_loss
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

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

        self.frame_count += 1

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
        # Get model predictions
        self.model.eval()
        self.optimizer.zero_grad()

        outputs = self.model(x)
        detections = self._get_detections_from_output(outputs)

        # Match Kalman predictions with model detections
        matches, unmatched_tracks, unmatched_detections = \
            self._match_tracks_to_detections(detections)

        # Compute tracking loss
        tracking_loss = self._compute_tracking_loss(matches, detections, outputs)

        # Update model if we have valid loss
        if tracking_loss > 0:
            tracking_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Update tracks
        self._update_tracks(detections, matches, unmatched_tracks, unmatched_detections)

        # Predict for next frame
        self._predict_next_frame()

        # Return predictions (run model again in eval mode for clean output)
        self.model.eval()
        with torch.no_grad():
            final_outputs = self.model(x)

        return final_outputs
