import torch
import torch.nn as nn
from typing import Optional, Tuple


class StateTransitionGate(nn.Module):
    """State transition gate with size velocity"""
    def __init__(self, dim_x: int = 8, dt: float = 1.0):
        super().__init__()
        self.dim_x = dim_x

        # State: [cx, cy, w, h, vx, vy, vw, vh]
        F = torch.eye(dim_x)
        F[0, 4] = dt  # cx += vx * dt
        F[1, 5] = dt  # cy += vy * dt
        F[2, 6] = dt  # w += vw * dt
        F[3, 7] = dt  # h += vh * dt
        self.register_buffer('F', F)

    def forward(self) -> torch.Tensor:
        return self.F


class ObservationGate(nn.Module):
    """Observation gate for 8-dim state"""
    def __init__(self, dim_x: int = 8, dim_z: int = 4):
        super().__init__()

        # Observe cx, cy, w, h (not velocities)
        H = torch.zeros(dim_z, dim_x)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        self.register_buffer('H', H)

    def forward(self) -> torch.Tensor:
        return self.H


class DetectionCovarianceGate(nn.Module):
    """Detection covariance gate (R matrix) for cxcywh format"""
    def __init__(
        self,
        std_weight_position: float = 1./20,
        std_weight_size: float = 1./20,
        learnable: bool = False
    ):
        super().__init__()

        if learnable:
            self.std_weight_position = nn.Parameter(torch.tensor(std_weight_position))
            self.std_weight_size = nn.Parameter(torch.tensor(std_weight_size))
        else:
            self.register_buffer('std_weight_position', torch.tensor(std_weight_position))
            self.register_buffer('std_weight_size', torch.tensor(std_weight_size))

    def forward(
            self,
            bbox: torch.Tensor,  # [batch, 4] or [4] - [cx, cy, w, h]
            confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            bbox: [batch, 4] or [4] tensor of [cx, cy, w, h]
            confidence: [batch] or scalar tensor of confidence scores

        Returns:
            R: [batch, 4, 4] or [4, 4] detection covariance matrix
        """
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)

        batch_size = bbox.shape[0]
        h = torch.clamp(bbox[:, 3], min=1.0)  # height

        # Bbox size proportional standard deviation
        std = torch.stack([
            self.std_weight_position * h,  # cx
            self.std_weight_position * h,  # cy
            self.std_weight_size * h,      # w
            self.std_weight_size * h       # h
        ], dim=1)  # [batch, 4]

        # Optional confidence weighting
        if confidence is not None:
            if confidence.dim() == 0:
                confidence = confidence.unsqueeze(0)
            conf_weight = 1.0 / (confidence.clamp(min=1e-6))
            std = std * conf_weight.unsqueeze(1)
        std = torch.clamp(std, min=0.1)

        # Create diagonal covariance matrices
        R = torch.diag_embed(std ** 2)  # [batch, 4, 4]

        return R.squeeze(0) if batch_size == 1 else R


class SystemCovarianceGate(nn.Module):
    """System covariance for 8-dim state"""
    def __init__(self, dim_x: int = 8, learnable: bool = False):
        super().__init__()

        Q = torch.eye(dim_x) * 0.01
        Q[4:, 4:] *= 0.01  # velocity uncertainties smaller

        if learnable:
            self.log_q_diag = nn.Parameter(torch.log(Q.diagonal()))
        else:
            self.register_buffer('Q', Q)

        self.learnable = learnable

    def forward(self) -> torch.Tensor:
        if self.learnable:
            return torch.diag(torch.exp(self.log_q_diag))
        else:
            return self.Q


class AdaptiveKalmanFilterCXCYWH(nn.Module):
    """
    Kalman Filter with size velocity
    State: [cx, cy, w, h, vx, vy, vw, vh]
    """
    def __init__(
        self,
        dim_x: int = 8,  # [cx, cy, w, h, vx, vy, vw, vh]
        dim_z: int = 4,  # [cx, cy, w, h]
        learnable_uncertainty: bool = False,
        min_variance: float = 1e-6
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.min_variance = min_variance

        # Gates
        self.transition_gate = StateTransitionGate(dim_x)
        self.observation_gate = ObservationGate(dim_x, dim_z)
        self.system_cov_gate = SystemCovarianceGate(dim_x, learnable=learnable_uncertainty)
        self.detection_cov_gate = DetectionCovarianceGate(learnable=learnable_uncertainty)

        # Identity matrix
        self.register_buffer('I', torch.eye(dim_x))

    def predict(
        self,
        mean: torch.Tensor,
        covariance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mean: [8, 1] state [cx, cy, w, h, vx, vy, vw, vh]
            covariance: [8, 8] state covariance
        """
        F = self.transition_gate().to(mean.device)
        Q = self.system_cov_gate().to(mean.device)

        mean_pred = F @ mean
        cov_pred = F @ covariance @ F.T + Q
        return mean_pred, cov_pred

    def update(
        self,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        measurement: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mean: [8, 1] state mean
            covariance: [8, 8] state covariance
            measurement: [4, 1] detection [cx, cy, w, h]
            confidence: scalar confidence
        """
        H = self.observation_gate().to(mean.device)

        if measurement.dim() == 3:
            bbox = measurement.squeeze(-1)
        else:
            bbox = measurement.squeeze(-1)

        R = self.detection_cov_gate(bbox, confidence).to(mean.device)
        R = R + torch.eye(R.shape[-1], device=R.device) * self.min_variance  # stabilize

        # Innovation
        innovation = measurement - H @ mean

        # Innovation covariance
        S = H @ covariance @ H.T + R
        S = S + torch.eye(S.shape[-1], device=S.device) * self.min_variance

        # Kalman gain / solve
        # K = covariance @ H.T @ inv(S) by solving `S @ K.T = (covariance @ H.T).T`
        cov_HT = covariance @ H.T  # [8, 4]
        try:
            # Cholesky: S = L @ L.T
            L = torch.linalg.cholesky(S)  # Lower triangular matrix

            # Solve twice (forward + backward substitution)
            # 1. L @ y = cov_HT.T
            y = torch.linalg.solve_triangular(L, cov_HT.T, upper=False)
            # 2. L.T @ K.T = y
            K = torch.linalg.solve_triangular(L.T, y, upper=True).T
        except RuntimeError:  # Fall-back
            # `solve(S, X)` solves `S @ X = B`
            # S.T @ K.T = cov_HT.T
            K = torch.linalg.solve(S.T, cov_HT.T).T  # [8, 4]

        # Update
        mean_new = mean + K @ innovation
        # Joseph form (Stable covariance update)
        # cov_new = (I - K@H) @ cov @ (I - K@H).T + K @ R @ K.T
        IKH = self.I - K @ H
        cov_new = IKH @ covariance @ IKH.T + K @ R @ K.T

        # Ensure covariance is symmetric
        cov_new = (cov_new + cov_new.T) / 2

        return mean_new, cov_new

    def forward(
        self,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        measurement: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_pred, cov_pred = self.predict(mean, covariance)
        mean_new, cov_new = self.update(mean_pred, cov_pred, measurement, confidence)
        return mean_new, cov_new


def initialize_state_cxcywh(bbox: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize state with size velocities

    Args:
        bbox: [4] tensor of [cx, cy, w, h]

    Returns:
        mean: [8, 1] state [cx, cy, w, h, vx, vy, vw, vh]
        covariance: [8, 8] covariance
    """
    mean = torch.zeros(8, 1, device=bbox.device)
    mean[:4, 0] = bbox  # [cx, cy, w, h]
    mean[4:, 0] = 0.0   # all velocities = 0

    covariance = torch.eye(8, device=bbox.device)
    covariance[:4, :4] *= 100.   # position/size uncertainty
    covariance[4:, 4:] *= 1000.  # velocity uncertainty

    return mean, covariance


# Custom Loss with Kalman Filter
class KalmanFilteredLoss(nn.Module):
    """
    Custom loss that applies Kalman filter before Hungarian matching
    """
    def __init__(self, learnable_uncertainty: bool = True):
        super().__init__()
        self.kalman_filter = AdaptiveKalmanFilterCXCYWH(learnable_uncertainty=learnable_uncertainty)

    def forward(
        self,
        pred_boxes: torch.Tensor,   # [batch, num_queries, 4] normalized [cx, cy, w, h]
        pred_logits: torch.Tensor,  # [batch, num_queries, num_classes]
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        track_states: Optional[dict] = None,
        image_sizes: Optional[torch.Tensor] = None  # [batch, 2] for denormalization
    ):
        """
        Args:
            pred_boxes: [batch, num_queries, 4] normalized [cx, cy, w, h] in [0, 1]
            pred_logits: [batch, num_queries, num_classes]
            gt_boxes: ground truth boxes
            gt_labels: ground truth labels
            track_states: dict of {query_idx: (mean, covariance)}
            image_sizes: [batch, 2] tensor of [height, width] for denormalization
        """
        batch_size, num_queries = pred_boxes.shape[:2]
        device = pred_boxes.device

        if track_states is None:
            track_states = {}

        if image_sizes is None:
            # Default to 640x640 if not provided
            image_sizes = torch.tensor([[640, 640]] * batch_size, device=device)

        # Apply Kalman filter to each prediction
        filtered_boxes = []
        new_track_states = {}

        for b in range(batch_size):
            batch_filtered = []
            h, w = image_sizes[b]

            for q in range(num_queries):
                # Get prediction (normalized)
                pred_box_norm = pred_boxes[b, q]  # [4] [cx, cy, w, h] in [0, 1]

                # Denormalize
                cx = pred_box_norm[0] * w
                cy = pred_box_norm[1] * h
                box_w = pred_box_norm[2] * w
                box_h = pred_box_norm[3] * h
                pred_box_abs = torch.stack([cx, cy, box_w, box_h])

                # Get confidence
                pred_logit = pred_logits[b, q]  # [num_classes]
                confidence = pred_logit.softmax(-1).max()

                # Get or initialize track state
                track_key = f"{b}_{q}"
                if track_key not in track_states:
                    mean, cov = initialize_state_cxcywh(pred_box_abs)
                else:
                    mean, cov = track_states[track_key]

                # Prepare measurement
                measurement = pred_box_abs.unsqueeze(-1)  # [4, 1]

                # Apply Kalman filter
                mean_new, cov_new = self.kalman_filter(
                    mean, cov, measurement, confidence
                )

                # Extract filtered bbox
                bbox_filtered_abs = mean_new[:4, 0]  # [4] [cx, cy, w, h]

                # Normalize back
                cx_norm = bbox_filtered_abs[0] / w
                cy_norm = bbox_filtered_abs[1] / h
                w_norm = bbox_filtered_abs[2] / w
                h_norm = bbox_filtered_abs[3] / h
                bbox_filtered_norm = torch.stack([cx_norm, cy_norm, w_norm, h_norm])

                batch_filtered.append(bbox_filtered_norm)
                new_track_states[track_key] = (mean_new.detach(), cov_new.detach())

            filtered_boxes.append(torch.stack(batch_filtered))

        filtered_boxes = torch.stack(filtered_boxes)  # [batch, num_queries, 4]

        # Now compute loss with filtered boxes
        # You need to implement Hungarian matching + loss computation here
        # This is a placeholder
        loss = self.compute_detr_loss(filtered_boxes, pred_logits, gt_boxes, gt_labels)

        return loss, new_track_states

    def compute_detr_loss(self, pred_boxes, pred_logits, gt_boxes, gt_labels):
        """
        Placeholder for DETR-style loss computation
        You should use RT-DETR's actual loss function here
        """
        # Hungarian matching + L1 + GIoU loss
        # This is where you'd call the actual RT-DETR loss
        pass
