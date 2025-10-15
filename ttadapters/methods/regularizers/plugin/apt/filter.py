import torch
import torch.nn as nn
from typing import Optional, Tuple


class SystemCovarianceGate(nn.Module):
    """
    System covariance gate (Q matrix).
    Represents uncertainty in the motion model.
    """
    def __init__(self, dim_x: int = 7, learnable: bool = False):
        super().__init__()
        self.dim_x = dim_x

        # Initialize Q
        Q = torch.eye(dim_x) * 0.01
        Q[4:, 4:] *= 0.01  # velocity uncertainty smaller

        if learnable:
            # Learnable log-space parameters (ensure positive)
            self.log_q_diag = nn.Parameter(torch.log(Q.diagonal()))
        else:
            self.register_buffer('Q', Q)

        self.learnable = learnable

    def forward(self) -> torch.Tensor:
        """
        Returns:
            Q: [dim_x, dim_x] system covariance matrix
        """
        if self.learnable:
            return torch.diag(torch.exp(self.log_q_diag))
        else:
            return self.Q


class DetectionCovarianceGate(nn.Module):
    """
    Detection covariance gate (R matrix).
    Computes uncertainty based on detection bbox size and confidence.
    """
    def __init__(
            self,
            std_weight_position: float = 1./20,
            std_weight_scale: float = 10.,
            learnable: bool = False
    ):
        super().__init__()

        if learnable:
            self.std_weight_position = nn.Parameter(torch.tensor(std_weight_position))
            self.std_weight_scale = nn.Parameter(torch.tensor(std_weight_scale))
        else:
            self.register_buffer('std_weight_position', torch.tensor(std_weight_position))
            self.register_buffer('std_weight_scale', torch.tensor(std_weight_scale))

    def forward(
            self,
            bbox: torch.Tensor,
            confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            bbox: [batch, 4] or [4] tensor of [x1, y1, x2, y2]
            confidence: [batch] or scalar tensor of confidence scores (optional)

        Returns:
            R: [batch, 4, 4] or [4, 4] detection covariance matrix
        """
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)

        batch_size = bbox.shape[0]
        h = bbox[:, 3] - bbox[:, 1]  # height [batch]

        # Bbox size proportional standard deviation
        std = torch.stack([
            2 * self.std_weight_position * h,                          # x
            2 * self.std_weight_position * h,                          # y
            self.std_weight_scale * self.std_weight_position * h,      # scale
            self.std_weight_scale * self.std_weight_position * h       # ratio
        ], dim=1)  # [batch, 4]

        # Optional: confidence weighting
        if confidence is not None:
            if confidence.dim() == 0:
                confidence = confidence.unsqueeze(0)
            conf_weight = 1.0 / (confidence.clamp(min=1e-6))
            std = std * conf_weight.unsqueeze(1)

        # Create diagonal covariance matrices
        R = torch.diag_embed(std ** 2)  # [batch, 4, 4]

        return R.squeeze(0) if batch_size == 1 else R


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
        h = bbox[:, 3]  # height

        # Bbox size proportional standard deviation
        std = torch.stack([
            self.std_weight_position * h,  # cx
            self.std_weight_position * h,  # cy
            self.std_weight_size * h,      # w
            self.std_weight_size * h       # h
        ], dim=1)  # [batch, 4]

        # Optional: confidence weighting
        if confidence is not None:
            if confidence.dim() == 0:
                confidence = confidence.unsqueeze(0)
            conf_weight = 1.0 / (confidence.clamp(min=1e-6))
            std = std * conf_weight.unsqueeze(1)

        # Create diagonal covariance matrices
        R = torch.diag_embed(std ** 2)  # [batch, 4, 4]

        return R.squeeze(0) if batch_size == 1 else R


class StateTransitionGate(nn.Module):
    """
    State transition gate (F matrix).
    Defines how the system state evolves (constant velocity model).
    """
    def __init__(self, dim_x: int = 7, dt: float = 1.0):
        super().__init__()
        self.dim_x = dim_x

        # Constant velocity model
        F = torch.eye(dim_x)
        F[0, 4] = F[1, 5] = F[2, 6] = dt
        self.register_buffer('F', F)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            F: [dim_x, dim_x] state transition matrix
        """
        return self.F


class ObservationGate(nn.Module):
    """
    Observation gate (H matrix).
    Maps system state to detection space.
    """
    def __init__(self, dim_x: int = 7, dim_z: int = 4):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z

        # Observe position, scale, ratio (not velocity)
        H = torch.zeros(dim_z, dim_x)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        self.register_buffer('H', H)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            H: [dim_z, dim_x] observation matrix
        """
        return self.H


class AdaptiveKalmanFilter(nn.Module):
    """
    Adaptive Kalman Filter with modular gate components.
    Filters detector outputs using motion model and adaptive uncertainty.
    """
    def __init__(
            self,
            dim_x: int = 7,
            dim_z: int = 4,
            learnable_uncertainty: bool = False
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z

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
        Prediction step: propagate state forward using motion model

        Args:
            mean: [batch, 7, 1] or [7, 1] state mean
            covariance: [batch, 7, 7] or [7, 7] state covariance

        Returns:
            mean_pred: predicted state mean
            cov_pred: predicted state covariance
        """
        F = self.transition_gate()
        Q = self.system_cov_gate()

        mean_pred = F @ mean
        cov_pred = F @ covariance @ F.T + Q
        return mean_pred, cov_pred

    def update(
            self,
            mean: torch.Tensor,
            covariance: torch.Tensor,
            measurement: torch.Tensor,
            bbox: torch.Tensor,
            confidence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update step: correct prediction with detection

        Args:
            mean: [batch, 7, 1] or [7, 1] state mean
            covariance: [batch, 7, 7] or [7, 7] state covariance
            measurement: [batch, 4, 1] or [4, 1] detection [cx, cy, s, r]
            bbox: [batch, 4] or [4] bounding box [x1, y1, x2, y2]
            confidence: [batch] or scalar detection confidence (optional)

        Returns:
            mean_new: updated state mean
            cov_new: updated state covariance
        """
        H = self.observation_gate()
        R = self.detection_cov_gate(bbox, confidence)

        # Innovation
        innovation = measurement - H @ mean

        # Innovation covariance
        S = H @ covariance @ H.T + R

        # Kalman gain
        K = covariance @ H.T @ torch.linalg.inv(S)

        # Update
        mean_new = mean + K @ innovation
        cov_new = (self.I - K @ H) @ covariance

        return mean_new, cov_new

    def forward(
            self,
            mean: torch.Tensor,
            covariance: torch.Tensor,
            measurement: torch.Tensor,
            bbox: torch.Tensor,
            confidence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full Kalman filter step: predict + update

        Args:
            mean: state mean
            covariance: state covariance
            measurement: detection measurement
            bbox: bounding box for adaptive uncertainty
            confidence: detection confidence (optional)

        Returns:
            mean_new: updated state mean
            cov_new: updated state covariance
        """
        # Predict using motion model
        mean_pred, cov_pred = self.predict(mean, covariance)

        # Update with detection
        mean_new, cov_new = self.update(mean_pred, cov_pred, measurement, bbox, confidence)

        return mean_new, cov_new


def convert_bbox_to_z(bbox: torch.Tensor) -> torch.Tensor:
    """
    Convert bbox [x1, y1, x2, y2] to measurement [cx, cy, s, r]

    Args:
        bbox: [..., 4] tensor of bounding boxes

    Returns:
        z: [..., 4, 1] measurement tensor
    """
    w = bbox[..., 2] - bbox[..., 0]
    h = bbox[..., 3] - bbox[..., 1]
    cx = bbox[..., 0] + w / 2.
    cy = bbox[..., 1] + h / 2.
    s = w * h  # scale (area)
    r = w / h  # aspect ratio

    z = torch.stack([cx, cy, s, r], dim=-1)
    return z.unsqueeze(-1)  # [..., 4, 1]


def convert_z_to_bbox(z: torch.Tensor) -> torch.Tensor:
    """
    Convert measurement [cx, cy, s, r] to bbox [x1, y1, x2, y2]

    Args:
        z: [..., 4, 1] or [..., 4] measurement tensor

    Returns:
        bbox: [..., 4] bounding box tensor
    """
    if z.shape[-1] == 1:
        z = z.squeeze(-1)

    cx, cy, s, r = z[..., 0], z[..., 1], z[..., 2], z[..., 3]
    w = torch.sqrt(s * r)
    h = s / w

    x1 = cx - w / 2.
    y1 = cy - h / 2.
    x2 = cx + w / 2.
    y2 = cy + h / 2.

    return torch.stack([x1, y1, x2, y2], dim=-1)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Adaptive Kalman Filter for Object Detection")
    print("=" * 60)

    # Create filter
    kf = AdaptiveKalmanFilter(learnable_uncertainty=False).to(device)

    # Initialize state
    bbox_init = torch.tensor([100., 100., 200., 300.], device=device)
    z_init = convert_bbox_to_z(bbox_init)

    mean = torch.zeros(7, 1, device=device)
    mean[:4] = z_init
    covariance = torch.eye(7, device=device) * 10.
    covariance[4:, 4:] *= 1000.  # high uncertainty for velocity

    # Detection from detector (with gradient)
    bbox_det = torch.tensor([105., 105., 205., 305.], device=device, requires_grad=True)
    conf = torch.tensor(0.9, device=device)
    z_det = convert_bbox_to_z(bbox_det)

    # Filter detection
    mean_new, cov_new = kf(mean, covariance, z_det, bbox_det, conf)
    bbox_filtered = convert_z_to_bbox(mean_new[:4])

    print(f"\nDetection bbox: {bbox_det.detach()}")
    print(f"Filtered bbox:  {bbox_filtered.detach()}")
    print(f"Gradient flow:  {bbox_filtered.requires_grad}")

    # Test backward
    loss = bbox_filtered.sum()
    loss.backward()
    print(f"Gradient computed: {bbox_det.grad is not None}")

    # Show gate outputs
    print("\n" + "=" * 60)
    print("Gate Components")
    print("=" * 60)
    print(f"System uncertainty (Q):    {kf.system_cov_gate().shape}")
    print(f"Detection uncertainty (R): {kf.detection_cov_gate(bbox_det, conf).shape}")
    print(f"State transition (F):      {kf.transition_gate().shape}")
    print(f"Observation model (H):     {kf.observation_gate().shape}")
