import torch
from torch import nn
import torch.nn.functional as F


class AssociativeMemory(nn.Module):
    """
    TTT-Linear inspired associative memory for adaptive parameter retrieval.
    
    Key idea: Weight matrix W acts as memory. Train W such that K @ W ≈ V.
    At test-time, Q @ W retrieves parameters based on similarity.
    """
    
    def __init__(self, feat_dim=128):
        super().__init__()
        self.feat_dim = feat_dim
        
        # QK projection (Shared!) - 16x16x3 = 768 → feat_dim
        # Shared QK aligns the "write" and "read" spaces
        self.qk_proj = nn.Linear(768, feat_dim)
        self.v_proj = nn.Linear(768, feat_dim)  # Same dim as K!
        
        # Memory weight matrix (THE memory itself!)
        self.W = nn.Linear(feat_dim, feat_dim, bias=False)
        nn.init.eye_(self.W.weight)  # Initialize to identity
        
        # Output projection (feat_dim → 2 parameters)
        self.out_proj = nn.Linear(feat_dim, 2)  # [log_gamma, log_temp]
        
        # Initialize output projection to reasonable values
        # Gamma: target 1.0 -> 0.5 + 1.5*sigmoid(b) = 1.0 -> sigmoid(b) = 1/3 -> b ≈ -0.69
        # Temp: target 0.01 -> exp(b) = 0.01 -> b ≈ -4.6
        nn.init.normal_(self.out_proj.weight, 0, 0.01)
        with torch.no_grad():
            self.out_proj.bias[0].fill_(-0.69)
            self.out_proj.bias[1].fill_(-4.6)
    
    def forward(self, img):
        """
        Args:
            img: (C, H, W) or (B, C, H, W) image tensor in [0, 255]
        Returns:
            params: (2,) or (B, 2) [gating_logit, log_temp]
            loss_mem: memory alignment loss
        """
        # Handle batch dimension
        if img.dim() == 3:
            img = img.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Downsample to 16x16 for efficiency
        img_tiny = F.interpolate(img, size=16, mode='bilinear', align_corners=False)
        feat = img_tiny.flatten(1)  # (B, 768)
        
        # Projections
        QK = self.qk_proj(feat)  # (B, feat_dim)
        V = self.v_proj(feat)  # (B, feat_dim)
        
        # Memory alignment loss: K @ W should equal V
        # This trains W to map K to V. V is NOT detached now!
        QK_transformed = self.W(QK)
        loss_mem = F.mse_loss(QK_transformed, V)
        
        # Orthogonality constraint (QK ⊥ V)
        cos_sim = F.cosine_similarity(QK.flatten(), V.flatten(), dim=0)
        loss_orth = -(cos_sim.abs())
        
        # Combined loss
        loss_total = loss_mem + 0.3 * loss_orth
        
        # Retrieval: Q @ W gets parameters
        retrieved = self.W(QK)  # (B, feat_dim)
        
        # Project to output parameters
        params = self.out_proj(retrieved)  # (B, 2)
        
        if squeeze_output:
            params = params.squeeze(0)  # (2,)
        
        return params, loss_mem, loss_orth
