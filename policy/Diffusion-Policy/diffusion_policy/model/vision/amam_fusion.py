import torch
import torch.nn as nn
import torch.nn.functional as F


class AMAMFusion(nn.Module):
    """
    AMAM (Adaptive Modality-Aware Module) fusion module.
    Dynamically computes importance weights for vision, tactile, and graph
    modalities through a gated attention mechanism.

    During approach phase: tactile ≈ 0, vision should dominate.
    During grasping phase: tactile becomes critical.
    Graph awareness: important when agents are in close proximity.

    Uses weighted concatenation (not weighted sum) to preserve feature dimensions.

    Args:
        vision_dim (int): Vision feature dimension (e.g., 512 from ResNet)
        tactile_dim (int): Tactile feature dimension (e.g., 64)
        graph_dim (int): Graph feature dimension (e.g., 64)
        temperature (float): Softmax temperature, lower = sharper attention
        lambda_reg (float): Entropy regularization weight
    """
    def __init__(self,
                 vision_dim=512,
                 tactile_dim=64,
                 graph_dim=64,
                 temperature=1.0,
                 lambda_reg=0.01):
        super().__init__()
        self.vision_dim = vision_dim
        self.tactile_dim = tactile_dim
        self.graph_dim = graph_dim
        self.temperature = temperature
        self.lambda_reg = lambda_reg

        # Attention MLP: takes concatenated features, outputs 3 importance scores
        concat_dim = vision_dim + tactile_dim + graph_dim
        self.attention_mlp = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 3 modality weights
        )

        # Store latest attention weights and reg loss for logging/training
        self.last_alpha = None
        self.last_reg_loss = None

    def forward(self, f_v, f_t, f_g):
        """
        Args:
            f_v: (B, vision_dim) - vision features
            f_t: (B, tactile_dim) - tactile features
            f_g: (B, graph_dim) - graph features

        Returns:
            (B, vision_dim + tactile_dim + graph_dim) - weighted concatenated features
        """
        # Compute attention weights
        concat = torch.cat([f_v, f_t, f_g], dim=-1)  # (B, 640)
        scores = self.attention_mlp(concat)  # (B, 3)
        alpha = F.softmax(scores / self.temperature, dim=-1)  # (B, 3)

        # Store for logging
        self.last_alpha = alpha.detach()

        # Weighted concatenation: each modality scaled by its attention weight
        f_v_weighted = alpha[:, 0:1] * f_v  # (B, vision_dim)
        f_t_weighted = alpha[:, 1:2] * f_t  # (B, tactile_dim)
        f_g_weighted = alpha[:, 2:3] * f_g  # (B, graph_dim)

        fused = torch.cat([f_v_weighted, f_t_weighted, f_g_weighted], dim=-1)

        # Compute entropy regularization loss
        # L_reg = -λ Σ α_m log(α_m) = λ * H(α)
        # Minimizing this encourages sparse (decisive) weight distribution
        entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=-1).mean()
        self.last_reg_loss = self.lambda_reg * entropy

        return fused

    def get_reg_loss(self):
        """Return the entropy regularization loss for adding to training loss."""
        if self.last_reg_loss is not None:
            return self.last_reg_loss
        return torch.tensor(0.0)
