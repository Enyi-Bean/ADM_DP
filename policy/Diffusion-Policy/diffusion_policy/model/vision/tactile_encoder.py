import torch
import torch.nn as nn
import torch.nn.functional as F


class TactileEncoder(nn.Module):
    """
    Tactile encoder for FSR gripper sensors.
    Each gripper has 2 fingers, each with a 4x4 FSR sensor grid (32 values total).

    Processing pipeline:
        1. Reshape (32,) -> (2, 4, 4) two fingers
        2. Log-normalization for numerical stability
        3. Positional encoding: concatenate (x, y) grid coordinates to each taxel
        4. Per-finger 1D convolution + adaptive pooling
        5. Physical features: resultant force, differential force, contact points
        6. FFN fusion -> output feature vector

    Args:
        input_dim (int): Raw tactile input dimension (default 32 = 2 fingers x 4 x 4)
        output_dim (int): Output feature dimension (default 64)
        conv_channels (list): 1D conv channel sizes per layer
        use_layernorm (bool): Whether to use LayerNorm in FFN
    """
    def __init__(self,
                 input_dim=32,
                 output_dim=64,
                 conv_channels=[16, 32],
                 use_layernorm=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Positional encoding grid for 4x4 sensor
        # (x, y) ∈ [-1, 1]^2, shape: (16, 2)
        grid_x = torch.linspace(-1, 1, 4).repeat(4)          # x coord for each taxel
        grid_y = torch.linspace(-1, 1, 4).repeat_interleave(4)  # y coord for each taxel
        pos_encoding = torch.stack([grid_x, grid_y], dim=-1)  # (16, 2)
        self.register_buffer('pos_encoding', pos_encoding)

        # Per-finger 1D convolution
        # Input: (B, 3, 16) — 3 channels = taxel_value + x + y, 16 taxels per finger
        self.finger_conv = nn.Sequential(
            nn.Conv1d(3, conv_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B, conv_channels[-1], 1)
        )

        # Physical features dimension:
        # resultant_force(1) + differential_force(1) + contact_point_finger0(2) + contact_point_finger1(2) = 6
        physical_dim = 6

        # Conv features: 2 fingers x conv_channels[-1]
        conv_feature_dim = 2 * conv_channels[-1]

        # FFN fusion
        fusion_input_dim = conv_feature_dim + physical_dim
        if use_layernorm:
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
            )

    def _compute_physical_features(self, fingers):
        """
        Compute hand-crafted physical features from tactile readings.

        Args:
            fingers: (B, 2, 4, 4) — two fingers, each 4x4 grid

        Returns:
            (B, 6) — [resultant_force, differential_force, contact_x0, contact_y0, contact_x1, contact_y1]
        """
        # Per-finger force: sum of all taxels
        finger0_force = fingers[:, 0].reshape(-1, 16).sum(dim=-1)  # (B,)
        finger1_force = fingers[:, 1].reshape(-1, 16).sum(dim=-1)  # (B,)

        # Resultant force: total force across both fingers
        resultant_force = finger0_force + finger1_force  # (B,)

        # Differential force: balance between two fingers
        differential_force = finger0_force - finger1_force  # (B,)

        # Contact point locations: force-weighted centroid per finger
        # pos_encoding: (16, 2) — (x, y) for each taxel
        pos = self.pos_encoding  # (16, 2)

        contact_points = []
        for f_idx in range(2):
            weights = fingers[:, f_idx].reshape(-1, 16)  # (B, 16)
            weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, 1) avoid div by 0
            # Weighted centroid: (B, 16) x (16, 2) -> (B, 2)
            centroid = torch.matmul(weights / weight_sum, pos)  # (B, 2)
            contact_points.append(centroid)

        # Stack: (B, 6)
        physical = torch.stack([
            resultant_force,
            differential_force,
            contact_points[0][:, 0], contact_points[0][:, 1],  # finger0 contact (x, y)
            contact_points[1][:, 0], contact_points[1][:, 1],  # finger1 contact (x, y)
        ], dim=-1)

        return physical

    def forward(self, x):
        """
        Args:
            x: (B, 32) — raw tactile readings from one gripper

        Returns:
            (B, output_dim) — tactile feature vector
        """
        B = x.shape[0]

        # 1. Reshape to (B, 2, 4, 4) — two fingers
        fingers = x.reshape(B, 2, 4, 4)

        # 2. Log-normalization for stability (FSR values can have large range)
        fingers_log = torch.log1p(fingers.clamp(min=0))

        # 3. Compute physical features (from log-normalized values)
        physical_features = self._compute_physical_features(fingers_log)  # (B, 6)

        # 4. Per-finger conv features
        finger_features = []
        for f_idx in range(2):
            finger_vals = fingers_log[:, f_idx].reshape(B, 16)  # (B, 16)
            # Concatenate positional encoding: (B, 16, 1) + (1, 16, 2) -> (B, 16, 3)
            pos = self.pos_encoding.unsqueeze(0).expand(B, -1, -1)  # (B, 16, 2)
            finger_input = torch.cat([finger_vals.unsqueeze(-1), pos], dim=-1)  # (B, 16, 3)
            # Conv1d expects (B, C, L) -> transpose
            finger_input = finger_input.permute(0, 2, 1)  # (B, 3, 16)
            conv_out = self.finger_conv(finger_input).squeeze(-1)  # (B, conv_channels[-1])
            finger_features.append(conv_out)

        # Concatenate both fingers' conv features: (B, 2 * conv_channels[-1])
        conv_features = torch.cat(finger_features, dim=-1)

        # 5. Fusion: conv features + physical features -> output
        fused = torch.cat([conv_features, physical_features], dim=-1)  # (B, conv_dim + 6)
        output = self.fusion(fused)  # (B, output_dim)

        return output
