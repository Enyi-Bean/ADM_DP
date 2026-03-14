import torch
import torch.nn as nn


class PointNetEncoderXYZ(nn.Module):
    """
    PointNet encoder for point cloud data (XYZ only, no color).
    Adapted from DP3: MLP + MaxPool + final projection.
    No T-Net, uses LayerNorm for stability.

    Args:
        in_channels (int): Input channels per point (3 for xyz)
        out_channels (int): Output feature dimension
        use_layernorm (bool): Whether to use LayerNorm in MLP
        final_norm (str): 'layernorm' or 'none' for final projection
    """
    def __init__(self,
                 in_channels=3,
                 out_channels=256,
                 use_layernorm=True,
                 final_norm='layernorm'):
        super().__init__()
        block_channel = [64, 128, 256]

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        else:
            self.final_projection = nn.Linear(block_channel[-1], out_channels)

    def forward(self, x):
        """
        Args:
            x: (B, N_points, 3)
        Returns:
            (B, out_channels)
        """
        x = self.mlp(x)             # (B, N_points, 256)
        x = torch.max(x, dim=1)[0]  # (B, 256) - max pool over points
        x = self.final_projection(x) # (B, out_channels)
        return x


class PointCloudFiLM(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) module.
    Uses point cloud features to modulate image features via learned scale and bias.

    Point cloud → PointNet → FiLM generator → scale & bias → modulate image features

    Args:
        pc_in_channels (int): Point cloud input channels (3 for xyz)
        pc_out_channels (int): PointNet output dimension (before FiLM projection)
        img_feature_dim (int): Image feature dimension to modulate (e.g., 512 for ResNet18)
        use_layernorm (bool): Whether to use LayerNorm in PointNet
    """
    def __init__(self,
                 pc_in_channels=3,
                 pc_out_channels=256,
                 img_feature_dim=512,
                 use_layernorm=True):
        super().__init__()

        self.pointnet = PointNetEncoderXYZ(
            in_channels=pc_in_channels,
            out_channels=pc_out_channels,
            use_layernorm=use_layernorm,
            final_norm='layernorm',
        )

        # FiLM generator: project point cloud features to scale and bias
        # Output 2 * img_feature_dim: first half is scale, second half is bias
        self.film_generator = nn.Linear(pc_out_channels, img_feature_dim * 2)

        # Initialize so modulation starts as near-identity (scale≈1, bias≈0)
        # Use small random weights (not exact zero) to allow gradient flow to PointNet
        nn.init.normal_(self.film_generator.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.film_generator.bias)
        # Set scale bias to 1 (so initial output ≈ 1 * img + 0 = img)
        self.film_generator.bias.data[:img_feature_dim] = 1.0

    def forward(self, point_cloud, img_features):
        """
        Args:
            point_cloud: (B, N_points, 3) - raw point cloud
            img_features: (B, img_feature_dim) - image features from ResNet

        Returns:
            (B, img_feature_dim) - modulated image features
        """
        # Encode point cloud
        pc_features = self.pointnet(point_cloud)  # (B, pc_out_channels)

        # Generate scale and bias
        film_params = self.film_generator(pc_features)  # (B, img_feature_dim * 2)
        scale, bias = film_params.chunk(2, dim=-1)      # each (B, img_feature_dim)

        # Modulate image features: scale * img + bias
        modulated = scale * img_features + bias

        return modulated
