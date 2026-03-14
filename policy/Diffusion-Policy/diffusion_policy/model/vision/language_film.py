"""
Language FiLM (Feature-wise Linear Modulation) conditioned on language.
Uses pre-computed CLIP text embeddings to modulate observation features:
    output = γ * obs_feature + β
where γ, β are generated from text embeddings via an MLP.

NOTE: This module is optional and not integrated into the main pipeline.
      It is useful for multi-task settings where language instructions
      differentiate tasks. For single-task training, it has no benefit.
"""
import torch
import torch.nn as nn


class LanguageFiLM(nn.Module):
    def __init__(self, text_emb_dim=512, obs_feature_dim=648, hidden_dim=256):
        """
        Args:
            text_emb_dim: CLIP text embedding dimension (512 for ViT-B/32)
            obs_feature_dim: observation feature dimension (AMAM + agent_pos)
            hidden_dim: hidden layer dimension in FiLM generator
        """
        super().__init__()
        self.text_emb_dim = text_emb_dim
        self.obs_feature_dim = obs_feature_dim

        self.film_generator = nn.Sequential(
            nn.Linear(text_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_feature_dim * 2),
        )

        # Near-identity initialization: γ≈1, β≈0
        nn.init.normal_(self.film_generator[-1].weight, std=1e-4)
        nn.init.zeros_(self.film_generator[-1].bias)

    def forward(self, obs_feature, text_emb):
        """
        Args:
            obs_feature: (B, obs_feature_dim) fused observation features
            text_emb: (B, text_emb_dim) pre-computed CLIP text embeddings
        Returns:
            (B, obs_feature_dim) language-conditioned features
        """
        params = self.film_generator(text_emb)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma + 1.0  # residual: starts as identity
        return gamma * obs_feature + beta

    @staticmethod
    def encode_texts(texts, clip_model_name='ViT-B/32', device='cpu'):
        """
        Utility: compute CLIP text embeddings for a list of strings.

        Args:
            texts: list of strings, e.g. ["pick up the red block"]
            clip_model_name: CLIP model name (default: ViT-B/32 → 512-dim)
            device: computation device
        Returns:
            (N, 512) float tensor of text embeddings
        """
        import clip
        model, _ = clip.load(clip_model_name, device=device)
        tokens = clip.tokenize(texts).to(device)
        with torch.no_grad():
            features = model.encode_text(tokens)
        return features.float().cpu()
