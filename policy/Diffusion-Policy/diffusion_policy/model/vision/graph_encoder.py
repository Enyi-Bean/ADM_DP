import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTCPEncoder(nn.Module):
    """
    GCN-based encoder for multi-agent TCP position data.
    Uses Graph Convolutional Network for stable and efficient training.

    Vectorized implementation: no per-batch or per-node loops.

    Args:
        input_dim (int): Input dimension for each node (TCP position), default 3 for [x, y, z]
        hidden_dim (int): Hidden dimension for node features
        output_dim (int): Output dimension of the final encoded features
        num_layers (int): Number of GCN layers for message passing
        num_agents (int): Number of agents/nodes in the graph
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=64,
                 num_layers=2, num_agents=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_agents = num_agents

        # Node feature encoding - maps raw TCP positions to hidden features
        # nn.Linear automatically handles batched input: (B, num_agents, input_dim) -> (B, num_agents, hidden_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Build normalized adjacency matrix for fully connected graph (with self-loops)
        # Register as buffer so it moves with model to GPU automatically
        adj = self._create_adjacency_matrix(num_agents)
        self.register_buffer('adj', adj)

        # GCN layers with residual connections
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(GCNLayer(hidden_dim))

        # Output layer - aggregates all node features into final representation
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * num_agents, output_dim),
            nn.ReLU()
        )

    def _create_adjacency_matrix(self, num_nodes):
        """
        Create normalized adjacency matrix with self-loops for GCN.
        A_hat = D^{-1/2} (A + I) D^{-1/2}
        For fully connected graph with self-loops, all nodes have same degree,
        so this simplifies to 1/num_nodes for all entries.
        """
        # A + I (fully connected with self-loops): all ones
        adj = torch.ones(num_nodes, num_nodes, dtype=torch.float32)
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        # For fully connected + self-loops, degree = num_nodes for every node
        deg_inv_sqrt = 1.0 / (num_nodes ** 0.5)
        adj = adj * (deg_inv_sqrt ** 2)  # = 1/num_nodes
        return adj  # [num_nodes, num_nodes]

    def forward(self, x):
        """
        Forward pass of the Graph TCP Encoder (fully vectorized).

        Args:
            x (torch.Tensor): Input tensor with shape [B*n_obs_steps, num_agents, 3]
                             where n_obs_steps=3 for historical observations

        Returns:
            torch.Tensor: Encoded graph features with shape [B*n_obs_steps, output_dim]
        """
        B = x.shape[0]

        # Encode node features: (B, num_agents, 3) -> (B, num_agents, hidden_dim)
        node_features = self.node_encoder(x)

        # Apply GCN layers (all batched via matrix multiply)
        for gnn_layer in self.gnn_layers:
            node_features = gnn_layer(node_features, self.adj)

        # Flatten node features: (B, num_agents, hidden_dim) -> (B, num_agents * hidden_dim)
        batch_features = node_features.reshape(B, -1)

        # Generate final output: (B, num_agents * hidden_dim) -> (B, output_dim)
        output = self.output_layer(batch_features)

        return output


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer (vectorized, with residual connection).

    Computes: h' = ReLU(A_hat @ h @ W + h)
    where A_hat is the normalized adjacency matrix.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x, adj):
        """
        Args:
            x: (B, num_nodes, in_dim) - batched node features
            adj: (num_nodes, num_nodes) - normalized adjacency matrix
        """
        # Aggregate neighbor features via matrix multiply: (num_nodes, num_nodes) @ (B, num_nodes, in_dim)
        # torch.matmul broadcasts: adj is (N, N), x is (B, N, D) -> out is (B, N, D)
        aggregated = torch.matmul(adj, x)

        # Linear transform
        out = self.linear(aggregated)

        # Residual connection + activation
        out = F.relu(out + x)

        return out


# ============================================================================
# GAT (Graph Attention Network) Implementation
# ----------------------------------------------------------------------------
# ============================================================================

# class GATLayer(nn.Module):
#     """
#     Graph Attention Network layer (vectorized, with residual connection).
#     Computes attention-weighted neighbor aggregation with multi-head attention.
#
#     Args:
#         in_dim (int): Input/output feature dimension
#         num_heads (int): Number of attention heads
#         dropout (float): Dropout rate on attention weights
#     """
#     def __init__(self, in_dim, num_heads=4, dropout=0.1):
#         super().__init__()
#         assert in_dim % num_heads == 0, "in_dim must be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim = in_dim // num_heads
#
#         # Linear projections for Q, K, V
#         self.W_q = nn.Linear(in_dim, in_dim)
#         self.W_k = nn.Linear(in_dim, in_dim)
#         self.W_v = nn.Linear(in_dim, in_dim)
#
#         # Attention score projection (LeakyReLU-based, following original GAT)
#         self.attn_fc = nn.Linear(2 * self.head_dim, 1)
#
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(in_dim)
#
#     def forward(self, x, adj=None):
#         """
#         Args:
#             x: (B, num_nodes, in_dim)
#             adj: (num_nodes, num_nodes) - adjacency matrix used as attention mask
#                  (optional, not strictly needed for fully connected graphs)
#         Returns:
#             (B, num_nodes, in_dim)
#         """
#         B, N, D = x.shape
#         H = self.num_heads
#         d = self.head_dim
#
#         # Project to Q, K, V: (B, N, D) -> (B, N, H, d) -> (B, H, N, d)
#         q = self.W_q(x).reshape(B, N, H, d).permute(0, 2, 1, 3)  # (B, H, N, d)
#         k = self.W_k(x).reshape(B, N, H, d).permute(0, 2, 1, 3)  # (B, H, N, d)
#         v = self.W_v(x).reshape(B, N, H, d).permute(0, 2, 1, 3)  # (B, H, N, d)
#
#         # Compute pairwise attention scores (GAT-style with LeakyReLU)
#         # For each pair (i, j): concat [q_i, k_j] -> linear -> LeakyReLU
#         # Expand for pairwise: q_i repeated for all j, k_j repeated for all i
#         q_expand = q.unsqueeze(3).expand(B, H, N, N, d)  # (B, H, N, N, d)
#         k_expand = k.unsqueeze(2).expand(B, H, N, N, d)  # (B, H, N, N, d)
#         pair_concat = torch.cat([q_expand, k_expand], dim=-1)  # (B, H, N, N, 2*d)
#         attn_scores = F.leaky_relu(self.attn_fc(pair_concat).squeeze(-1), negative_slope=0.2)  # (B, H, N, N)
#
#         # Mask out non-edges if adjacency matrix provided
#         if adj is not None:
#             mask = (adj == 0).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
#             attn_scores = attn_scores.masked_fill(mask, float('-inf'))
#
#         # Normalize attention weights
#         attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, N, N)
#         attn_weights = self.dropout(attn_weights)
#
#         # Weighted aggregation: (B, H, N, N) @ (B, H, N, d) -> (B, H, N, d)
#         out = torch.matmul(attn_weights, v)
#
#         # Merge heads: (B, H, N, d) -> (B, N, H*d) = (B, N, D)
#         out = out.permute(0, 2, 1, 3).reshape(B, N, D)
#
#         # Residual connection + layer norm
#         out = self.layer_norm(out + x)
#
#         return out


# class GraphTCPEncoderGAT(nn.Module):
#     """
#     GAT-based encoder for multi-agent TCP position data.
#     Uses multi-head attention for neighbor aggregation.
#
#     To use: replace GraphTCPEncoder with GraphTCPEncoderGAT in robot_dp.yaml:
#         graph_model:
#           _target_: diffusion_policy.model.vision.graph_encoder.GraphTCPEncoderGAT
#           input_dim: 3
#           hidden_dim: 64
#           output_dim: 64
#           num_layers: 2
#           num_agents: 3
#           num_heads: 4
#     """
#     def __init__(self, input_dim=3, hidden_dim=64, output_dim=64,
#                  num_layers=2, num_agents=2, num_heads=4):
#         super().__init__()
#         self.num_agents = num_agents
#
#         self.node_encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#
#         # Fully connected adjacency (with self-loops) as attention mask
#         adj = torch.ones(num_agents, num_agents, dtype=torch.float32)
#         self.register_buffer('adj', adj)
#
#         self.gat_layers = nn.ModuleList()
#         for _ in range(num_layers):
#             self.gat_layers.append(GATLayer(hidden_dim, num_heads=num_heads))
#
#         self.output_layer = nn.Sequential(
#             nn.Linear(hidden_dim * num_agents, output_dim),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         """
#         Args:
#             x: (B, num_agents, 3)
#         Returns:
#             (B, output_dim)
#         """
#         B = x.shape[0]
#         node_features = self.node_encoder(x)
#         for gat_layer in self.gat_layers:
#             node_features = gat_layer(node_features, self.adj)
#         batch_features = node_features.reshape(B, -1)
#         output = self.output_layer(batch_features)
#         return output
