"""
Dummy test for GraphTCPEncoder dimension check.
Tests the encoder standalone and simulates the full obs_encoder concat flow.

Usage:
    conda activate ADM-DP
    cd /NEW_EDS/JJ_Group/wangey2512/code/Mres_Project/RoboFactory
    python test_graph_encoder.py
"""
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'policy', 'Diffusion-Policy'))

from diffusion_policy.model.vision.graph_encoder import GraphTCPEncoder

# ========== Config (matches robot_dp.yaml + default_task.yaml) ==========
batch_size = 64
n_obs_steps = 3
num_agents = 2
tcp_dim = 3          # [x, y, z]
hidden_dim = 64
output_dim = 64
num_layers = 2

B = batch_size * n_obs_steps  # 192, obs_encoder sees flattened B*T

# ========== Test 1: GraphTCPEncoder standalone ==========
print("=" * 50)
print("Test 1: GraphTCPEncoder standalone")
print("=" * 50)

model = GraphTCPEncoder(
    input_dim=tcp_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_layers=num_layers,
    num_agents=num_agents,
)
print(f"Model created: {model.__class__.__name__}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

x = torch.randn(B, num_agents, tcp_dim)  # (192, 2, 3)
print(f"  Input shape:  {tuple(x.shape)}")

out = model(x)
print(f"  Output shape: {tuple(out.shape)}")

assert out.shape == (B, output_dim), \
    f"FAIL: expected ({B}, {output_dim}), got {tuple(out.shape)}"
print("  PASS\n")

# ========== Test 2: Simulate obs_encoder concat ==========
print("=" * 50)
print("Test 2: Simulate obs_encoder feature concat")
print("=" * 50)

# ResNet18 output (after pool+fc): 512-dim per image
rgb_feature = torch.randn(B, 512)
# low_dim (agent_pos): shape [8]
low_dim_feature = torch.randn(B, 8)
# GNN output
gnn_feature = out  # (192, 64)

concat = torch.cat([rgb_feature, low_dim_feature, gnn_feature], dim=-1)
print(f"  RGB feature:     {tuple(rgb_feature.shape)}")
print(f"  Low-dim feature: {tuple(low_dim_feature.shape)}")
print(f"  GNN feature:     {tuple(gnn_feature.shape)}")
print(f"  Concat shape:    {tuple(concat.shape)}")

expected_dim = 512 + 8 + output_dim  # 584
assert concat.shape == (B, expected_dim), \
    f"FAIL: expected ({B}, {expected_dim}), got {tuple(concat.shape)}"
print(f"  global_cond dim = {expected_dim}  PASS\n")

# ========== Test 3: Gradient flow ==========
print("=" * 50)
print("Test 3: Gradient flow")
print("=" * 50)

x2 = torch.randn(B, num_agents, tcp_dim, requires_grad=True)
out2 = model(x2)
loss = out2.sum()
loss.backward()
print(f"  Input grad shape: {tuple(x2.grad.shape)}")
assert x2.grad is not None and x2.grad.abs().sum() > 0, "FAIL: no gradient"
print("  PASS\n")

# ========== Test 4: Small batch (edge case) ==========
print("=" * 50)
print("Test 4: Small batch (B=1)")
print("=" * 50)

x3 = torch.randn(1, num_agents, tcp_dim)
out3 = model(x3)
print(f"  Input: {tuple(x3.shape)} -> Output: {tuple(out3.shape)}")
assert out3.shape == (1, output_dim)
print("  PASS\n")

print("All tests passed!")
