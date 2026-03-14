"""
Test Tactile Encoder + full training forward pass.
Verifies that tactile features are correctly extracted and integrated
into the multi-modal observation encoder.

Usage:
    conda activate ADM-DP
    cd /NEW_EDS/JJ_Group/wangey2512/code/Mres_Project/RoboFactory
    python test_tactile_encoder.py
"""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'policy', 'Diffusion-Policy'))

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.graph_encoder import GraphTCPEncoder
from diffusion_policy.model.vision.pointcloud_encoder import PointCloudFiLM
from diffusion_policy.model.vision.tactile_encoder import TactileEncoder
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer

# ========== Config ==========
batch_size = 4
n_obs_steps = 3
horizon = 8
n_action_steps = 8
action_dim = 8
agent_pos_dim = 8
image_shape = (3, 240, 320)
tcp_dim = 3
num_agents = 3
n_points = 512
tactile_dim = 32  # 2 fingers x 4x4 FSR

shape_meta = {
    'obs': {
        'head_cam': {'shape': list(image_shape), 'type': 'rgb'},
        'agent_pos': {'shape': [agent_pos_dim], 'type': 'low_dim'},
        'tcp_agent0': {'shape': [tcp_dim], 'type': 'tcp'},
        'tcp_agent1': {'shape': [tcp_dim], 'type': 'tcp'},
        'tcp_agent2': {'shape': [tcp_dim], 'type': 'tcp'},
        'point_cloud': {'shape': [n_points, 3], 'type': 'pointcloud'},
        'tactile': {'shape': [tactile_dim], 'type': 'tactile'},
    },
    'action': {'shape': [action_dim]},
}

# ========== Test 1: TactileEncoder standalone ==========
print("=" * 60)
print("Test 1: TactileEncoder standalone")
print("=" * 60)

tactile_enc = TactileEncoder(input_dim=32, output_dim=64, conv_channels=[16, 32], use_layernorm=True)
tactile_input = torch.rand(4, 32)  # Simulated FSR readings (non-negative)
tactile_output = tactile_enc(tactile_input)
print(f"  Input: {tuple(tactile_input.shape)} -> Output: {tuple(tactile_output.shape)}")
assert tactile_output.shape == (4, 64), f"FAIL: expected (4, 64), got {tuple(tactile_output.shape)}"
print(f"  Parameters: {sum(p.numel() for p in tactile_enc.parameters()):,}")
print("  PASS\n")

# ========== Test 2: Physical features ==========
print("=" * 60)
print("Test 2: Physical features computation")
print("=" * 60)

# Create known tactile input: finger0 has force on top-left, finger1 uniform
tactile_known = torch.zeros(1, 32)
tactile_known[0, 0] = 10.0   # finger0, top-left taxel only
tactile_known[0, 16:] = 1.0  # finger1, uniform force

fingers = tactile_known.reshape(1, 2, 4, 4)
fingers_log = torch.log1p(fingers.clamp(min=0))

physical = tactile_enc._compute_physical_features(fingers_log)
print(f"  Physical features shape: {tuple(physical.shape)}")
print(f"  Resultant force: {physical[0, 0].item():.4f}")
print(f"  Differential force (f0-f1): {physical[0, 1].item():.4f}")
print(f"  Finger0 contact point: ({physical[0, 2].item():.4f}, {physical[0, 3].item():.4f})")
print(f"  Finger1 contact point: ({physical[0, 4].item():.4f}, {physical[0, 5].item():.4f})")

# Finger0 has force only at top-left -> contact point should be near (-1, -1)
assert physical[0, 2].item() < 0, "FAIL: finger0 contact x should be negative (top-left)"
assert physical[0, 3].item() < 0, "FAIL: finger0 contact y should be negative (top-left)"
# Finger1 has uniform force -> contact point should be near (0, 0)
assert abs(physical[0, 4].item()) < 0.1, "FAIL: finger1 contact x should be near 0 (uniform)"
assert abs(physical[0, 5].item()) < 0.1, "FAIL: finger1 contact y should be near 0 (uniform)"
print("  PASS\n")

# ========== Test 3: Full obs_encoder with tactile ==========
print("=" * 60)
print("Test 3: Full obs_encoder (ResNet18 + GNN + FiLM + Tactile)")
print("=" * 60)

rgb_model = get_resnet('resnet18', weights=None)
graph_model = GraphTCPEncoder(input_dim=3, hidden_dim=64, output_dim=64, num_layers=2, num_agents=num_agents)
film_model = PointCloudFiLM(pc_in_channels=3, pc_out_channels=256, img_feature_dim=512, use_layernorm=True)
tactile_model = TactileEncoder(input_dim=32, output_dim=64, conv_channels=[16, 32], use_layernorm=True)

obs_encoder = MultiImageObsEncoder(
    shape_meta=shape_meta,
    rgb_model=rgb_model,
    graph_model=graph_model,
    pointcloud_film_model=film_model,
    tactile_model=tactile_model,
    resize_shape=None, crop_shape=None, random_crop=True,
    use_group_norm=True, share_rgb_model=False, imagenet_norm=True,
)

obs_feature_dim = obs_encoder.output_shape()[0]
print(f"  obs_feature_dim = {obs_feature_dim}")
# 512 (ResNet, FiLM) + 8 (agent_pos) + 64 (GNN) + 64 (tactile) = 648
expected_dim = 512 + agent_pos_dim + 64 + 64
assert obs_feature_dim == expected_dim, \
    f"FAIL: expected {expected_dim}, got {obs_feature_dim}"
print(f"  = ResNet(512) + agent_pos({agent_pos_dim}) + GNN(64) + Tactile(64)")
print("  PASS\n")

# ========== Test 4: Full policy forward (compute_loss) ==========
print("=" * 60)
print("Test 4: Full training forward with Tactile Encoder")
print("=" * 60)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
    beta_schedule='squaredcos_cap_v2', variance_type='fixed_small',
    clip_sample=True, prediction_type='epsilon',
)
policy = DiffusionUnetImagePolicy(
    shape_meta=shape_meta, noise_scheduler=noise_scheduler,
    obs_encoder=obs_encoder, horizon=horizon,
    n_action_steps=n_action_steps, n_obs_steps=n_obs_steps,
    num_inference_steps=100, obs_as_global_cond=True,
    diffusion_step_embed_dim=128, down_dims=[256, 512, 1024],
    kernel_size=5, n_groups=8, cond_predict_scale=True,
)

total_params = sum(p.numel() for p in policy.parameters())
print(f"  Total parameters: {total_params:,}")

global_cond_dim = obs_feature_dim * n_obs_steps
unet_cond_dim = 128 + global_cond_dim
print(f"  obs_feature_dim = {obs_feature_dim}")
print(f"  global_cond_dim = {global_cond_dim} ({obs_feature_dim} x {n_obs_steps})")
print(f"  UNet cond_dim = {unet_cond_dim} (128 + {global_cond_dim})")

# Setup normalizer
normalizer = LinearNormalizer()
normalizer.fit(data={
    'head_cam': torch.rand(50, *image_shape),
    'agent_pos': torch.randn(50, agent_pos_dim),
    'tcp_agent0': torch.randn(50, tcp_dim),
    'tcp_agent1': torch.randn(50, tcp_dim),
    'tcp_agent2': torch.randn(50, tcp_dim),
    'point_cloud': torch.randn(50, n_points, 3),
    'tactile': torch.rand(50, tactile_dim),
    'action': torch.randn(50, action_dim),
})
policy.set_normalizer(normalizer)

# Training batch
batch = {
    'obs': {
        'head_cam': torch.rand(batch_size, horizon, *image_shape),
        'agent_pos': torch.randn(batch_size, horizon, agent_pos_dim),
        'tcp_agent0': torch.randn(batch_size, horizon, tcp_dim),
        'tcp_agent1': torch.randn(batch_size, horizon, tcp_dim),
        'tcp_agent2': torch.randn(batch_size, horizon, tcp_dim),
        'point_cloud': torch.randn(batch_size, horizon, n_points, 3),
        'tactile': torch.rand(batch_size, horizon, tactile_dim),
    },
    'action': torch.randn(batch_size, horizon, action_dim),
}

loss = policy.compute_loss(batch)
print(f"  Loss: {loss.item():.4f}")
assert not torch.isnan(loss), "FAIL: loss is NaN"
print("  PASS\n")

# ========== Test 4.5: Normalizer value check ==========
print("=" * 60)
print("Test 4.5: Normalizer value verification (tactile + TCP)")
print("=" * 60)

test_normalizer = LinearNormalizer()
# 触觉数据: FSR传感器值范围 [0, 100]（模拟真实FSR）
tactile_data = torch.rand(100, tactile_dim) * 100  # [0, 100]
# TCP数据
tcp_data = torch.randn(100, tcp_dim) * 0.5 + 1.0

test_normalizer.fit(data={
    'tactile': tactile_data,
    'tcp_agent0': tcp_data,
    'agent_pos': torch.randn(100, agent_pos_dim),
    'action': torch.randn(100, action_dim),
})

# 验证触觉归一化
tactile_sample = tactile_data[:5]
tactile_normalized = test_normalizer['tactile'].normalize(tactile_sample)
tactile_unnormalized = test_normalizer['tactile'].unnormalize(tactile_normalized)

print(f"  Tactile original range: [{tactile_data.min():.3f}, {tactile_data.max():.3f}]")
print(f"  Tactile normalized range: [{tactile_normalized.min():.3f}, {tactile_normalized.max():.3f}]")

# 验证归一化后范围在[-1, 1]
assert tactile_normalized.min() >= -1.1, f"FAIL: normalized min {tactile_normalized.min():.3f} < -1.1"
assert tactile_normalized.max() <= 1.1, f"FAIL: normalized max {tactile_normalized.max():.3f} > 1.1"
print(f"  Normalized range check: PASS (within [-1, 1])")

# 验证反归一化能恢复原始值
recon_error = (tactile_unnormalized - tactile_sample).abs().max().item()
print(f"  Unnormalize reconstruction error: {recon_error:.8f}")
assert recon_error < 1e-5, f"FAIL: reconstruction error {recon_error:.8f} too large"
print(f"  Reconstruction check: PASS")

# 验证TCP归一化
tcp_sample = tcp_data[:5]
tcp_normalized = test_normalizer['tcp_agent0'].normalize(tcp_sample)
tcp_unnormalized = test_normalizer['tcp_agent0'].unnormalize(tcp_normalized)
tcp_recon_error = (tcp_unnormalized - tcp_sample).abs().max().item()
print(f"  TCP normalized range: [{tcp_normalized.min():.3f}, {tcp_normalized.max():.3f}]")
print(f"  TCP reconstruction error: {tcp_recon_error:.8f}")
assert tcp_recon_error < 1e-5, "FAIL: TCP reconstruction error too large"
print(f"  TCP check: PASS")
print("  PASS\n")

# ========== Test 5: Gradient flow ==========
print("=" * 60)
print("Test 5: Gradient flow (all components)")
print("=" * 60)

loss.backward()

grad_checks = {
    'Tactile Conv': tactile_model.finger_conv[0].weight.grad,
    'Tactile Fusion': tactile_model.fusion[0].weight.grad,
    'PointNet (film)': policy.obs_encoder.pointcloud_film_model.pointnet.mlp[0].weight.grad,
    'FiLM generator': policy.obs_encoder.pointcloud_film_model.film_generator.weight.grad,
    'ResNet': list(policy.obs_encoder.key_model_map.values())[0].layer1[0].conv1.weight.grad,
    'GNN': policy.obs_encoder.graph_model.node_encoder[0].weight.grad,
    'UNet': list(policy.model.parameters())[0].grad,
}

all_ok = True
for name, grad in grad_checks.items():
    has_grad = grad is not None and grad.abs().sum() > 0
    status = "PASS" if has_grad else "FAIL (no gradient!)"
    print(f"  {name}: {status}")
    if not has_grad:
        all_ok = False

assert all_ok, "FAIL: some components have no gradient"

print("\n" + "=" * 60)
print("All Tactile Encoder tests passed!")
print("=" * 60)
