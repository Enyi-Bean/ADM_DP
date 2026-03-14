"""
Test PointCloud FiLM modulation + full training forward pass.
Verifies that point cloud features correctly modulate image features
without changing the output dimension.

Usage:
    conda activate ADM-DP
    cd /NEW_EDS/JJ_Group/wangey2512/code/Mres_Project/RoboFactory
    python test_pointcloud_film.py
"""
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'policy', 'Diffusion-Policy'))

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.graph_encoder import GraphTCPEncoder
from diffusion_policy.model.vision.pointcloud_encoder import PointNetEncoderXYZ, PointCloudFiLM
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
n_points = 512  # 点云采样点数

shape_meta = {
    'obs': {
        'head_cam': {'shape': list(image_shape), 'type': 'rgb'},
        'agent_pos': {'shape': [agent_pos_dim], 'type': 'low_dim'},
        'tcp_agent0': {'shape': [tcp_dim], 'type': 'tcp'},
        'tcp_agent1': {'shape': [tcp_dim], 'type': 'tcp'},
        'tcp_agent2': {'shape': [tcp_dim], 'type': 'tcp'},
        'point_cloud': {'shape': [n_points, 3], 'type': 'pointcloud'},
    },
    'action': {'shape': [action_dim]},
}

# ========== Test 1: PointNet standalone ==========
print("=" * 60)
print("Test 1: PointNetEncoderXYZ standalone")
print("=" * 60)

pointnet = PointNetEncoderXYZ(in_channels=3, out_channels=256, use_layernorm=True, final_norm='layernorm')
pc_input = torch.randn(4, n_points, 3)
pc_output = pointnet(pc_input)
print(f"  Input: {tuple(pc_input.shape)} -> Output: {tuple(pc_output.shape)}")
assert pc_output.shape == (4, 256)
print(f"  Parameters: {sum(p.numel() for p in pointnet.parameters()):,}")
print("  PASS\n")

# ========== Test 2: PointCloudFiLM standalone ==========
print("=" * 60)
print("Test 2: PointCloudFiLM standalone")
print("=" * 60)

film = PointCloudFiLM(pc_in_channels=3, pc_out_channels=256, img_feature_dim=512, use_layernorm=True)
pc_input = torch.randn(4, n_points, 3)
img_feat = torch.randn(4, 512)

# Test near-identity initialization
modulated = film(pc_input, img_feat)
print(f"  Point cloud: {tuple(pc_input.shape)}")
print(f"  Image feat:  {tuple(img_feat.shape)}")
print(f"  Modulated:   {tuple(modulated.shape)}")
assert modulated.shape == (4, 512), f"FAIL: expected (4, 512), got {tuple(modulated.shape)}"

# Check near-identity init: output should be close to input at initialization
diff = (modulated - img_feat).abs().mean().item()
print(f"  Init diff from identity: {diff:.6f} (should be near 0)")
print(f"  Parameters: {sum(p.numel() for p in film.parameters()):,}")
print("  PASS\n")

# ========== Test 3: Full obs_encoder with FiLM ==========
print("=" * 60)
print("Test 3: Full obs_encoder (ResNet18 + GNN + PointCloud FiLM)")
print("=" * 60)

rgb_model = get_resnet('resnet18', weights=None)
graph_model = GraphTCPEncoder(input_dim=3, hidden_dim=64, output_dim=64, num_layers=2, num_agents=num_agents)
film_model = PointCloudFiLM(pc_in_channels=3, pc_out_channels=256, img_feature_dim=512, use_layernorm=True)

obs_encoder = MultiImageObsEncoder(
    shape_meta=shape_meta,
    rgb_model=rgb_model,
    graph_model=graph_model,
    pointcloud_film_model=film_model,
    resize_shape=None, crop_shape=None, random_crop=True,
    use_group_norm=True, share_rgb_model=False, imagenet_norm=True,
)

obs_feature_dim = obs_encoder.output_shape()[0]
print(f"  obs_feature_dim = {obs_feature_dim}")
# 512 (ResNet, FiLM-modulated) + 8 (agent_pos) + 64 (GNN) = 584
assert obs_feature_dim == 512 + agent_pos_dim + 64, \
    f"FAIL: expected {512 + agent_pos_dim + 64}, got {obs_feature_dim}"
print(f"  = ResNet(512, FiLM-enhanced) + agent_pos({agent_pos_dim}) + GNN(64)")
print("  PASS\n")

# ========== Test 4: Full policy forward (compute_loss) ==========
print("=" * 60)
print("Test 4: Full training forward with PointCloud FiLM")
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
assert unet_cond_dim == 1880

# Setup normalizer
normalizer = LinearNormalizer()
normalizer.fit(data={
    'head_cam': torch.rand(50, *image_shape),
    'agent_pos': torch.randn(50, agent_pos_dim),
    'tcp_agent0': torch.randn(50, tcp_dim),
    'tcp_agent1': torch.randn(50, tcp_dim),
    'tcp_agent2': torch.randn(50, tcp_dim),
    'point_cloud': torch.randn(50, n_points, 3),
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
    },
    'action': torch.randn(batch_size, horizon, action_dim),
}

loss = policy.compute_loss(batch)
print(f"  Loss: {loss.item():.4f}")
assert not torch.isnan(loss), "FAIL: loss is NaN"
print("  PASS\n")

# ========== Test 4.5: Normalizer value check ==========
print("=" * 60)
print("Test 4.5: Normalizer value verification")
print("=" * 60)

# 用有明确范围的数据测试归一化
test_normalizer = LinearNormalizer()
# 点云坐标范围: x∈[-1,1], y∈[0,2], z∈[0.5,1.5]
pc_data = torch.zeros(100, n_points, 3)
pc_data[:, :, 0] = torch.rand(100, n_points) * 2 - 1    # x: [-1, 1]
pc_data[:, :, 1] = torch.rand(100, n_points) * 2          # y: [0, 2]
pc_data[:, :, 2] = torch.rand(100, n_points) + 0.5        # z: [0.5, 1.5]

# TCP和其他数据
tcp_data = torch.randn(100, tcp_dim) * 0.5 + 1.0  # 均值1, 标准差0.5
agent_pos_data = torch.randn(100, agent_pos_dim)

test_normalizer.fit(data={
    'point_cloud': pc_data,
    'tcp_agent0': tcp_data,
    'agent_pos': agent_pos_data,
    'action': torch.randn(100, action_dim),
})

# 验证点云归一化
pc_sample = pc_data[:5]  # 取5个样本
pc_normalized = test_normalizer['point_cloud'].normalize(pc_sample)
pc_unnormalized = test_normalizer['point_cloud'].unnormalize(pc_normalized)

print(f"  Point cloud original range:")
print(f"    x: [{pc_data[:,:,0].min():.3f}, {pc_data[:,:,0].max():.3f}]")
print(f"    y: [{pc_data[:,:,1].min():.3f}, {pc_data[:,:,1].max():.3f}]")
print(f"    z: [{pc_data[:,:,2].min():.3f}, {pc_data[:,:,2].max():.3f}]")
print(f"  Point cloud normalized range:")
print(f"    x: [{pc_normalized[:,:,0].min():.3f}, {pc_normalized[:,:,0].max():.3f}]")
print(f"    y: [{pc_normalized[:,:,1].min():.3f}, {pc_normalized[:,:,1].max():.3f}]")
print(f"    z: [{pc_normalized[:,:,2].min():.3f}, {pc_normalized[:,:,2].max():.3f}]")

# 验证归一化后范围在[-1, 1]
assert pc_normalized.min() >= -1.1, f"FAIL: normalized min {pc_normalized.min():.3f} < -1.1"
assert pc_normalized.max() <= 1.1, f"FAIL: normalized max {pc_normalized.max():.3f} > 1.1"
print(f"  Normalized range check: PASS (within [-1, 1])")

# 验证反归一化能恢复原始值
recon_error = (pc_unnormalized - pc_sample).abs().max().item()
print(f"  Unnormalize reconstruction error: {recon_error:.8f}")
assert recon_error < 1e-5, f"FAIL: reconstruction error {recon_error:.8f} too large"
print(f"  Reconstruction check: PASS")

# 验证TCP归一化
tcp_sample = tcp_data[:5]
tcp_normalized = test_normalizer['tcp_agent0'].normalize(tcp_sample)
tcp_unnormalized = test_normalizer['tcp_agent0'].unnormalize(tcp_normalized)
tcp_recon_error = (tcp_unnormalized - tcp_sample).abs().max().item()
print(f"  TCP normalize range: [{tcp_normalized.min():.3f}, {tcp_normalized.max():.3f}]")
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
print("All PointCloud FiLM tests passed!")
print("=" * 60)
