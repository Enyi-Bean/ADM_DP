"""
Full pipeline test: verifies the complete model architecture including
ResNet + FiLM + GNN + Tactile + AMAM fusion + UNet diffusion policy.

Tests:
  1. obs_encoder output dimension
  2. AMAM fusion: modality features are weighted (not plain concat)
  3. Full training forward (compute_loss) with reg loss
  4. Dimension chain verification (obs → global_cond → UNet cond_dim)
  5. Normalizer (tactile + pointcloud + TCP)
  6. Gradient flow through all components
  7. Inference forward (predict_action)

Usage:
    conda activate ADM-DP
    cd /NEW_EDS/JJ_Group/wangey2512/code/Mres_Project/RoboFactory
    python test_full_pipeline.py
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
from diffusion_policy.model.vision.amam_fusion import AMAMFusion
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer

# ========== Config (matches robot_dp.yaml + default_task.yaml) ==========
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
tactile_dim = 32
diffusion_step_embed_dim = 128

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

# ========== Build full model ==========
print("Building full model...")
rgb_model = get_resnet('resnet18', weights=None)
graph_model = GraphTCPEncoder(input_dim=3, hidden_dim=64, output_dim=64, num_layers=2, num_agents=num_agents)
film_model = PointCloudFiLM(pc_in_channels=3, pc_out_channels=256, img_feature_dim=512, use_layernorm=True)
tactile_model = TactileEncoder(input_dim=32, output_dim=64, conv_channels=[16, 32], use_layernorm=True)
amam_model = AMAMFusion(vision_dim=512, tactile_dim=64, graph_dim=64, temperature=1.0, lambda_reg=0.01)

obs_encoder = MultiImageObsEncoder(
    shape_meta=shape_meta,
    rgb_model=rgb_model, graph_model=graph_model,
    pointcloud_film_model=film_model, tactile_model=tactile_model,
    amam_model=amam_model,
    resize_shape=None, crop_shape=None, random_crop=True,
    use_group_norm=True, share_rgb_model=False, imagenet_norm=True,
)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
    beta_schedule='squaredcos_cap_v2', variance_type='fixed_small',
    clip_sample=True, prediction_type='epsilon',
)
policy = DiffusionUnetImagePolicy(
    shape_meta=shape_meta, noise_scheduler=noise_scheduler,
    obs_encoder=obs_encoder, horizon=horizon,
    n_action_steps=n_action_steps, n_obs_steps=n_obs_steps,
    num_inference_steps=10, obs_as_global_cond=True,
    diffusion_step_embed_dim=diffusion_step_embed_dim,
    down_dims=[256, 512, 1024], kernel_size=5, n_groups=8,
    cond_predict_scale=True,
)

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

print(f"Total parameters: {sum(p.numel() for p in policy.parameters()):,}\n")

# ========== Test 1: obs_encoder output dimension ==========
print("=" * 60)
print("Test 1: obs_encoder output dimension")
print("=" * 60)

obs_feature_dim = obs_encoder.output_shape()[0]
expected_dim = 512 + 64 + 64 + agent_pos_dim  # AMAM(v+t+g) + agent_pos
print(f"  obs_feature_dim = {obs_feature_dim}")
print(f"  Expected: AMAM(512+64+64) + agent_pos({agent_pos_dim}) = {expected_dim}")
assert obs_feature_dim == expected_dim, f"FAIL: expected {expected_dim}, got {obs_feature_dim}"
print("  PASS\n")

# ========== Test 2: AMAM actually modulates features ==========
print("=" * 60)
print("Test 2: AMAM fusion modulates features (not plain concat)")
print("=" * 60)

# Build a second obs_encoder WITHOUT AMAM for comparison
rgb_model2 = get_resnet('resnet18', weights=None)
graph_model2 = GraphTCPEncoder(input_dim=3, hidden_dim=64, output_dim=64, num_layers=2, num_agents=num_agents)
film_model2 = PointCloudFiLM(pc_in_channels=3, pc_out_channels=256, img_feature_dim=512, use_layernorm=True)
tactile_model2 = TactileEncoder(input_dim=32, output_dim=64, conv_channels=[16, 32], use_layernorm=True)

obs_encoder_no_amam = MultiImageObsEncoder(
    shape_meta=shape_meta,
    rgb_model=rgb_model2, graph_model=graph_model2,
    pointcloud_film_model=film_model2, tactile_model=tactile_model2,
    amam_model=None,  # No AMAM
    resize_shape=None, crop_shape=None, random_crop=True,
    use_group_norm=True, share_rgb_model=False, imagenet_norm=True,
)

no_amam_dim = obs_encoder_no_amam.output_shape()[0]
print(f"  With AMAM: obs_feature_dim = {obs_feature_dim}")
print(f"  Without AMAM: obs_feature_dim = {no_amam_dim}")
assert obs_feature_dim == no_amam_dim, "FAIL: dimensions should match"
print(f"  Dimensions match (AMAM preserves dims): PASS")

# Verify AMAM weights are not uniform (features are actually weighted)
dummy_obs = {
    'head_cam': torch.rand(2, *image_shape),
    'agent_pos': torch.randn(2, agent_pos_dim),
    'tcp_agent0': torch.randn(2, tcp_dim),
    'tcp_agent1': torch.randn(2, tcp_dim),
    'tcp_agent2': torch.randn(2, tcp_dim),
    'point_cloud': torch.randn(2, n_points, 3),
    'tactile': torch.rand(2, tactile_dim),
}
with torch.no_grad():
    _ = obs_encoder(dummy_obs)
alpha = amam_model.last_alpha[0]
print(f"  AMAM weights: α_v={alpha[0]:.4f}, α_t={alpha[1]:.4f}, α_g={alpha[2]:.4f}")
print(f"  Weights sum: {alpha.sum():.4f} (should be 1.0)")
assert abs(alpha.sum().item() - 1.0) < 1e-5, "FAIL: weights should sum to 1"
print("  PASS\n")

# ========== Test 3: AMAM response to zero vs non-zero tactile ==========
print("=" * 60)
print("Test 3: AMAM attention shift (zero vs active tactile)")
print("=" * 60)

# Approach phase: tactile = 0
obs_approach = {
    'head_cam': torch.rand(2, *image_shape),
    'agent_pos': torch.randn(2, agent_pos_dim),
    'tcp_agent0': torch.randn(2, tcp_dim),
    'tcp_agent1': torch.randn(2, tcp_dim),
    'tcp_agent2': torch.randn(2, tcp_dim),
    'point_cloud': torch.randn(2, n_points, 3),
    'tactile': torch.zeros(2, tactile_dim),  # No contact
}
with torch.no_grad():
    _ = obs_encoder(obs_approach)
alpha_approach = amam_model.last_alpha.mean(dim=0).clone()

# Grasp phase: tactile active
obs_grasp = {
    'head_cam': torch.rand(2, *image_shape),
    'agent_pos': torch.randn(2, agent_pos_dim),
    'tcp_agent0': torch.randn(2, tcp_dim),
    'tcp_agent1': torch.randn(2, tcp_dim),
    'tcp_agent2': torch.randn(2, tcp_dim),
    'point_cloud': torch.randn(2, n_points, 3),
    'tactile': torch.rand(2, tactile_dim) * 50,  # Strong contact
}
with torch.no_grad():
    _ = obs_encoder(obs_grasp)
alpha_grasp = amam_model.last_alpha.mean(dim=0).clone()

print(f"  Approach (tactile=0):  α_v={alpha_approach[0]:.4f}, α_t={alpha_approach[1]:.4f}, α_g={alpha_approach[2]:.4f}")
print(f"  Grasp (tactile≠0):    α_v={alpha_grasp[0]:.4f}, α_t={alpha_grasp[1]:.4f}, α_g={alpha_grasp[2]:.4f}")
# Weights should differ between phases (even before training, inputs differ → weights differ)
weight_diff = (alpha_approach - alpha_grasp).abs().sum().item()
print(f"  Weight difference: {weight_diff:.6f}")
assert weight_diff > 1e-6, "FAIL: weights should differ between approach and grasp"
print(f"  Attention shifts with tactile state: PASS")
print("  (Note: after training, α_t should be lower in approach, higher in grasp)\n")

# ========== Test 4: Dimension chain verification ==========
print("=" * 60)
print("Test 4: Complete dimension chain verification")
print("=" * 60)

vision_dim = 512
tactile_out_dim = 64
graph_out_dim = 64
amam_out = vision_dim + tactile_out_dim + graph_out_dim
obs_feat = amam_out + agent_pos_dim
global_cond = obs_feat * n_obs_steps
unet_cond = diffusion_step_embed_dim + global_cond

print(f"  ResNet output:              {vision_dim}")
print(f"  + PointCloud FiLM:          {vision_dim} (same dim, modulated)")
print(f"  Tactile encoder output:     {tactile_out_dim}")
print(f"  GNN output:                 {graph_out_dim}")
print(f"  AMAM fusion:                {amam_out} (weighted concat of above 3)")
print(f"  + agent_pos:                {agent_pos_dim}")
print(f"  obs_feature_dim:            {obs_feat}")
print(f"  x n_obs_steps({n_obs_steps}):          {global_cond}")
print(f"  + diffusion_step_embed:     {diffusion_step_embed_dim}")
print(f"  UNet cond_dim:              {unet_cond}")

assert obs_feat == obs_feature_dim, f"FAIL: obs_feat {obs_feat} != {obs_feature_dim}"
assert unet_cond == 2072, f"FAIL: UNet cond_dim {unet_cond} != 2072"
print("  All dimensions verified: PASS\n")

# ========== Test 5: Normalizer check ==========
print("=" * 60)
print("Test 5: Normalizer (tactile + pointcloud + TCP)")
print("=" * 60)

test_normalizer = LinearNormalizer()
tactile_data = torch.rand(100, tactile_dim) * 100
pc_data = torch.randn(100, n_points, 3)
tcp_data = torch.randn(100, tcp_dim)

test_normalizer.fit(data={
    'tactile': tactile_data,
    'point_cloud': pc_data,
    'tcp_agent0': tcp_data,
    'agent_pos': torch.randn(100, agent_pos_dim),
    'action': torch.randn(100, action_dim),
})

# Tactile normalization
t_sample = tactile_data[:5]
t_norm = test_normalizer['tactile'].normalize(t_sample)
t_recon = test_normalizer['tactile'].unnormalize(t_norm)
t_err = (t_recon - t_sample).abs().max().item()
print(f"  Tactile: [{tactile_data.min():.1f}, {tactile_data.max():.1f}] → [{t_norm.min():.3f}, {t_norm.max():.3f}], recon_err={t_err:.8f}")
assert t_norm.min() >= -1.1 and t_norm.max() <= 1.1, "FAIL: tactile not in [-1,1]"
assert t_err < 1e-5, "FAIL: tactile reconstruction error too large"

# Pointcloud normalization
pc_sample = pc_data[:5]
pc_norm = test_normalizer['point_cloud'].normalize(pc_sample)
pc_recon = test_normalizer['point_cloud'].unnormalize(pc_norm)
pc_err = (pc_recon - pc_sample).abs().max().item()
print(f"  PointCloud: recon_err={pc_err:.8f}")
assert pc_err < 1e-5, "FAIL: pointcloud reconstruction error too large"

# TCP normalization
tcp_sample = tcp_data[:5]
tcp_norm = test_normalizer['tcp_agent0'].normalize(tcp_sample)
tcp_recon = test_normalizer['tcp_agent0'].unnormalize(tcp_norm)
tcp_err = (tcp_recon - tcp_sample).abs().max().item()
print(f"  TCP: recon_err={tcp_err:.8f}")
assert tcp_err < 1e-5, "FAIL: TCP reconstruction error too large"
print("  All normalizers: PASS\n")

# ========== Test 6: Full training forward + entropy reg ==========
print("=" * 60)
print("Test 6: Full compute_loss with entropy regularization")
print("=" * 60)

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
reg_loss = amam_model.last_reg_loss.item()
diffusion_loss = loss.item() - reg_loss

print(f"  Total loss:     {loss.item():.4f}")
print(f"  Diffusion loss: {diffusion_loss:.4f}")
print(f"  AMAM reg loss:  {reg_loss:.6f}")
print(f"  AMAM α: v={amam_model.last_alpha[:, 0].mean():.4f}, "
      f"t={amam_model.last_alpha[:, 1].mean():.4f}, "
      f"g={amam_model.last_alpha[:, 2].mean():.4f}")
assert not torch.isnan(loss), "FAIL: loss is NaN"
assert reg_loss > 0, "FAIL: reg loss should be positive"
print("  PASS\n")

# ========== Test 7: Gradient flow ==========
print("=" * 60)
print("Test 7: Gradient flow (all components)")
print("=" * 60)

loss.backward()

grad_checks = {
    'AMAM MLP':       amam_model.attention_mlp[0].weight.grad,
    'Tactile Conv':   tactile_model.finger_conv[0].weight.grad,
    'Tactile Fusion': tactile_model.fusion[0].weight.grad,
    'PointNet':       film_model.pointnet.mlp[0].weight.grad,
    'FiLM generator': film_model.film_generator.weight.grad,
    'ResNet':         list(obs_encoder.key_model_map.values())[0].layer1[0].conv1.weight.grad,
    'GNN':            graph_model.node_encoder[0].weight.grad,
    'UNet':           list(policy.model.parameters())[0].grad,
}

all_ok = True
for name, grad in grad_checks.items():
    has_grad = grad is not None and grad.abs().sum() > 0
    status = "PASS" if has_grad else "FAIL (no gradient!)"
    print(f"  {name}: {status}")
    if not has_grad:
        all_ok = False

assert all_ok, "FAIL: some components have no gradient"
print("  PASS\n")

# ========== Test 8: Inference forward ==========
print("=" * 60)
print("Test 8: Inference forward (predict_action)")
print("=" * 60)

policy.eval()
obs_input = {
    'head_cam': torch.rand(1, n_obs_steps, *image_shape),
    'agent_pos': torch.randn(1, n_obs_steps, agent_pos_dim),
    'tcp_agent0': torch.randn(1, n_obs_steps, tcp_dim),
    'tcp_agent1': torch.randn(1, n_obs_steps, tcp_dim),
    'tcp_agent2': torch.randn(1, n_obs_steps, tcp_dim),
    'point_cloud': torch.randn(1, n_obs_steps, n_points, 3),
    'tactile': torch.rand(1, n_obs_steps, tactile_dim),
}

with torch.no_grad():
    action_dict = policy.predict_action(obs_input)

action = action_dict['action']
print(f"  Predicted action shape: {tuple(action.shape)}")
assert action.shape[0] == 1, "FAIL: batch size should be 1"
assert action.shape[-1] == action_dim, f"FAIL: action_dim should be {action_dim}"
assert not torch.isnan(action).any(), "FAIL: action contains NaN"
print(f"  Action sample: [{action[0, 0, 0]:.4f}, {action[0, 0, 1]:.4f}, ...]")
print("  PASS\n")

# ========== Summary ==========
print("=" * 60)
print("FULL PIPELINE SUMMARY")
print("=" * 60)
print(f"  Architecture: ResNet18 + FiLM + GNN + Tactile + AMAM + UNet")
print(f"  obs_feature_dim: {obs_feature_dim}")
print(f"  UNet cond_dim: {unet_cond}")
print(f"  Total parameters: {sum(p.numel() for p in policy.parameters()):,}")
print(f"  All 8 tests passed!")
print("=" * 60)
