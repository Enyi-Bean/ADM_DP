"""
Dummy test for full training forward pass (compute_loss).
Simulates exactly what happens during one training step.

Usage:
    conda activate ADM-DP
    cd /NEW_EDS/JJ_Group/wangey2512/code/Mres_Project/RoboFactory
    python test_training_forward.py
"""
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'policy', 'Diffusion-Policy'))

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.graph_encoder import GraphTCPEncoder
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer

# ========== Config (from robot_dp.yaml + default_task.yaml) ==========
batch_size = 4          # 用小 batch 测试，不需要 64
n_obs_steps = 3
horizon = 8
n_action_steps = 8
action_dim = 8
agent_pos_dim = 8
image_shape = (3, 240, 320)
tcp_dim = 3
num_agents = 3
gnn_hidden = 64
gnn_output = 64

shape_meta = {
    'obs': {
        'head_cam': {'shape': list(image_shape), 'type': 'rgb'},
        'agent_pos': {'shape': [agent_pos_dim], 'type': 'low_dim'},
        'tcp_agent0': {'shape': [tcp_dim], 'type': 'tcp'},
        'tcp_agent1': {'shape': [tcp_dim], 'type': 'tcp'},
        'tcp_agent2': {'shape': [tcp_dim], 'type': 'tcp'},
    },
    'action': {'shape': [action_dim]},
}

print("=" * 60)
print("Step 1: Build obs_encoder (ResNet18 + GNN)")
print("=" * 60)

rgb_model = get_resnet('resnet18', weights=None)
graph_model = GraphTCPEncoder(
    input_dim=tcp_dim, hidden_dim=gnn_hidden,
    output_dim=gnn_output, num_layers=2, num_agents=num_agents,
)
obs_encoder = MultiImageObsEncoder(
    shape_meta=shape_meta,
    rgb_model=rgb_model,
    graph_model=graph_model,
    resize_shape=None,
    crop_shape=None,
    random_crop=True,
    use_group_norm=True,
    share_rgb_model=False,
    imagenet_norm=True,
)

obs_feature_dim = obs_encoder.output_shape()[0]
print(f"  obs_feature_dim = {obs_feature_dim}")
# 512 (resnet18) + 8 (agent_pos) + 64 (gnn) = 584
assert obs_feature_dim == 512 + agent_pos_dim + gnn_output, \
    f"FAIL: expected {512 + agent_pos_dim + gnn_output}, got {obs_feature_dim}"
print("  PASS\n")

print("=" * 60)
print("Step 2: Build DiffusionUnetImagePolicy")
print("=" * 60)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=100,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule='squaredcos_cap_v2',
    variance_type='fixed_small',
    clip_sample=True,
    prediction_type='epsilon',
)

policy = DiffusionUnetImagePolicy(
    shape_meta=shape_meta,
    noise_scheduler=noise_scheduler,
    obs_encoder=obs_encoder,
    horizon=horizon,
    n_action_steps=n_action_steps,
    n_obs_steps=n_obs_steps,
    num_inference_steps=100,
    obs_as_global_cond=True,
    diffusion_step_embed_dim=128,
    down_dims=[256, 512, 1024],
    kernel_size=5,
    n_groups=8,
    cond_predict_scale=True,
)

total_params = sum(p.numel() for p in policy.parameters())
print(f"  Policy created, total parameters: {total_params:,}")
print(f"  obs_feature_dim = {policy.obs_feature_dim}")

global_cond_dim = policy.obs_feature_dim * n_obs_steps
diffusion_step_embed_dim = 128
unet_cond_dim = diffusion_step_embed_dim + global_cond_dim

print(f"  global_cond_dim = {global_cond_dim} (obs_feature_dim {policy.obs_feature_dim} x n_obs_steps {n_obs_steps})")
print(f"  diffusion_step_embed_dim = {diffusion_step_embed_dim}")
print(f"  UNet cond_dim = {unet_cond_dim} (diffusion_step_embed {diffusion_step_embed_dim} + global_cond {global_cond_dim})")
print(f"    = ResNet({512}) + agent_pos({agent_pos_dim}) + GNN({gnn_output}) = {512+agent_pos_dim+gnn_output} per step")
print(f"    x {n_obs_steps} obs_steps = {global_cond_dim}")
print(f"    + {diffusion_step_embed_dim} diffusion timestep embed = {unet_cond_dim}")

# Verify UNet internal cond_dim matches
actual_unet_cond_dim = policy.model.cond_dim if hasattr(policy.model, 'cond_dim') else None
if actual_unet_cond_dim is not None:
    assert actual_unet_cond_dim == unet_cond_dim, \
        f"FAIL: UNet cond_dim {actual_unet_cond_dim} != expected {unet_cond_dim}"
    print(f"  UNet internal cond_dim verified: {actual_unet_cond_dim}")

assert unet_cond_dim == 1880, f"FAIL: expected UNet cond_dim 1880, got {unet_cond_dim}"
print("  PASS\n")

print("=" * 60)
print("Step 3: Setup normalizer with dummy data")
print("=" * 60)

# normalizer needs to be fit before compute_loss
dummy_data = {
    'obs': {
        'head_cam': torch.rand(100, *image_shape),
        'agent_pos': torch.randn(100, agent_pos_dim),
        'tcp_agent0': torch.randn(100, tcp_dim),
        'tcp_agent1': torch.randn(100, tcp_dim),
        'tcp_agent2': torch.randn(100, tcp_dim),
    },
    'action': torch.randn(100, action_dim),
}
normalizer = LinearNormalizer()
normalizer.fit(data={
    'head_cam': dummy_data['obs']['head_cam'],
    'agent_pos': dummy_data['obs']['agent_pos'],
    'tcp_agent0': dummy_data['obs']['tcp_agent0'],
    'tcp_agent1': dummy_data['obs']['tcp_agent1'],
    'tcp_agent2': dummy_data['obs']['tcp_agent2'],
    'action': dummy_data['action'],
})
policy.set_normalizer(normalizer)
print("  Normalizer fitted and set")
print("  PASS\n")

print("=" * 60)
print("Step 4: compute_loss (training forward pass)")
print("=" * 60)

# Simulate a training batch
# In real training: obs has shape [B, horizon, ...], action has shape [B, horizon, action_dim]
batch = {
    'obs': {
        'head_cam': torch.rand(batch_size, horizon, *image_shape),
        'agent_pos': torch.randn(batch_size, horizon, agent_pos_dim),
        'tcp_agent0': torch.randn(batch_size, horizon, tcp_dim),
        'tcp_agent1': torch.randn(batch_size, horizon, tcp_dim),
        'tcp_agent2': torch.randn(batch_size, horizon, tcp_dim),
    },
    'action': torch.randn(batch_size, horizon, action_dim),
}

loss = policy.compute_loss(batch)
print(f"  Loss value: {loss.item():.4f}")
print(f"  Loss shape: {tuple(loss.shape)} (scalar)")
assert loss.dim() == 0, "FAIL: loss should be scalar"
assert not torch.isnan(loss), "FAIL: loss is NaN"
assert not torch.isinf(loss), "FAIL: loss is Inf"
print("  PASS\n")

print("=" * 60)
print("Step 5: Backward pass (gradient check)")
print("=" * 60)

loss.backward()

# Check gradients exist for key components
grad_checks = {
    'obs_encoder.graph_model': policy.obs_encoder.graph_model.node_encoder[0].weight.grad,
    'obs_encoder.resnet': list(policy.obs_encoder.key_model_map.values())[0].layer1[0].conv1.weight.grad,
    'unet_model': list(policy.model.parameters())[0].grad,
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
print("All tests passed!")
print("=" * 60)
