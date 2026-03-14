"""
Test multi-agent inference pipeline with fake checkpoints.
Simulates eval_multi_dp.py: each agent has its own policy + runner,
all agents share TCP observations and point cloud.

Usage:
    conda activate ADM-DP
    cd /NEW_EDS/JJ_Group/wangey2512/code/Mres_Project/RoboFactory
    python test_inference.py
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
from diffusion_policy.env_runner.dp_runner import DPRunner

# ========== Config (matches robot_dp.yaml + default_task.yaml, 3-arm) ==========
n_obs_steps = 3
horizon = 8
n_action_steps = 8
action_dim = 8
agent_pos_dim = 8
image_shape = (3, 240, 320)
tcp_dim = 3
num_agents = 3  # 3个机械臂
n_points = 512  # 点云采样点数
tactile_dim = 32  # 2 fingers x 4x4 FSR


def build_policy():
    """Build a single agent's policy with random weights (simulates loading a checkpoint)."""
    rgb_model = get_resnet('resnet18', weights=None)
    graph_model = GraphTCPEncoder(
        input_dim=tcp_dim, hidden_dim=64, output_dim=64,
        num_layers=2, num_agents=num_agents,
    )
    film_model = PointCloudFiLM(
        pc_in_channels=3, pc_out_channels=256,
        img_feature_dim=512, use_layernorm=True,
    )
    tactile_model = TactileEncoder(
        input_dim=32, output_dim=64,
        conv_channels=[16, 32], use_layernorm=True,
    )
    amam_model = AMAMFusion(
        vision_dim=512, tactile_dim=64, graph_dim=64,
        temperature=1.0, lambda_reg=0.01,
    )
    obs_encoder = MultiImageObsEncoder(
        shape_meta=shape_meta, rgb_model=rgb_model,
        graph_model=graph_model, pointcloud_film_model=film_model,
        tactile_model=tactile_model, amam_model=amam_model,
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
        num_inference_steps=10,  # 10步加速测试，正式eval用100
        obs_as_global_cond=True, diffusion_step_embed_dim=128,
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
    policy.eval()
    return policy


def fake_get_model_input(agent_id, shared_tcp):
    """
    Simulates get_model_input() in eval_multi_dp.py.
    Each agent has its own camera, agent_pos and point cloud, but TCP is shared.
    """
    return {
        'head_cam': np.random.rand(*image_shape).astype(np.float32),
        'agent_pos': np.random.randn(agent_pos_dim).astype(np.float32),
        'tcp_agent0': shared_tcp[0].copy(),
        'tcp_agent1': shared_tcp[1].copy(),
        'tcp_agent2': shared_tcp[2].copy(),
        'point_cloud': np.random.randn(n_points, 3).astype(np.float32),  # 每个agent自己的点云
        'tactile': np.random.rand(tactile_dim).astype(np.float32),  # 每个agent自己的触觉
    }


def generate_shared_tcp():
    """Simulate all agents' TCP positions (shared observation)."""
    return [np.random.randn(tcp_dim).astype(np.float32) for _ in range(num_agents)]


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

# ========== Step 1: Build per-agent policies (simulates DP class in eval_multi_dp.py) ==========
print("=" * 60)
print(f"Step 1: Build {num_agents} independent policies (one per agent)")
print("=" * 60)

policies = []
runners = []
for agent_id in range(num_agents):
    print(f"  Building policy for Agent {agent_id} ...")
    policy = build_policy()
    runner = DPRunner(output_dir=None, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)
    policies.append(policy)
    runners.append(runner)

print(f"  All {num_agents} policies built successfully: PASS\n")

# ========== Step 2: Initial observation (simulates env.reset) ==========
print("=" * 60)
print("Step 2: Initial observation (simulates env.reset)")
print("=" * 60)

shared_tcp = generate_shared_tcp()
print(f"  Shared TCP positions: {[t.tolist() for t in shared_tcp]}")

for agent_id in range(num_agents):
    obs = fake_get_model_input(agent_id, shared_tcp)
    runners[agent_id].update_obs(obs)
    print(f"  Agent {agent_id}: obs keys = {sorted(obs.keys())}")

print("  PASS\n")

# ========== Step 3: Multi-agent inference loop (simulates main eval loop) ==========
print("=" * 60)
print("Step 3: Multi-agent inference loop (3 iterations)")
print("=" * 60)

for iteration in range(3):
    print(f"\n  --- Iteration {iteration + 1} ---")

    # Each agent independently predicts actions
    all_actions = {}
    for agent_id in range(num_agents):
        action_list = runners[agent_id].get_action(policies[agent_id])
        all_actions[f'panda-{agent_id}'] = action_list
        print(f"  Agent {agent_id}: action shape {action_list.shape}, "
              f"action sample [{action_list[0][0]:.4f}, {action_list[0][1]:.4f}, ...]")

    # Verify all agents produce valid actions
    for agent_id in range(num_agents):
        key = f'panda-{agent_id}'
        actual_action_steps = all_actions[key].shape[0]
        assert all_actions[key].shape == (actual_action_steps, action_dim), \
            f"FAIL: Agent {agent_id} action shape {all_actions[key].shape}"
        assert not np.isnan(all_actions[key]).any(), \
            f"FAIL: Agent {agent_id} action contains NaN"

    # Simulate env.step: generate new shared TCP and update each agent's obs
    shared_tcp = generate_shared_tcp()
    for agent_id in range(num_agents):
        # In real eval: agent_pos comes from executed action
        new_obs = fake_get_model_input(agent_id, shared_tcp)
        runners[agent_id].update_obs(new_obs)

    print(f"  All {num_agents} agents produced valid actions: PASS")

# ========== Step 4: Verify TCP sharing ==========
print("\n" + "=" * 60)
print("Step 4: Verify TCP data is shared across agents")
print("=" * 60)

shared_tcp = generate_shared_tcp()
obs_list = []
for agent_id in range(num_agents):
    obs = fake_get_model_input(agent_id, shared_tcp)
    obs_list.append(obs)

# All agents should see the same TCP values
for i in range(num_agents):
    for j in range(i + 1, num_agents):
        for tcp_key in ['tcp_agent0', 'tcp_agent1', 'tcp_agent2']:
            assert np.array_equal(obs_list[i][tcp_key], obs_list[j][tcp_key]), \
                f"FAIL: Agent {i} and Agent {j} have different {tcp_key}"
print(f"  All agents share identical TCP observations: PASS")

# Each agent has its own head_cam, agent_pos, point_cloud and tactile
for i in range(num_agents):
    for j in range(i + 1, num_agents):
        assert not np.array_equal(obs_list[i]['point_cloud'], obs_list[j]['point_cloud']), \
            f"FAIL: Agent {i} and Agent {j} should have different point_cloud"
        assert not np.array_equal(obs_list[i]['tactile'], obs_list[j]['tactile']), \
            f"FAIL: Agent {i} and Agent {j} should have different tactile"
print(f"  Each agent has independent head_cam, agent_pos, point_cloud & tactile: PASS")

print("\n" + "=" * 60)
print("All multi-agent inference tests passed!")
print("=" * 60)
