"""
Test dataset TCP auto-detection and normalization for 2-arm and 3-arm.
Creates dummy zarr files and verifies the dataset + normalizer work correctly.

Usage:
    conda activate ADM-DP
    cd /NEW_EDS/JJ_Group/wangey2512/code/Mres_Project/RoboFactory
    python test_normalizer.py
"""
import numpy as np
import zarr
import torch
import shutil
import os
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'policy', 'Diffusion-Policy'))

from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset


def create_dummy_zarr(zarr_path, num_agents, num_frames=200, num_episodes=4):
    """Create a dummy zarr dataset with the given number of TCP agents."""
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)

    root = zarr.open(zarr_path, mode='w')
    data = root.create_group('data')
    meta = root.create_group('meta')

    H, W = 240, 320
    state_dim = 8
    action_dim = 8

    # head_camera: (N, H, W, 3) uint8
    data.create_dataset('head_camera', data=np.random.randint(0, 255, (num_frames, H, W, 3), dtype=np.uint8))
    # state: (N, state_dim)
    data.create_dataset('state', data=np.random.randn(num_frames, state_dim).astype(np.float32))
    # action: (N, action_dim)
    data.create_dataset('action', data=np.random.randn(num_frames, action_dim).astype(np.float32))

    # tcp_agent0, tcp_agent1, ... tcp_agent{num_agents-1}
    for i in range(num_agents):
        data.create_dataset(f'tcp_agent{i}', data=np.random.randn(num_frames, 3).astype(np.float32))

    # episode_ends
    frames_per_ep = num_frames // num_episodes
    episode_ends = np.array([frames_per_ep * (i + 1) for i in range(num_episodes)], dtype=np.int64)
    meta.create_dataset('episode_ends', data=episode_ends)

    print(f"  Created zarr: {zarr_path}")
    print(f"  Keys in data/: {sorted(data.keys())}")
    print(f"  Frames: {num_frames}, Episodes: {num_episodes}")


def test_dataset(zarr_path, expected_num_agents):
    """Test dataset loading, tcp key detection, normalizer, and postprocess."""
    print(f"\n  Loading dataset from {zarr_path} ...")
    dataset = RobotImageDataset(
        zarr_path=zarr_path,
        horizon=8,
        pad_before=2,
        pad_after=7,
        seed=42,
        val_ratio=0.02,
        batch_size=4,
    )

    # Check tcp_keys auto-detection
    print(f"  Detected tcp_keys: {dataset.tcp_keys}")
    expected_keys = [f'tcp_agent{i}' for i in range(expected_num_agents)]
    assert dataset.tcp_keys == expected_keys, \
        f"FAIL: expected {expected_keys}, got {dataset.tcp_keys}"
    print(f"  tcp_keys detection: PASS")

    # Check normalizer
    normalizer = dataset.get_normalizer()
    print(f"  Normalizer keys: {sorted(normalizer.params_dict.keys())}")
    for key in expected_keys:
        assert key in normalizer.params_dict, f"FAIL: {key} not in normalizer"
    print(f"  Normalizer contains all tcp keys: PASS")

    # Test normalization round-trip for each tcp key
    for key in expected_keys:
        dummy = torch.randn(10, 3)
        normed = normalizer[key].normalize(dummy)
        unnormed = normalizer[key].unnormalize(normed)
        diff = (dummy - unnormed).abs().max().item()
        assert diff < 1e-5, f"FAIL: round-trip error for {key}: {diff}"
    print(f"  Normalization round-trip: PASS")

    # Test postprocess (simulates a training batch)
    sample = dataset[0]  # get one sample to check structure
    # Simulate batch from dataloader
    batch_samples = {}
    for k, v in dataset.sampler.replay_buffer.items():
        shape = (4, 8) + v.shape[1:]  # batch=4, horizon=8
        batch_samples[k] = torch.randn(*shape)
    # Make head_camera uint8-like
    batch_samples['head_camera'] = torch.randint(0, 255, (4, 8, 240, 320, 3)).float()

    result = dataset.postprocess(batch_samples, device='cpu')
    print(f"  postprocess obs keys: {sorted(result['obs'].keys())}")
    for key in expected_keys:
        assert key in result['obs'], f"FAIL: {key} not in postprocess output"
        assert result['obs'][key].shape == (4, 8, 3), \
            f"FAIL: {key} shape {result['obs'][key].shape}, expected (4, 8, 3)"
    print(f"  postprocess TCP shapes: PASS")


# ========== Run Tests ==========
print("=" * 60)
print("Test 1: 2-arm (tcp_agent0, tcp_agent1)")
print("=" * 60)
zarr_2arm = '/tmp/test_2arm.zarr'
create_dummy_zarr(zarr_2arm, num_agents=2)
test_dataset(zarr_2arm, expected_num_agents=2)

print("\n" + "=" * 60)
print("Test 2: 3-arm (tcp_agent0, tcp_agent1, tcp_agent2)")
print("=" * 60)
zarr_3arm = '/tmp/test_3arm.zarr'
create_dummy_zarr(zarr_3arm, num_agents=3)
test_dataset(zarr_3arm, expected_num_agents=3)

# Cleanup
shutil.rmtree(zarr_2arm, ignore_errors=True)
shutil.rmtree(zarr_3arm, ignore_errors=True)

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
