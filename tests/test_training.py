"""
Test training pipeline with fake zarr data.
Creates a minimal zarr dataset, then runs 2 epochs of training in debug mode.

Usage:
    conda activate ADM-DP
    cd /NEW_EDS/JJ_Group/wangey2512/code/Mres_Project/ADM-DP
    python test_training.py
"""
import numpy as np
import zarr
import os
import shutil
import subprocess
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def create_fake_zarr(save_dir, num_episodes=5, steps_per_episode=20):
    """Create a minimal zarr dataset matching default_task.yaml."""
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    root = zarr.group(save_dir)
    data = root.create_group('data')
    meta = root.create_group('meta')

    total_steps = num_episodes * steps_per_episode

    # Image: (N, 3, 240, 320) uint8
    head_camera = np.random.randint(0, 255, (total_steps, 3, 240, 320), dtype=np.uint8)
    # State/agent_pos: (N, 8)
    state = np.random.randn(total_steps, 8).astype(np.float32)
    # Action: (N, 8)
    action = np.random.randn(total_steps, 8).astype(np.float32)
    # TCP positions: (N, 3) x 3 agents
    tcp_agent0 = np.random.randn(total_steps, 3).astype(np.float32)
    tcp_agent1 = np.random.randn(total_steps, 3).astype(np.float32)
    tcp_agent2 = np.random.randn(total_steps, 3).astype(np.float32)
    # Point cloud: (N, 512, 3)
    point_cloud = np.random.randn(total_steps, 512, 3).astype(np.float32)
    # Tactile: (N, 32)
    tactile = np.random.rand(total_steps, 32).astype(np.float32)

    # Episode ends
    episode_ends = np.array([(i + 1) * steps_per_episode for i in range(num_episodes)], dtype=np.int64)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

    data.create_dataset('head_camera', data=head_camera, chunks=(10, 3, 240, 320), compressor=compressor)
    data.create_dataset('state', data=state, chunks=(100, 8), dtype='float32', compressor=compressor)
    data.create_dataset('action', data=action, chunks=(100, 8), dtype='float32', compressor=compressor)
    data.create_dataset('tcp_agent0', data=tcp_agent0, chunks=(100, 3), dtype='float32', compressor=compressor)
    data.create_dataset('tcp_agent1', data=tcp_agent1, chunks=(100, 3), dtype='float32', compressor=compressor)
    data.create_dataset('tcp_agent2', data=tcp_agent2, chunks=(100, 3), dtype='float32', compressor=compressor)
    data.create_dataset('point_cloud', data=point_cloud, chunks=(10, 512, 3), dtype='float32', compressor=compressor)
    data.create_dataset('tactile', data=tactile, chunks=(100, 32), dtype='float32', compressor=compressor)
    meta.create_dataset('episode_ends', data=episode_ends, dtype='int64', compressor=compressor)

    print(f"Created fake zarr at {save_dir}")
    print(f"  Episodes: {num_episodes}, Steps/ep: {steps_per_episode}, Total: {total_steps}")
    print(f"  Keys: head_camera, state, action, tcp_agent0/1/2, point_cloud, tactile")


if __name__ == '__main__':
    zarr_path = os.path.join(project_root, 'data/zarr_data/FakeTask_Agent0_5.zarr')
    os.makedirs(os.path.join(project_root, 'data/zarr_data'), exist_ok=True)

    # Step 1: Create fake zarr
    print("=" * 60)
    print("Step 1: Creating fake zarr dataset")
    print("=" * 60)
    create_fake_zarr(zarr_path, num_episodes=5, steps_per_episode=20)

    # Step 2: Run training in debug mode (2 epochs, 3 steps each)
    print("\n" + "=" * 60)
    print("Step 2: Running training (debug mode: 2 epochs, 3 steps)")
    print("=" * 60)

    cmd = [
        sys.executable, os.path.join(project_root, 'policy', 'Diffusion-Policy', 'train.py'),
        '--config-name=robot_dp.yaml',
        'task.name=FakeTask-rf',
        f'task.dataset.zarr_path={zarr_path}',
        'training.debug=True',
        'training.seed=42',
        'training.device=cuda:0',
        'training.resume=False',
        'logging.mode=offline',
        'task.dataset.val_ratio=0.2',
        'task.dataset.max_train_episodes=4',
    ]

    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=project_root)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("Training test PASSED!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print(f"Training test FAILED (exit code {result.returncode})")
        print("=" * 60)

    # Cleanup
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
        print(f"Cleaned up {zarr_path}")
