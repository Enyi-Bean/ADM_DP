<div align="center">
<h1>ADM-DP: Adaptive Dynamic Modality Diffusion Policy through Vision-Tactile-Graph Fusion<br>for Multi-Agent Manipulation</h1>
</div>

## Overview
ADM-DP is an Adaptive Dynamic Modality Diffusion Policy that adaptively fuses vision, tactile, and graph modalities for multi-arm manipulation. Each arm trains independently with its own observations while sharing TCP positions for coordination.

**Key Design:**
- **Decoupled Training / Coupled Inference**: independent per-arm policies, coordinated through shared TCP at inference
- **AMAM Fusion**: adaptive attention over vision, tactile, and graph modalities (e.g., suppresses tactile during approach, amplifies during grasp)
- **Multi-Modal**: RGB + PointCloud FiLM + GNN (TCP) + Tactile Encoder

## Installation
```bash
git clone https://github.com/xxx/ADM_DP.git
conda create -n ADM_DP python=3.9
conda activate ADM_DP
cd ADM_DP
pip install -r requirements.txt
```

## Data Preparation

ADM-DP expects training data in `.zarr` format. Each arm's data should be stored separately.

### Zarr Data Structure

```
data/zarr_data/{TaskName}_Agent{id}_{num}.zarr/
├── data/
│   ├── head_camera      (N, 3, H, W)       uint8      RGB image (CHW format)
│   ├── state            (N, D_action)       float32    joint positions (agent_pos)
│   ├── action           (N, D_action)       float32    joint action labels
│   ├── tcp_agent0       (N, 3)              float32    TCP position of arm 0
│   ├── tcp_agent1       (N, 3)              float32    TCP position of arm 1
│   ├── ...                                             (one key per arm)
│   ├── point_cloud      (N, N_points, 3)    float32    point cloud (xyz)
│   ├── tactile          (N, 32)             float32    FSR tactile readings (2 fingers x 4x4)
│   └── language_emb     (N, D_emb)          float32    [Optional] CLIP text embedding
└── meta/
    └── episode_ends     (E,)                int64      cumulative frame index per episode
```

- `N`: total number of frames across all episodes
- `E`: number of episodes
- **TCP keys are shared**: every arm's dataset contains the same `tcp_agent*` fields from all arms
- **Other observations are per-arm**: each arm only sees its own `head_camera`, `point_cloud`, `tactile`
- `language_emb`: optional, pre-computed CLIP text embeddings for language-conditioned multi-task settings

## Configuration: 2-Arm vs. 3-Arm

The default configuration supports 3 arms. To adapt for a different number of arms, modify the following:

**`policy/Diffusion-Policy/diffusion_policy/config/task/default_task.yaml`**
```yaml
obs:
  # Add or remove tcp_agent keys to match your setup
  tcp_agent0:
    shape: [3]
    type: tcp
  tcp_agent1:
    shape: [3]
    type: tcp
  # tcp_agent2:       # uncomment for 3-arm
  #   shape: [3]
  #   type: tcp
```

**`policy/Diffusion-Policy/diffusion_policy/config/robot_dp.yaml`**
```yaml
obs_encoder:
  graph_model:
    num_agents: 2     # set to match the number of arms
```

## Training

Each arm is trained independently with its own dataset:
```bash
bash policy/Diffusion-Policy/train.sh <task_name> <num_episodes> <agent_id> <seed> <gpu_id>
```

Checkpoints are saved to `checkpoints/{TaskName}_Agent{id}_{num}/`.

### Quick Verification

To verify the training pipeline works correctly:
```bash
python tests/test_training.py
```

## Evaluation

After training, evaluate all arms together. Each arm loads its own checkpoint, and they coordinate through shared TCP observations at inference time:
```bash
bash policy/Diffusion-Policy/eval_multi.sh <config_path> <num_episodes> <checkpoint_epoch> <debug_mode> <task_name>
```

### Quick Verification

To verify the model architecture and inference pipeline:
```bash
python tests/test_full_pipeline.py
```
