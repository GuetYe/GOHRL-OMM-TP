# GOHRL-OMM-TP: Goal-Oriented Hierarchical Reinforcement Learning for Overlay Multicast With Transformer-Based Traffic Prediction

This repository provides the implementation of **GOHRL-OMM-TP**, an intelligent overlay multicast routing framework that integrates **SDN-enabled global network state awareness**, **goal-oriented hierarchical reinforcement learning**, and **Transformer-based traffic prediction**. The method is designed for dynamic overlay multicast scenarios where multicast members may frequently join or leave, and where link bandwidth, delay, and loss may vary over time.

The implementation contains two main parts:

1. **Traffic prediction module**: predicts future link-state information, especially link bandwidth, from historical network snapshots.
2. **Goal-oriented hierarchical reinforcement learning module**: constructs and maintains an overlay multicast tree through a two-level decision process, where the high-level policy selects source-destination node pairs and the low-level policy plans constrained routing paths.

---

## 1. Main Functions

The repository implements the following functions:

- Parse topology and link-state data from XML and pickle-based graph snapshots.
- Build network-state matrices containing QoS information such as bandwidth, delay, loss, and path-related states.
- Train a Transformer-based prediction model using historical link-state sequences.
- Use predicted link bandwidth to update the routing environment before reinforcement learning decisions.
- Select overlay multicast subgoals through an upper-level actor-critic policy.
- Generate next-hop routing actions through a lower-level actor-critic policy under neighbor and path constraints.
- Maintain replay buffers for both high-level and low-level training.
- Save training rewards, average rewards, and constructed multicast paths.
- Visualize reward curves and QoS comparison results.

---

## 2. Repository Structure

```text
GOHRL-OMM-TP/
├── config.py                         # Global configuration for topology, data paths, rewards, and training settings
├── config/
│   ├── external_config.yaml           # Hyperparameters for the upper-level policy network
│   └── internal_config.yaml           # Hyperparameters for the lower-level policy network
├── dataset.py                         # Topology parsing, graph loading, neighbor extraction, and shortest-hop calculation
├── dataset_seq.py                     # Sequence dataset construction for traffic prediction
├── net.py                             # Neural network definitions for high-level and low-level actor-critic models
├── UpperAC.py                         # Upper-level actor-critic agent for source-destination subgoal selection
├── UnderAC.py                         # Lower-level actor-critic agent and joint training process
├── replay_buffer.py                   # Replay buffers for high-level and low-level agents
├── parse_config.py                    # YAML configuration parser and training-parameter saving tools
├── utils.py                           # Common utility functions such as advantage calculation and model saving
├── train_main.py                      # Earlier training script; the current joint training process is mainly in UnderAC.py
├── predict/
│   ├── transform/
│   │   ├── transformer_model.py        # Transformer prediction model
│   │   ├── transformer_train.py        # Transformer training script
│   │   ├── transformer_test.py         # Transformer testing script
│   │   ├── dataset.py                  # Prediction dataset loader
│   │   ├── config.py                   # Prediction data path configuration
│   │   ├── utils.py                    # Prediction metrics and plotting tools
│   │   └── requirements.txt            # Core package versions used by the prediction module
│   ├── lstm/                           # LSTM prediction baseline
│   ├── gru/                            # GRU prediction baseline
│   ├── process_data.py                 # Data processing script for prediction results
│   └── draw.py                         # Prediction-result plotting script
├── draw/
│   ├── node14/                         # Plotting and result-analysis scripts for 14-node topology
│   └── node21/                         # Plotting and result-analysis scripts for 21-node topology
└── train_data/                         # Saved training logs, reward curves, and path records
```

> Note: The training scripts call the routing environment module `env.py` and the plotting utility module `draw_tools.py`. Before running the complete training workflow, make sure these files are included in the repository root or are available in the Python path.

---

## 3. Method Overview

### 3.1 SDN-Enabled Overlay Multicast State Construction

The network topology and dynamic link-state snapshots are represented as NetworkX graphs. Each edge may contain QoS-related attributes such as:

- `free_bw`: available bandwidth,
- `delay`: link delay,
- `loss`: packet loss rate,
- `used_bw`: used bandwidth,
- `pkt_err`: packet error count,
- `pkt_drop`: packet drop count,
- `distance`: hop or distance-related information.

The topology information is parsed by `dataset.py`. The function `parse_topo_links_info()` reads `links_info.xml` and builds the basic graph structure. The functions `get_node_neighbors()` and `get_all_pairs_dijkstra_path_length()` are then used to construct neighbor constraints and shortest-hop information for action masking.

The dynamic graph snapshots are loaded from pickle files through `read_pkl()` or `read_pickle()`. These snapshots are used as the time-varying network states for both prediction and routing optimization.

---

### 3.2 Transformer-Based Traffic Prediction

The traffic prediction module is located in:

```text
predict/transform/
```

The main model is implemented in:

```text
predict/transform/transformer_model.py
```

The Transformer uses historical link-state sequences as input and predicts the future link bandwidth vector. In the current implementation, the prediction dataset mainly extracts `free_bw` from each graph snapshot and forms a sequence-learning dataset.

The training script is:

```text
predict/transform/transformer_train.py
```

The default training process is:

1. Read a sequence of graph snapshots from the configured dataset path.
2. Extract bandwidth features from each snapshot.
3. Build fixed-length sequence samples with a sliding window.
4. Normalize the input and target tensors.
5. Train the Transformer using mean squared error loss.
6. Save model weights after each epoch.

The saved prediction model can then be loaded by the reinforcement learning training process. In `UnderAC.py`, the trained Transformer is used to predict future bandwidth states, and the predicted values are used to update the routing environment through `env.update_bw(outputs)`.

---

### 3.3 Goal-Oriented Hierarchical Reinforcement Learning

The reinforcement learning framework is divided into two levels.

#### High-Level Policy: Source-Destination Pair Selection

The high-level policy is implemented in:

```text
UpperAC.py
```

Its role is to select a source-destination node pair as the current subgoal for overlay multicast tree construction. The source node is selected from the current multicast tree or source set, while the destination node is selected from the remaining multicast member set.

The high-level policy uses an actor-critic structure:

- `PPOPolicyNet` in `net.py` outputs action logits for node selection.
- `PPOValueNet` estimates the value of the current global multicast state.
- `UpperAcReplayBuffer` stores high-level transitions.

Invalid actions are masked according to the current source-node set and destination-node set, so that the high-level agent only selects feasible subgoals.

#### Low-Level Policy: Constrained Path Planning

The low-level policy is implemented in:

```text
UnderAC.py
```

After a source-destination pair is selected by the high-level policy, the low-level agent constructs the routing path hop by hop. At each step, the action corresponds to selecting the next-hop node.

The low-level policy uses:

- `ACPolicyNet` in `net.py` as the actor network,
- `ACValueNet` in `net.py` as the critic network,
- `UnderAcReplayBuffer` for low-level transition storage.

The low-level action space is constrained by neighbor information and shortest-hop information. Specifically, the candidate next-hop nodes are restricted to neighbors that move closer to the destination node. This reduces the invalid action space and helps avoid routing loops and inefficient exploration.

---

### 3.4 Joint Training Procedure

The main joint training logic is implemented in the `__main__` section of:

```text
UnderAC.py
```

The overall process is as follows:

1. Initialize the high-level actor-critic agent using `config/external_config.yaml`.
2. Initialize the low-level actor-critic agent using `config/internal_config.yaml`.
3. Initialize the overlay multicast routing environment.
4. Load the trained Transformer prediction model.
5. Build the sequence dataset for traffic prediction.
6. For each training episode:
   - Read one dynamic graph snapshot.
   - Reset the routing environment.
   - Predict future bandwidth using the Transformer model.
   - Update the environment with predicted bandwidth.
   - Use the high-level policy to select a source-destination subgoal.
   - Use the low-level policy to construct the corresponding path.
   - Update the multicast source set and destination set after a subgoal is completed.
   - Store low-level and high-level transitions into replay buffers.
   - Update actor and critic networks when replay buffers have enough samples.
   - Record rewards and path information.
7. Save reward curves, average reward curves, and constructed path records.

This process allows the routing policy to jointly use current network state and predicted future traffic state, improving proactive routing adaptability in dynamic overlay multicast scenarios.

---

## 4. Data Preparation

The implementation expects topology and dynamic graph data to be arranged in a structure similar to the following:

```text
DataSet/
└── node21/
    ├── links_info.xml
    └── weight_pickle/
        ├── 1-xxxx.pkl
        ├── 2-xxxx.pkl
        ├── 3-xxxx.pkl
        └── ...
```

The XML file provides the static topology and link attributes. The pickle files store time-varying graph snapshots.

The default paths are configured in `config.py`:

```python
LINKS_INFO = WORK_DIR / "Overlay-Multicast-predict-RL/DataSet/node21/links_info.xml"
PICKLE_PATH = WORK_DIR / "Overlay-Multicast-predict-RL/DataSet/node21/weight_pickle"
```

For a new local environment, modify these paths according to the actual location of the dataset.

The Transformer prediction module also has its own dataset path in:

```text
predict/transform/config.py
```

```python
DATASET_PATH = WORK_DIR / 'DataSet/node21/weight_pickle'
```

Make sure the path points to the same dynamic graph snapshot directory.

---

## 5. Environment Setup

### 5.1 Recommended Python Environment

The code was developed with PyTorch and NetworkX. The core package versions used by the prediction module are listed in `predict/transform/requirements.txt`:

```text
matplotlib==3.2.2
networkx==2.8.8
numpy==1.22.4
torch==1.11.0+cu113
```

Additional packages used by the whole project include:

```text
PyYAML
pandas
```

### 5.2 Create Environment

A typical installation process is:

```bash
conda create -n gohrl_omm_tp python=3.8 -y
conda activate gohrl_omm_tp
pip install networkx==2.8.8 numpy==1.22.4 matplotlib==3.2.2 PyYAML pandas
```

Install PyTorch according to the CUDA version of the machine. For example, for CUDA 11.3:

```bash
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

For CPU-only execution, install a CPU version of PyTorch instead.

---

## 6. How to Run

### 6.1 Train the Transformer Prediction Model

Enter the Transformer prediction directory:

```bash
cd predict/transform
```

Check the dataset path in:

```text
config.py
```

Then run:

```bash
python transformer_train.py
```

The training script saves the model weights under a time-stamped directory, such as:

```text
predict/transform/model_weight/YYYY_MM_DD_HH_MM_SS/
```

The loss curve is saved under:

```text
predict/transform/loss_graph/
```

### 6.2 Test the Transformer Prediction Model

Update the model weight path in `transformer_test.py`, then run:

```bash
python transformer_test.py
```

This script loads a saved Transformer model and compares predicted link states with the target sequence.

### 6.3 Train the GOHRL-OMM-TP Routing Policy

Return to the repository root:

```bash
cd ../../
```

Check the following settings before running:

- `config.py`: topology path, pickle dataset path, node number, source node, destination nodes, training range.
- `config/external_config.yaml`: high-level actor-critic hyperparameters.
- `config/internal_config.yaml`: low-level actor-critic hyperparameters.
- `UnderAC.py`: Transformer model weight path.

Then run:

```bash
python UnderAC.py
```

During training, the script will:

1. Load dynamic graph snapshots.
2. Predict future bandwidth values.
3. Update the routing environment.
4. Select source-destination subgoals.
5. Plan paths hop by hop.
6. Update both high-level and low-level actor-critic networks.
7. Save reward and path results.

### 6.4 Analyze and Plot Results

The result-analysis scripts are located in:

```text
draw/node14/
draw/node21/
```

Training results are saved in:

```text
train_data/
```

The typical saved files include:

```text
rewards.csv
average_reward.csv
all_paths.txt
```

These files can be used to draw reward curves and compare QoS metrics such as bandwidth, delay, and packet loss.

---

## 7. Key Configuration Items

### 7.1 Global Configuration

`config.py` controls the main routing and training parameters:

| Parameter                                      | Description                                                  |
| ---------------------------------------------- | ------------------------------------------------------------ |
| `DEVICE`                                       | Use GPU if CUDA is available; otherwise use CPU              |
| `LINKS_INFO`                                   | Path of the static topology XML file                         |
| `PICKLE_PATH`                                  | Path of dynamic graph snapshot files                         |
| `STATE_NUM`                                    | Number of QoS/state channels                                 |
| `ACTION_NUM`                                   | Number of nodes in the topology                              |
| `PARAM_WEIGHT`                                 | Weights used in reward calculation                           |
| `INTERNAL_REWARD`                              | Reward values for low-level path planning                    |
| `EXTERNAL_REWARD`                              | Reward values passed from the lower level to the upper level |
| `src_node`                                     | Initial source node                                          |
| `dst_lst`                                      | Multicast destination node list                              |
| `episodes`                                     | Number of training episodes                                  |
| `picture_start`, `picture_end`, `picture_step` | Range and step for reading graph snapshots                   |

### 7.2 High-Level Policy Configuration

`config/external_config.yaml` controls the high-level agent:

```yaml
policy_lr: 0.005
value_lr: 0.005
reward_decay: 0.9
batch_size: 128
memory_capacity: 500
```

### 7.3 Low-Level Policy Configuration

`config/internal_config.yaml` controls the low-level agent:

```yaml
policy_lr: 0.0005
value_lr: 0.0008
reward_decay: 0.9
batch_size: 128
memory_capacity: 500
```

---

## 8. Implementation Flow

The complete implementation flow can be summarized as follows:

```text
Topology XML + dynamic graph snapshots
                │
                ▼
       Network-state parsing
                │
                ▼
 Historical link-state sequence construction
                │
                ▼
       Transformer traffic prediction
                │
                ▼
 Predicted bandwidth updates routing environment
                │
                ▼
 High-level policy selects source-destination subgoal
                │
                ▼
 Low-level policy plans next-hop path under constraints
                │
                ▼
 Multicast tree is expanded and updated
                │
                ▼
 Rewards and paths are stored for training and analysis
```

---

## 9. Output Files

The training process may produce:

```text
train_data/
├── rewards.csv
├── average_reward.csv
├── all_paths.txt
└── generated figures
```

The prediction process may produce:

```text
predict/transform/
├── model_weight/
├── loss_graph/
├── performance_graph/
└── result/
```

These outputs can be used to verify the training convergence, path-construction behavior, and prediction performance.

---

## 10. Reproducibility Notes

To reproduce the experiments, please check the following items:

1. Confirm that `links_info.xml` and `weight_pickle/` are correctly placed.
2. Confirm that `ACTION_NUM` matches the number of topology nodes.
3. For 14-node and 21-node experiments, adjust the fully connected layer dimensions in `net.py` if necessary.
4. Confirm that the Transformer weight path in `UnderAC.py` points to a valid `.pth` file.
5. Confirm that the environment module provides the following methods or attributes:
   - `Environment()`
   - `update_pkl_graph()`
   - `reset()`
   - `step()`
   - `subgoal_map_inter_state()`
   - `update_bw()`
   - `Mt`
   - `combined_normal_matrix`
   - `internal_state_matrix`
   - `src_lst`
   - `dst_lst`
6. Use NetworkX 2.x, because the code uses `networkx.read_gpickle()`.

---

## 11. Suggested Public-Release Checklist

Before publishing the repository, it is recommended to check the following:

- Remove local absolute paths and replace them with relative paths or configuration arguments.
- Remove cache files such as `__pycache__/` and `*.pyc`.
- Avoid uploading unnecessary intermediate training weights unless they are required for reproduction.
- Add a `.gitignore` file for cache files, model checkpoints, logs, and large datasets.
- Make sure no private information, local machine paths, account names, or personal contact information remains in the code comments.
- Include a small example dataset or explain how to obtain the full dataset.
- Provide the exact command used to reproduce the main experiment.

A typical `.gitignore` may include:

```gitignore
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.venv/
venv/
.idea/
.vscode/
*.log
*.tmp
.DS_Store
Thumbs.db
model_weight/
checkpoints/
*.pth
*.pt
```

If model weights are needed for reproducibility, keep only the final selected checkpoint and document its purpose clearly.

---

## 12. Project Summary

GOHRL-OMM-TP implements a proactive intelligent routing framework for overlay multicast. It combines the global state-awareness capability of SDN, the prediction capability of Transformer models, and the structural decomposition capability of hierarchical reinforcement learning. Through high-level source-destination subgoal selection and low-level constrained path planning, the method aims to reduce action-space complexity, improve training stability, and enhance overlay multicast adaptability under dynamic network states.
