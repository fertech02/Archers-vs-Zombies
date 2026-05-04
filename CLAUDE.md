s# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KU Leuven ML course project (2025-2026) with three tasks:
1. **Task 1** — Single-agent RL in the Knights Archers Zombies (KAZ) visual environment
2. **Task 2** — Multi-agent Q-learning on matrix games (game theory)
3. **Task 3** — Two-archer cooperative play; final tournament submission

The environment is PettingZoo's `knights_archers_zombies_v10` with procedural visual distortion applied on top.

## Setup

```bash
pip install -r requirements.txt
# Python 3.11 or 3.12 required
```

Deployment target: `/cw/lvs/NoCsBack/vakken/H0T25A/ml-project/venv/` (pre-installed venv on departmental servers). Disk limit: **50 MB** per student.

## Common Commands

```bash
# Train PPO policy on KAZ
python train_policy.py

# Train zombie detection CNN
python zombie_detection/train.py

# Run Task 2 (game theory Q-learning)
python task2/main.py

# Evaluate an agent (100 episodes, distortion level 5)
python evaluation.py -l submission.py --episodes 100 --distortion 5

# Evaluate with visual rendering
python evaluation.py -l submission.py -s --distortion 5

# Evaluate zombie detector only
python evaluation.py -l submission.py --zombies

# Save JSONL results
python evaluation.py -l submission.py -o results.jsonl
```

### `evaluation.py` CLI flags
| Flag | Description |
|------|-------------|
| `-l FILE` | Agent module to load |
| `-s` | Enable pygame rendering |
| `--episodes N` | Number of episodes (default: 100) |
| `--seed N` | Master random seed (default: 42) |
| `--distortion N` | Visual distortion level 0–5 (default: 5) |
| `-o FILE` | Save results as JSONL |
| `--zombies` | Zombie detector evaluation only |
| `-v` | Verbose output |

## Architecture

### Data flow

```
Raw frame (1280×720 RGB)
  → visual_utils.VisualWrapper  (distortion, heat-haze, cloud overlays, levels 0–5)
  → resize to (90, 160)
  → ZombieCNN  →  detection head: (8, 5) = [confidence, x, y, w, h] normalized
                →  feature extractor: 512-dim vector fed to RLlib policy
```

### Key modules

| File / Dir | Role |
|---|---|
| `submission.py` | **Tournament entry point** — implement three interfaces here |
| `evaluation.py` | Loads submission module and runs evaluation harness |
| `utils.py` | `create_environment()` factory; `iou()` helper |
| `visual_utils.py` | `VisualWrapper` — procedural distortion pipeline |
| `train_policy.py` | Ray RLlib PPO training; saves checkpoints to `results/ppo_kaz/` |
| `zombie_detection/cnn.py` | `ZombieCNN` — backbone + detection head |
| `zombie_detection/rllib_model.py` | `KAZVisionModel` — RLlib TorchModelV2 wrapping ZombieCNN |
| `zombie_detection/preprocessing.py` | `preprocess_obs()`, `decode_detections()` |
| `zombie_detection/train.py` | Supervised training for ZombieCNN |
| `task2/` | Matrix game Q-learning (ε-greedy, Boltzmann, Lenient Boltzmann) |

### Submission interfaces (in `submission.py`)

Three classes must be implemented:
- `CustomWrapper` — wraps the environment (e.g., frame stacking, resize)
- `CustomPredictFunction` — maps observations → actions for both archers
- `CustomZombieDetectorFunction` — maps observation → list of bounding boxes

Agents can expose `ENV_SETTINGS = {"frame_stack": N, "resize_dim": (H, W), "distortion_level": N}` to influence environment construction.

### ZombieCNN architecture

- Input: `(3, 90, 160)` float32 tensor in [0, 1]
- 3 conv layers → FC(512) shared backbone
- **Detection head**: output shape `(MAX_ZOMBIES=8, 5)` — each slot: `[confidence, x, y, w, h]` (values in [0, 1])
- **Feature extractor** (`extract_features()`): returns the 512-dim backbone vector for RL use
- Pretrained checkpoint: `zombie_detection/zombie_cnn.pth`

### RLlib checkpoint loading

```python
# Pattern used in submission_example_rllib.py
checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "results/learner_group/learner/rl_module/")
```

Always use `os.path.dirname(os.path.abspath(__file__))` for model paths — absolute paths are required for tournament deployment.

## Task 2 (Game Theory)

`task2/` is self-contained. Implements independent Q-learning on four 2×2 matrix games (StagHunt, SubsidyGame, PrisonerDilemma, BiasedRockPaperScissor) with three exploration agents (EpsilonAgent, BoltzmannAgent, LenientAgent). Run with `python task2/main.py`; plots saved to `task2/results/plots/`.

## Tournament Constraints

- Submission runs on Python 3.12 departmental machines
- Model loading must complete in **< 10 seconds**
- All file references must be absolute paths (use `__file__`)
- 50 MB total disk quota
- 2 GB RAM limit for remote users
