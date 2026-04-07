# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CAS4160 Homework 3: Q-Learning (DQN and Double DQN) implementation for RL with Gymnasium environments (CartPole, LunarLander, Atari BankHeist).

## Environment Setup

```bash
# Install dependencies (uses uv)
uv sync

# Or with pip
pip install -e .
```

Python 3.10 is used (see `.python-version`). Key deps: PyTorch 2.6, Gymnasium 1.0, ale-py 0.10.2.

## Running Training

The script must be run from `cas4160/scripts/` (or with that directory on the path) because `scripting_utils.py` is imported without a package prefix:

```bash
cd cas4160/scripts
python run_hw3.py -cfg ../../experiments/dqn/cartpole.yaml --seed 1
python run_hw3.py -cfg ../../experiments/dqn/bankheist.yaml --seed 1
python run_hw3.py -cfg ../../experiments/dqn/bankheist_ddqn.yaml --seed 1
```

Logs go to `data/` (TensorBoard). View with:
```bash
tensorboard --logdir data/
```

## Architecture

### Config system
YAML files in `experiments/dqn/` specify a `base_config` (`dqn_basic` or `dqn_atari`) plus overrides. `scripting_utils.make_config()` loads the YAML, pops `base_config`, and calls the matching factory function in `cas4160/env_configs/`. The factory returns a flat dict with keys: `agent_kwargs`, `make_critic`, `make_optimizer`, `make_lr_schedule`, `exploration_schedule`, `make_env`, `total_steps`, `batch_size`, `learning_starts`, `log_name`.

### Training loop (`run_hw3.py`)
`run_training_loop` drives everything: creates envs, instantiates `DQNAgent`, picks replay buffer type, then loops for `total_steps`:
1. Get action via `agent.get_action(obs, epsilon)` (epsilon from `exploration_schedule`)
2. Step env, insert transition into replay buffer
3. After `learning_starts` steps: sample batch → `agent.update(...)` → log
4. Periodically evaluate on `eval_env` and log `eval_return`

### Agent (`cas4160/agents/dqn_agent.py`)
`DQNAgent(nn.Module)` holds `critic`, `target_critic`, optimizer, and lr scheduler. Key methods:
- `get_action(obs, epsilon)`: epsilon-greedy using `self.critic`
- `update_critic(obs, action, reward, next_obs, done)`: computes TD targets with `target_critic`, trains `critic` with MSE loss, clips gradients if `clip_grad_norm` is set
- `update_target_critic()`: hard copy of critic weights to target
- `update(...)`: calls `update_critic`, then `update_target_critic` every `target_update_period` steps

### Double DQN
Controlled by `use_double_q` flag. In `update_critic`, when `use_double_q=True`: select next action using online `critic` (argmax), but evaluate it using `target_critic`. When `False`: both select and evaluate using `target_critic`.

### Replay buffers (`cas4160/infrastructure/replay_buffer.py`)
- `ReplayBuffer`: standard; stores full `(obs, action, reward, next_obs, done)` tuples
- `MemoryEfficientReplayBuffer`: for stacked-frame Atari observations; stores individual frames and indices. Requires calling `on_reset(obs[-1])` at episode start, then `insert(action, reward, next_obs[-1], done)` per step (no `observation` arg — it was already stored)

### `done` flag semantics
Store `terminated` (not `terminated or truncated`) in the replay buffer. Truncated episodes should not bootstrap with `done=True` for TD updates. The env is reset in both cases, but only `terminated` is stored as `done`.

### Atari preprocessing
`wrap_deepmind()` applies: AtariPreprocessing (grayscale, 84×84, frame-skip=4, noop_max=30) → FrameStack(4). Observations are `uint8` shape `(4, 84, 84)`. `PreprocessAtari` module normalizes to `[0,1]` inside the CNN.

### Adding new experiments
Create a YAML in `experiments/dqn/` with `base_config: dqn_basic` or `dqn_atari` plus any overrides matching the factory function's keyword arguments (see `dqn_basic_config.py` or `dqn_atari_config.py` for available params).
