# DeepSet v2 — Step 0 Facts File

Generated: 2026-10-01 session.  
All items verified by running commands or reading source this session.

---

## 0a. Package Versions

| Package     | Version  |
|-------------|----------|
| Python      | 3.12.13  |
| highway-env | 1.12.1   |
| gymnasium   | 1.3.0    |
| torch       | 2.14.0   |
| numpy       | 2.5.2    |

**CONFIRMED** — `pip show` + `python --version`.

---

## 0b. Installed highway_env Source Constants

| Constant | Value | Source |
|----------|-------|--------|
| `Vehicle.MAX_SPEED` | 40.0 | `highway_env/vehicle/kinematics.py` |
| `AbstractLane.DEFAULT_WIDTH` | 4 | `highway_env/road/lane.py` |
| `PERCEPTION_DISTANCE` | `5.0 * Vehicle.MAX_SPEED = 200.0` | `highway_env/envs/common/abstract.py:56` (class attr on `AbstractEnv`) |
| `MDPVehicle.DEFAULT_TARGET_SPEEDS` | `[20. 25. 30.]` | `highway_env/vehicle/controller.py` |

### KinematicObservation defaults
- `features`: uses `self.FEATURES` if not provided (class default)
- `vehicles_count`: 5
- `normalize`: True
- `absolute`: False
- `see_behind`: False
- `clip`: True
- `order`: "sorted"
- Normalization ranges (lazy, computed at first `observe()`):
  - x: `[-5*MAX_SPEED, 5*MAX_SPEED]` = [-200, 200] → rx = 200.0
  - y: `[-DEFAULT_WIDTH*n_side_lanes, DEFAULT_WIDTH*n_side_lanes]` = [-16, 16] for 4 lanes → ry = 16.0
  - vx: `[-2*MAX_SPEED, 2*MAX_SPEED]` = [-80, 80]
  - vy: `[-2*MAX_SPEED, 2*MAX_SPEED]` = [-80, 80]

### Vehicle.to_dict features
`presence, x, y, vx, vy, heading, cos_h, sin_h, cos_d, sin_d, long_off, lat_off, ang_off`

### MDPVehicle.act()
- FASTER: `speed_index = speed_to_index(speed) + 1`, clipped to `[0, len(target_speeds)-1]`
- SLOWER: `speed_index = speed_to_index(speed) - 1`, clipped

### DiscreteMetaAction indices
```
ACTIONS_ALL = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
```
Produces `Discrete(5)` action space.

### HighwayEnvFast defaults
```python
simulation_frequency: 5     # overrides HighwayEnv's 15
lanes_count: 3              # overrides HighwayEnv's 4
vehicles_count: 20          # overrides HighwayEnv's 50
duration: 30                # overrides HighwayEnv's 40
ego_spacing: 1.5            # overrides HighwayEnv's 2
```
Also disables collision check for uncontrolled vehicles.

### Default action configs for other scenarios (record only)

| Scenario | action type | target_speeds | duration | policy_freq | sim_freq |
|----------|-------------|---------------|----------|-------------|----------|
| merge-v1 | DiscreteMetaAction | [20, 25, 30] (default) | (no explicit) | 1 | 15 |
| roundabout-v1 | DiscreteMetaAction | [0, 8, 16] | 11 | 1 | 15 |
| intersection-v2 | DiscreteMetaAction (longitudinal=True, lateral=False) | [0, 4.5, 9] | 13 | 1 | 15 |

**CONFIRMED** — `inspect.getsource()` and `env.unwrapped.config`.

---

## 0c. env_config.py Key Structures

### make_optimal_env config literals ([env_config.py:953-967](file:///home/meerpi/curr_project/capstone/env_config.py#L953-L967))
```python
cfg = copy.deepcopy(DEFAULT_ENV_CONFIG)  # observation=Kinematics[presence,x,y,vx,vy], policy_freq=5, sim_freq=15, duration=100
cfg.update({
    "lanes_count": 4,
    "vehicles_density": 1.4,
    "vehicles_count": 14,
    "duration": duration,           # default 100
    "simulation_frequency": 5,      # overrides DEFAULT_ENV_CONFIG's 15!
    "collision_reward": 0.0,        # all native rewards zeroed
    "high_speed_reward": 0.0,
    "right_lane_reward": 0.0,
    "on_road_reward": 0.0,
    "lane_change_reward": 0.0,
    "normalize_reward": False,
    "reward_speed_range": [18, 30],
})
```
Then wraps with `TacticalLaneObservationWrapper` (reward_shaping=True, include_continuous_features=False by default).

### SensorTierWrapper ([env_config.py:210-236](file:///home/meerpi/curr_project/capstone/env_config.py#L210-L236))
- `gym.ObservationWrapper` subclass
- Calls `apply_sensor_tier(observation, self.tier, self.env)` in `observation()` method

### get_scaling_factors ([env_config.py:60-80](file:///home/meerpi/curr_project/capstone/env_config.py#L60-L80))
- Reads `env.unwrapped.observation_type.features_range` to compute `rx`, `ry`
- Falls back to `(200.0, 16.0)` if unavailable

### filter_full_adas ([env_config.py:83-96](file:///home/meerpi/curr_project/capstone/env_config.py#L83-L96))
- Docstring says "Full 360 degree radar, 100m range" but implementation is `return obs.copy()` — **passes everything through unmodified**
- The 100m range claim in the docstring is aspirational/misleading; no distance gate
- **CONFIRMED**: filter_full_adas applies NO filtering whatsoever

### TacticalLaneObservationWrapper reset ([env_config.py:642-679](file:///home/meerpi/curr_project/capstone/env_config.py#L642-L679))
- Calls `self.env.reset()`, **discards the returned obs** (`_, info = ...`)
- Builds obs from `self._build_tactical_obs()` which reads `unwrapped.road.vehicles`
- Returns `(obs, info)` with info keys: `action_mask`, `overtake_count`, `speed`, `cost`

### TacticalLaneObservationWrapper step ([env_config.py:681-811](file:///home/meerpi/curr_project/capstone/env_config.py#L681-L811))
- Calls `self.env.step(action)`, **discards the returned obs** (`_, base_reward, ...`)
- Rebuilds obs from `self._build_tactical_obs()`
- Computes shaped reward from scratch (when `apply_reward_shaping=True` and action space is discrete)
- Info keys: `action_mask`, `overtake_count`, `speed`, `crashed`, `cost`

### TacticalLaneObservationWrapper reward constants (defaults, [env_config.py:437-451](file:///home/meerpi/curr_project/capstone/env_config.py#L437-L451))
| Parameter | Value |
|-----------|-------|
| collision_penalty | -50.0 |
| overtake_bonus | 1.0 |
| blockage_coef | 0.35 |
| lane_change_penalty | 0.05 |
| **danger_zone** | **8.0** |
| **danger_penalty** | **0.3** |
| **jitter_penalty** | **0.12** |

**CONFIRMED** — read from source file.

---

## 0d. Legacy OptimalAgent Parameter Counts

| obs_dim | Actor | Critic | Total |
|---------|-------|--------|-------|
| 81      | 153,861 | 152,833 | **306,694** |
| 90      | 156,165 | 155,137 | 311,302 |

Architecture: 3×256 Tanh hidden, orthogonal init, actor head std=0.01, critic head std=1.0.
Separate actor and critic (no weight sharing).

**CONFIRMED** — instantiated `OptimalAgent` and counted `sum(p.numel())`.

---

## 0e. Git History: simulation_frequency and Checkpoint Dating

### simulation_frequency=5 introduction
```
git log -S'"simulation_frequency": 5' --oneline -- env_config.py
→ a679bb9 feat: add optimal tactical overtaker PPO, anti-jitter regularization, ...
```
Commit date: **Tue Sep 8 00:35:07 2026 +0530**

### Checkpoint dating
```
git log --follow --format='%h %ad' -- models/ppo_optimal_overtaker_4lane_best.pt
→ a679bb9 Tue Sep 8 00:35:07 2026 +0530
```

**Both in the same commit** (a679bb9). The simulation_frequency=5 change and the checkpoint were introduced together. The checkpoint was trained WITH simulation_frequency=5.

### Checkpoint args
```python
ckpt['args'] = {
    'lanes': 4, 'density': 1.4, 'vehicles': 14, 'frame_stack': 3, 'seed': 42,
    'total_timesteps': 1000000, 'learning_rate': 0.00025, 'lr_end': 1e-05,
    'num_envs': 8, 'num_steps': 512, 'gamma': 0.995, 'gae_lambda': 0.95,
    'num_minibatches': 4, 'update_epochs': 4, 'clip_coef': 0.2,
    'ent_coef': 0.02, 'ent_end': 0.005, 'vf_coef': 0.5, 'max_grad_norm': 0.5,
    'cuda': True, 'eval_only': False, 'model_path': None,
    'eval_episodes': 30, 'eval_seed_start': 2000
}
```
**Contains CLI args only — no env config.**

**CONFIRMED** — `torch.load` and key inspection.

---

## 0f. deepset_encoder.py Keep/Discard Analysis

### DISCARD (wrong for v2)
| Item | Line | Issue |
|------|------|-------|
| danger_zone=15.0 | L264 | Legacy is 8.0 — **MISMATCH** |
| danger_penalty=0.5 | L265 | Legacy is 0.3 — **MISMATCH** |
| jitter_penalty=0.02 | L266 | Legacy is 0.12 — **MISMATCH** |
| No sensor tier filter | entire file | No SensorTierWrapper or apply_sensor_tier usage |
| No tier tests | tests/test_deepset_encoder.py | No tier-switching tests |
| Native rewards NOT zeroed | L468-486 (make_deepset_env config) | Missing collision_reward=0, high_speed_reward=0 etc — double-counting rewards |
| Shared encoder for actor/critic | L176 | Single `self.encoder` used by both — legacy has separate networks |
| DeepSetEncoder architecture | L47-161 | Separate ego_mlp (32-dim) + phi (not ego-conditioned) + rho — differs from v2 spec |

### KEEP (reusable patterns)
| Item | Line | Notes |
|------|------|-------|
| `layer_init` | L38-44 | Orthogonal init + constant bias — identical pattern |
| Permutation-invariant structure | L102-161 | Core φ/pool/ρ pattern correct |
| Masked mean + max pooling | L140-157 | Same approach (v2 uses sum/N instead of mean) |
| Action masking with -1e8 | L222-226 | Identical to legacy OptimalAgent |
| `get_action_and_value` signature | L211-236 | Compatible with PPO loop |

### MISSING (v2 needs, not in deepset_encoder.py)
- Ego-conditioned phi: `phi(x_veh ; ego)` (conditioning ego features into per-vehicle MLP)
- Separate actor/critic encoders (no weight sharing)
- 3×256 Tanh rho backbone (matching legacy depth)
- Sensor tier integration
- Reward-equivalence with legacy
- Range gate (|x_rel| > range_m masking)
- Physical-unit feature scaling (x/100, y/4, vx/30, vy/30)

**CONFIRMED** — read from deepset_encoder.py and compared with env_config.py.

---

## 0g. Golden Legacy Baseline

```bash
# Generated: scratch/golden_legacy.npz
# Seeds: 4000-4004, 100 steps each, seeded random valid actions
# Frame stack: 0 (raw 27-dim obs)
# Env: make_optimal_env(frame_stack_k=0) with default flags
```

File contents:
- `obs`: 5 arrays of shape (101, 27) — 101 = reset obs + 100 step obs
- `rewards`: 5 arrays of shape (100,)
- `terminated`: 5 arrays of shape (100,)
- `truncated`: 5 arrays of shape (100,)
- `action_mask`: 5 arrays of shape (101, 5)

**CONFIRMED** — generated and verified shapes.

---

## BACKGROUND Verification

| Claim | Status | Evidence |
|-------|--------|----------|
| KinematicObservation features presence/x/y/vx/vy | **CONFIRMED** | DEFAULT_OBSERVATION_CONFIG in env_config.py:27 |
| Normalization anisotropic (x ±5*MAX_SPEED, y ±WIDTH*n_side_lanes, vx/vy ±2*MAX_SPEED) | **CONFIRMED** | normalize_obs source, verified rx=200, ry=16 |
| Missing rows zero-filled | **CONFIRMED** | KinematicObservation.observe source |
| Ego row ABSOLUTE, others RELATIVE (absolute=False) | **CONFIRMED** | observe source: `origin = self.observer_vehicle if not self.absolute else None` |
| MDPVehicle.DEFAULT_TARGET_SPEEDS = [20,25,30] | **CONFIRMED** | `[20. 25. 30.]` numpy array |
| highway-v0 defaults: duration 40, policy_freq 1, sim_freq 15 | **CONFIRMED** | HighwayEnv.default_config (no explicit policy_freq override → parent's 1), sim_freq 15 |
| highway-fast-v0 defaults: sim_freq 5, lanes 3, vehicles 20, duration 30 | **CONFIRMED** | HighwayEnvFast.default_config source |
| danger_zone 8.0 / jitter 0.12 in legacy | **CONFIRMED** | env_config.py:444-446 |
| danger_zone 15.0 / jitter 0.02 in deepset_encoder.py | **CONFIRMED** | deepset_encoder.py:264-266 |
| Report claim "15m proximity zone / -0.02 jitter" | **CONFIRMED locally** | These are the deepset_encoder.py values, NOT the legacy values |
| make_optimal_env never applies SensorTierWrapper | **CONFIRMED** | env_config.py:953-992, no SensorTierWrapper call |
| Legacy scales MAX_LEAD_DIST=MAX_LAG_DIST=100m, SPEED_NORM=30 | **CONFIRMED** | env_config.py:425-431 |
| OptimalAgent: separate actor/critic, 3×256 Tanh, orthogonal init, actor head std 0.01, critic head std 1.0 | **CONFIRMED** | train_optimal_overtaker.py:174-217 |
| Masking via logits -1e8 | **CONFIRMED** | train_optimal_overtaker.py:240-246 |
| Assumed: sum/14 + max pooling, phi width 64 | **UNVERIFIED** | These are design choices for v2, not verified claims |
| Assumed: legacy ~306.7k params | **CONFIRMED** at 306,694 |

---

## Models Directory SHA256 Baseline

```
07b097de...  models/ppo_highway_full_adas_seed101.pt
3d0cb21f...  models/rainbow_highway_full_adas_seed101.pt
746faea3...  models/ppo_lstm_highway_front_only_seed101.pt
90b87214...  models/ppo_optimal_overtaker_4lane_best.pt
a775eaf3...  models/ppo_optimal_overtaker_4lane.pt
d5f1cb9a...  models/rainbow_highway_front_only_seed101.pt
df6bcfc5...  models/ppo_highway_front_only_seed101.pt
e2759653...  models/ppo_lstm_highway_full_adas_seed101.pt
e37640ab...  models/test_decoupled.pt
```
