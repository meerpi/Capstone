# DeepSet v2 — Implementation & Verification Report (GATE STOP)

**Date**: 2026-10-01  
**Repo**: github.com/meerpi/Capstone  
**Task**: DeepSet v2 feedforward, sensor-tier-aware, reward-matched implementation and unit test suite for `highway-fast-v0`.  
**Hard Rules Status**: FULLY COMPLIED. Zero edits/writes to `models/` (all hashes verified bit-identical). Zero edits to `env_config.py`, `continuous_reward.py`, `ppo.py`, `ppo_lstm.py`, `train_optimal_overtaker.py`, `deepset_encoder.py`, or existing tests. All new code exclusively in `deepset_v2/`, `tests/deepset_v2/`, `eval_out/`, and `scratch/`.

---

## 1. Executive Summary & Gate Status

- **STOP AT GATE**: We have completed Steps 0, 1, 2, 3, and 4. As instructed, **NO training** was launched, **NO checkpoint evaluation** was executed, and **NO trainer** was written.
- **Unit Tests**: **26 of 26 tests PASSED** in 61.3s on real `highway-env` environments (seeds 4000–4099).
- **Parameter Parity**: DeepSet v2 has **342,278 parameters** (actor: 171,653, critic: 170,625), exactly **+11.60%** vs the legacy 81-dim `OptimalAgent` (306,694 params), within the ±15% budget.
- **Throughput Parity**: Under 8 `AsyncVectorEnv` workers (2,000 steps), DeepSet v2 achieves **618.4 SPS (full_adas)** and **688.6 SPS (front_only)**, outperforming the legacy 81-dim wrapper (566.5 SPS) because frame stacking is omitted.
- **GPU Timing (RTX 3060 Laptop, batch 4096)**: Forward 3.67 ms, Forward+Backward 8.24 ms (vs legacy 1.18 ms / 3.26 ms).

---

## 2. Step 0: Fact-Checking & Item-by-Item Verification

Every claim was verified locally this session:

| Item | Claim / Feature | Status | Evidence / Command / Location |
|---|---|---|---|
| **0a** | Package versions: Python 3.12, highway-env 1.12.1, gym 1.3.0, torch 2.14.0, numpy 2.5.2 | **CONFIRMED** | `pip show highway-env gymnasium torch numpy` |
| **0b** | `Vehicle.MAX_SPEED` = 40.0, `AbstractLane.DEFAULT_WIDTH` = 4.0 | **CONFIRMED** | `highway_env/vehicle/kinematics.py`, `highway_env/road/lane.py` |
| **0b** | `PERCEPTION_DISTANCE` = 200.0 (`5.0 * MAX_SPEED`) | **CONFIRMED** | `highway_env/envs/common/abstract.py:56` |
| **0b** | `MDPVehicle.DEFAULT_TARGET_SPEEDS` = `[20., 25., 30.]` m/s (72, 90, 108 km/h) | **CONFIRMED** | `highway_env/vehicle/controller.py:16` |
| **0b** | `HighwayEnvFast` defaults: `sim_freq: 5`, `lanes: 3`, `vehicles: 20`, `duration: 30` | **CONFIRMED** | `highway_env/envs/highway_env.py:321` (`HighwayEnvFast.default_config`) |
| **0b** | Other scenarios: merge-v1 (sim 15, target [20,25,30]); roundabout-v1 (sim 15, target [0,8,16], dur 11); intersection-v2 (sim 15, target [0,4.5,9], dur 13) | **CONFIRMED** | Inspected scenario configs in `site-packages/highway_env/envs/` |
| **0c** | `make_optimal_env` config literals: `lanes: 4`, `density: 1.4`, `vehicles: 14`, `duration: 100`, `sim_freq: 5`, native rewards zeroed | **CONFIRMED** | [env_config.py:953-967](file:///home/meerpi/curr_project/capstone/env_config.py#L953-L967) |
| **0c** | `filter_full_adas` docstring claims "100m range", but code is `return obs.copy()` (no filter) | **CONFIRMED** | [env_config.py:83-96](file:///home/meerpi/curr_project/capstone/env_config.py#L83-L96) |
| **0c** | `TacticalLaneObservationWrapper`: discards base env obs in `reset()` & `step()`; rebuilds from `unwrapped.road.vehicles` | **CONFIRMED** | [env_config.py:642-811](file:///home/meerpi/curr_project/capstone/env_config.py#L642-L811) |
| **0c** | Legacy reward defaults: `collision_penalty: -50`, `overtake_bonus: 1.0`, `danger_zone: 8.0`, `jitter_penalty: 0.12` | **CONFIRMED** | [env_config.py:437-451](file:///home/meerpi/curr_project/capstone/env_config.py#L437-L451) |
| **0d** | Legacy `OptimalAgent` parameter counts: 81-dim = 306,694; 90-dim = 311,302 | **CONFIRMED** | `sum(p.numel())` on freshly instantiated models |
| **0e** | `simulation_frequency: 5` and `models/ppo_optimal_overtaker_4lane_best.pt` introduced in same commit (`a679bb9`, 2026-09-08) | **CONFIRMED** | `git log -S'"simulation_frequency": 5'`, `git log --follow` |
| **0e** | Checkpoint `ckpt['args']` holds CLI arguments only, no environment config | **CONFIRMED** | `torch.load` inspection of checkpoint dict keys |
| **0f** | `deepset_encoder.py`: uses `danger_zone=15.0`, `jitter=0.02`, misses sensor tiers, leaves native rewards unzeroed, shares encoder | **CONFIRMED** | [deepset_encoder.py:47-266](file:///home/meerpi/curr_project/capstone/deepset_encoder.py#L47-L266) |
| **0g** | Golden legacy baseline generated on seeds 4000–4004 (100 steps each) | **CONFIRMED** | `scratch/golden_legacy.npz` |

---

## 3. Background Assumptions: Contradictions & Discoveries

### Refuted Background Assumption: IDLE Survival on Seeds 4000–4020
- **Reviewer Assumption (Prompt Test 9)**: *"max episode length 500 steps (run IDLE on seeds 4000-4020 until one survives to truncation; assert 500)"*.
- **Empirical Reality**: Under `density=1.4` and `vehicles=14`, an agent executing pure `IDLE` (action 1, which commands target speed 25 m/s) **crashes on 100% of seeds 4000–4099** (0/100 survival, average episode length 20–50 steps). In `highway-env`, `MDPVehicle` has no automatic collision avoidance in `IDLE`; because other traffic drives between 20 and 30 m/s, ego inevitably catches up to a 20 m/s lead car and rear-ends it.
- **Resolution**: Under action 4 (`SLOWER`, which drops target speed to the 20 m/s floor), the ego does not overtake lead cars. On **seed 4002 with SLOWER**, the episode survives to truncation at **exactly step 500** (`trunc=True, term=False, steps=500`). Test 9 verifies direct config math (100s × 5Hz = 500), records the 0/21 IDLE survival, and confirms the 500-step truncation using seed 4002.

### Observation on `env_config.py` Working Tree State
- Prior to the start of this session (at 01:19:45 on 2026-10-01, vs session start at 14:04), `env_config.py` had uncommitted local edits that added `_credited_overtakes` and `_dwell_counters` to debounce overtake farming.
- Per the Hard Rules: *"Do NOT edit env_config.py... New files only under deepset_v2/, tests/deepset_v2/, eval_out/, scratch/"*, we did not modify or revert `env_config.py`.

---

## 4. Architecture & Implementation Summary

### `deepset_v2/env.py`
- **Env Factory**: `make_deepset_env_v2(tier="full_adas", seed=None, range_m=100.0, render_mode=None)`
- **Base Environment**: Explicitly configured with `lanes_count=4`, `vehicles_density=1.4`, `vehicles_count=14`, `duration=100`, `policy_frequency=5`, `simulation_frequency=5`, and native rewards zeroed.
- **Compositional Reward & Masking**: Wraps with legacy `TacticalLaneObservationWrapper(apply_reward_shaping=True, include_continuous_features=False)`. Reward and action masks are 100% bit-identical to legacy.
- **Zero Simulator State Leaks**: Stashes raw normalized `KinematicsObservation` below the legacy wrapper in `info["_raw_kinematics"]`. The policy observation builder accesses **ONLY** `info["_raw_kinematics"]` and ego proprioception (`speed`, `lane_pos`, `can_left`, `can_right`). It **never** reads `unwrapped.road.vehicles`.
- **Sensor Tier Pipeline**:
  1. `apply_sensor_tier(raw_kinematics, tier=tier, env=env)` (reuses canonical tier filter).
  2. Range gating: Vehicles with $|x_{\text{rel}}| > 100.0\text{ m}$ are masked out.
  3. Feature normalization:
     - Vehicle row: $[x_{\text{rel}}/100.0,\, y_{\text{rel}}/4.0,\, v_{x,\text{rel}}/30.0,\, v_{y,\text{rel}}/30.0]$
     - Ego state: $[v_{\text{ego}}/30.0,\, \text{lane\_pos} \in [0,1],\, \text{can\_left},\, \text{can\_right}]$
  4. Output: Flat Box of shape `(74,)` = 4 ego + 14 × 5 `[presence, x, y, vx, vy]`.

### `deepset_v2/model.py`
- **Model**: `DeepSetAgent(n_vehicles=14, feat_dim=4, ego_dim=4, phi_hidden=(64, 64), rho_hidden=(256, 256, 256), n_actions=5)`
- **Ego-Conditioned $\phi$**: $\phi([v_i; \text{ego}])$ maps $(B, 14, 8) \to (B, 14, 64)$. Tanh activations, orthogonal initialization.
- **Masked Pooling**: Concatenation of masked sum normalized by 14 and masked max:
  $$\text{pool} = \left[\frac{1}{14}\sum_{i=1}^N m_i \phi_i \;\|\; \max_{i: m_i=1} \phi_i\right] \in \mathbb{R}^{128}$$
  When all rows are masked, pooling produces strictly finite zeros (no $-\infty$ or NaN).
- **Backbone $\rho$**: $[\text{pool}; \text{ego}] \to 3 \times 256 \text{ Tanh} \to \text{head}$. Ego is injected directly into $\rho$ so an empty sensor field retains full ego state.
- **Separate Encoders**: Independent actor and critic encoders (no weight sharing), matching legacy.
- **Action Masking**: Large negative constant (`-1e8`) masking over invalid actions.

---

## 5. Parameter Count Comparison

| Model | Actor Parameters | Critic Parameters | Total Parameters | vs Legacy 81-dim |
|---|---|---|---|---|
| **Legacy OptimalAgent (obs_dim=81)** | 153,861 | 152,833 | **306,694** | baseline (0.0%) |
| **Legacy OptimalAgent (obs_dim=90)** | 156,165 | 155,137 | **311,302** | +1.50% |
| **DeepSet v2 Agent (obs_dim=74)** | 171,653 | 170,625 | **342,278** | **+11.60%** (budget: $\pm 15\%$) |

---

## 6. Step 3: Test Suite Execution Results

Command executed:
```bash
source .venv/bin/activate && python -m pytest tests/deepset_v2/test_deepset_v2.py -v --tb=short
```

Output:
```text
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/meerpi/curr_project/capstone/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/meerpi/curr_project/capstone
collected 26 items

tests/deepset_v2/test_deepset_v2.py::TestPermutationInvariance::test_shuffled_rows_same_output PASSED [  3%]
tests/deepset_v2/test_deepset_v2.py::TestPermutationInvariance::test_on_real_env_observations PASSED [  7%]
tests/deepset_v2/test_deepset_v2.py::TestMaskInvariance::test_garbage_in_masked_rows_identical PASSED [ 11%]
tests/deepset_v2/test_deepset_v2.py::TestMaskInvariance::test_all_masked_finite PASSED [ 15%]
tests/deepset_v2/test_deepset_v2.py::TestVariableN::test_variable_vehicle_count[1] PASSED [ 19%]
tests/deepset_v2/test_deepset_v2.py::TestVariableN::test_variable_vehicle_count[4] PASSED [ 23%]
tests/deepset_v2/test_deepset_v2.py::TestVariableN::test_variable_vehicle_count[14] PASSED [ 26%]
tests/deepset_v2/test_deepset_v2.py::TestVariableN::test_variable_vehicle_count[40] PASSED [ 30%]
tests/deepset_v2/test_deepset_v2.py::TestGradientFlow::test_all_params_have_grad PASSED [ 34%]
tests/deepset_v2/test_deepset_v2.py::TestGradientFlow::test_phi_weights_nonzero_grad PASSED [ 38%]
tests/deepset_v2/test_deepset_v2.py::TestDeterminism::test_same_seed_identical_init PASSED [ 42%]
tests/deepset_v2/test_deepset_v2.py::TestDeterminism::test_same_seed_identical_trajectory PASSED [ 46%]
tests/deepset_v2/test_deepset_v2.py::TestTierInvariance::test_front_only_ignores_behind PASSED [ 50%]
tests/deepset_v2/test_deepset_v2.py::TestTierInvariance::test_full_adas_sees_behind PASSED [ 53%]
tests/deepset_v2/test_deepset_v2.py::TestTierInvariance::test_no_velocity_zeroes_vxvy PASSED [ 57%]
tests/deepset_v2/test_deepset_v2.py::TestTierInvariance::test_short_range_masks_distant PASSED [ 61%]
tests/deepset_v2/test_deepset_v2.py::TestTierInvariance::test_ego_features_identical_across_tiers PASSED [ 65%]
tests/deepset_v2/test_deepset_v2.py::TestRewardEquivalence::test_rewards_match_legacy PASSED [ 69%]
tests/deepset_v2/test_deepset_v2.py::TestActionMaskEquality::test_masks_match_legacy PASSED [ 73%]
tests/deepset_v2/test_deepset_v2.py::TestConfigAssertions::test_config_values PASSED [ 76%]
tests/deepset_v2/test_deepset_v2.py::TestConfigAssertions::test_config_matches_make_optimal_env PASSED [ 80%]
tests/deepset_v2/test_deepset_v2.py::TestConfigAssertions::test_max_episode_length_500 PASSED [ 84%]
tests/deepset_v2/test_deepset_v2.py::TestPolicyInputLeak::test_no_road_vehicles_in_deepset_obs PASSED [ 88%]
tests/deepset_v2/test_deepset_v2.py::TestLegacyRegression::test_golden_legacy_bitidentical PASSED [ 92%]
tests/deepset_v2/test_deepset_v2.py::TestLegacyRegression::test_git_status_clean PASSED [ 96%]
tests/deepset_v2/test_deepset_v2.py::TestModelsGuard::test_models_unchanged PASSED [100%]

======================== 26 passed in 61.33s (0:01:01) =========================
```

---

## 7. Step 4: Throughput & Timing Measurements

### A. Environment Throughput (8 AsyncVectorEnv Workers, 2,000 Steps)
Measured with random valid actions on seeds starting at 4000:
- **`make_optimal_env (k=3, 81-dim legacy)`**: 566.5 SPS (2,000 steps in 3.53 s)
- **`make_optimal_env (k=0, 27-dim legacy)`**: 608.1 SPS (2,000 steps in 3.29 s)
- **`make_deepset_env_v2 (full_adas)`**: 618.4 SPS (2,000 steps in 3.23 s)
- **`make_deepset_env_v2 (front_only)`**: 688.6 SPS (2,000 steps in 2.90 s)

*Observation*: DeepSet v2 is ~9% faster than legacy k=3 because it eliminates the deque frame-stacking overhead while maintaining identical reward calculation.

### B. GPU Forward & Backward Timing (batch_size = 4096)
Device: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM)
- **Legacy OptimalAgent (obs_dim=81)**:
  - Forward: **1.17 ms**
  - Forward + Backward: **3.26 ms**
- **DeepSet v2 Agent (obs_dim=74)**:
  - Forward: **3.67 ms**
  - Forward + Backward: **8.24 ms**

*Observation*: DeepSet v2 requires ~5 ms more per PPO update epoch at batch 4096. For 4 epochs over 4 minibatches per rollout (16 passes per 4096 steps), this adds only ~0.08 s of GPU time per PPO iteration.

---

## 8. Repository State & Models Guard

### Git Status
```text
 M env_config.py (pre-existing modification from 01:19)
?? deepset_encoder.py (pre-existing untracked)
?? deepset_v2/
?? eval_out/
?? scratch/
?? tests/deepset_v2/
?? tests/test_deepset_encoder.py (pre-existing untracked)
?? tests/test_overtake_flicker_discrete.py (pre-existing untracked)
?? train_deepset_eval.py (pre-existing untracked)
```

### Models Guard
All 9 checkpoints in `models/` verified unchanged against their pre-session SHA256 hashes:
- `ppo_highway_full_adas_seed101.pt`: `07b097de...` (MATCH)
- `rainbow_highway_full_adas_seed101.pt`: `3d0cb21f...` (MATCH)
- `ppo_lstm_highway_front_only_seed101.pt`: `746faea3...` (MATCH)
- `ppo_optimal_overtaker_4lane_best.pt`: `90b87214...` (MATCH)
- `ppo_optimal_overtaker_4lane.pt`: `a775eaf3...` (MATCH)
- `rainbow_highway_front_only_seed101.pt`: `d5f1cb9a...` (MATCH)
- `ppo_highway_front_only_seed101.pt`: `df6bcfc5...` (MATCH)
- `ppo_lstm_highway_full_adas_seed101.pt`: `e2759653...` (MATCH)
- `test_decoupled.pt`: `e37640ab...` (MATCH)

---

## 9. Deliverables Created

1. `deepset_v2/env.py`: Sensor-tier aware environment wrapper with legacy reward composition.
2. `deepset_v2/model.py`: Ego-conditioned DeepSet agent with separate actor/critic encoders.
3. `deepset_v2/__init__.py`: Package initialization.
4. `tests/deepset_v2/conftest.py`: Test runner path configuration.
5. `tests/deepset_v2/test_deepset_v2.py`: 26 comprehensive unit tests.
6. `eval_out/deepset_v2_facts.md`: Step 0 facts verification log.
7. `eval_out/deepset_v2_report.md`: This comprehensive gate report.
8. `scratch/golden_legacy.npz`: Regression baseline on seeds 4000–4004.
9. `scratch/measure_throughput.py`: SPS and GPU benchmark script.

---

## 10. Open Questions (Max 3)

1. **Pre-existing `env_config.py` uncommitted changes**: The working tree contained uncommitted edits to `env_config.py` adding overtake dwell debouncing. Should we keep `env_config.py` as-is, or restore it to `git checkout HEAD` so the legacy baseline uses the unmodified overtake reward?
2. **Action space target speed floor in POMDP**: Since `IDLE` crashes on 100% of seeds under density 1.4 while `SLOWER` reaches step 500, should future training curriculums for `front_only` initialize target speed at 20 m/s or allow the agent to learn deceleration before lane changes?
3. **Training script destination**: For the upcoming matched ablation (Phase 3 Step 2), do you prefer the PPO trainer to be written under `deepset_v2/train.py` or `scripts/train_deepset_v2.py`?

---

## 11. Proposed Next Step

**GATE**: Halt execution and wait for user approval of this report before proceeding to Phase 3 Step 2 (training script / ablation run) or Phase 4.
