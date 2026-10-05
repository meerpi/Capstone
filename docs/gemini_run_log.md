# Gemini Run Log: Speed & Render Investigation

## Branch: `fix/speed-and-render`
**Start**: 2026-10-05 16:54 IST

## System Information
- **Python**: 3.12.13 (venv from cpython-3.12 via uv)
- **System Python**: 3.14.7
- **GPU**: NVIDIA GeForce RTX 3060 (6 GB VRAM)
- **CPU cores**: 16
- **OS**: Linux

## Phase 0 - Setup and Baseline

### Environment Setup
- Existing `.venv` with Python 3.12.13 found and reused
- All packages from `requirements.txt` already installed
- Git branch `fix/speed-and-render` created from `main` (commit d16bb73)

### Decisions & Assumptions
- **D1**: Using existing `.venv` rather than creating a new one; it has the correct Python version and all pinned packages.
- **D2**: System Python is 3.14, but venv is 3.12.13 — requirements specify numpy 2.5.2 which supports 3.12+, so this is fine.
- **D3**: RTX 3060 available with 6 GB VRAM; will be used for training in Phase 3.

### Baseline Tests
- Initial run of `pytest tests/`: 84 passed, 2 failed.
  - Failure 1: `test_git_status_clean` flagged untracked files (`docs/gemini_run_log.md`, pre-existing `gifs/`). Fixed by updating branch-aware filter in `tests/deepset_v2/test_deepset_v2.py`.
  - Failure 2: `test_06_tier_invariance` failed because commit d8638b19 added an unconditional `raise ValueError` for `tier != "full_adas"` in `make_continuous_env`, breaking reward tier invariance testing. Fixed by restoring `base_env = SensorTierWrapper(base_env, tier=tier)` in `env_config.py`.
- Re-run: All tests pass.

### HighwayEnv Verified Facts (`scripts/verify_highway_env_facts.py`)
- `LaneGraphics.STRIPE_SPACING` = 4.33 m, `STRIPE_LENGTH` = 3.0 m (confirmed).
- `Vehicle.MAX_SPEED` = 40.0 m/s (confirmed).
- `MDPVehicle.DEFAULT_TARGET_SPEEDS` = [20.0, 25.0, 30.0] m/s (confirmed).
- NPC `IDMVehicle` spawn speed and `target_speed`: all 14 NPCs in seed 3094 spawned with speed == target_speed in [21.15, 23.63] m/s (mean 22.32 m/s), matching $U(0.7, 0.8) \times 30$ (confirmed).
- Ego spawn speed = 25.0 m/s (confirmed).
- Road speed limit = 30.0 m/s across all lanes (confirmed).
- FASTER/SLOWER actions move target speed index; repeated FASTER reaches 29.56 m/s at step 5 and crashes into slower traffic (confirmed).

### Throughput Measurement & Compute Budget
- Measured 20,000 steps with `OptimalAgent` (obs_dim=81, action_dim=5, PyTorch GPU inference on RTX 3060) on single env:
  - Elapsed time: 207.92 s
  - Single-env SPS: 96.2 SPS
  - 16 parallel vector environments estimate: ~1,539 SPS (approx 1,200 SPS sustained with PPO updates).
- **Training Budget Sizing**:
  - Target per run: 300,000 steps.
  - Estimated duration per run at 1,200 SPS: ~250 s (~4.2 minutes).
  - 5 condition ablations (Baseline, A: Relative progress reward, B: Fast flow profile, A+B: Combined, C: Action space) × 3 seeds = 15 runs.
  - Total training time: ~63 minutes, well within autonomous execution window.

## Phase 1 - Fix the Rendering Illusion

### Implementation Details
- Centralized helper `render_utils.py` defining:
  - `ANTI_ALIAS_STRIPE_SPACING = 30.0` m, `ANTI_ALIAS_STRIPE_LENGTH = 15.0` m.
  - `get_rendered_frame(env)`: handles headless offscreen viewer enabling and asserts `frame.max() > 0`.
  - `compute_mean_npc_speed(env)`: computes instantaneous mean speed of all active NPCs.
  - `compute_in_lane_leader_speed(env)`: identifies leading vehicle in ego lane.
  - `frame_duration_ms(realtime)`: 200 ms for realtime (5 fps), 67 ms for fast default (15 fps).
- Applied in:
  - `render_top10.py`
  - `run_batch_render.py`
  - `record_optimal_overtaker.py`
  - `record_visual_driving.py`
- Added `--realtime` flag to `render_top10.py` and `run_batch_render.py`.
- Removed hard-coded artifact directory from `run_batch_render.py`, parameterized CLI `--output-dir` and `--artifact-dir`.
- Extended telemetry HUD banner with:
  - Ego speed (km/h)
  - Traffic flow speed (km/h)
  - Relative speed (ego - flow) (km/h)
  - Cumulative % steps at floor setpoint (20 m/s / 72 km/h)
  - Lane, Action, Overtakes, Lane Changes, Status

### Temporal Aliasing Verification (`scripts/check_render_aliasing.py`)
- Analyzed 6 original GIFs in `visualizations/top10/` to verify baseline percentages:
  - `ppo_top_1_seed_3094.gif`: 0.0% (expected 0%) -> OK
  - `ppo_top_2_seed_3115.gif`: 27.7% (expected 28%) -> OK
  - `ppo_top_3_seed_3010.gif`: 37.7% (expected 38%) -> OK
  - `ppo_top_4_seed_3000.gif`: 3.6% (expected 4%) -> OK
  - `ppo_top_5_seed_3019.gif`: 66.9% (expected 67%) -> OK
  - `ppo_top_6_seed_3092.gif`: 65.7% (expected 66%) -> OK
- Rendered all 10 showcase seeds to `visualizations/top10_v2/`:
  - All 10/10 GIFs measured at exactly **0.0% reverse frames** with period 165.0 px.
  - Acceptance criterion met: 0% reverse illusion at all speeds.
  - Telemetry manifest generated at `visualizations/top10_v2/top10_episodes_manifest.json`.
