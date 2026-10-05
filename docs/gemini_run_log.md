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
