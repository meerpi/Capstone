# Sensor-Budgeted Autonomous Highway Driving Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Handbook](https://img.shields.io/badge/Documentation-Handbook%20PDF-red.svg)](autonomous_driving_capstone_handbook.pdf)

An empirical benchmark evaluating autonomous highway decision-making under sensor-budgeted perception, classical heuristics, reinforcement learning, and constrained safe-RL in [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv).

> 📖 **Comprehensive Manual Available:** For the complete A–Z mathematical derivations, architectural proofs, and line-by-line code defense guide, see the 23-page [**Capstone Handbook & Code Defense Guide (PDF)**](autonomous_driving_capstone_handbook.pdf).

---

## Visual Demonstrations

| Champion Tactical Overtaker | Classical MOBIL Blindspot Cut-in |
|:---:|:---:|
| ![Optimal Overtaker](visualizations/top10/ppo_top_1_seed_3094.gif) | ![MOBIL Failure](visualizations/classical_mobil_front_only_seed2002.gif) |
| *Anti-aliased stripes (`STRIPE_SPACING = 30.0m`) with live telemetry HUD.* | *Under front-only radar, unobserved trailing traffic causes blind cut-ins.* |

*Complete video catalogs and MP4 recordings are available in [visualizations/](visualizations/) and [gifs/INDEX.md](gifs/INDEX.md).*

---

## Unified Empirical Benchmark Results

Evaluated on the standardized production environment (`highway-fast-v0`, 4 lanes, 14 NPC vehicles, density 1.4, 5 Hz, 100s / 500 steps) across **100 fresh, held-out deterministic seeds (4000–4099)**. CIs are exact Clopper-Pearson 95% binomial intervals; $p_{\text{IDM}}$ is two-sided Fisher's exact test against `IDM Only`.

| Policy | Perception / Observation | Crashes (95% CI) | Crashes / 1000 km | Surviving Speed | RMS Accel | RMS Jerk | Lane Changes / 100 km | $p_{\text{IDM}}$ | Checkpoint / Source |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **DeepSet v2 (2M)** | Sensor Set (74-dim Full-ADAS) | **0/100** [0.0%, 3.6%] | **0.0** | 71.3 km/h | 7.06 m/s² | 22.51 m/s³ | 79.8 | **< 0.0001** | [`checkpoints/deepset_v2_highway_2m_best.pt`](checkpoints/deepset_v2_highway_2m_best.pt) |
| **Optimal Overtaker (Seed 42)** | Privileged Ground Truth (81-dim) | **1/100** [0.0%, 5.5%] | **4.9** | 73.9 km/h | 5.69 m/s² | 16.89 m/s³ | 280.1 | **< 0.0001** | [`checkpoints/optimal_overtaker_extended_speeds_best.pt`](checkpoints/optimal_overtaker_extended_speeds_best.pt) |
| **Optimal Overtaker (Seed 137)** | Privileged Ground Truth (81-dim) | **5/100** [1.6%, 11.3%] | 25.5 | 75.0 km/h | 6.13 m/s² | 18.32 m/s³ | 5179.4 | **0.0007** | [`checkpoints/optimal_overtaker_extended_speeds_seed137_best.pt`](checkpoints/optimal_overtaker_extended_speeds_seed137_best.pt) |
| **Optimal Overtaker (Best)** | Privileged Ground Truth (81-dim) | **10/100** [4.9%, 17.6%] | 53.0 | 74.2 km/h | 1.62 m/s² | 9.54 m/s³ | 98.5 | **0.0327** | [`models/ppo_optimal_overtaker_4lane_best.pt`](models/ppo_optimal_overtaker_4lane_best.pt) |
| **always_SLOWER (Floor Speed)** | None (Trivial Drift) | 22/100 [14.3%, 31.4%] | 136.0 | 72.1 km/h | 0.72 m/s² | 1.21 m/s³ | 0.0 | 1.0000 | Baseline |
| **IDM Only (Lane-Keeping)** | Sensor (Full-ADAS) | 22/100 [14.3%, 31.4%] | 128.1 | 76.8 km/h | 2.93 m/s² | 16.28 m/s³ | 0.0 | — (Ref) | [`baseline_classical.py`](baseline_classical.py) |
| **Feedforward PPO** | Sensor (75-dim Full-ADAS) | 25/100 [16.9%, 34.7%] | 157.4 | 72.8 km/h | 1.25 m/s² | 6.12 m/s³ | 0.0 | 0.7390 | [`models/ppo_highway_full_adas_seed101.pt`](models/ppo_highway_full_adas_seed101.pt) |
| **IDM + MOBIL** | Sensor (Full-ADAS) | 30/100 [21.2%, 40.0%] | 188.4 | 77.8 km/h | 3.23 m/s² | 17.85 m/s³ | 223.5 | 0.2590 | [`baseline_classical.py`](baseline_classical.py) |
| **PPO-Lagrangian (Sep Norm)** | Privileged Ground Truth (81-dim) | 48/100 [37.9%, 58.2%] | 407.6 | 73.7 km/h | 0.57 m/s² | 2.30 m/s³ | 152.0 | 0.0002 | [`checkpoints/ppo_lagrangian_sep_norm_best.pt`](checkpoints/ppo_lagrangian_sep_norm_best.pt) |
| **always_IDLE** | None (Fixed Throttle) | 100/100 [96.4%, 100.0%] | 3081.0 | 90.0 km/h | 0.00 m/s² | 0.00 m/s³ | 0.0 | < 0.0001 | Baseline |
| **Random Action** | None (Stochastic) | 100/100 [96.4%, 100.0%] | 8065.4 | 88.6 km/h | 5.10 m/s² | 28.78 m/s³ | 2766.4 | < 0.0001 | Baseline |

*Full telemetry JSON saved to [eval_out/av_benchmark_heldout_seeds4000.json](eval_out/av_benchmark_heldout_seeds4000.json).*

---

## Core Empirical Takeaways

1. **The Kinematic Floor Speed Trap:** Under `DiscreteMetaAction`, minimum speed clamps at 20 m/s (72.0 km/h). Because traffic queues drop lead speeds to 6–15 m/s, `always_SLOWER` achieves 22/100 crashes simply by letting traffic pull away. Standard feedforward RL (`Feedforward PPO`) collapses into this exact floor-speed drift (0.0 lane changes, $p = 0.7390$ vs IDM).
2. **Permutation Invariance Breakthrough:** `DeepSet v2` achieves 0/100 crashes (0.0 crashes/1000km) under raw sensor sets without privileged ground-truth lane arrays by processing traffic via symmetric dual pooling ($\frac{1}{N_{\max}}\sum \oplus \max$).
3. **The Vacuous Classical Blindspot:** Classical IDM+MOBIL crashes 71% under forward-only radar because its safety check treats unobserved trailing traffic as absent (`safe_lag = True`), triggering cut-ins directly in front of overtaking vehicles.
4. **Action Space & Training Seed Sensitivity:** Widening target speeds to $[10, 15, 20, 25, 30]$ m/s enables slow queue matching, but exhibits high policy variance: Seed 42 achieves 1/100 crashes with fluid overtakes, whereas Seed 137 suffers lateral hunting (5,179 lane changes/100km).
5. **Constrained Safe-RL (PPO-Lagrangian):** Enforcing a 5% crash budget ($\kappa = 0.05$) via dual-critic PPO-Lagrangian suffered multiplier oscillations in high density, yielding 48/100 crashes on held-out seeds.

---

<details>
<summary><b>Click to view Historical Diagnostic Benchmarks (Seeds 2000–2099)</b></summary>

| Policy | Profile | Episodes | Crashes (95% CI) | Speed | Artifact |
|:---|:---|:---:|:---:|:---:|:---|
| **Optimal Overtaker (Ext. Spd, Seed 42)** | Current (14 veh, dens 1.4) | 100 | **0/100** [0.0%, 3.6%] | 73.5 km/h | [eval_out/crash_breakdown_optimal_overtaker_extended_speeds.json](eval_out/crash_breakdown_optimal_overtaker_extended_speeds.json) |
| **Optimal Overtaker (Ext. Spd, Seed 137)** | Current (14 veh, dens 1.4) | 100 | **7/100** [2.9%, 13.8%] | 74.9 km/h | [eval_out/crash_breakdown_optimal_overtaker_seed137_best.json](eval_out/crash_breakdown_optimal_overtaker_seed137_best.json) |
| **Optimal Overtaker (Best)** | Current (14 veh, dens 1.4) | 100 | **10/100** [4.9%, 17.6%] | 73.6 km/h | [eval_out/crash_breakdown_optimal_overtaker.json](eval_out/crash_breakdown_optimal_overtaker.json) |
| **IDM Only** | Current (14 veh, dens 1.4) | 100 | **11/100** [5.6%, 18.8%] | 76.2 km/h | [eval_out/classical_baselines_current.json](eval_out/classical_baselines_current.json) |
| **always_SLOWER** | Current (14 veh, dens 1.4) | 100 | **12/100** [6.4%, 20.0%] | 71.8 km/h | [parity/report.md](parity/report.md) |
| **Feedforward PPO (75-dim)** | Current (14 veh, dens 1.4) | 100 | **12/100** [6.4%, 20.0%] | 72.7 km/h | [eval_out/learned_policy_evals_current.json](eval_out/learned_policy_evals_current.json) |
| **IDM + MOBIL** | Current (14 veh, dens 1.4) | 100 | **16/100** [9.4%, 24.7%] | 77.3 km/h | [eval_out/classical_baselines_current.json](eval_out/classical_baselines_current.json) |
| **DeepSet v2 (2M)** | Current (14 veh, dens 1.4) | 100 | **27/100** [18.6%, 36.8%] | 72.6 km/h | [eval_out/crash_breakdown_2m_seeds2000.json](eval_out/crash_breakdown_2m_seeds2000.json) |
| **PPO-Lagrangian** | Current (14 veh, dens 1.4) | 100 | **34/100** [24.8%, 44.2%] | 78.4 km/h | [eval_out/crash_breakdown_ppo_lagrangian_sep_norm.json](eval_out/crash_breakdown_ppo_lagrangian_sep_norm.json) |

*Legacy 40s profile (10 veh, dens 1.0, 50-dim): Feedforward PPO (front_only) achieved 2/100 crashes (74.2 km/h); PPO-LSTM (front_only) achieved 1/100 crashes (72.5 km/h); full_adas LSTM checkpoint collapsed to 100/100 crashes ([eval_out/learned_policy_evals_legacy.json](eval_out/learned_policy_evals_legacy.json)).*
</details>

---

## Quickstart & Reproduction

```bash
# Setup virtual environment (Python 3.12+)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run complete 100-seed held-out benchmark (4000-4099)
python scripts/evaluate_av_metrics.py --episodes 100 --seed-start 4000 --workers 8

# Run test suite
pytest tests/deepset_v2/test_deepset_v2.py tests/test_render_utils.py
```

---

## Repository Map

| Path | Description |
|:---|:---|
| [`checkpoints/`](checkpoints/) | Top-performing model weights (DeepSet v2, PPO-Lagrangian, Extended-Speed runs) |
| [`models/`](models/) | Baseline PPO and Optimal Overtaker checkpoints |
| [`deepset_v2/`](deepset_v2/) | Permutation-invariant set wrapper, dual-pooling encoder, and unit tests |
| [`scripts/`](scripts/) | AV evaluation engine (`evaluate_av_metrics.py`), anti-aliasing utilities (`render_utils.py`) |
| [`eval_out/`](eval_out/) | Raw empirical evaluation JSON telemetry and Fisher's test results |
| [`visualizations/`](visualizations/) | Anti-aliased telemetry recordings and GIFs |
| [`autonomous_driving_capstone_handbook.pdf`](autonomous_driving_capstone_handbook.pdf) | Complete 23-page mathematical & code defense handbook |

---

