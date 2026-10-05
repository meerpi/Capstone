# Sensor-budgeted autonomous highway driving benchmark

This repository provides an experimental testbed evaluating autonomous highway motion planning under degraded sensor perception. It benchmarks classical heuristics, recurrent neural architectures, constrained safe reinforcement learning, and set-based observations across multi-lane traffic scenarios in HighwayEnv.

## Visual demonstrations

![MOBIL front-only cut-in collision](visualizations/classical_mobil_front_only_seed2002.gif)

*Under front-only radar, the MOBIL heuristic treats unobserved trailing vehicles as absent, resulting in cut-in collisions during lane changes.*

![Optimal overtaker trajectory](visualizations/top10/ppo_top_1_seed_3094.gif)

*The optimal overtaker policy navigates multi-lane traffic using anti-jitter regularized actions and adjacent-corridor overtake accounting.*

Additional trajectory animations and video telemetry overlays are available in [visualizations/](visualizations/) and [visualizations/top10/](visualizations/top10/).

## Results

The table below summarizes performance across deterministic benchmark seeds. The legacy environment profile runs 10 vehicles at density 1.0 for 40 seconds (200 steps); the current profile runs 14 vehicles at density 1.4 for 100 seconds (500 steps).

| Policy | Sensor tier | Environment profile | Episodes | Crashes (95% CI) | Mean speed | Artifact path |
|:---|:---|:---|:---:|:---:|:---:|:---|
| IDM Only | full_adas | legacy | 30 | 0/30 (0.0% [0.0%, 11.6%]) | 81.4 km/h | [benchmark_results.md](benchmark_results.md) |
| IDM + MOBIL | full_adas | legacy | 30 | 2/30 (6.7% [0.8%, 22.1%]) | 83.4 km/h | [benchmark_results.md](benchmark_results.md) |
| Feedforward PPO | full_adas | legacy | 30 | 0/30 (0.0% [0.0%, 11.6%]) | 73.3 km/h | [benchmark_results.md](benchmark_results.md) |
| Recurrent PPO-LSTM | full_adas | legacy | 30 | 0/30 (0.0% [0.0%, 11.6%]) | 72.6 km/h | [benchmark_results.md](benchmark_results.md) |
| IDM Only | front_only | legacy | 30 | 0/30 (0.0% [0.0%, 11.6%]) | 81.4 km/h | [benchmark_results.md](benchmark_results.md) |
| IDM + MOBIL | front_only | legacy | 30 | 15/30 (50.0% [31.3%, 68.7%]) | 84.2 km/h | [benchmark_results.md](benchmark_results.md) |
| Feedforward PPO | front_only | legacy | 30 | 3/30 (10.0% [2.1%, 26.5%]) | 79.3 km/h | [benchmark_results.md](benchmark_results.md) |
| Recurrent PPO-LSTM | front_only | legacy | 30 | 0/30 (0.0% [0.0%, 11.6%]) | 72.6 km/h | [benchmark_results.md](benchmark_results.md) |
| Optimal Overtaker | full_adas | current | 100 | 10/100 (10.0% [4.9%, 17.6%]) | 73.6 km/h | [eval_out/crash_breakdown_optimal_overtaker.json](eval_out/crash_breakdown_optimal_overtaker.json) |
| PPO-Lagrangian (separate norm) | full_adas | current | 100 | 34/100 (34.0% [24.8%, 44.2%]) | 78.4 km/h | [eval_out/crash_breakdown_ppo_lagrangian_sep_norm.json](eval_out/crash_breakdown_ppo_lagrangian_sep_norm.json) |
| DeepSet v2 (2M baseline) | full_adas | current | 100 | 27/100 (27.0% [18.6%, 36.8%]) | 72.8 km/h | [eval_out/crash_breakdown_2m_seeds2000.json](eval_out/crash_breakdown_2m_seeds2000.json) |
| DeepSet v2 (retrained) | full_adas | current | 100 | 0/100 (0.0% [0.0%, 3.6%]) | 71.8 km/h | [eval_out/deepset_v2_retrain_seed42_100seeds.json](eval_out/deepset_v2_retrain_seed42_100seeds.json) |
| Optimal Overtaker (extended speeds) | full_adas | current | 100 | 0/100 (0.0% [0.0%, 3.6%]) | 73.5 km/h | [eval_out/crash_breakdown_optimal_overtaker_extended_speeds.json](eval_out/crash_breakdown_optimal_overtaker_extended_speeds.json) |

Confidence intervals are exact Clopper-Pearson 95% binomial intervals. Detailed empirical analyses and failure breakdowns are documented in [docs/findings.md](docs/findings.md).

## What is in the repo

| Path | Description |
|:---|:---|
| [models/](models/) | Pre-trained PyTorch checkpoints for PPO, PPO-LSTM, and Optimal Overtaker policies |
| [deepset_v2/](deepset_v2/) | Permutation-invariant vehicle set observation wrapper and neural network agent |
| [docs/](docs/) | Technical findings, known issues, environment specifications, and audit reports |
| [PROJECT_CHRONICLE_AND_POSTMORTEM.md](PROJECT_CHRONICLE_AND_POSTMORTEM.md) | Comprehensive engineering trajectory, empirical post-mortem, and pathology analysis |
| [eval_out/](eval_out/) | Evaluation telemetry, crash diagnostics, and checkpoint profile manifests |
| [scripts/](scripts/) | Diagnostic utilities for pre-impact telemetry analysis and benchmark generation |
| [tests/](tests/) | Unit tests verifying observation bounds, reward invariants, and model execution |
| [visualizations/](visualizations/) | Recorded MP4 telemetry videos and animated GIFs |
| [baseline_classical.py](baseline_classical.py) | Implementations of IDM car-following and MOBIL lane-changing heuristics |
| [env_config.py](env_config.py) | Scenario configurations, sensor tier filtering, and observation wrappers |
| [ppo.py](ppo.py) | Feedforward on-policy actor-critic architecture with invalid action masking |
| [ppo_lstm.py](ppo_lstm.py) | Recurrent on-policy actor-critic architecture with decoupled LSTM recurrent cells |
| [train_optimal_overtaker.py](train_optimal_overtaker.py) | Training pipeline for discrete tactical overtaker policies and PPO-Lagrangian |
| [run_comparison_benchmark.py](run_comparison_benchmark.py) | Benchmark execution harness across classical and learned controllers |

## Setup

Dependency resolution with the package resolver requires Python 3.12 or newer because numpy 2.5.2 requires Python 3.12+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

The following evaluation commands execute in a clean environment created from requirements.txt:

Run classical heuristic baseline (IDM only on seed 9000, expected duration 500 steps, speed 73.5 km/h, 0 crashes):
```bash
python baseline_classical.py --episodes 1 --seed-start 9000 --scenario highway --tier front_only --controller idm_only
```

Run comparative classical benchmark across legacy and current profiles (writes JSON outputs to eval_out/):
```bash
python run_comparison_benchmark.py --classical-only --episodes 1 --seed-start 9000
```

Run crash diagnostic on pre-trained optimal overtaker checkpoint (evaluates seed 9000 and prints telemetry summary):
```bash
python scripts/diagnose_legacy_crashes.py --model-path models/ppo_optimal_overtaker_4lane_best.pt --episodes 1 --seed-start 9000
```

Training scripts were verified using the --help flag to confirm parameter configuration without initiating full multi-hour training runs:
```bash
python ppo.py --help
python ppo_lstm.py --help
python train_optimal_overtaker.py --help
python scripts/train_deepset_v2.py --help
```

## Limitations

The repository exhibits the following technical limitations documented in [docs/known-issues.md](docs/known-issues.md):
- Checkpoint evaluation discrepancy: `models/ppo_optimal_overtaker_4lane_best.pt` exhibits different crash rates across evaluation horizons and configurations.
- Velocity quantization floor: The default discrete action space sets minimum speed to 20.0 m/s (72.0 km/h), causing in-lane rear-end collisions when lead traffic travels at 13 to 18 m/s and adjacent lanes are blocked.
- Ground-truth simulator access: Tactical observation wrappers query internal vehicle arrays directly rather than processing degraded sensor matrices.
- Autoreset boundary transitions: Checkpoints trained prior to valid-step masking included post-reset initial states in loss computation.
- Unresolved encoder parity: DeepSet v2 (11/100 crashes) and the legacy encoder (10/100 crashes) have not been assigned an adoption or parity verdict.

## Citation

```bibtex
@misc{mahanty2026sensorbudgeted,
  author = {Aritra Mahanty},
  title = {Sensor-Budgeted Autonomous Highway Driving: Multi-Paradigm Motion Planning & Reinforcement Learning Benchmark},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/meerpi/Capstone}}
}
```
