# Sensor-Budgeted Autonomous Highway Driving Benchmark (Round 1)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29.1-brightgreen.svg)](https://gymnasium.farama.org/)
[![Highway-Env](https://img.shields.io/badge/Highway--Env-v1.8.2-orange.svg)](https://github.com/Farama-Foundation/HighwayEnv)
[![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)](https://pytorch.org/)

This repository presents the **Round 1 Benchmark** investigating how classical control heuristics and deep reinforcement learning policies perform when operating under **degraded radar sensor coverage** on autonomous highway environments.

Specifically, we evaluate what happens when an autonomous vehicle equipped with **Full 360° ADAS Perception** is stripped down to **Front-Only Forward Radar** (inducing physical blind spots for trailing traffic), comparing:
1. **Classical Heuristics:** Intelligent Driver Model (**IDM**) car-following and **IDM + MOBIL** lane-changing.
2. **On-Policy Policy Gradient RL:** Feedforward **PPO** (MLP + Invalid Action Masking).
3. **On-Policy Recurrent RL:** Decoupled **PPO + LSTM** with temporal belief state tracking.

---

## 1. Executive Summary & Headline Results

Across identical, reproducible test seeds ($N = 30$ episodes per condition, seeds 2000–2029) in dense 4-lane highway traffic:

| Sensor Tier | Policy Paradigm | Controller / Model | Crash Rate ($N/30$) | Episode Duration | Mean Speed | Lane Changes / ep | Primary Failure Mode / Behavioral Dynamic |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Full ADAS** (360° Omnidirectional) | Classical Heuristic | **IDM Only** | **0.0%** (0/30) | 200.0 ± 0.0 | 81.4 km/h | 0.00 | Passive car-following; zero overtaking initiative. |
| | Classical Heuristic | **IDM + MOBIL** | **6.7%** (2/30) | 193.6 ± 32.5 | 83.4 km/h | 5.80 | High mobility with active overtaking; occasional tight squeeze. |
| | On-Policy Feedforward RL | **PPO (MLP)** | **0.0%** (0/30) | 200.0 ± 0.0 | 73.3 km/h | 0.67 | Prudent defensive cruising; tactical lane changes when blocked. |
| | On-Policy Recurrent RL | **PPO + LSTM** | **0.0%** (0/30) | 200.0 ± 0.0 | 72.6 km/h | 1.53 | Fluid overtaking; stable velocity and zero crashes. |
| **Front-Only Radar** (Blind Spots $x < 0$) | Classical Heuristic | **IDM Only** | **0.0%** (0/30) | 200.0 ± 0.0 | 81.4 km/h | 0.00 | Front-only sensing suffices for longitudinal car-following. |
| | Classical Heuristic | **IDM + MOBIL** | **50.0%** (15/30) | 147.1 ± 64.3 | 84.2 km/h | 4.40 | **Catastrophic Blind-Spot Collapse (7.5x crash surge)**. |
| | On-Policy Feedforward RL | **PPO (MLP)** | **10.0%** (3/30) | 187.8 ± 40.9 | 79.3 km/h | 1.53 | **80% crash reduction vs MOBIL** via learned margin compliance. |
| | On-Policy Recurrent RL | **PPO + LSTM** | **0.0%** (0/30) | 200.0 ± 0.0 | 72.6 km/h | 1.53 | **Hidden state memory tracks trailing traffic through blind spots**. |

---

## 2. Key Insights: What Happens Under Sensor Ablation?

### 2.1 The Classical MOBIL Blind-Spot Collapse
In classical traffic theory, MOBIL calculates whether changing lanes is safe by evaluating the deceleration imposed on the trailing vehicle in the target lane:
$$\tilde{a}_{\text{new\_lag}} \ge -b_{\text{safe}} \quad (b_{\text{safe}} = 2.0\text{ m/s}^2)$$

- **Under Full ADAS:** The vehicle accurately senses trailing vehicles, yielding a low **6.7% crash rate** (2/30).
- **Under Front-Only Radar:** All trailing vehicles behind the ego vehicle ($x_{\text{rel}} < 0$) are masked to zero. The MOBIL safety condition evaluates trailing distance as infinite ($s \to \infty$), causing the safety check to **vacuously pass (`True`)**.
- **Consequence:** MOBIL cuts in directly in front of fast-approaching blind-spot traffic, causing crashes in **50.0% of episodes (15/30)**.

### 2.2 Why On-Policy PPO Exhibits Defensive Compliance
To avoid degenerate behaviors (such as riding road shoulders or freezing behind slow traffic), the training environment incorporates three tactical mechanisms (`TacticalOvertakingWrapper` in `env_config.py`):
1. **Invalid Action Masking (Mechanism C):** Prevents boundary violations by masking out-of-bounds lane changes (`LANE_LEFT` in lane 0, `LANE_RIGHT` in lane 3) with $-\infty$ logit clamping.
2. **Headway Penalty (Mechanism A):** Imposes a continuous tailgating penalty when following slower vehicles within 25 meters, incentivizing active lane-changing.
3. **Discrete Overtake Bonus (Mechanism B):** Rewards successfully passing surrounding vehicles ($+1.0$ per passed vehicle).

Equipped with these constraints, PPO learns a defensive policy that does not recklessly cut across lanes unless clear frontal clearance exists, cutting the crash rate to **10.0% (3/30)** under front-only radar.

### 2.3 Why Recurrent PPO (PPO-LSTM) Resolves the POMDP
A front-only sensor turns the driving task from a Markov Decision Process (MDP) into a **Partially Observable Markov Decision Process (POMDP)**.
- Memoryless policies (Classical MOBIL and Feedforward PPO) only observe the instantaneous frame $o_t$.
- **PPO-LSTM** maintains an internal recurrent hidden state belief vector $h_t = f(h_{t-1}, o_t)$. When a trailing vehicle accelerates from behind, PPO-LSTM retains the temporal memory of that vehicle's approach even after it enters the radar blind spot, achieving **0.0% crashes** under both Full ADAS and Front-Only tiers.

---

## 3. Visual Demonstrations Gallery

All episodes recorded at 10 FPS with complete telemetry HUD overlays displaying step count, speed (km/h), lane index, discrete action, cumulative overtakes, and crash status.

### 3.1 Classical IDM + MOBIL
| Full ADAS (Safe Overtaking) | Front-Only Radar (Blind-Spot Cut-In Crash) |
| :---: | :---: |
| ![MOBIL Full ADAS](visualizations/classical_mobil_full_adas_seed2002.gif) | ![MOBIL Front Only](visualizations/classical_mobil_front_only_seed2002.gif) |
| *Smooth lane changes with full 360° awareness.* | *Vacuous safety check causes immediate cut-in crash.* |
| [Download MP4](visualizations/classical_mobil_full_adas_seed2002.mp4) | [Download MP4](visualizations/classical_mobil_front_only_seed2002.mp4) |

### 3.2 Feedforward PPO (MLP)
| Full ADAS (Seed 2015) | Front-Only Radar (Seed 2015) |
| :---: | :---: |
| ![PPO Full ADAS](visualizations/ppo_full_adas_seed2015.gif) | ![PPO Front Only](visualizations/ppo_front_only_seed2015.gif) |
| *Accurate speed regulation and lane centering.* | *Learned defensive spacing without boundary violation.* |
| [Download MP4](visualizations/ppo_full_adas_seed2015.mp4) | [Download MP4](visualizations/ppo_front_only_seed2015.mp4) |

### 3.3 Recurrent PPO (PPO-LSTM)
| Full ADAS (Seed 2015) | Front-Only Radar (Seed 2015) |
| :---: | :---: |
| ![PPO-LSTM Full ADAS](visualizations/ppo_lstm_full_adas_seed2015.gif) | ![PPO-LSTM Front Only](visualizations/ppo_lstm_front_only_seed2015.gif) |
| *High-speed overtaking with temporal awareness.* | *Maintains tracking through blind spots with zero collisions.* |
| [Download MP4](visualizations/ppo_lstm_full_adas_seed2015.mp4) | [Download MP4](visualizations/ppo_lstm_front_only_seed2015.mp4) |

### 3.4 Classical IDM (Pure Longitudinal Car-Following)
| Full ADAS (Seed 2002) | Front-Only Radar (Seed 2002) |
| :---: | :---: |
| ![IDM Full ADAS](visualizations/classical_idm_full_adas_seed2002.gif) | ![IDM Front Only](visualizations/classical_idm_front_only_seed2002.gif) |
| *Zero lane changes; strictly maintains headway.* | *Identical performance; unaffected by rear sensor removal.* |
| [Download MP4](visualizations/classical_idm_full_adas_seed2002.mp4) | [Download MP4](visualizations/classical_idm_front_only_seed2002.mp4) |

---

## 4. Repository Structure

```
Capstone/
├── .gitignore                      # Clean exclusion of temporary/venv/shelved files
├── requirements.txt                # Pinned dependencies
├── README.md                       # Round 1 documentation & benchmark summary
├── benchmark_results.md            # Comprehensive quantitative benchmark report
├── ultimate_comparison_results.json # Full numerical logs across all 30 test seeds
├── env_config.py                   # Sensor tiers & TacticalOvertakingWrapper
├── ppo.py                          # Feedforward PPO architecture and training pipeline
├── ppo_lstm.py                     # Recurrent PPO (LSTM) with decoupled Actor-Critic
├── baseline_classical.py           # Observation-driven IDM and IDM+MOBIL baselines
├── run_comparison_benchmark.py     # Reproducible 30-seed automated benchmark suite
├── record_visual_driving.py        # Telemetry HUD renderer exporting MP4 & GIF
├── models/                         # Trained model checkpoints
│   ├── ppo_highway_full_adas_seed101.pt
│   ├── ppo_highway_front_only_seed101.pt
│   ├── ppo_lstm_highway_full_adas_seed101.pt
│   └── ppo_lstm_highway_front_only_seed101.pt
└── visualizations/                 # 12 MP4 videos and 12 animated GIFs with telemetry
    ├── classical_idm_*.mp4 / .gif
    ├── classical_mobil_*.mp4 / .gif
    ├── ppo_*.mp4 / .gif
    └── ppo_lstm_*.mp4 / .gif
```

---

## 5. Quickstart & Reproduction Guide

### 5.1 Environment Setup
```bash
git clone git@github.com:meerpi/Capstone.git
cd Capstone
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 Running the Full 30-Seed Benchmark Suite
To re-evaluate all 4 models across both sensor tiers across seeds 2000–2029:
```bash
python run_comparison_benchmark.py
```

### 5.3 Recording Driving Episodes with Telemetry HUD
To record an episode and generate high-definition GIF and MP4 outputs:
```bash
# Record Classical MOBIL under Front-Only Radar (demonstrating cut-in failure)
python record_visual_driving.py --agent classical_mobil --tier front_only --seed 2002

# Record Recurrent PPO under Front-Only Radar
python record_visual_driving.py --agent ppo_lstm --tier front_only --seed 2015

# Record Feedforward PPO under Full ADAS
python record_visual_driving.py --agent ppo --tier full_adas --seed 2015
```

---

## 6. Citation & Reference

If you use this benchmark or codebase in your research, please cite:
```bibtex
@misc{mahanty2026sensorbudgeted,
  author = {Aritra Mahanty},
  title = {Sensor-Budgeted Autonomous Highway Driving: Multi-Paradigm Motion Planning & Reinforcement Learning Benchmark},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/meerpi/Capstone}}
}
```
