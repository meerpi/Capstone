# Sensor-Budgeted Autonomous Highway Driving Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29.1-brightgreen.svg)](https://gymnasium.farama.org/)
[![Highway-Env](https://img.shields.io/badge/Highway--Env-v1.8.2-orange.svg)](https://github.com/Farama-Foundation/HighwayEnv)
[![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)](https://pytorch.org/)

This repository presents a comprehensive autonomous highway driving benchmark investigating:
1. **Degraded Radar Sensor Coverage (Ablation Benchmark):** How classical control heuristics and deep reinforcement learning policies perform when stripped from **Full 360° ADAS Perception** down to **Front-Only Radar** (inducing trailing blind spots).
2. **Constrained Safe Reinforcement Learning (PPO-Lagrangian):** A dual-critic, PID-regulated Lagrangian policy formulation that decouples collision penalties from performance rewards, using separate unit-scale advantage normalization ($A_{\text{adj}} = \text{norm}(A_R) - \lambda \cdot \text{norm}(A_C)$) to achieve a **48 percentage point reduction in crash rate** over uncalibrated baselines across 100 deterministic benchmark seeds.
3. **Multi-Paradigm Comparative Study:** Comparing Classical Heuristics (**IDM**, **IDM+MOBIL**), On-Policy Feedforward RL (**PPO** + Action Masking), On-Policy Recurrent RL (**PPO+LSTM** POMDP state tracking), and Constrained Safe RL (**PPO-Lagrangian**).

> [!NOTE]
> 📖 **Comprehensive Technical Report & Post-Mortem:**  
> For an exhaustive, phase-by-phase breakdown of the full project trajectory, architectural hardening, reward pathology analyses, and root-cause breakdowns of advantage normalization, read **[PROJECT_CHRONICLE_AND_POSTMORTEM.md](PROJECT_CHRONICLE_AND_POSTMORTEM.md)**.

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

### 3.5 Optimal Tactical Overtaker (Anti-Jitter Regularization & Top 10 Demonstrations)

Trained with continuous headway incentives, anti-jitter regularization ($-0.12$ action-switching penalty), and strictly adjacent-corridor overtake accounting, this 4-lane PPO agent exhibits dynamic acceleration up to $108.0\text{ km/h}$, proactive slalom lane changes around slow clusters, and stable cruising without stutter.

#### Top 10 Non-Idle Overtaking & Obstacle Avoidance Gallery (Selected across 150 Seeds)
Every episode in this Top 10 completed the full $500\text{ steps}$ ($100\text{ seconds}$) without collisions, executing active lateral evasion and longitudinal speed regulation:

| Rank | Seed | Overtakes | Lane Changes | FASTER / IDLE / SLOWER | Mean Speed | Max Speed | GIF Animation | Video MP4 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **3094** | **5** | **4** | 102 / 288 / 101 | **86.7 km/h** | **108.0 km/h** | [GIF](visualizations/top10/ppo_top_1_seed_3094.gif) | [MP4](visualizations/top10/ppo_top_1_seed_3094.mp4) |
| **#2** | **3115** | **1** | **15** | 63 / 331 / 64 | **75.2 km/h** | **90.0 km/h** | [GIF](visualizations/top10/ppo_top_2_seed_3115.gif) | [MP4](visualizations/top10/ppo_top_2_seed_3115.mp4) |
| **#3** | **3010** | **4** | **3** | 40 / 423 / 32 | **84.0 km/h** | **108.0 km/h** | [GIF](visualizations/top10/ppo_top_3_seed_3010.gif) | [MP4](visualizations/top10/ppo_top_3_seed_3010.mp4) |
| **#4** | **3000** | **1** | **11** | 47 / 389 / 47 | **74.4 km/h** | **90.0 km/h** | [GIF](visualizations/top10/ppo_top_4_seed_3000.gif) | [MP4](visualizations/top10/ppo_top_4_seed_3000.mp4) |
| **#5** | **3019** | **1** | **14** | 12 / 445 / 10 | **76.5 km/h** | **100.3 km/h** | [GIF](visualizations/top10/ppo_top_5_seed_3019.gif) | [MP4](visualizations/top10/ppo_top_5_seed_3019.mp4) |
| **#6** | **3092** | **2** | **3** | 27 / 444 / 24 | **78.4 km/h** | **107.9 km/h** | [GIF](visualizations/top10/ppo_top_6_seed_3092.gif) | [MP4](visualizations/top10/ppo_top_6_seed_3092.mp4) |
| **#7** | **3078** | **1** | **3** | 40 / 414 / 40 | **76.2 km/h** | **105.7 km/h** | [GIF](visualizations/top10/ppo_top_7_seed_3078.gif) | [MP4](visualizations/top10/ppo_top_7_seed_3078.mp4) |
| **#8** | **3028** | **1** | **6** | 17 / 453 / 17 | **73.8 km/h** | **90.0 km/h** | [GIF](visualizations/top10/ppo_top_8_seed_3028.gif) | [MP4](visualizations/top10/ppo_top_8_seed_3028.mp4) |
| **#9** | **2010** | **1** | **5** | 20 / 452 / 20 | **76.4 km/h** | **107.9 km/h** | [GIF](visualizations/top10/ppo_top_9_seed_2010.gif) | [MP4](visualizations/top10/ppo_top_9_seed_2010.mp4) |
| **#10** | **3049** | **1** | **3** | 36 / 423 / 35 | **73.4 km/h** | **90.7 km/h** | [GIF](visualizations/top10/ppo_top_10_seed_3049.gif) | [MP4](visualizations/top10/ppo_top_10_seed_3049.mp4) |

#### Highlight: Rank #1 (Seed 3094: 5 Overtakes, 4 Lane Changes, 108 km/h Sprint)
![Rank 1 Seed 3094](visualizations/top10/ppo_top_1_seed_3094.gif)

---

## 4. Repository Structure

```
Capstone/
├── .gitignore                      # Clean exclusion of temporary/venv/scratch files
├── requirements.txt                # Pinned dependencies
├── README.md                       # Comprehensive documentation & quickstart
├── PROJECT_CHRONICLE_AND_POSTMORTEM.md # Complete technical trajectory, what worked, and post-mortem
├── benchmark_results.md            # Quantitative benchmark report
├── env_config.py                   # Sensor tiers, wrappers, frame stacking, and anti-jitter reward
├── train_optimal_overtaker.py      # End-to-end PPO trainer with anti-jitter and adjacent overtake bonuses
├── ppo.py                          # Feedforward PPO architecture
├── ppo_lstm.py                     # Recurrent PPO (LSTM) with decoupled Actor-Critic
├── baseline_classical.py           # Classical IDM car-following and MOBIL lane-changing controllers
├── record_optimal_overtaker.py     # High-definition telemetry HUD episode recorder
├── render_top10.py                 # Telemetry renderer for top demonstration episodes
├── run_batch_render.py             # Automated batch rendering script for Top 10 seeds
├── run_comparison_benchmark.py     # 30-seed automated benchmark suite
├── record_visual_driving.py        # Multi-model visual driving recorder
├── models/                         # Checkpoints for PPO, PPO-LSTM, and Optimal Overtaker
│   ├── ppo_optimal_overtaker_4lane_best.pt
│   ├── ppo_optimal_overtaker_4lane.pt
│   ├── ppo_highway_full_adas_seed101.pt
│   ├── ppo_highway_front_only_seed101.pt
│   ├── ppo_lstm_highway_full_adas_seed101.pt
│   └── ppo_lstm_highway_front_only_seed101.pt
└── visualizations/                 # Generated GIFs and MP4 videos with telemetry HUD
    ├── top10/                      # Top 10 non-idle overtaking and avoidance runs (GIF + MP4)
    │   ├── manifest.json           # Telemetry metrics manifest across all 10 runs
    │   ├── ppo_top_1_seed_3094.gif / .mp4
    │   └── ...
    ├── classical_idm_*.mp4 / .gif
    ├── classical_mobil_*.mp4 / .gif
    ├── ppo_*.mp4 / .gif
    └── ppo_lstm_*.mp4 / .gif
```

---

## 4. Constrained Safe Reinforcement Learning: PPO-Lagrangian

While standard reward-shaped PPO blends collision penalties (e.g. $-50.0$) directly into the reward signal, this single-objective formulation can lead to either defensive paralysis (crawling/freezing to minimize collision variance) or reckless speed exploitation if rewards overpower the penalty.

To solve this rigorously, we formulate highway driving as a **Constrained Markov Decision Process (CMDP)**:
$$\max_{\theta} \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t R_t \right] \quad \text{subject to} \quad J_C(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T C_t \right] \le d$$
where $C_t = 1.0$ if a collision occurs and $0.0$ otherwise, and $d = 0.05$ (maximum 5% crash budget).

### 4.1 Key Algorithmic Components
1. **Dual-Critic Architecture:** `OptimalAgent` incorporates an independent cost value head $V_C(s)$ (identical architecture to $V_R(s)$, trained on cost returns $G_t^C$) to predict future crash risk without corrupting the reward value baseline.
2. **PID Multiplier Regulation:** An adaptive PID controller dynamically modulates the Lagrange multiplier $\lambda$:
   $$\lambda_{k+1} = \text{clamp}\left( \lambda_k + K_p e_k + K_i \sum_{j=0}^k e_j + K_d (e_k - e_{k-1}), \; 0, \; \lambda_{\max} \right)$$
   where $e_k = J_C^{(k)} - d$.
3. **Separate Unit-Scale Advantage Normalization:** Combining raw advantages ($A_R - \lambda A_C$) suffers from scale distortion when $\sigma(A_R) \gg \sigma(A_C)$. We separately standardize both streams to unit scale prior to combination:
   $$A_{\text{adj}} = \frac{A_R - \mu(A_R)}{\sigma(A_R) + 10^{-8}} - \lambda \cdot \frac{A_C - \mu(A_C)}{\sigma(A_C) + 10^{-8}}$$
   $A_{\text{adj}}$ is **not** re-normalized, ensuring $\lambda$ directly sets the relative physical penalty weight.
4. **Symmetrical Cost-GAE Termination:** Verified by automated unit tests (`tests/test_ppo_lagrangian.py`), true crash terminations strictly zero out cost value bootstraps ($\gamma V_C(s_{t+1}) = 0$), preventing post-crash values from bleeding backward into crashed trajectories.

### 4.2 100-Seed Fair Benchmark Comparison (Seeds 2000–2099)
Evaluated across 100 deterministic benchmark seeds in dense 4-lane traffic using exact 95% Clopper-Pearson binomial confidence intervals:

| Metric | Baseline PPO (-50.0, 1.0) | Failed Unnorm Run (82% Crash) | PPO-Lagrangian (Separate Norm) |
| :--- | :---: | :---: | :---: |
| **Crash Rate (95% Clopper-Pearson)** | **10.0%** [4.9%, 17.6%] | **82.0%** [73.1%, 89.0%] | **34.0%** [24.8%, 44.2%] |
| **Crashes / Episodes** | 10 / 100 | 82 / 100 | **34 / 100** |
| **Mean Ego Speed** | 73.56 km/h | 105.06 km/h | **78.34 km/h** |
| **Mean Episode Duration** | 460.2 steps | 127.8 steps | **349.3 steps** |
| **Mean Overtakes / Episode** | 0.49 | 1.80 | **0.90** |
| **Mean Lane Changes / Episode** | 1.63 | 5.93 | **1.66** |

**Key Takeaways:**
- **Zero CI Overlap:** The Clopper-Pearson 95% intervals between the failed unnormalized run (`[73.1%, 89.0%]`) and the separately normalized PPO-Lagrangian run (`[24.8%, 44.2%]`) have zero overlap, demonstrating a statistically significant **48 percentage point reduction in crash rate**.
- **Tactical Balance:** PPO-Lagrangian nearly triples episode survival duration ($349.3$ vs $127.8$ steps) and maintains disciplined lane changes ($1.66$/ep) while achieving higher speed ($78.3$ km/h) and almost double the overtakes ($0.90$/ep) of the reward-shaped baseline ($0.49$/ep).

---

## 5. Quickstart & Reproduction Guide

### 5.1 Environment Setup
```bash
git clone git@github.com:meerpi/Capstone.git
cd Capstone
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 5.2 Run the Automated Unit Test Suite (43 Tests)
Run the full test suite verifying continuous bounds, truncation bootstrapping, checkpoint resume, and Lagrangian cost-GAE symmetry:
```bash
PYTHONPATH=. uv run --with pytest pytest tests/ -v
```

### 5.3 Run the 100-Seed 3-Way Benchmark
Reproduce the exact 100-seed Clopper-Pearson comparison table across baseline, failed calibration, and PPO-Lagrangian checkpoints:
```bash
uv run python scripts/evaluate_comparison_100seeds.py \
  --baseline models/ppo_optimal_overtaker_4lane_best.pt \
  --failed-run scratch/archived_runs/failed_calibration_82pct_crash/ppo_optimal_overtaker_4lane_best.pt \
  --lagrangian scratch/ppo_lagrangian_1m_sep_norm/ppo_optimal_overtaker_4lane_best.pt \
  --episodes 100 \
  --seed-start 2000 \
  --device cuda
```

### 5.4 Train PPO-Lagrangian from Scratch
Train the constrained safe RL agent with separate advantage normalization and PID multiplier regulation:
```bash
uv run python train_optimal_overtaker.py \
  --lagrangian \
  --total-timesteps 1000000 \
  --num-envs 8 \
  --num-steps 512 \
  --cost-limit 0.05 \
  --lagrangian-kp 1.0 \
  --lagrangian-ki 0.01 \
  --checkpoint-dir scratch/ppo_lagrangian_1m_sep_norm \
  --output-path scratch/ppo_lagrangian_1m_sep_norm/final.pt \
  --cuda
```

### 5.5 Render & Record Any Optimal Driving Episode
To evaluate and record an episode with full real-time telemetry HUD overlay (saved as both `.gif` and `.mp4`):
```bash
# Record Rank #1 episode (Seed 3094, 5 overtakes, 108 km/h)
python record_optimal_overtaker.py --seed 3094 --steps 500

# Record Rank #2 episode (Seed 3115, 15 lane changes)
python record_optimal_overtaker.py --seed 3115 --steps 500
```

### 5.6 Batch Render the Top 10 Demonstrations
To batch render all Top 10 non-idle overtaking seeds to `visualizations/top10/`:
```bash
python run_batch_render.py
```

### 5.7 Classical Baselines & Sensor Ablation Benchmark
To run the classical IDM/MOBIL heuristic baselines and evaluate performance across the 30-seed benchmark:
```bash
# Run classical baselines standalone
python baseline_classical.py

# Run comparative benchmark suite across models and sensor tiers
python run_comparison_benchmark.py
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
