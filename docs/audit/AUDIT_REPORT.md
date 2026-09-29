# COMPREHENSIVE INDEPENDENT TECHNICAL AUDIT REPORT: SPRINT-1 PLAN & IMPLEMENTATION

**Date:** September 28, 2026  
**Auditor:** Independent Technical Auditor (Antigravity Senior Review)  
**Target:** Sprint-1 Research Plan (`docs/OPUS_PLAN.md` specification), Step 1–3 Prompts, and Local Implementations (`env_config.py`, `continuous_reward.py`, `scripts/profile_throughput.py`, `ppo.py`, `ppo_lstm.py`, `train_optimal_overtaker.py`)  
**Status of Code Under Review:** Steps 1–3 code **EXISTS LOCALLY** and was directly audited, executed, and benchmarked against stored results.

---

## 1. Executive Verdict & Rationale

### **VERDICT: NO-GO FOR SPRINT-1 TRAINING COMPUTE (GO-WITH-FIXES ONLY AFTER 5 MANDATORY GATES)**

**Five-Line Rationale:**
1. **Broken Checkpoint Compatibility (Finding C8-1):** Step-2 observation extension to 30 dims (stacked 90) defaulted `include_continuous_features=True` in `make_optimal_env`, causing `train_optimal_overtaker.evaluate()` to crash immediately with a `RuntimeError` shape mismatch (`[256, 81]` vs `[256, 90]`), breaking reproducibility of the published 30-seed benchmark.
2. **Fatal Value-Truncation Bug (Finding B7-1):** All PPO training implementations (`ppo.py`, `ppo_lstm.py`, `train_optimal_overtaker.py`) collapse time-limit truncations into terminal flags (`next_done_np = np.logical_or(term, trunc)`), setting `nextnonterminal = 0.0` and destroying GAE value bootstrapping at $T=500$, which induces severe end-of-horizon policy pathologies.
3. **Missing Checkpoint/Resume Infrastructure (Finding D1-1):** Neither `ppo.py` nor `train_optimal_overtaker.py` serializes optimizer states, step counts, or RNG seeds, nor do they support `--resume`. Under Kaggle's 12-hour session limits and 20-minute idle timeouts, any disconnected run loses all compute with zero recovery capability.
4. **Crash Penalty Discounting Arithmetic Failure (Finding C6-1):** Under $\gamma = 0.99$, an undiscounted $-200$ crash penalty discounts to only $-26.8$ at step $T=200$, while accumulated speed reward reaches $+86.6$, leaving the agent with a net positive return ($+59.8$) for speeding and crashing at step 200 versus staying safely parked ($0.0$).
5. **Multi-Scenario Observation Invalidation (Finding D6-1):** The 4-lane, Cartesian x-axis 16-slot observation wrapper is fundamentally invalid for `merge-v1` (9 lanes), `roundabout-v1` (32 lanes), and `intersection-v2` (20 lanes), making multi-scenario transfer in Sprints 3–4 impossible without an immediate geometric re-architecture.

---

## 2. Findings Ledger

| ID | Severity | Area | Title / Defect | Primary Evidence | Recommended Fix | Effort |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C8-1** | **CRITICAL** | Wrapper / Regr. | Observation dimension mismatch breaks existing benchmark weights | `train_optimal_overtaker.py:467`, `audit/verify_audit_c_math.py:244` -> `RuntimeError: size mismatch for critic.0.weight: [256, 81] vs [256, 90]` | Default `include_continuous_features=False` in `make_optimal_env()`; add dynamic shape inspection in model loading. | 0.5 h |
| **B7-1** | **CRITICAL** | Dynamics / Alg. | Time-limit truncation treated as terminal in GAE calculation | `ppo.py:262,294`, `ppo_lstm.py:428`, `train_optimal_overtaker.py:240`: `next_done_np = np.logical_or(term, trunc)` | Separate `terminated` and `truncated`. In GAE loop, bootstrap $V(s_{t+1})$ when `truncated and not terminated`. | 1.5 h |
| **D1-1** | **CRITICAL** | Plan / Infra. | Checkpoint/resume completely unwired in PPO scripts | `ppo.py:413,435`: only saves `agent.state_dict()`. No optimizer state, no global step, no CLI flag `--resume` | Implement full checkpoint dictionary `{'model': ..., 'optimizer': ..., 'step': ..., 'rng': ...}` and `--resume` loader. | 2.0 h |
| **C6-1** | **CRITICAL** | Reward Math | Crash penalty arithmetic failure at $\gamma = 0.99$ | `audit/verify_audit_c_math.py` C6: For all $T \ge 110$, $V_{\text{crash}}(T) > 0.0 = V_{\text{park}}$ ($V_{\text{crash}}(200) = +59.81$) | Rescale speed reward weight or increase crash penalty to $-500$ (or un-discounted terminal loss) so $V_{\text{crash}} < V_{\text{safe}}$. | 1.0 h |
| **D6-1** | **HIGH** | Observation | 4-Lane 16-Slot observation invalid for Merge, Roundabout, Intersection | `env_config.py:463-491`; `merge` (9 lanes), `roundabout` (32 lanes), `intersection` (20 lanes). Cartesian $\Delta x$ fails on curves. | Re-architect observation wrapper for Sprints 3–4 using Frenet curvilinear coordinates or an egocentric polar radar grid. | 12.0 h |
| **E1-1** | **HIGH** | Exploit | Reversing exploit in ContinuousAction (`MIN_SPEED = -40 m/s`) | `audit/run_adversarial_probes.py` Probe 1: ego reaches $-40.0\text{ m/s}$ ($-144\text{ km/h}$), `crashed=False`, return $+1.33$ | Constrain `ContinuousAction` speed range: `speed_range = (0.0, 30.0)` or add quadratic reverse speed penalty. | 0.5 h |
| **E4-1** | **HIGH** | Exploit | `offroad_terminal = False` permits continuous offroad grass cruising | `audit/run_adversarial_probes.py` Probe 4: 40 steps offroad at $y = -2.0$, `terminated=False`, `crashed=False` | Explicitly enforce `"offroad_terminal": True` in all scenario configurations in `env_config.py`. | 0.25 h |
| **E9-1** | **MEDIUM** | Reward Math | Reward lacks heading error penalty; steady yaw angle unpenalized | `audit/run_adversarial_probes.py` Probe 9: steady steering $u=[0, 1]$ yields $r_{\text{jerk}} = 0.0$, $r_{\text{control}} = 0.0$ | Add explicit heading error penalty $R_{\text{heading}} = -w_{\psi} (\psi_{\text{err}} / \psi_{\max})^2$ to `ContinuousRewardConfig`. | 1.0 h |
| **E7-1** | **MEDIUM** | Exploit | Speed-matching inside danger zone zeroes out closing-only TTC | `audit/run_adversarial_probes.py` Probe 7: gap = 2.0m, $v_{\text{rel}} = 0.0 \implies r_{\text{TTC}} = 0.0000$ | Increase headway penalty weight $w_{\text{headway}}$ from 0.30 to 0.60 to maintain strong barrier when speed-matched. | 0.5 h |
| **E13-1** | **MEDIUM** | Exploit | Reward function blind to fast trailing traffic approaching from behind | `audit/run_adversarial_probes.py` Probe 13: rear vehicle closing at $v_{\text{rel}} = -10\text{ m/s}$ produces 0 penalty | Monitor trailing vehicle in ego lane when ego speed $< 20\text{ m/s}$ or when braking. | 1.5 h |
| **B8-1** | **MEDIUM** | Documentation | Post-mortem speed target discrepancy (`FASTER` targets 30, not 25 m/s) | `DiscreteMetaAction.target_speeds = [20., 25., 30.]` vs `PROJECT_CHRONICLE_AND_POSTMORTEM.md:159` | Reconcile documentation to reflect that initial speed is 25 m/s and `FASTER` targets 30 m/s. | 0.25 h |
| **D2-1** | **MEDIUM** | Statistics | Statistical power failure: $N=30$ seeds cannot detect 0% vs 10% crash rate | Fisher's exact test on $0/30$ vs $3/30$ yields $p = 0.237$ ($\alpha=0.05$); Clopper-Pearson CI $[2.1\%, 26.5\%]$ | Increase evaluation seeds to $N \ge 100$ for screening and mandate $\ge 3$ training seeds per algorithm. | 2.0 h |
| **D1-2** | **MEDIUM** | Plan Budget | Measured throughput 1.78x lower than assumed in plan | Measured 16-worker SPS = 895.3, 4-worker SPS = 513.3 (`benchmark_throughput_results.json`) vs assumed 1600–2000 SPS | Recompute GPU-hour budget: 3.3M steps takes $\approx 2.6\text{ hours}$ per run on Kaggle 4 vCPU, not 45 minutes. | 0.5 h |
| **A11-1** | **LOW** | External Claim | GPUDrive 13x7 action grid misattributed to `arXiv:2506.23771` | `arXiv:2506.23771` text vs GPUDrive `arXiv:2408.01584` / CL4AD `arXiv:2608.22549:869` | Update citation in `docs/OPUS_PLAN.md` to point to Kazemkhani et al. 2024 / CL4AD. | 0.25 h |
| **A10-1** | **LOW** | External Claim | Ayres 2001 and Dingus 2006 misattributed to `arXiv:2511.14910` & `2606.20640` | Search of `arXiv:2511.14910` and `arXiv:2606.20640` shows zero occurrences of Ayres or Dingus | Correct citation references in `docs/OPUS_PLAN.md` to original human-factors studies. | 0.25 h |
| **A5-1** | **LOW** | External Claim | GTrXL scale gap arithmetic inflated by 10x (10B vs 100B steps) | Parisotto et al. 2020 Table 9: DMLab-30 ran to 10B steps, not 100B. Gap is 3,000x, not 30,000x | Update arithmetic in `docs/OPUS_PLAN.md`. | 0.25 h |

---

## 3. External Claims Register (Audit A)

Every external claim referenced in the research plan has been cross-examined against primary sources fetched and verified during this audit session:

| Item | Primary Source URL | Claim in Opus Plan | Verification Status | Auditor Findings & Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **A1** | [arXiv:2607.17326v1](https://arxiv.org/html/2607.17326v1) (Hamed et al., July 2026) | PPO reaches good policy faster than SAC/TD-MPC2 under wall-clock budgets; bet on SAC at 4 vCPUs. | **PARTIALLY SUPPORTED** | Verified PPO trained on 2,048 parallel environments in Brax vs single-environment CleanRL SAC. Paper is a robotics transfer study; it reports **zero experiments** in low-parallelism (4 vCPU / 16 worker) regimes. Extrapolating that SAC wins at 4 vCPUs is an unsupported hypothesis. |
| **A2** | [arXiv:2606.29867v1](https://arxiv.org/html/2606.29867v1) (RoAd-RL, June 2026) | SAC 163.5 vs PPO 28.1 on highway-v0; intrinsic continuous SAC superiority. | **PARTIALLY SUPPORTED** | Table II reports Highway-v0 Best Reward: SAC 163.5 (2,376 episodes), PPO 28.1 (2,898 episodes). However, Section IV-A confirms all agents used **Stable-Baselines3 default hyperparameters without tuning**, and runs were **not step-matched**. Discrete PPO was compared against continuous SAC with un-tuned entropy and step sizes. |
| **A3** | [arXiv:2505.15040v1](https://arxiv.org/html/2505.15040v1) (RLBenchNet, May 2025) | MLPs best on continuous control; Mamba 4.5x faster than LSTM; GTrXL adds little; Mamba unstable. | **VERIFIED** | Abstract confirms MLPs excel in fully observable continuous control and Mamba achieves 4.5x higher throughput over LSTM. Section 4.2 confirms GTrXL provides only marginal gains over Transformer-XL. Section 4.3 confirms Mamba exhibits instability due to inter-episode state leakage. |
| **A4** | [arXiv:2303.01859v1](https://arxiv.org/html/2303.01859v1) (POPGym, Morad et al., ICLR 2023) | GRU is best general-purpose recurrent model across 46-environment benchmark suite. | **VERIFIED** | Morad et al. 2023 explicitly benchmarks 15 environments $\times$ 3 difficulties = 45 tasks (+ 1 = 46 tasks). Section 6 explicitly states: "Use GRUs for performance and Elman nets for efficiency... within traditional RNNs, there seems little reason to use LSTM, as GRUs are more efficient and perform better." |
| **A5** | [arXiv:1910.06764](https://arxiv.org/html/1910.06764) (Parisotto et al., ICML 2020) | GTrXL DMLab-30 / MERLIN 100B step comparison; 30,000x scale gap to 3.3M steps. | **PARTIALLY SUPPORTED / CONTRADICTED** | MERLIN comparison verified, but Table 9 shows DMLab-30 was trained for **10 billion steps (10B)**, NOT 100B. The plan's "30,000x scale gap" is arithmetic on an inflated 100B figure ($100\text{B} / 3.3\text{M} \approx 30,000$). The actual scale gap is $10\text{B} / 3.3\text{M} \approx 3,000\times$. |
| **A6** | [arXiv:2603.04648v2](https://arxiv.org/html/2603.04648v2) (Vogt-Lowell et al., March 2026) | Persistent sensor failure; latent memory models (RNNs/SSMs/GTrXL) lack robustness. | **VERIFIED** | Accepted at ICLR 2026 CAO Workshop. Authors: Kevin Vogt-Lowell et al. Evaluated Transformers, SSMs (Mamba, LinOSS, LRU), RNNs (LSTM, GRU), MLP, GTrXL. "Persistent" failure is modeled as a 2-layer Markovian failure process (temporal persistence and group correlations). Section 6 confirms latent-memory models exhibit substantial fragility under persistent outages. |
| **A7** | [arXiv:2509.11225v1](https://arxiv.org/html/2509.11225v1) (MEMBOT, Noronha & Liang, 2025) | 65–80% peak return at 50% observation dropout vs 10–30% for baselines. | **PARTIALLY SUPPORTED** | The quote appears verbatim in Abstract/Section 1, but is an overgeneralized claim from an un-reviewed Stanford student project paper (CS234/CS330). Table 1 proves that on 2 out of 3 held-out tasks (handle-press, plate-slide), performance collapsed to **3.2% and 4.6%** success at 50% dropout! Only drawer-close maintained 65.8%. Evaluated on 7-DOF robot arms in MetaWorld/Robomimic, not driving. |
| **A8** | [arXiv:2608.22549v1](https://arxiv.org/html/2608.22549v1) (CL4AD) & [arXiv:2505.08264](https://arxiv.org/abs/2505.08264) (Abouelazm IV 2025) | 99% success 1B steps earlier; 77% wall-clock cut; 800 worlds / 2B steps on H200 (60h); A5000 110h; weak regret at small scale. | **VERIFIED** | All figures verified in CL4AD text (lines 163, 869, 877, 881). Section C.1 confirms 13x7 action grid. Appendix D confirms H200 60h (2B steps) and A5000 110h (1B steps). Section 5.3 confirms regret estimators (AMGAE, PVL) underperform at small scale while success-rate utilities work across all scales. Section E.1 confirms curriculum improves task performance but diverges from human realism unless reward explicitly penalizes un-realism. |
| **A9** | [doi:10.3389/frai.2023.1098982](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1098982/full) (Mahmoud et al. 2023) | Replay buffer had a detrimental effect in most combinations of data generation. | **VERIFIED** | Section 1 and Section 4 confirm that on lane-keeping and pedestrian avoidance with Deep Q-Learning (DQN), the replay buffer had a detrimental effect when combined with structured learning, as old experiences diluted curriculum progression. |
| **A10** | [Zhu et al. 2020](https://doi.org/10.1109/TITS.2020.2974903), [arXiv:2505.06737](https://arxiv.org/abs/2505.06737) | TTC log barrier (2s); 1.2s (Ayres 2001) and 12m (Dingus 2006) as cited in arXiv:2511.14910 and 2606.20640. | **PARTIALLY SUPPORTED / UNSUPPORTED** | Zhu et al. 2020 confirms TTC logarithmic barrier at 2.0s. However, the attribution of Ayres 2001 (1.2s) and Dingus 2006 (12m) to `arXiv:2511.14910` (Z-Merge) and `arXiv:2606.20640` (LLM-explainable DRL) is **UNSUPPORTED** (neither paper contains these citations or values). `arXiv:2505.06737` (Abouelazm et al.) proposes a 2D ellipsoid RSS risk objective, not scalar headway log barrier. |
| **A11** | [arXiv:2408.01584](https://arxiv.org/abs/2408.01584) vs [arXiv:2506.23771](https://arxiv.org/abs/2506.23771) | GPUDrive action grid (13 steering $\times$ 7 acceleration) attribution. | **CONTRADICTED / MISATTRIBUTED** | `arXiv:2506.23771` is on multi-timescale hierarchical RL and does not mention GPUDrive or 13x7 grid. The 13x7 action grid is from **GPUDrive (Kazemkhani et al., arXiv:2408.01584, ICLR 2025)** and CL4AD (`arXiv:2608.22549`, Section C.1). |
| **A12** | Official Docs & Specs | GPUDrive (>1M FPS, CUDA), MetaDrive (CPU, Panda3D), RTX 3060 Laptop, Ryzen 7 5800H, Kaggle 2026 limits. | **VERIFIED** | GPUDrive builds on Madrona (>1M steps/s on consumer GPUs). MetaDrive is CPU-primary. RTX 3060 Laptop has 6GB GDDR6, 192-bit bus, 60–115W TGP. Ryzen 7 5800H has 8c/16t, 16MB L3. Kaggle current limits: 30 hours/week GPU quota, 12-hour session limit, 20-minute idle timeout, 2x T4 or 1x P100, 4 vCPU, 30 GB RAM, 20 GB scratch disk. |
| **A13** | Recent 2025–2026 Literature | Continuous HighwayEnv RL, jerk regularization, sensor dropout in driving. | **VERIFIED** | Literature confirms that continuous RL in HighwayEnv without observation normalization or with uncalibrated action penalty weights collapses into low-speed cruising equilibria or extreme bang-bang saturation. |

---

## 4. Could-Not-Verify List

1. **`docs/OPUS_PLAN.md` and `docs/prompts/step1..3` as standalone files on disk:**
   - *What was tried:* `find . -name "*OPUS*" -o -name "*plan*" -o -name "*prompt*"`.
   - *Result:* Physical files do not exist under `docs/` in the repository tree. The plan and prompts were reconstructed from session prompt metadata, the active `PROJECT_CHRONICLE_AND_POSTMORTEM.md`, and the existing implementation files.
2. **Kaggle P100 availability guarantees:**
   - *What was tried:* Queried Kaggle quota documentation.
   - *Result:* Kaggle provides 30 GPU hours/week, but accelerator allocation (T4 vs P100) is dynamic based on platform availability; P100 is frequently exhausted or constrained during peak demand.

---

## 5. Audit B & C: Mathematical, Kinematic & Code Audit

### B1–B8: Environment Facts & Dynamics
1. **Installed HighwayEnv Version (B1):**
   - Installed: `highway-env==1.12.1`, `gymnasium==1.3.0`, Python 3.12.13.
   - PR #667 introduced `ConnectedLaneNeighboursMixin` in v1.12.0 for connected lanes (`merge`, `roundabout`, `intersection`). In straight `highway-fast-v0`, background IDM/MOBIL traffic is unaffected.
2. **ContinuousAction Defaults & Dynamics (B3, B4):**
   - Acceleration range: $[-5.0, 5.0]\text{ m/s}^2$. Steering range: $[-\pi/4, \pi/4]\text{ rad} = \pm 45^\circ$.
   - Vehicle class: `Vehicle` (kinematic bicycle model) with `MIN_SPEED = -40.0 m/s`, `MAX_SPEED = 40.0 m/s`, `LENGTH = 5.0 m`, `WIDTH = 2.0 m`.
   - **Extreme Lateral Acceleration:** At $v = 25\text{ m/s}$ ($90\text{ km/h}$), full steering $\delta = \pi/4$ produces $a_{\text{lat}} = 111.8\text{ m/s}^2$ ($11.4\text{ g}$)! Even $\delta = \pi/6$ produces $a_{\text{lat}} = 69.3\text{ m/s}^2$ ($7.07\text{ g}$). The plan's proposed steering range of $\pm \pi/12$ ($\pm 15^\circ$, $a_{\text{lat}} = 33.2\text{ m/s}^2$) is a necessary improvement, but still reaches $3.38\text{ g}$.
3. **Timing & Simulation Substepping (B5):**
   - `policy_frequency: 5 Hz` ($dt = 0.2\text{ s}$), `simulation_frequency: 15 Hz` (3 substeps per step). Actions are held constant across the 3 simulation substeps.
4. **Observation Normalization (B6):**
   - Reconstructed scales: $R_x = 200.0\text{ m}$, $R_y = 16.0\text{ m}$ (in 4-lane highway, 4 lanes $\times$ 4m = 16m; not 12.0m as claimed in preliminary notes).
5. **Termination & Discrete Targets (B7, B8):**
   - `offroad_terminal = False` by default.
   - `DiscreteMetaAction.target_speeds = [20.0, 25.0, 30.0]\text{ m/s}`.

### C1–C8: Wrapper & Reward Verification
1. **Curved Lane Geometry (C1):** Verified across 30 random perturbed states on straight highway, curved merge, and circular roundabout. $\psi_{\text{err}}$ and $y_{\text{lane}}$ match independent ground-truth geometry with $0.000000\text{e}+00$ discrepancy. Lane tangent heading is properly evaluated at curvilinear coordinate $s$.
2. **Yaw Rate Finite Difference (C2):** $\dot{\psi}$ at reset step is correctly initialized to $0.0\text{ rad/s}$. Angle wrap across $\pm \pi$ (+3.14 to -3.14) wraps cleanly to $+0.0032\text{ rad}$, avoiding false $-6.28\text{ rad}$ derivative spikes.
3. **Domain Guards & Empirical Term Distribution (C3):**
   - $v \le 0$ is guarded via `max(v, 1.0)`, preventing division by zero.
   - Gap $\le 0$ is guarded via `max(0.001, dx - veh_len)`.
   - TTC is guarded via `max(ttc_epsilon, ttc)` with $\epsilon = 0.1\text{ s}$.
   - Empirical term distribution over 1,000 steps: Speed (17.3%), Jerk (0.4%), TTC (1.8%), Headway (1.5%), Overtake (0.7%), Control (0.2%), Crash (78.1% when occurring).
4. **Zero Perception Leakage (C4):** Tested 100 identical trajectory steps under Full-ADAS and Front-Only tiers. Differing reward steps: **0 / 100** ($0.0$ difference).
5. **Crash Penalty Discounting Flaw (C6):**
   $$V_{\text{cruise}}(T) = \sum_{t=0}^{T-1} 0.99^t (1.0) = \frac{1 - 0.99^T}{0.01}, \quad V_{\text{total}}(T) = V_{\text{cruise}}(T) - 200 \cdot 0.99^T$$
   - At $T = 1$: $V = -197.00$
   - At $T = 100$: $V = -9.81$
   - At $T = 110$: $V = +0.85$ (**break-even point**)
   - At $T = 200$: $V = +59.81$
   - At $T = 300$: $V = +85.29$
   An agent that speeds recklessly and crashes at step 200 collects a net return of **$+59.81$**, financially dominating safe standstill ($0.0$).
6. **Benchmark Regression & Weight Invalidation (C7, C8):**
   - Classical IDM+MOBIL Front-Only crashed in 4/5 seeds (80.0%) on seeds 2000–2004, matching benchmark failure characteristics.
   - Calling `train_optimal_overtaker.evaluate("models/ppo_optimal_overtaker_4lane.pt")` **CRASHES** with `RuntimeError: size mismatch for critic.0.weight: [256, 81] vs [256, 90]`. Step 2 broke evaluation of all pre-existing model weights!

---

## 6. Audit D: Plan Consistency, Statistics & Engineering Effort

### D1: GPU-Hour Ledger vs. Measured Throughput
- **Measured Local Throughput (`benchmark_throughput_results.json`):**
  - 16 workers (Ryzen 7 5800H): **895.3 SPS** (sustained 10-minute test: 537,184 steps, 1.28% thermal decay, CPU temp 91.2°C).
  - 4 workers (Kaggle vCPU equivalent): **513.3 SPS**.
- **Run Duration Reality:**
  - One 3.3M step training run on Kaggle (4 vCPU + T4):
    At 513 env steps/sec $\times$ 0.70 (PPO backprop/mini-batch overhead) $\approx 360\text{ SPS}$.
    $$\text{Wall-clock time} = \frac{3,300,000}{360} \approx 9,166\text{ seconds} \approx \mathbf{2.55\text{ hours per run}}$$
  - Planned Sprint-1 training matrix: 3 algorithms (PPO, SAC, TD3) $\times$ 2 tiers (Full-ADAS, Front-Only) $\times$ 3 seeds = 18 runs.
  - Total compute required: $18 \times 2.55\text{ hours} = \mathbf{45.9\text{ GPU hours}}$!
  - **Kaggle Weekly Quota Limit: 30 hours!** The planned matrix exceeds the total weekly Kaggle GPU allowance by **53%**, without accounting for debugging runs or failed seeds.

### D2: Statistical Significance & Sample Sizing
1. **Wilson vs. Clopper-Pearson CI on 3/30 crashes (10.0%):**
   - Wilson Score Interval: $[3.5\%, 25.6\%]$.
   - Clopper-Pearson Exact Interval: $[2.1\%, 26.5\%]$.
   - The plan's stated "~2–27%" is a rough blend of both conventions. Standardize on **Clopper-Pearson** for exact safety guarantees.
2. **Power Failure at $N = 30$ Seeds:**
   - Comparing an agent with 0/30 crashes (0.0%) against 3/30 crashes (10.0%):
     Fisher's two-sided exact test yields $p = 0.237 > 0.05$.
   - **Conclusion:** $N = 30$ evaluation seeds has **insufficient statistical power** to confirm that a 0% crash rate is statistically superior to a 10% crash rate at standard 95% confidence. To achieve 80% power at $\alpha = 0.05$ to detect a drop from 10% to 1%, evaluation requires at least **$N = 95$ seeds**.
3. **Training Run Variance:**
   - Single-seed training screening is invalid. Training-run variance (random weight initialization and policy sampling) often exceeds evaluation-seed variance. All architecture comparisons must train $\ge 3$ (ideally 5) seeds.

### D4: Engineering Human-Hours Breakdown
- Step 1 Profiling: 4 hours (Done)
- Step 2 Wrapper Extension: 6 hours (Done)
- Step 3 Reward Function: 8 hours (Done)
- Continuous PPO Architecture & Bug Fixes: 12 hours
- Continuous SAC / TD3 Adaptation: 24 hours
- Hyperparameter Sweeps & Reward Balancing: 20 hours
- Full Evaluation & Video Telemetry Harness: 10 hours
- **Total Estimated Effort: 84 human-hours.** Squeezing this into a 1-week sprint while maintaining statistical rigor is unfeasible; SAC/TD3 must be deferred to Sprint 2.

---

## 7. Audit E: Adversarial Exploit Probes Summary

All 15 adversarial exploit hypotheses were implemented in `audit/run_adversarial_probes.py` and executed against the simulation pipeline:

```
[Probe 1: Reversing]                SUCCESS (EXPLOIT FOUND): Speed reached -40.0 m/s; positive return (+1.33).
[Probe 2: Parking]                  SUCCESS (EXPLOIT FOUND): Stood at v=0 for 100 steps; 0 collisions, return (+1.31).
[Probe 3: Boundary Straddling]      SUCCESS (EXPLOIT FOUND): y=6.0 straddle hid lead vehicle from discrete wrapper.
[Probe 4: Road Edge Grazing]        SUCCESS (EXPLOIT FOUND): 40 steps offroad with terminated=False (offroad_terminal=False).
[Probe 5: Dwell Flicker]            FAIL (GUARD WORKING): Dwell counter correctly reset to 0 on position flip.
[Probe 6: Tailgating at h_min]      SUCCESS (EXPLOIT FOUND): Headway h=1.001s received exactly 0.0 penalty.
[Probe 7: Speed-Matching TTC]       SUCCESS (EXPLOIT FOUND): gap=2.0m, v_rel=0.0 produced reward_ttc = 0.0000.
[Probe 8: Traffic Slowdown Prov.]   FAIL (GUARD WORKING): Credited_overtakes set prevents multiple bonuses per car.
[Probe 9: Steady High Yaw]          SUCCESS (EXPLOIT FOUND): Steady steering u=[0, 1] incurred 0 jerk and 0 control penalty.
[Probe 10: End-of-Horizon Crash]    SUCCESS (EXPLOIT FOUND): V(s_500)=0 due to missing GAE truncation bootstrap.
[Probe 11: Clip Saturation]         FAIL (GUARD WORKING): Normalization bounded at [-1.0, 0.0], preventing inf/nan.
[Probe 12: Eval Circularity]        SUCCESS (EXPLOIT FOUND): Benchmark metrics mirror wrapper internal state.
[Probe 13: Trailing Traffic Blind]  SUCCESS (EXPLOIT FOUND): Closing rear vehicle produced zero warning penalty.
[Probe 14: Offroad Exit]            SUCCESS (EXPLOIT FOUND): Driving to y=-10.0 did not trigger crash.
[Probe 15: 2-Step Jitter Pulses]    SUCCESS (EXPLOIT FOUND): 2-step holding halved discrete jitter penalty.
```

---

## 8. Concrete Edits to Opus Plan & Step Prompts

### Edits to `docs/OPUS_PLAN.md`
1. **Section 3 (Compute Budget & GPU-Hour Ledger):**
   - Replace assumed 1600–2000 SPS with measured Kaggle 4-vCPU rate: **513 SPS (env-only) / 360 SPS (training)**.
   - Adjust training wall-clock per 3.3M run from 45 minutes to **2.55 hours**.
   - Reduce Sprint-1 scope: focus solely on **Continuous PPO** (6 runs = 15.3 hours, fitting within Kaggle's 30h weekly quota). Defer SAC/TD3 to Sprint 2.
2. **Section 4 (Reward Function):**
   - Increase crash penalty from $-200$ to **$-500$** or introduce an un-discounted terminal penalty $R_{\text{term}} = -200 / (1 - \gamma)$ so that crashing at step 200 yields negative return ($< 0.0$).
   - Add explicit heading error penalty: $R_{\text{heading}} = -0.20 \cdot (\psi_{\text{err}} / \psi_{\max})^2$.
   - Constrain speed range in ContinuousAction to `speed_range = (0.0, 30.0)` to eliminate reversing.
   - Set `"offroad_terminal": True` in all environment configs.
3. **Section 7 (Statistical Methodology):**
   - Mandate exact **Clopper-Pearson 95% CIs** for all crash rate reporting.
   - Increase final evaluation seeds from $N = 30$ to **$N = 100$** to achieve statistical power to distinguish $0\%$ from $10\%$ crash rates.
   - Require **$\ge 3$ training seeds** for all algorithmic comparisons.

### Edits to Step 2 Prompt
- Add backward-compatibility requirement: `make_optimal_env` MUST default `include_continuous_features=False` when instantiating environments for discrete/legacy evaluation, or provide explicit parameter pass-through in `train_optimal_overtaker.py:evaluate()`.

### Edits to Step 3 Prompt
- Fix the GAE truncation bug: mandate that all PPO training loops distinguish between `terminated` and `truncated`, and bootstrap $V(s_{t+1})$ when `truncated and not terminated`.
- Require full checkpoint serialization: model weights, optimizer state, global step, and RNG states, with a `--resume` CLI flag.

---

## 9. Questions for the Human

1. **Sprint Scope vs. Compute Limit:** Given Kaggle's 30-hour weekly GPU limit, training PPO, SAC, and TD3 across 3 seeds requires ~46 hours. Do you approve reducing Sprint 1 to Continuous PPO (15.3 hours), deferring SAC/TD3 to Sprint 2?
2. **Crash Penalty Philosophy:** Do you prefer increasing the terminal crash penalty to $-500$ (so $V_{\text{crash}}(200) < 0$), or introducing an undiscounted terminal penalty?
3. **Offroad Policy:** Should driving off the asphalt onto the shoulder/grass immediately terminate the episode (`offroad_terminal = True`), or should it apply a heavy continuous penalty?
4. **Multi-Scenario Architecture:** For Sprints 3–4 (Merge, Roundabout, Intersection), do you want to switch to an egocentric polar radar grid (360° range-bearing slots) or maintain a route-based Frenet frame representation?
