# Technical findings

This document records empirical findings and algorithmic analyses from the project codebase. Each entry details the question investigated, the evaluation method, the artifact source, the quantitative result, and applicable caveats. Entries are organized by topic.

## Perception and coordinate transformation

### Metric coordinate reconstruction in sensor cone filters

Question: Why did the bearing angle filter in the forward sensor tier retain perpendicular vehicles while excluding vehicles directly ahead in adjacent lanes?

Method: In `env_config.py`, compare bearing angle calculations evaluated on normalized observation coordinates against calculations evaluated on metric coordinates reconstructed via `get_scaling_factors()`.

Evidence: `env_config.py:get_scaling_factors`, `scripts/audit/audit_b_environment_facts.py:65`.

Result: HighwayEnv normalizes longitudinal coordinates by $R_x = 200.0$ m and lateral coordinates by $R_y = 12.0$ m (or $16.0$ m depending on lane count). Computing bearing angles directly as $\text{atan2}(y_{\text{norm}}, x_{\text{norm}})$ evaluates $\text{atan2}(y/R_y, x/R_x)$, distorting the aspect ratio by a factor of $200.0 / 12.0 \approx 16.7$. This anisotropic scaling produced angular distortions up to $50^\circ$, causing lateral vehicles to fall within the nominal $120^\circ$ ($|\theta| \le 60^\circ$) forward cone while excluding forward vehicles in adjacent lanes. Reconstructing physical coordinates ($x = x_{\text{norm}} \cdot R_x$, $y = y_{\text{norm}} \cdot R_y$) prior to trigonometric calculations restored the intended geometric cone.

Caveats: Coordinate reconstruction requires querying scenario-specific scaling factors from the underlying environment configuration.

## Classical motion planning under degraded perception

### Trailing-vehicle blind-spot collision under front-only radar

Question: Why does the IDM+MOBIL controller crash in 50.0% of episodes under front-only radar compared to 6.7% under full omnidirectional perception?

Method: Execute `baseline_classical.py --scenario highway --tier front_only --controller idm_mobil --episodes 30 --seed-start 2000 --profile legacy` and compare with `baseline_classical.py --scenario highway --tier full_adas --controller idm_mobil --episodes 30 --seed-start 2000 --profile legacy`.

Evidence: `benchmark_results.md:43,48`.

Result: Under full ADAS across seeds 2000 to 2029, IDM+MOBIL recorded 2/30 crashes (6.7%), mean duration $193.6 \pm 32.5$ steps, mean speed 83.4 km/h, and 5.80 lane changes per episode. Under front-only radar across the same seeds, IDM+MOBIL recorded 15/30 crashes (50.0%), mean duration $147.1 \pm 64.3$ steps, mean speed 84.2 km/h, and 4.40 lane changes per episode. In `baseline_classical.py`, the MOBIL implementation treats an unobserved follower as no follower. When trailing traffic ($x < 0$) is masked by the front-only sensor cone, the distance to the follower evaluates to infinity ($s \to \infty$). Consequently, the deceleration penalty term in the MOBIL safety condition evaluates to zero, causing the safety check $\tilde{a}_{\text{new\_lag}} \ge -b_{\text{safe}}$ to evaluate to true unconditionally. The controller therefore cuts into the target lane directly in front of approaching unobserved traffic. This failure is a consequence of that design choice (treating an unobserved follower as absent), rather than an inherent property of the MOBIL model. Under identical conditions, IDM without lane changes (IDM only) recorded 0/30 crashes (0.0%) at 81.4 km/h (`benchmark_results.md:42,47`), because longitudinal car-following relies only on forward lead vehicles.

Caveats: Memoryless controllers cannot track vehicles once they enter blind spots.

## Tactical observation and metric formulation

### Longitudinal coordinate overtake metric exploitation

Question: Why did the agent register overtake events without executing lane changes or passing adjacent vehicles?

Method: Mathematical inspection of the overtake detection logic in `TacticalLaneObservationWrapper` and `TacticalOvertakingWrapper`.

Evidence: `env_config.py:TacticalLaneObservationWrapper._update_overtakes`.

Result: The initial overtake metric registered an overtake whenever any tracked vehicle's relative longitudinal coordinate transitioned to $x_{\text{rel}} < 0$, regardless of lateral lane assignment. On a four-lane highway, an ego vehicle cruising in lane 0 received overtake bonuses when vehicles in lane 3 decelerated. Constraining valid overtakes to adjacent corridors ($|\Delta \text{lane}| \le 1$ and $|x_{\text{rel}}| \le 30.0$ m) eliminated unearned rewards from distant lanes.

Caveats: Metrics based purely on longitudinal position require lateral proximity constraints in multi-lane environments.

### Action switching from discrete velocity quantization

Question: Why did the policy alternate between acceleration and deceleration actions on 77.0% of steps during steady car-following?

Method: Telemetry analysis of action sequences and vehicle velocities in `train_optimal_overtaker.py`.

Evidence: `env_config.py:TacticalLaneObservationWrapper.__init__` line 441.

Result: HighwayEnv's discrete action space defines target speeds of 20.0 m/s (72.0 km/h) for `SLOWER` and 25.0 m/s (90.0 km/h) for `FASTER`. When lead traffic traveled at an intermediate velocity (such as 22.1 m/s), the agent had no intermediate action. The policy alternated between `FASTER` and `SLOWER` every step, executing 384 action switches in a 500-step episode (a 77.0% switching rate). Adding an action switching penalty of $-0.12 \cdot \mathbb{I}(a_t \ne a_{t-1})$ reduced action switching to 6 switches in 500 steps (1.2%).

Caveats: The penalty regularizes policy output but does not alter the underlying speed discretization of the action space.

### Forward perception horizon and crawling deadlock

Question: Why did the policy decelerate to 15 to 20 km/h or stop on the highway when trailing lead traffic?

Method: Diagnostic review of blockage detection range and reward weights in `env_config.py`.

Evidence: `env_config.py:TacticalLaneObservationWrapper.__init__`.

Result: With a forward perception limit of 25.0 m, the agent detected lead vehicles late. At speeds of 20.0 to 25.0 m/s, 25.0 m provides 1.0 to 1.25 seconds of reaction time. The collision penalty of $-50.0$ dominated value estimates, while lane change incentives were insufficient to encourage a maneuver. The agent repeatedly selected `SLOWER`, dropping speed until it halted. Extending forward blockage sensing from 25.0 m to 55.0 m and adding an impatience penalty proportional to consecutive steps spent blocked enabled the policy to initiate lane changes before closing the headway distance.

Caveats: Expanding the sensing horizon increases observation computation when checking road vehicles.

## Horizon length and advantage estimation

### Environment duration configuration and episode truncation

Question: Why did evaluation episodes terminate at 200 steps when evaluation runs requested 500 steps?

Method: Inspect `duration` parameter settings in `highway-fast-v0` and step accounting in vectorized environment wrappers.

Evidence: `env_config.py:SCENARIO_CONFIGS['highway']`, `eval_out/checkpoint_profiles.json`.

Result: HighwayEnv defines a default `duration: 40` seconds. At policy frequency 5 Hz, the environment signals `truncated=True` at step 200 ($40 \times 5$). Overriding `duration = 100` allows 500-step episodes, and `duration = 200` allows 1,000-step episodes.

Caveats: Longer horizons require verifying that traffic generation maintains consistent density over extended roadway distances.

### Generalized Advantage Estimation bootstrapping across truncation boundaries

Question: How must value bootstrapping differ between time-limit truncations and terminal collision states?

Method: Unit test evaluation in `tests/test_truncation_bootstrapping.py` checking value bootstrap assignments across terminal flags.

Evidence: `train_optimal_overtaker.py`, `ppo.py`, `tests/test_truncation_bootstrapping.py`.

Result: When an episode ends due to a time limit (`truncated=True`), the agent did not crash, so the expected future return is non-zero. Generalized Advantage Estimation must bootstrap from the critic estimate $\gamma V(s_{t+1})$. When an episode terminates due to a collision (`terminated=True`), the continuation value is physically zero, so the bootstrap term must be zeroed ($\gamma V(s_{t+1}) = 0$). Setting bootstrap to zero on truncation introduces an artificial penalty against surviving to the time limit.

Caveats: Environments must correctly distinguish between `terminated` (physical collision) and `truncated` (time limit reached).

## Constrained reinforcement learning and advantage scaling

### Advantage variance disparity and separate unit standardization

Question: Why did combining reward and cost advantages in PPO-Lagrangian without separate normalization cause constraint failure?

Method: Compare policy optimization using combined advantage $A_{\text{combined}} = A_R - \lambda A_C$ normalized jointly against $A_{\text{adj}} = \text{norm}(A_R) - \lambda \cdot \text{norm}(A_C)$ normalized separately.

Evidence: `eval_out/crash_breakdown_ppo_lagrangian_sep_norm.json`, `tests/test_ppo_lagrangian.py`.

Result: In HighwayEnv, dense speed rewards yield a reward advantage standard deviation $\sigma(A_R) \approx 5\text{ to }15$, whereas sparse crash costs (1.0 on collision, 0.0 otherwise) yield $\sigma(A_C) \approx 0.3\text{ to }0.5$. With multiplier $\lambda \approx 0.14$, the cost penalty term $\lambda A_C \approx 0.04$ represented less than 1% of the reward advantage magnitude. Normalizing the combined advantage after subtraction caused the reward stream to swamp the cost gradient. Standardizing $A_R$ and $A_C$ independently to unit variance before combining preserved the cost constraint gradient. Evaluated on 100 benchmark seeds (2000 to 2099), the model with separate normalization achieved 34/100 crashes (34.0%, 95% Clopper-Pearson CI [24.8%, 44.2%]), mean speed 78.37 km/h, and mean duration 349.3 steps (`eval_out/crash_breakdown_ppo_lagrangian_sep_norm.json`).

Caveats: Independent standardization requires non-zero advantage variance for both streams within each minibatch.

### Cost continuation value zeroing on terminal collision states

Question: Does cost value estimation require zeroing continuation values upon collision termination?

Method: Test value bootstrap behavior on crash states using `TestCostGAETermination` in `tests/test_ppo_lagrangian.py`.

Evidence: `train_optimal_overtaker.py`, `tests/test_ppo_lagrangian.py`.

Result: In cost Generalized Advantage Estimation, setting $\gamma V_C(s_{t+1}) = 0$ on true collisions (`terminated=True`) prevents post-collision cost estimates from leaking backward into pre-collision transitions. The cost of a collision is assigned strictly to the terminal transition that caused it.

Caveats: The cost critic must be trained with identical terminal masking conventions to the reward critic.

### Lexicographic safety-first model selection under asymmetric risk

Question: Why does linear scalar score ranking fail to select the safest model checkpoint?

Method: Compare checkpoint selection under linear scalar scoring against lexicographic safety-first ordering in `train_optimal_overtaker.py`.

Evidence: `train_optimal_overtaker.py`, `tests/test_model_selection.py`.

Result: Linear scalar scoring allows checkpoints with higher mean speed to outscore zero-crash checkpoints despite suffering collisions. Lexicographic ordering ranks candidates first by crash count ($C_A < C_B$), breaking ties by mean speed ($\bar{v}_A > \bar{v}_B$). Under lexicographic selection, a candidate with zero crashes strictly dominates any candidate with one or more crashes, regardless of speed.

Caveats: Lexicographic selection requires at least one candidate to achieve zero crashes during validation; otherwise it selects the candidate with minimum crashes.

## Action space discretization and velocity floors

### Velocity quantization floor and in-lane rear-end collisions

Question: What caused 80.0% of crashes in the baseline optimal overtaker across 100 benchmark seeds?

Method: Run collision diagnostic classification across 100 evaluation seeds (2000 to 2099) using `scripts/diagnose_legacy_crashes.py`.

Evidence: `eval_out/crash_breakdown_optimal_overtaker.json`, `eval_out/crash_breakdown_optimal_overtaker_extended_speeds.json`.

Result: In `eval_out/crash_breakdown_optimal_overtaker.json`, `models/ppo_optimal_overtaker_4lane_best.pt` crashed in 10/100 episodes (10.0%, 95% Clopper-Pearson CI [4.9%, 17.6%]). Of those 10 crashes, 8 (80.0%) were in-lane rear-end collisions where the lead vehicle traveled at less than 20.0 m/s (such as 16.3 m/s on seed 2003). Because `DiscreteMetaAction` defines target speeds as $[20.0, 25.0, 30.0]$ m/s, action `SLOWER` cannot command a speed below 20.0 m/s. When adjacent lanes were blocked by surrounding vehicles, the ego vehicle could not match the lead vehicle's speed, making in-lane collisions physically unavoidable. Expanding `target_speeds` to $[10.0, 15.0, 20.0, 25.0, 30.0]$ m/s allowed the agent to decelerate to 10.0 or 15.0 m/s, yielding 0/100 crashes (0.0%, 95% Clopper-Pearson CI [0.0%, 3.6%]) in `eval_out/crash_breakdown_optimal_overtaker_extended_speeds.json` and `eval_out/deepset_v2_retrain_seed42_100seeds.json`.

Caveats: Expanding discrete target speeds alters the action space and requires retraining policies on the modified speed setpoints.

## Empirical benchmark comparisons

### Sensor perception ablation across 30 seeds

Question: How do classical controllers and reinforcement learning policies compare across full ADAS and front-only perception?

Method: Evaluate IDM, IDM+MOBIL, feedforward PPO, and recurrent PPO-LSTM across seeds 2000 to 2029 under full ADAS and front-only radar.

Evidence: `benchmark_results.md:40-52`.

Result: The performance metrics across seeds 2000 to 2029 under the legacy profile (10 vehicles, density 1.0, 40 s duration) are:

Full ADAS perception:
- IDM Only: 0/30 crashes (0.0%), duration $200.0 \pm 0.0$ steps, speed 81.4 km/h, 0.00 lane changes, 0.00 overtakes per episode.
- IDM + MOBIL: 2/30 crashes (6.7%), duration $193.6 \pm 32.5$ steps, speed 83.4 km/h, 5.80 lane changes, 1.87 overtakes per episode.
- Feedforward PPO: 0/30 crashes (0.0%), duration $200.0 \pm 0.0$ steps, speed 73.3 km/h, 0.67 lane changes, 0.20 overtakes per episode.
- Recurrent PPO-LSTM: 0/30 crashes (0.0%), duration $200.0 \pm 0.0$ steps, speed 72.6 km/h, 1.53 lane changes, 0.47 overtakes per episode.

Front-only radar perception:
- IDM Only: 0/30 crashes (0.0%), duration $200.0 \pm 0.0$ steps, speed 81.4 km/h, 0.00 lane changes, 0.00 overtakes per episode.
- IDM + MOBIL: 15/30 crashes (50.0%), duration $147.1 \pm 64.3$ steps, speed 84.2 km/h, 4.40 lane changes, 1.20 overtakes per episode.
- Feedforward PPO: 3/30 crashes (10.0%), duration $187.8 \pm 40.9$ steps, speed 79.3 km/h, 1.53 lane changes, 0.53 overtakes per episode.
- Recurrent PPO-LSTM: 0/30 crashes (0.0%), duration $200.0 \pm 0.0$ steps, speed 72.6 km/h, 1.53 lane changes, 0.53 overtakes per episode.

Caveats: Evaluation is restricted to 30 deterministic test seeds with 10 surrounding vehicles.

### Benchmark comparison under extended horizon on 100 seeds

Question: What are the crash rates, speeds, and failure distributions of PPO baselines across 100 seeds under the 100 s horizon?

Method: Evaluate `models/ppo_optimal_overtaker_4lane_best.pt` and `scratch/ppo_lagrangian_1m_sep_norm/ppo_optimal_overtaker_4lane_best.pt` across seeds 2000 to 2099 under 14 vehicles, density 1.4, and duration 100 s.

Evidence: `eval_out/crash_breakdown_optimal_overtaker.json`, `eval_out/crash_breakdown_ppo_lagrangian_sep_norm.json`.

Result:
- Baseline PPO (`models/ppo_optimal_overtaker_4lane_best.pt`): 10/100 crashes (10.0%, 95% Clopper-Pearson CI [4.9%, 17.6%]), mean speed 73.62 km/h, mean lane changes 1.63, mean return 34.33. Crash breakdown: 8 floor-speed rear-ends (80.0%), 2 lane-change collisions (20.0%).
- PPO-Lagrangian with separate normalization: 34/100 crashes (34.0%, 95% Clopper-Pearson CI [24.8%, 44.2%]), mean speed 78.37 km/h, mean lane changes 1.61, mean return 6.83. Crash breakdown: 8 floor-speed rear-ends (23.5%), 13 lane-change collisions (38.2%), 13 blind-spot or other contacts (38.2%).

Caveats: Both models were evaluated with the default $[20.0, 25.0, 30.0]$ m/s action space.

### Demonstration trajectory extraction across traffic seeds

Question: Which simulation seeds demonstrate non-idle tactical maneuvers without collisions?

Method: Evaluate candidate runs from 150 seeds using composite score: $(\text{overtakes} \times 35) + (\text{lane changes} \times 12) + (\text{FASTER} \times 1.5) + (\bar{v} - 70.0) \times 2.0$. Filter for zero collisions, $\ge 12$ FASTER actions, $\ge 3$ lane changes, and $\ge 1$ overtake over 500 steps.

Evidence: `PROJECT_CHRONICLE_AND_POSTMORTEM.md:378-390`, `render_top10.py`, `visualizations/optimal_seed*.mp4`.

Result: The top 10 demonstration seeds and their performance metrics are:
- Seed 3094: 5 overtakes, 4 lane changes, actions 102 FASTER / 288 IDLE / 101 SLOWER, mean speed 86.7 km/h, max speed 108.0 km/h, score 409.5.
- Seed 3115: 1 overtake, 15 lane changes, actions 63 FASTER / 331 IDLE / 64 SLOWER, mean speed 75.2 km/h, max speed 90.0 km/h, score 320.0.
- Seed 3010: 4 overtakes, 3 lane changes, actions 40 FASTER / 423 IDLE / 32 SLOWER, mean speed 84.0 km/h, max speed 108.0 km/h, score 264.1.
- Seed 3000: 1 overtake, 11 lane changes, actions 47 FASTER / 389 IDLE / 47 SLOWER, mean speed 74.4 km/h, max speed 90.0 km/h, score 246.2.
- Seed 3019: 1 overtake, 14 lane changes, actions 12 FASTER / 445 IDLE / 10 SLOWER, mean speed 76.5 km/h, max speed 100.3 km/h, score 234.1.
- Seed 3092: 2 overtakes, 3 lane changes, actions 27 FASTER / 444 IDLE / 24 SLOWER, mean speed 78.4 km/h, max speed 107.9 km/h, score 163.4.
- Seed 3078: 1 overtake, 3 lane changes, actions 40 FASTER / 414 IDLE / 40 SLOWER, mean speed 76.2 km/h, max speed 105.7 km/h, score 143.4.
- Seed 3028: 1 overtake, 6 lane changes, actions 17 FASTER / 453 IDLE / 17 SLOWER, mean speed 73.8 km/h, max speed 90.0 km/h, score 140.0.
- Seed 2010: 1 overtake, 5 lane changes, actions 20 FASTER / 452 IDLE / 20 SLOWER, mean speed 76.4 km/h, max speed 107.9 km/h, score 137.8.
- Seed 3049: 1 overtake, 3 lane changes, actions 36 FASTER / 423 IDLE / 35 SLOWER, mean speed 73.4 km/h, max speed 90.7 km/h, score 131.9.

Caveats: These seeds represent filtered demonstration runs selected from 150 candidate seeds, not an average across random traffic.
