# Known issues and limitations

This document records limitations, open questions, and known discrepancies in the repository. Each entry details the observed symptom, the underlying cause (or identifies it as unknown), the affected files or checkpoints, and the current workaround.

## Evaluation discrepancy on optimal overtaker checkpoint

Symptom: The checkpoint `models/ppo_optimal_overtaker_4lane_best.pt` was initially reported in project logs at 0/30 crashes (0.0%), 76.4 km/h mean speed, and 1.67 overtakes per episode on seeds 2000 to 2029 under a 500-step horizon. A subsequent re-evaluation on the same seeds gave 4/30 crashes (13.3%), 73.3 km/h mean speed, and 0.57 overtakes per episode. On the 100-seed benchmark (seeds 2000 to 2099), `eval_out/crash_breakdown_optimal_overtaker.json` records 10/100 crashes (10.0%), 73.62 km/h mean speed, and 1.63 lane changes per episode.

Cause: Unknown. The cause of the performance divergence between the earlier 0/30 report and the subsequent 4/30 re-evaluation on seeds 2000 to 2029 is unconfirmed. Possible contributing factors include environment profile differences (such as vehicle counts, density, or reward wrapper parameters) across commits. No separate 30-seed JSON artifact for the 13.3% result exists in `eval_out/`.

Affected files or checkpoints: `models/ppo_optimal_overtaker_4lane_best.pt`, `train_optimal_overtaker.py`.

Workaround: Evaluate checkpoints using the explicit environment profile configurations documented in `eval_out/checkpoint_profiles.json`, specify exact evaluation seeds, and record the git commit hash with each evaluation output.

## Advantage scale distortion in unnormalized constrained calibration run

Symptom: An initial PID-Lagrangian training run (crash cost 1.0, constraint budget 0.05, $K_p = 0.05$, $K_i = 0.001$) failed with 82/100 crashes (82.0%, 95% Clopper-Pearson CI [73.1%, 89.0%]) and a mean speed of 105.06 km/h across seeds 2000 to 2099.

Cause: Combining unnormalized reward and cost advantage streams ($A_{\text{combined}} = A_R - \lambda A_C$) mixed two disparate variance distributions. Dense speed rewards had advantage standard deviation $\sigma(A_R) \approx 5\text{ to }15$, while sparse collision costs had $\sigma(A_C) \approx 0.3\text{ to }0.5$. With multiplier $\lambda \approx 0.142$, the cost term accounted for less than 1% of the reward advantage magnitude. Normalizing the combined advantage after subtraction caused the reward stream to swamp the cost penalty, neutralizing the constraint gradient.

Affected files or checkpoints: The artifact for this run resides under `scratch/archived_runs/failed_calibration_82pct_crash/README.txt`, which is gitignored, so no committed artifact exists in the repository. Checkpoints for the separate-normalization run (which achieved 34/100 crashes) also reside in gitignored `scratch/ppo_lagrangian_1m_sep_norm/`, although its evaluation summary is committed in `eval_out/crash_breakdown_ppo_lagrangian_sep_norm.json`.

Workaround: Standardize reward and cost advantages independently to unit scale prior to combination ($A_{\text{adj}} = \text{norm}(A_R) - \lambda \cdot \text{norm}(A_C)$) without joint re-normalization. Direct all future benchmark evaluation logs to tracked paths under `eval_out/`.

## DeepSet encoder versus legacy encoder evaluation at 1M steps

Symptom: Evaluated at 1,000,000 training steps across 100 benchmark seeds (2000 to 2099), DeepSet v2 recorded 11/100 crashes (11.0%, 95% Clopper-Pearson CI [5.6%, 18.8%]), while the legacy OptimalAgent recorded 10/100 crashes (10.0%, 95% Clopper-Pearson CI [4.9%, 17.6%]).

Cause: Both models experienced in-lane rear-end collisions behind sub-20 m/s lead traffic (10 of 11 crashes for DeepSet v2; 8 of 10 crashes for legacy OptimalAgent) because the default discrete action space cannot command speeds below 20.0 m/s.

Affected files or checkpoints: `deepset_v2/model.py`, `eval_out/crash_diagnostics/crash_diagnosis_report.json`, `eval_out/crash_breakdown_optimal_overtaker.json`.

Workaround: This comparison is an open question. Neither parity nor an adoption verdict has been accepted. The 1-crash difference between 11/100 and 10/100 is within binomial sampling error, and adoption remains undecided.

## Ground-truth observation in tactical observation wrapper

Symptom: The optimal overtaker policy accesses complete simulator road state rather than processing sensor-limited observations.

Cause: `TacticalLaneObservationWrapper._build_tactical_obs()` iterates directly over `env.unwrapped.road.vehicles`, bypassing `SensorTierWrapper` and the raw `KinematicsObservation` matrix.

Affected files or checkpoints: `env_config.py:TacticalLaneObservationWrapper`, `train_optimal_overtaker.py`, `models/ppo_optimal_overtaker_4lane.pt`, `models/ppo_optimal_overtaker_4lane_best.pt`.

Workaround: The DeepSet v2 pipeline (`deepset_v2/env.py`) reads exclusively from `info["_raw_kinematics"]` through `apply_sensor_tier`. However, the trained checkpoints in `models/` remain dependent on direct access to `road.vehicles`.

## Autoreset boundary transition contamination in pre-fix checkpoints

Symptom: In vectorized environments, episodes that terminate return an initial observation $s_0$ from the reset environment on the same step as the terminal reward $r_T$ and done flag $d_T$ from the preceding episode.

Cause: Gymnasium `AsyncVectorEnv` and `SyncVectorEnv` default to `autoreset="NEXT_STEP"`. In training implementations without boundary transition masking, the transition $(s_0, r_T, d_T)$ was treated as a normal step, causing policy gradient and value loss updates to link the terminal state of one episode with the initial state of the next.

Affected files or checkpoints: The codebase now masks phantom samples via $b_{\text{valid}} = 1.0 - b_{\text{dones}}$ in `train_optimal_overtaker.py:613`, `ppo.py:440`, and `scripts/train_deepset_v2.py:470`. Checkpoints trained prior to this fix were trained without masking:
- `models/ppo_highway_full_adas_seed101.pt`
- `models/ppo_highway_front_only_seed101.pt`
- `models/ppo_lstm_highway_full_adas_seed101.pt`
- `models/ppo_lstm_highway_front_only_seed101.pt`
- `models/ppo_optimal_overtaker_4lane.pt`
- `models/ppo_optimal_overtaker_4lane_best.pt`

Workaround: Verify boundary masking using `tests/test_ppo_lstm_phantom.py`. Retrain policies with $b_{\text{valid}}$ active if boundary sample purity is required.

## Velocity quantization floor in default action space

Symptom: Policies cannot decelerate below 20.0 m/s (72.0 km/h) in-lane, leading to unavoidable rear-end collisions when lead vehicles travel at 15.0 to 19.0 m/s and adjacent lanes are occupied.

Cause: `DiscreteMetaAction` defines target speeds as $[20.0, 25.0, 30.0]$ m/s. The minimum selectable target speed is 20.0 m/s.

Affected files or checkpoints: `env_config.py`, `models/ppo_optimal_overtaker_4lane.pt`, `models/ppo_optimal_overtaker_4lane_best.pt`.

Workaround: Expand `target_speeds` to $[10.0, 15.0, 20.0, 25.0, 30.0]$ m/s as implemented in `scripts/train_deepset_v2.py:108` and verified in `eval_out/crash_breakdown_optimal_overtaker_extended_speeds.json`.

## Simulation throughput limits and multi-agent realism trade-offs

Symptom: Simulation throughput on CPU is bounded to 500 to 900 steps per second across parallel workers, limiting scaling for long-horizon multi-agent training.

Cause: HighwayEnv executes vehicle kinematics, IDM updates, and bounding-box collision checks in Python on CPU.

Affected files or checkpoints: `env_config.py`, `benchmark_throughput_results.json`.

Workaround: GPUDrive provides GPU-accelerated simulation ($>1,000,000$ steps per second) using Waymo Open Motion Dataset logs, but replays static log trajectories with limited multi-agent reactivity. HighwayEnv was retained because it provides interactive IDM traffic, configurable sensor cones, and discrete action spaces.

## Observation dimension incompatibility across continuous feature flags

Symptom: Loading an 81-dimensional checkpoint into an environment configured with continuous stability features raises shape mismatch errors (expected 81, received 90).

Cause: Adding lateral offset $y_{\text{lane}}$, heading error $\psi_{\text{err}}$, and yaw rate $\dot{\psi}$ expands the base observation from 27 to 30 dimensions, which expands a 3-frame stack from 81 to 90 dimensions.

Affected files or checkpoints: `env_config.py:TacticalLaneObservationWrapper`, `models/ppo_optimal_overtaker_4lane.pt`, `models/ppo_optimal_overtaker_4lane_best.pt`.

Workaround: Set `include_continuous_features=False` when evaluating 81-dimensional models, or inspect checkpoint shape metadata in `eval_out/checkpoint_profiles.json` prior to instantiating the environment.
