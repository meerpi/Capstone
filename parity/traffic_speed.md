# Step 1: Traffic Speed Check Report

## 1. Experimental Setup
We evaluated 500 episodes each across three environments with the ego vehicle held strictly constant by the `IDLE` action (action 1):
1. **Official `highway-fast-v0`** (30 s duration, 1 Hz policy, 5 Hz sim, 3 lanes, 20 vehicles, density 1.0)
2. **Official `highway-v0`** (40 s duration, 1 Hz policy, 15 Hz sim, 4 lanes, 50 vehicles, density 1.0)
3. **Our `make_optimal_env`** (100 s duration, 5 Hz policy, 5 Hz sim, 4 lanes, 14 vehicles, density 1.4)

Seeds: 1000 to 1499 (500 episodes).
Measurement code: [`parity/check_traffic_speed.py`](file:///home/meerpi/curr_project/capstone/parity/check_traffic_speed.py)
Raw output data: [`parity/traffic_speed_results.json`](file:///home/meerpi/curr_project/capstone/parity/traffic_speed_results.json)

---

## 2. Quantitative Results

### Summary Table

| Metric | `highway-fast-v0` | `highway-v0` | Our `make_optimal_env` |
| :--- | :--- | :--- | :--- |
| **Episodes** | 500 | 500 | 500 |
| **Total Steps** | 7,605 | 9,985 | 36,556 |
| **Ego Crashes (under IDLE)** | 472 / 500 (**94.4%**) | 474 / 500 (**94.8%**) | 498 / 500 (**99.6%**) |
| **Non-ego speed at $t=0$ (mean $\pm$ std)** | $22.50 \pm 0.86$ m/s | $22.50 \pm 0.86$ m/s | $22.49 \pm 0.86$ m/s |
| **Non-ego speed at $t=0$ [min, max]** | $[21.00, 24.00]$ m/s | $[21.00, 24.00]$ m/s | $[21.00, 24.00]$ m/s |
| **Leader speed over time (mean $\pm$ std)**| $20.11 \pm 1.59$ m/s | $20.76 \pm 1.38$ m/s | $20.65 \pm 2.22$ m/s |
| **Leader speed minimum** | **10.93 m/s** | **12.08 m/s** | **6.37 m/s** |
| **Steps with in-lane leader ahead** | 7,605 / 7,605 (100.0%) | 9,985 / 9,985 (100.0%) | 36,509 / 36,556 (99.9%) |
| **Steps leader $< 20$ m/s (ego floor setpoint)**| 3,165 / 7,605 (**41.62%**) | 2,316 / 9,985 (**23.19%**) | 10,303 / 36,509 (**28.22%**) |
| **Steps leader $< 25$ m/s (ego start speed)** | 7,605 / 7,605 (**100.0%**) | 9,985 / 9,985 (**100.0%**) | 36,509 / 36,509 (**100.0%**) |

---

## 3. Percentile Distributions

### Non-Ego Vehicle Initial Speed ($t=0$) [m/s]
All environments match the theoretical sampling distribution:
$U(0.7 \times 30, 0.8 \times 30) = U(21.0, 24.0)$ m/s.

| Percentile | `highway-fast-v0` | `highway-v0` | `make_optimal_env` |
| :--- | :--- | :--- | :--- |
| **Min** | 21.00 | 21.00 | 21.00 |
| **p10** | 21.30 | 21.30 | 21.30 |
| **p25** | 21.75 | 21.75 | 21.74 |
| **p50 (Median)** | 22.49 | 22.50 | 22.49 |
| **p75** | 23.24 | 23.24 | 23.24 |
| **p90** | 23.71 | 23.70 | 23.70 |
| **Max** | 24.00 | 24.00 | 24.00 |

### In-Lane Leader Vehicle Speed Over Time [m/s]

| Percentile | `highway-fast-v0` | `highway-v0` | `make_optimal_env` |
| :--- | :--- | :--- | :--- |
| **Min** | 10.93 | 12.08 | **6.37** |
| **p10** | 18.07 | 19.15 | 17.72 |
| **p25** | 19.33 | 20.07 | 19.75 |
| **p50 (Median)** | 20.24 | 20.89 | 21.05 |
| **p75** | 21.10 | 21.68 | 22.07 |
| **p90** | 21.91 | 22.28 | 23.13 |
| **Max** | 23.99 | 23.99 | 23.99 |

---

## 4. Hypothesis H-B Verdict

> **Hypothesis H-B**: In our config traffic is slower than 20 m/s often enough that in-lane driving at the floor speed cannot avoid rear-ending.

### Verdict: **SUPPORTED**

### Empirical Evidence:
1. **Sub-20 m/s Traffic Occurrence**: In `make_optimal_env`, the lead vehicle is slower than 20 m/s on **28.22% of steps** (10,303 out of 36,509 steps where a leader exists ahead). Lead speeds plunge as low as **6.37 m/s** (22.9 km/h) due to vehicle queueing and deceleration waves.
2. **Floor-Speed Limit**: In the discrete action space (`DiscreteMetaAction`), `DEFAULT_TARGET_SPEEDS = [20.0, 25.0, 30.0]` m/s. The ego vehicle **physically cannot select a speed setpoint lower than 20.0 m/s** via `SLOWER`.
3. **Inevitability of Rear-Ending**: When an in-lane lead vehicle decelerates to 6–19 m/s, an ego vehicle that stays in its lane—even if it selects `SLOWER` until it hits the floor setpoint (20 m/s)—still has a positive closing speed relative to the leader of $\Delta v \in [1, 14]$ m/s. Because `ControlledVehicle` tracks its target speed without an emergency braking stop in `act()` when following road, in-lane driving at the floor speed **cannot avoid rear-ending the slower leader without a lateral lane change**.
4. **IDLE Policy Failure**: Holding ego at `IDLE` (initial speed 25 m/s) results in a **99.6% crash rate** in `make_optimal_env` (498/500 episodes crash into the lead vehicle). Even in official `highway-fast-v0`, `IDLE` crashes in **94.4%** of episodes.
