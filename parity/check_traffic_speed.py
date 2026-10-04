"""Step 1: Traffic Speed Check across 3 environments:
- official highway-fast-v0
- official highway-v0
- our make_optimal_env config

Runs 500 episodes per environment with ego held constant by IDLE.
Measures:
1. Distribution of non-ego vehicle speeds at t=0
2. Distribution of speed of vehicle directly ahead of ego over time
3. Fraction of steps where leader is slower than ego minimum setpoint (20 m/s)
4. Crash rate and collision stats under IDLE
"""

import copy
import sys
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401

# Import make_optimal_env from project root
sys.path.insert(0, ".")
from env_config import make_optimal_env


def run_eval(env_name: str, n_episodes: int = 500, seed_start: int = 1000):
    t0_speeds = []
    leader_speeds = []
    crashes = 0
    total_steps = 0
    steps_with_leader = 0
    steps_leader_below_20 = 0
    steps_leader_below_25 = 0

    for ep in range(n_episodes):
        seed = seed_start + ep
        if env_name == "highway-fast-v0":
            env = gym.make("highway-fast-v0")
            obs, info = env.reset(seed=seed)
        elif env_name == "highway-v0":
            env = gym.make("highway-v0")
            obs, info = env.reset(seed=seed)
        elif env_name == "make_optimal_env":
            # 0 frame stack for speed inspection
            env = make_optimal_env(seed=seed, frame_stack_k=0)
            obs, info = env.reset(seed=seed)
        else:
            raise ValueError(f"Unknown env: {env_name}")

        unwrapped = env.unwrapped
        ego = unwrapped.vehicle

        # Collect t=0 non-ego speeds
        for v in unwrapped.road.vehicles:
            if v is not ego:
                t0_speeds.append(float(v.speed))

        # Run episode with IDLE (action 1)
        done = False
        truncated = False
        while not (done or truncated):
            total_steps += 1
            # Check vehicle directly ahead of ego in ego's current lane
            front, _ = unwrapped.road.neighbour_vehicles(ego, ego.lane_index)
            if front is not None and front.position[0] > ego.position[0]:
                steps_with_leader += 1
                sp = float(front.speed)
                leader_speeds.append(sp)
                if sp < 20.0:
                    steps_leader_below_20 += 1
                if sp < 25.0:
                    steps_leader_below_25 += 1

            action = 1  # IDLE
            obs, reward, done, truncated, info = env.step(action)

        if ego.crashed:
            crashes += 1

        env.close()

    t0_arr = np.array(t0_speeds)
    leader_arr = np.array(leader_speeds)

    def stats(arr):
        if len(arr) == 0:
            return {}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(np.max(arr)),
        }

    return {
        "env_name": env_name,
        "n_episodes": n_episodes,
        "total_steps": total_steps,
        "crashes": crashes,
        "crash_rate": crashes / n_episodes,
        "t0_stats": stats(t0_arr),
        "leader_stats": stats(leader_arr),
        "steps_with_leader": steps_with_leader,
        "frac_steps_with_leader": steps_with_leader / max(total_steps, 1),
        "steps_leader_below_20": steps_leader_below_20,
        "frac_leader_below_20": steps_leader_below_20 / max(steps_with_leader, 1),
        "frac_all_steps_leader_below_20": steps_leader_below_20 / max(total_steps, 1),
        "steps_leader_below_25": steps_leader_below_25,
        "frac_leader_below_25": steps_leader_below_25 / max(steps_with_leader, 1),
    }


if __name__ == "__main__":
    envs = ["highway-fast-v0", "highway-v0", "make_optimal_env"]
    results = {}
    for e in envs:
        print(f"Running {e} (500 episodes)...", flush=True)
        res = run_eval(e, n_episodes=500, seed_start=1000)
        results[e] = res
        print(f"Done {e}: crash rate={res['crash_rate']:.3f} ({res['crashes']}/{res['n_episodes']})", flush=True)
        print(f"  t0 non-ego speed mean={res['t0_stats'].get('mean', 0):.2f}, min={res['t0_stats'].get('min', 0):.2f}, max={res['t0_stats'].get('max', 0):.2f}")
        print(f"  Leader speed mean={res['leader_stats'].get('mean', 0):.2f}, min={res['leader_stats'].get('min', 0):.2f}")
        print(f"  Steps with leader: {res['steps_with_leader']}/{res['total_steps']} ({res['frac_steps_with_leader']:.1%})")
        print(f"  Leader < 20 m/s: {res['steps_leader_below_20']}/{res['steps_with_leader']} ({res['frac_leader_below_20']:.2%}) of leader steps")

    import json
    with open("parity/traffic_speed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved parity/traffic_speed_results.json successfully.")
