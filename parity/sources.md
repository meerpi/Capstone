# SOTA-Parity Ladder: Sources and Reference Configurations

## 0. Overview
This document records exact source code snippets, configuration dictionaries, and published benchmark hyperparameters across:
- Installed `highway-env` package (v1.10.1 / site-packages)
- Farama-Foundation `scripts/sb3_highway_ppo.py`
- `github.com/Yousefz-z/highway-rl-generalization`
- `eleurent/rl-agents` (Ego-Attention DQN on Highway)
- Comparison against our repository's `env_config.py` and `make_optimal_env`.

---

## 1. Installed highway-env Source

File: `/home/meerpi/curr_project/capstone/.venv/lib/python3.12/site-packages/highway_env/envs/highway_env.py`

### 1.1 HighwayEnv.default_config
```python
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        utils.update_config(
            config,
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            },
        )
        return config
```

### 1.2 HighwayEnvFast
```python
class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        utils.update_config(
            cfg,
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            },
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
```
- `simulation_frequency`: 5 (vs 15 in `HighwayEnv`)
- `lanes_count`: 3 (vs 4 in `HighwayEnv`)
- `vehicles_count`: 20 (vs 50 in `HighwayEnv`)
- `duration`: 30 s (vs 40 s in `HighwayEnv`)
- `vehicle.check_collisions = False` for all non-controlled vehicles: **uncontrolled vehicles ghost through each other without collision checks**.

### 1.3 `_reward` and `normalize_reward`
From `highway_env/envs/highway_env.py`:
```python
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/Farama-Foundation/HighwayEnv/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }
```
When `normalize_reward=True`, raw reward in `[-1.0, 0.4 + 0.1]` = `[-1.0, 0.5]` is linearly mapped to `[0, 1]`:
$R_{\text{norm}} = \frac{R - (-1)}{0.5 - (-1)} = \frac{R + 1}{1.5}$.
Hence collision gives $R_{\text{norm}} = 0$, while max speed in rightmost lane gives $1.0$.

### 1.4 MDPVehicle Target Speeds and Lane Change Handling
From `highway_env/vehicle/controller.py`:
```python
class MDPVehicle(ControlledVehicle):
    """A controlled vehicle with a specified discrete range of allowed target speeds."""

    DEFAULT_TARGET_SPEEDS = np.linspace(20, 30, 3)
```
Target speeds are `[20.0, 25.0, 30.0]` m/s.

Lane index updates in `ControlledVehicle.act`:
```python
        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                self.position
            ):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                self.position
            ):
                self.target_lane_index = target_lane_index
```
From `highway_env/road/lane.py`:
```python
    def is_reachable_from(self, position: np.ndarray) -> bool:
        if self.forbidden:
            return False
        longitudinal, lateral = self.local_coordinates(position)
        is_close = (
            np.abs(lateral) <= 2 * self.width_at(longitudinal)
            and 0 <= longitudinal < self.length + self.VEHICLE_LENGTH
        )
        return is_close
```
**Repeated lane changes**: `target_lane_index` increments/decrements `_id` relative to `self.target_lane_index`, NOT `self.lane_index`. Since `is_reachable_from` accepts lateral distance up to $2 \times \text{width}$, an agent that issues `LANE_RIGHT` twice in rapid succession will advance `target_lane_index` by 2 lanes immediately. A third immediate call while still in the initial lane is rejected (`lateral > 2 * width`), keeping `target_lane_index` at +2 until the vehicle physically shifts.

### 1.5 Background Vehicle Speed Initialisation and IDM Target Speed
From `highway_env/vehicle/kinematics.py` lines 82-90:
```python
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(
                    0.7 * lane.speed_limit, 0.8 * lane.speed_limit
                )
            else:
                speed = road.np_random.uniform(
                    Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1]
                )
```
In `HighwayEnv._create_road`:
```python
self.road = Road(
    network=RoadNetwork.straight_road_network(
        self.config["lanes_count"], speed_limit=30
    ), ...
)
```
Since `lane.speed_limit = 30`, initial speed of non-ego vehicles is sampled uniformly from:
$[0.7 \times 30, 0.8 \times 30] = [21.0, 24.0]\text{ m/s}$.

Target speed of non-ego vehicles:
When `Vehicle.create_random` creates an `IDMVehicle`, `target_speed` is initialized in `ControlledVehicle.__init__` to `self.target_speed = target_speed or self.speed`.
Thus, the non-ego vehicle's IDM target speed is set to its sampled initial speed ($21.0$ to $24.0$ m/s).

### 1.6 `env.unwrapped.config` for `highway-fast-v0`
Command:
```python
import gymnasium as gym
import highway_env
env = gym.make("highway-fast-v0")
print(env.unwrapped.config)
```
Exact printed output:
```python
{'action': {'type': 'DiscreteMetaAction'},
 'centering_position': [0.3, 0.5],
 'collision_reward': -1,
 'controlled_vehicles': 1,
 'duration': 30,
 'ego_spacing': 1.5,
 'high_speed_reward': 0.4,
 'initial_lane_id': None,
 'lane_change_reward': 0,
 'lanes_count': 3,
 'manual_control': False,
 'neighbour_vehicles_connected_lanes': False,
 'normalize_reward': True,
 'observation': {'type': 'Kinematics'},
 'offroad_terminal': False,
 'offscreen_rendering': True,
 'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
 'policy_frequency': 1,
 'real_time_rendering': False,
 'render_agent': True,
 'reward_speed_range': [20, 30],
 'right_lane_reward': 0.1,
 'scaling': 5.5,
 'screen_height': 150,
 'screen_width': 600,
 'show_trajectories': False,
 'simulation_frequency': 5,
 'vehicles_count': 20,
 'vehicles_density': 1}
```

---

## 2. Farama `scripts/sb3_highway_ppo.py`
URL: `https://raw.githubusercontent.com/Farama-Foundation/HighwayEnv/master/scripts/sb3_highway_ppo.py`

Verbatim script:
```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env  # noqa: F401


# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    train = True
    if train:
        n_cpu = 6
        batch_size = 64
        env = make_vec_env("highway-fast-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=batch_size * 12 // n_cpu,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            tensorboard_log="highway_ppo/",
        )
        # Train the agent
        model.learn(total_timesteps=int(2e4))
        # Save the agent
        model.save("highway_ppo/model")

    model = PPO.load("highway_ppo/model")
    env = gym.make("highway-fast-v0")
    for _ in range(5):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
```

Hyperparameter summary:
- `n_envs`: 6 (`SubprocVecEnv`)
- `n_steps`: $64 \times 12 // 6 = 128$ steps per env (total rollout batch $128 \times 6 = 768$ steps)
- `batch_size`: 64
- `n_epochs`: 10
- `learning_rate`: 5e-4
- `gamma`: 0.8
- `net_arch`: `pi=[256, 256]`, `vf=[256, 256]`
- `total_timesteps`: $20,000$ (2e4)

---

## 3. Published Generalization & Ego-Attention Sources

### 3.1 `github.com/Yousefz-z/highway-rl-generalization`
Repository: `https://github.com/Yousefz-z/highway-rl-generalization`
- **Density Settings**:
  - `low`: `vehicles_density = 0.5`
  - `medium`: `vehicles_density = 1.0` (standard `highway-fast-v0`)
  - `high`: `vehicles_density = 1.5`
  - Note: Leaves `vehicles_count = 20` unchanged because extra vehicles spawn far down the road and are never met in a 30-second episode.
- **Evaluation Protocol**:
  - 100 evaluation episodes per density on shared seeds starting at `10_000` (i.e. `10000..10099`).
  - Action space: `DiscreteMetaAction` (5 actions: `LANE_LEFT=0, IDLE=1, FASTER=2, SLOWER=3, LANE_RIGHT=4`).
  - Metrics tracked: `mean_reward`, `crash_rate`, `mean_speed`, `dominant_action`, `dominant_action_frac`.
- **Key Published Result**:
  - For standard PPO trained on medium density (1x) with 50,000 steps:
    - Mean reward: $20.81 \pm 0.01$
    - Crash rate: $0.02 \pm 0.00$ ($2\%$)
    - Mean speed: $20.03 \pm 0.02\text{ m/s}$ (exactly the minimum floor speed setpoint)
    - Dominant action: `LANE_RIGHT` on $95\%$ of all steps!
  - At high density (1.5x):
    - Crash rate escalates to $0.69 \pm 0.01$ ($69\%$), still playing `LANE_RIGHT` $96\%$ of steps.
  - Published Verdict: Standard PPO on `highway-fast-v0` converges to a trivial, open-loop policy that hugs the right lane at floor speed ($20\text{ m/s}$).

### 3.2 `eleurent/rl-agents` Ego-Attention Configuration
Repository: `https://github.com/eleurent/rl-agents`
File: `scripts/configs/HighwayEnv/agents/DQNAgent/ego_attention.json`
- **Environment**: `highway-v0`
- **Discount Factor**: `gamma = 0.99`
- **Observation Features**: `in = 7` features per vehicle (`presence, x, y, vx, vy, cos_h, sin_h`).
- **Architecture**:
  - Embedding MLP: `[64, 64]`
  - Others Embedding MLP: `[64, 64]`
  - Multi-head Ego-Attention: 2 heads, feature size 64
  - Value Head MLP: `[64, 64]`

---

## 4. Systematic Comparison: Official vs Our Pipeline (`env_config.py` / `make_optimal_env`)

| Parameter / Feature | Official `highway-fast-v0` | Official `highway-v0` | Our `make_optimal_env` |
| :--- | :--- | :--- | :--- |
| **Lanes count** | 3 | 4 | 4 |
| **Vehicles count** | 20 | 50 | 14 |
| **Vehicles density** | 1.0 | 1.0 | 1.4 |
| **Episode duration** | 30 s (30 steps @ 1 Hz) | 40 s (40 steps @ 1 Hz) | 100 s (500 steps @ 5 Hz) |
| **Policy frequency** | 1 Hz | 1 Hz | 5 Hz |
| **Simulation frequency**| 5 Hz | 15 Hz | 5 Hz (1:1 with policy) |
| **Non-ego collisions** | Disabled (`check_collisions=False`)| Enabled | Inherits `HighwayEnvFast` (disabled) |
| **Observation type** | `Kinematics` (5 vehicles x 5 features) | `Kinematics` (5 x 5) | `TacticalLaneObservation` (27-dim slot grid) |
| **Observation scaling**| Linear normalized $[-1, 1]$ | Linear normalized $[-1, 1]$ | Per-slot relative coordinates & lane deltas |
| **Reward Function** | Scaled speed + right lane + collision ($[-1, 0.5]$ normalized to $[0, 1]$) | Scaled speed + right lane + collision ($[-1, 0.5]$ normalized to $[0, 1]$) | Complex custom shaping: collision penalty (-50), survival (+0.15), speed bonus, danger zone quadratic penalty, blockage impatience penalty, jitter penalty |
| **Discount factor $\gamma$**| 0.8 (Farama) / 0.99 (rl-agents) | 0.8 / 0.99 | 0.99 (or 0.95) @ 5 Hz |
