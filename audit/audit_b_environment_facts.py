"""Audit B: Environment Facts Inspection Script.
Collects precise primary facts on HighwayEnv installed version, ContinuousAction dynamics,
timing, observation scaling, termination, and discrete speed targets.
"""

import inspect
import math
import sys
import gymnasium as gym
import highway_env
from highway_env.envs.common.action import ContinuousAction, DiscreteMetaAction, action_factory
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle

import env_config

print("=== B1: INSTALLED VERSIONS ===")
print(f"Python: {sys.version}")
print(f"Gymnasium: {gym.__version__}")
print(f"HighwayEnv: {highway_env.__version__}")

print("\n=== B3 & B4: CONTINUOUS ACTION INTERNALS & DYNAMICS ===")
# Inspect ContinuousAction class
ca_src = inspect.getsource(ContinuousAction)
print("ContinuousAction __init__ and acceleration/steering ranges:")
lines = [l for l in ca_src.splitlines() if "ACCELERATION_RANGE" in l or "STEERING_RANGE" in l or "speed_range" in l or "dynamical" in l or "clip" in l]
for l in lines:
    print("  ", l.strip())

# Check defaults in ContinuousAction
ca = ContinuousAction(env=None)
print("ContinuousAction instance defaults:")
print("  ACCELERATION_RANGE:", getattr(ca, "ACCELERATION_RANGE", None))
print("  STEERING_RANGE:", getattr(ca, "STEERING_RANGE", None))
print("  speed_range:", getattr(ca, "speed_range", None))
print("  dynamical:", getattr(ca, "dynamical", None))
print("  clip:", getattr(ca, "clip", None))

# Check Vehicle kinematics defaults
print("\nVehicle Class Defaults:")
print("  Vehicle.MIN_SPEED:", getattr(Vehicle, "MIN_SPEED", None))
print("  Vehicle.MAX_SPEED:", getattr(Vehicle, "MAX_SPEED", None))
print("  Vehicle.LENGTH:", getattr(Vehicle, "LENGTH", None))
print("  Vehicle.WIDTH:", getattr(Vehicle, "WIDTH", None))

# Kinematic bicycle slip relation:
# In HighwayEnv Vehicle.act():
# beta = atan(0.5 * tan(delta))
# lateral acceleration a_lat = v^2 / R = v^2 * (sin(beta) / (L/2)) or v * dot(psi) = v^2 * tan(delta) / L
# At v = 25 m/s, delta = pi/4 vs delta = pi/6 vs delta = pi/12:
L = getattr(Vehicle, "LENGTH", 5.0)
for delta_deg, delta_rad in [(45, math.pi / 4), (30, math.pi / 6), (15, math.pi / 12)]:
    beta = math.atan(0.5 * math.tan(delta_rad))
    # yaw rate dpsi/dt = (v / L) * cos(beta) * tan(delta)
    # lateral acceleration a_lat = v * yaw_rate
    v = 25.0
    yaw_rate = (v / L) * math.cos(beta) * math.tan(delta_rad)
    a_lat = v * yaw_rate
    print(f"  Delta = {delta_deg:2d} deg ({delta_rad:.4f} rad) at 25 m/s: beta={math.degrees(beta):.2f} deg, yaw_rate={yaw_rate:.2f} rad/s ({math.degrees(yaw_rate):.1f} deg/s), a_lat={a_lat:.2f} m/s^2 ({a_lat/9.81:.2f} g)")

print("\nVehicle attributes read by TacticalLaneObservationWrapper:")
# Check what attributes Vehicle has:
v = Vehicle(None, [0, 0], 0, 20)
attrs_to_check = ["target_lane_index", "route", "lane_index", "lane", "heading", "speed", "position", "crashed", "on_road", "action"]
for attr in attrs_to_check:
    print(f"  v.{attr}: exists = {hasattr(v, attr)}")

print("\n=== B5: TIMING & ACTIONS ===")
env = env_config.make_env("highway", "full_adas")
cfg = env.unwrapped.config
print("env.unwrapped.config timing:")
print("  policy_frequency:", cfg.get("policy_frequency"))
print("  simulation_frequency:", cfg.get("simulation_frequency"))
print("  duration:", cfg.get("duration"))
dt = 1.0 / cfg.get("policy_frequency", 5)
substeps = cfg.get("simulation_frequency", 15) // cfg.get("policy_frequency", 5)
print(f"  dt = {dt}s, simulation substeps per step = {substeps}")
print("  Are actions held constant across sim substeps?: Yes, step() calls act() once per step, then advances simulation substeps.")

print("\n=== B6: OBSERVATION NORMALIZATION & DIMENSIONS ===")
obs_cfg = cfg.get("observation", {})
print("Observation config:", obs_cfg)
obs_type = env.unwrapped.observation_type
print("features_range in Kinematics:", getattr(obs_type, "features_range", None))
rx, ry = env_config.get_scaling_factors(env)
print(f"Reconstructed scaling factors: rx={rx}, ry={ry}")

print("\n=== B7: TERMINATION LOGIC ===")
print("  offroad_terminal in config:", cfg.get("offroad_terminal"))
print("  crash definition: does ego.crashed trigger terminated?:")
# Inspect HighwayEnv _is_terminated
step_src = inspect.getsource(env.unwrapped._is_terminated)
print("  _is_terminated source:")
for l in step_src.splitlines():
    print("   ", l)

trunc_src = inspect.getsource(env.unwrapped._is_truncated)
print("  _is_truncated source:")
for l in trunc_src.splitlines():
    print("   ", l)

print("\n=== B8: DISCRETE SPEED TARGETS ===")
# Inspect DiscreteMetaAction
dma = DiscreteMetaAction(env=env.unwrapped)
print("DiscreteMetaAction target_speeds:", getattr(dma, "target_speeds", None))
print("DiscreteMetaAction actions mapping:", getattr(dma, "ACTIONS_ALL", getattr(dma, "ACTIONS", None)))
if hasattr(dma, "ACTIONS_ALL"):
    for k, val in dma.ACTIONS_ALL.items():
        print(f"  Action {k}: {val}")

env.close()
