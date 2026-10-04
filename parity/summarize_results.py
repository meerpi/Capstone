import json
import glob
from pathlib import Path

def summarize():
    # Baselines
    b_files = sorted(glob.glob("parity/baseline_results/*.json"))
    print(f"==========================================================================================")
    print(f"LADDER BASELINES EVALUATION (Seeds 6000-6099, 100 Episodes)")
    print(f"==========================================================================================")
    print(f"| {'Rung':<6} | {'Method':<18} | {'Crash%':<7} | {'95% CI':<16} | {'Speed':<10} | {'Return':<8} | {'Lane Chg':<8} | {'Dom Action':<22} |")
    print(f"|{'-'*8}|{'-'*20}|{'-'*9}|{'-'*18}|{'-'*12}|{'-'*10}|{'-'*10}|{'-'*24}|")
    for f in b_files:
        with open(f) as fp:
            d = json.load(fp)
        rung = d.get("rung", "?")
        method = d.get("baseline_name", Path(f).stem.split("_", 1)[1])
        cr_pct = f"{d.get('crash_rate', 0)*100:.1f}%"
        ci0 = d.get("ci_95", [0,0])[0] * 100
        ci1 = d.get("ci_95", [0,0])[1] * 100
        ci_str = f"[{ci0:.1f}%, {ci1:.1f}%]"
        speed = f"{d.get('mean_speed_ms', 0):.2f} m/s"
        ret = f"{d.get('mean_return', 0):.2f}"
        lc = f"{d.get('mean_lane_changes', 0):.2f}"
        dom = f"{d.get('dominant_action', '?')} ({d.get('dominant_action_frac', 0)*100:.1f}%)"
        print(f"| {rung:<6} | {method:<18} | {cr_pct:<7} | {ci_str:<16} | {speed:<10} | {ret:<8} | {lc:<8} | {dom:<22} |")

    # Learned Models
    m_files = sorted(glob.glob("parity/results_*.json"))
    print(f"\n==========================================================================================")
    print(f"LEARNED MODELS EVALUATION (Seeds 6000-6099, 100 Episodes)")
    print(f"==========================================================================================")
    print(f"| {'Rung':<6} | {'Method':<12} | {'Seed':<4} | {'Steps':<7} | {'Crash%':<7} | {'95% CI':<16} | {'Speed':<10} | {'Return':<8} | {'Lane Chg':<8} | {'Dom Action':<22} |")
    print(f"|{'-'*8}|{'-'*14}|{'-'*6}|{'-'*9}|{'-'*9}|{'-'*18}|{'-'*12}|{'-'*10}|{'-'*10}|{'-'*24}|")
    for f in m_files:
        with open(f) as fp:
            d = json.load(fp)
        rung = d.get("rung", "?")
        method = d.get("method", "?")
        seed = str(d.get("seed", "?"))
        steps = str(d.get("total_timesteps", "?"))
        cr_pct = f"{d.get('crash_rate', 0)*100:.1f}%"
        ci0 = d.get("ci_95", [0,0])[0] * 100
        ci1 = d.get("ci_95", [0,0])[1] * 100
        ci_str = f"[{ci0:.1f}%, {ci1:.1f}%]"
        speed = f"{d.get('mean_speed_ms', 0):.2f} m/s"
        ret = f"{d.get('mean_return', 0):.2f}"
        lc = f"{d.get('mean_lane_changes', 0):.2f}"
        dom = f"{d.get('dominant_action', '?')} ({d.get('dominant_action_frac', 0)*100:.1f}%)"
        print(f"| {rung:<6} | {method:<12} | {seed:<4} | {steps:<7} | {cr_pct:<7} | {ci_str:<16} | {speed:<10} | {ret:<8} | {lc:<8} | {dom:<22} |")

if __name__ == "__main__":
    summarize()
