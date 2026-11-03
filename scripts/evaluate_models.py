"""
Compare traditional rules with the trained PPO agent on the 2026 El Nino days
(data the agent never saw during training).

Run:  python -m evaluation.evaluate           (or  python evaluate.py)
      python evaluate.py --no-ppo             (baselines only)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from environment.smart_grid_env import SmartGridEnv   # change to "from smart_grid_env import SmartGridEnv" if needed

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "ppo_el_nino"
BEST_PATH = BASE_DIR / "models" / "best_el_nino" / "best_model"
STATS_PATH = BASE_DIR / "models" / "ppo_el_nino_vecnormalize.pkl"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------- baselines
def rule(sharp, rotate, battery):
    """sharp: -1 proportional ... +1 strict priority | rotate: -1 off, +1 on
    battery: 'none' or 'greedy' (discharge when short, charge when surplus)"""
    def policy(obs):
        if battery == "greedy":
            b = -1.0 if obs[0] < 0 else 1.0       # obs[0] = supply ratio - 1
        else:
            b = 0.0
        return np.array([b, sharp, rotate], dtype=np.float32)
    return policy


BASELINES = {
    "Proportional (no priority)":       rule(-1, -1, "none"),
    "Priority rule":                    rule(+1, -1, "none"),
    "Priority + greedy battery":        rule(+1, -1, "greedy"),
    "Priority + battery + rotation":    rule(0.0, +1, "greedy"),
}


# ---------------------------------------------------------------- runner
def run(policy, split="test", enso_phase="El Nino"):
    env = SmartGridEnv(enso_phase=enso_phase, split=split, eval_mode=True)
    obs, _ = env.reset(seed=0)
    rows = []
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        rows.append(info)
        if terminated or truncated:
            break
    df = pd.DataFrame(rows)
    city_short = np.sum(np.stack(df["city_short"]), axis=0)
    city_dem = np.sum(np.stack(df["city_dem"]), axis=0)
    return {
        "Total shortage (MWh)":              df["total_shortage_mwh"].sum(),
        "Priority-weighted shortage (MWh)":  df["priority_weighted_shortage"].sum(),
        "Critical-load shortage (MWh)":      df["critical_shortage_mwh"].sum(),
        "Avg weighted unmet (%)":            df["weighted_unmet_pct"].mean(),
        "City unfairness (std of % unserved)": float(np.std(100 * city_short / city_dem)),
    }


def ppo_policy(model_file):
    # imported here so the baselines run even if stable-baselines3 is missing
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    dummy = DummyVecEnv([lambda: SmartGridEnv(split="test", eval_mode=True)])
    vn = VecNormalize.load(str(STATS_PATH), dummy)     # SAME normalisation as training
    vn.training = False
    vn.norm_reward = False
    model = PPO.load(str(model_file))

    def policy(obs):
        action, _ = model.predict(vn.normalize_obs(obs), deterministic=True)
        return action
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ppo", action="store_true")
    ap.add_argument("--model", default=str(MODEL_PATH), help="use --model models/best_el_nino/best_model to try the best checkpoint")
    args = ap.parse_args()

    policies = dict(BASELINES)
    if not args.no_ppo:
        policies["PPO agent"] = ppo_policy(args.model)

    results = pd.DataFrame({name: run(p) for name, p in policies.items()}).T
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print("\nRESULTS ON 2026 EL NINO DAYS (lower is better)\n")
    print(results.to_string())
    results.to_csv(RESULT_DIR / "policy_comparison_2026.csv")

    # ---- chart: each metric scaled so the best method = 1.0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = ["Total shortage (MWh)", "Priority-weighted shortage (MWh)",
               "City unfairness (std of % unserved)"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5))
    for ax, m in zip(axes, metrics):
        vals = results[m]
        ax.bar(range(len(vals)), vals.values)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=30, ha="right", fontsize=8)
        ax.set_title(m, fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "policy_comparison_2026.png", dpi=200)
    print("\nSaved:", RESULT_DIR / "policy_comparison_2026.csv")
    print("Saved:", RESULT_DIR / "policy_comparison_2026.png")

    if not args.no_ppo:
        best_base = results.drop(index="PPO agent")["Priority-weighted shortage (MWh)"].min()
        ppo_val = results.loc["PPO agent", "Priority-weighted shortage (MWh)"]
        print(f"\nPPO vs best baseline (priority-weighted shortage): {100*(ppo_val-best_base)/best_base:+.1f}%")
        print("Report this number honestly, whichever direction it goes.")


if __name__ == "__main__":
    main()