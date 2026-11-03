from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


# ----------------------------------------------------------------------------
# Allocation helper: water-filling.
# Gives every load a share proportional to its weight, never more than its
# demand, and hands any unused energy to the loads that are still short.
# (The old code skipped this last step, so energy was wasted and PPO looked bad.)
# ----------------------------------------------------------------------------
def waterfill(supply, demand, weights, passes=12):
    alloc = np.zeros_like(demand)
    remaining = float(supply)
    for _ in range(passes):
        open_ = alloc < demand - 1e-9
        w = weights * open_
        s = w.sum()
        if s <= 0 or remaining <= 1e-9:
            break
        alloc += np.minimum(remaining * w / s, demand - alloc)
        remaining = supply - alloc.sum()
    return alloc


class SmartGridEnv(gym.Env):
    """
    One step = one day.

    The agent does NOT output 105 raw allocation numbers. It chooses 3 knobs:
      a[0]  battery        -1 = discharge, 0 = idle, +1 = charge
      a[1]  priority sharpness  (0 = share in proportion to demand,
                                 3 = almost strict priority order)
      a[2]  rotation strength   (0 = ignore past cuts,
                                 1 = give recently-cut cities more power;
                                 never applied to hospitals / water works)
    The allocator then does the rest, so supply limits are always respected
    and the government priorities (hospitals, water, ...) stay in force.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 enso_phase="El Nino",       # None = use all days
                 split="train",              # "train" 2015-23, "val" 2024-25, "test" 2026, "all"
                 episode_len=60,             # random window length while training
                 eval_mode=False,            # True = run the whole split in order
                 data_path=None,
                 seed=0):
        super().__init__()

        base = Path(__file__).resolve().parent.parent
        data_path = Path(data_path) if data_path else base / "data" / "maharashtra_electricity_2015_2026.csv"

        self.enso_phase, self.split = enso_phase, split
        self.episode_len, self.eval_mode = episode_len, eval_mode
        self.rng = np.random.default_rng(seed)

        # Battery: MODEL assumption (dataset has no battery measurements)
        self.cap = 50_000.0
        self.max_rate = 10_000.0
        self.eff = 0.95

        # ---------------- load data once, as numpy arrays ----------------
        df = pd.read_csv(data_path, parse_dates=["date"])
        if enso_phase is not None:
            df = df[df["enso_phase"] == enso_phase]
        if split == "train":
            df = df[df["date"] < "2024-01-01"]                       # 2015-2023
        elif split == "val":
            df = df[(df["date"] >= "2024-01-01") & (df["date"] < "2026-01-01")]   # 2024-2025
        elif split == "test":
            df = df[df["date"] >= "2026-01-01"]                      # 2026 (never used for training)
        df = df.sort_values(["date", "city", "category"])

        self.cities = sorted(df["city"].unique())
        self.categories = sorted(df["category"].unique())
        nC, nK = len(self.cities), len(self.categories)
        self.nC, self.nK, self.n_loads = nC, nK, nC * nK

        self.dates = np.array(sorted(df["date"].unique()))
        T = len(self.dates)
        self.demand = df["demand_mwh"].values.reshape(T, self.n_loads).astype(np.float64)
        self.priority = df["priority_weight"].values.reshape(T, self.n_loads).astype(np.float64)

        g = df.groupby("date").first().loc[self.dates]
        self.state_avail = g["total_available_mwh"].values.astype(np.float64)
        self.state_dem = g["state_demand_mwh"].values.astype(np.float64)
        self.oni = g["oni"].values.astype(np.float64)
        self.temp = g["avg_temp_c"].values.astype(np.float64)

        # Energy available to the 15 cities = demand x (state supply / state demand)
        self.supply_ratio = self.state_avail / self.state_dem
        self.avail_loads = self.demand.sum(1) * self.supply_ratio

        # city index of every load (loads are ordered city-major)
        self.city_of_load = np.repeat(np.arange(nC), nK)
        self.critical = self.priority[0] >= 9          # hospitals + public water works
        self.tier_masks = [self.priority[0] == w for w in sorted(set(self.priority[0]), reverse=True)]

        # windows that are made of consecutive calendar days
        gaps = np.diff(self.dates).astype("timedelta64[D]").astype(int)
        run_id = np.concatenate([[0], np.cumsum(gaps != 1)])
        self.run_id = run_id

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.obs_dim = 7 + nC + 4
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(self.obs_dim,), dtype=np.float32)

    # ------------------------------------------------------------------
    def _valid_starts(self):
        T = len(self.dates)
        return [i for i in range(T - self.episode_len + 1)
                if self.run_id[i] == self.run_id[i + self.episode_len - 1]]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        T = len(self.dates)
        if self.eval_mode:
            self.t0, self.t_end = 0, T
        else:
            starts = self._valid_starts()
            self.t0 = int(self.rng.choice(starts)) if starts else 0
            self.t_end = min(self.t0 + self.episode_len, T)
        self.t = self.t0
        self.soc = 0.5 * self.cap
        self.ema = np.zeros(self.nC)            # recent shortage % of each city
        self.last_tier_short = np.zeros(4)
        self.history = []
        return self._obs(), {}

    # ------------------------------------------------------------------
    def _obs(self):
        t = self.t
        ratio = self.supply_ratio[t]
        ahead = self.supply_ratio[t + 1:t + 8]
        if len(ahead) == 0:
            ahead = self.supply_ratio[t:t + 1]
        ahead = ahead + self.rng.normal(0, 0.01, len(ahead))     # forecast with error
        v = np.concatenate([
            [ratio - 1.0, ahead.mean() - 1.0, ahead.min() - 1.0,
             self.soc / self.cap, self.oni[t] / 2.5, (self.temp[t] - 28) / 8,
             (self.state_dem[t] / self.state_dem.mean()) - 1.0],
            self.ema / 10.0,
            self.last_tier_short,
        ])
        return np.clip(v, -10, 10).astype(np.float32)

    # ------------------------------------------------------------------
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        battery_a = a[0]
        sharp = 1.5 * (a[1] + 1.0)              # 0 .. 3
        rotate = 0.5 * (a[2] + 1.0)             # 0 .. 1

        t = self.t
        demand, prio = self.demand[t], self.priority[t]
        supply = self.avail_loads[t]

        # --- battery discharge (adds supply) ---
        if battery_a < -0.05:
            out = min(-battery_a * self.max_rate, self.soc * self.eff)
            self.soc -= out / self.eff
            supply += out

        # --- allocation: demand x priority^sharp x rotation, then water-filling ---
        if self.ema.std() > 1e-9:
            z = np.clip((self.ema - self.ema.mean()) / self.ema.std(), -3, 3)
        else:
            z = np.zeros(self.nC)
        rot = np.exp(+rotate * 0.5 * np.clip(z, -2, 2))[self.city_of_load]   # cut-more-recently => get more
        rot = np.where(self.critical, 1.0, rot)          # hospitals / water are never rotated
        weights = demand * np.power(prio, sharp) * rot
        alloc = waterfill(supply, demand, weights)

        # --- battery charge from what is left over ---
        surplus = max(0.0, supply - alloc.sum())
        if battery_a > 0.05:
            cin = min(battery_a * self.max_rate, surplus, (self.cap - self.soc) / self.eff)
            self.soc += cin * self.eff

        # --- scoring ---
        short = demand - alloc
        w_unmet = 100 * (short * prio).sum() / (demand * prio).sum()
        crit = 100 * short[self.critical].sum() / max(demand[self.critical].sum(), 1e-9)
        city_short = short.reshape(self.nC, self.nK).sum(1)
        city_dem = demand.reshape(self.nC, self.nK).sum(1)
        self.ema = 0.8 * self.ema + 0.2 * (100 * city_short / city_dem)
        fair = float(self.ema.std())

        tier_short = [short[m].sum() / max(demand[m].sum(), 1e-9) for m in self.tier_masks[:4]]
        self.last_tier_short = np.array(tier_short + [0] * (4 - len(tier_short)))

        reward = -(w_unmet + 2.0 * crit + 0.5 * fair)

        self.history.append(dict(date=str(self.dates[t])[:10], total_shortage_mwh=short.sum(),
                                 priority_weighted_shortage=(short * prio).sum(),
                                 weighted_unmet_pct=w_unmet, critical_shortage_mwh=short[self.critical].sum(),
                                 critical_unmet_pct=crit, unfairness=fair, soc=self.soc / self.cap,
                                 city_short=city_short, city_dem=city_dem))

        self.t += 1
        terminated = self.t >= self.t_end and self.eval_mode
        truncated = self.t >= self.t_end and not self.eval_mode
        done = terminated or truncated
        obs = np.zeros(self.obs_dim, np.float32) if done else self._obs()
        info = {"total_shortage_mwh": float(short.sum()),
                "priority_weighted_shortage": float((short * prio).sum()),
                "critical_shortage_mwh": float(short[self.critical].sum()),
                "weighted_unmet_pct": float(w_unmet),
                "unfairness": fair,
                "battery_soc": float(self.soc / self.cap),
                "city_short": city_short, "city_dem": city_dem}
        return obs, float(reward), terminated, truncated, info