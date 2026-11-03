from pathlib import Path
from typing import Dict

import gymnasium as gym
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# IMPORTANT:
# Reuse the SAME waterfill function from the project environment.
from environment.smart_grid_env import waterfill


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "ppo_el_nino.zip"
NORM_PATH = ROOT / "models" / "ppo_el_nino_vecnormalize.pkl"


# ============================================================
# PROJECT LOAD ORDER
# ============================================================

CITIES = [
    "Ahilyanagar",
    "Akola",
    "Amravati",
    "Chh. Sambhajinagar",
    "Jalgaon",
    "Kolhapur",
    "Latur",
    "Mumbai",
    "Nagpur",
    "Nanded",
    "Nashik",
    "Pune",
    "Sangli",
    "Solapur",
    "Thane",
]

CATEGORIES = [
    "Agriculture",
    "Commercial",
    "Educational institutions",
    "Hospitals",
    "Industry",
    "Public water works",
    "Residential",
]


# Same priority weights used by the project.
PRIORITY = {
    "Agriculture": 1,
    "Commercial": 3,
    "Educational institutions": 6,
    "Hospitals": 10,
    "Industry": 2,
    "Public water works": 9,
    "Residential": 5,
}


N_CITIES = len(CITIES)
N_CATEGORIES = len(CATEGORIES)
N_LOADS = N_CITIES * N_CATEGORIES

# Trained PPO observation:
# 7 global + 15 city + 4 tier = 26
OBS_DIM = 7 + N_CITIES + 4


# ============================================================
# REQUEST FORMAT
# ============================================================

class ScenarioRequest(BaseModel):
    available_mwh: float = Field(..., ge=0)

    demands: Dict[str, Dict[str, float]]


# ============================================================
# REAL-TIME ENVIRONMENT
# ============================================================

class RealtimeSmartGridEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        demands: np.ndarray,
        available_mwh: float,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Demand
        # ----------------------------------------------------

        self.demand = np.asarray(
            demands,
            dtype=np.float64
        ).reshape(N_LOADS)

        self.available = float(
            available_mwh
        )

        # ----------------------------------------------------
        # Priority array
        # Order:
        # city 1 -> 7 categories
        # city 2 -> 7 categories
        # ...
        # ----------------------------------------------------

        self.priority = np.array(
            [
                PRIORITY[category]
                for _city in CITIES
                for category in CATEGORIES
            ],
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # City index for every load
        # ----------------------------------------------------

        self.city_of_load = np.repeat(
            np.arange(N_CITIES),
            N_CATEGORIES,
        )

        # ----------------------------------------------------
        # Critical loads
        #
        # Hospitals = priority 10
        # Public water = priority 9
        # ----------------------------------------------------

        self.critical = (
            self.priority >= 9
        )

        # ----------------------------------------------------
        # Battery assumptions
        # Same as SmartGridEnv
        # ----------------------------------------------------

        self.cap = 50_000.0
        self.max_rate = 10_000.0
        self.eff = 0.95

        # Start at 50% SOC
        self.soc = 0.5 * self.cap

        # ----------------------------------------------------
        # No historical data in live scenario
        # ----------------------------------------------------

        self.ema = np.zeros(
            N_CITIES,
            dtype=np.float64
        )

        self.last_tier_short = np.zeros(
            4,
            dtype=np.float64
        )

        # ----------------------------------------------------
        # Allocation storage
        # ----------------------------------------------------

        self.alloc = np.zeros(
            N_LOADS,
            dtype=np.float64
        )

        self.short = self.demand.copy()

        # ----------------------------------------------------
        # MUST match trained PPO spaces
        # ----------------------------------------------------

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

    # ========================================================
    # OBSERVATION
    # ========================================================

    def _obs(self):

        total_demand = max(
            float(self.demand.sum()),
            1e-9
        )

        ratio = (
            self.available /
            total_demand
        )

        # ----------------------------------------------------
        # Live scenario does not provide:
        #
        # - future forecast
        # - historical shortage
        # - actual ONI
        # - actual temperature
        #
        # Therefore use fixed El Nino reference context
        # while preserving the trained 26-dimensional input.
        # ----------------------------------------------------

        observation = np.concatenate(
            [
                [
                    # Current supply ratio
                    ratio - 1.0,

                    # Future supply ratio reference
                    ratio - 1.0,

                    # Minimum future ratio reference
                    ratio - 1.0,

                    # Battery SOC
                    self.soc / self.cap,

                    # El Nino reference ONI
                    1.5 / 2.5,

                    # Temperature reference
                    0.0,

                    # State-demand reference
                    0.0,
                ],

                # City shortage history
                self.ema / 10.0,

                # Tier shortage history
                self.last_tier_short,
            ]
        )

        return np.clip(
            observation,
            -10.0,
            10.0
        ).astype(np.float32)

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )

        self.soc = (
            0.5 *
            self.cap
        )

        self.ema[:] = 0.0

        self.last_tier_short[:] = 0.0

        self.alloc[:] = 0.0

        self.short = (
            self.demand.copy()
        )

        return (
            self._obs(),
            {}
        )

    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        action
    ):

        # ----------------------------------------------------
        # Clip PPO action
        # ----------------------------------------------------

        a = np.clip(
            np.asarray(
                action,
                dtype=np.float64
            ),
            -1.0,
            1.0,
        )

        # ----------------------------------------------------
        # PPO controls
        #
        # a[0] = battery
        # a[1] = priority sharpness
        # a[2] = rotation
        # ----------------------------------------------------

        battery_a = float(
            a[0]
        )

        sharp = (
            1.5 *
            (float(a[1]) + 1.0)
        )

        rotate = (
            0.5 *
            (float(a[2]) + 1.0)
        )

        # ----------------------------------------------------
        # Initial supply
        # ----------------------------------------------------

        supply = float(
            self.available
        )

        # ----------------------------------------------------
        # Battery discharge
        # ----------------------------------------------------

        if battery_a < -0.05:

            out = min(
                -battery_a *
                self.max_rate,

                self.soc *
                self.eff,
            )

            self.soc -= (
                out /
                self.eff
            )

            supply += out

        # ----------------------------------------------------
        # Rotation logic
        # ----------------------------------------------------

        if self.ema.std() > 1e-9:

            z = np.clip(
                (
                    self.ema -
                    self.ema.mean()
                )
                /
                self.ema.std(),

                -3,
                3,
            )

        else:

            z = np.zeros(
                N_CITIES
            )

        rot = np.exp(
            rotate *
            0.5 *
            np.clip(
                z,
                -2,
                2,
            )
        )[
            self.city_of_load
        ]

        # Critical loads are not rotated
        rot = np.where(
            self.critical,
            1.0,
            rot
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # This is the SAME allocation equation as the project
        # environment.
        # ----------------------------------------------------

        weights = (
            self.demand
            *
            np.power(
                self.priority,
                sharp
            )
            *
            rot
        )

        # ----------------------------------------------------
        # Allocate electricity
        #
        # Reuse the actual waterfill() from
        # environment/smart_grid_env.py
        # ----------------------------------------------------

        self.alloc = waterfill(
            supply,
            self.demand,
            weights,
        )

        # ----------------------------------------------------
        # Numerical safety
        # ----------------------------------------------------

        self.alloc = np.clip(
            self.alloc,
            0.0,
            self.demand,
        )

        if (
            self.alloc.sum()
            >
            supply + 1e-6
        ):

            self.alloc *= (
                supply /
                self.alloc.sum()
            )

        # ----------------------------------------------------
        # Shortage
        # ----------------------------------------------------

        self.short = np.maximum(
            self.demand -
            self.alloc,

            0.0,
        )

        # ====================================================
        # SCORING
        # ====================================================

        weighted_den = max(
            float(
                np.sum(
                    self.demand *
                    self.priority
                )
            ),
            1e-9,
        )

        weighted_unmet = (
            100.0
            *
            float(
                np.sum(
                    self.short *
                    self.priority
                )
            )
            /
            weighted_den
        )

        # ----------------------------------------------------
        # Critical shortage
        # ----------------------------------------------------

        critical_demand = max(
            float(
                self.demand[
                    self.critical
                ].sum()
            ),
            1e-9,
        )

        critical_unmet = (
            100.0
            *
            float(
                self.short[
                    self.critical
                ].sum()
            )
            /
            critical_demand
        )

        # ----------------------------------------------------
        # City shortage
        # ----------------------------------------------------

        city_short = (
            self.short
            .reshape(
                N_CITIES,
                N_CATEGORIES
            )
            .sum(axis=1)
        )

        city_dem = (
            self.demand
            .reshape(
                N_CITIES,
                N_CATEGORIES
            )
            .sum(axis=1)
        )

        city_pct = (
            100.0
            *
            city_short
            /
            np.maximum(
                city_dem,
                1e-9
            )
        )

        # ----------------------------------------------------
        # EMA fairness
        # ----------------------------------------------------

        self.ema = (
            0.8 *
            self.ema
            +
            0.2 *
            city_pct
        )

        unfairness = float(
            self.ema.std()
        )

        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        reward = -(
            weighted_unmet
            +
            2.0 *
            critical_unmet
            +
            0.5 *
            unfairness
        )

        # ----------------------------------------------------
        # Save information BEFORE VecEnv resets the environment
        #
        # This is VERY IMPORTANT.
        #
        # DummyVecEnv automatically calls reset() after
        # terminated=True. Therefore backend must read
        # allocation from info["alloc"], not self.alloc later.
        # ----------------------------------------------------

        info = {

            "alloc":
                self.alloc.copy(),

            "short":
                self.short.copy(),

            "weighted_unmet_pct":
                float(
                    weighted_unmet
                ),

            "critical_unmet_pct":
                float(
                    critical_unmet
                ),

            "unfairness":
                float(
                    unfairness
                ),

            "battery_soc":
                float(
                    self.soc /
                    self.cap
                ),

            "sharp":
                float(
                    sharp
                ),

            "rotate":
                float(
                    rotate
                ),

            "battery_action":
                float(
                    battery_a
                ),

            "effective_supply_mwh":
                float(
                    supply
                ),
        }

        # ----------------------------------------------------
        # One-step environment
        # ----------------------------------------------------

        observation = np.zeros(
            OBS_DIM,
            dtype=np.float32
        )

        terminated = True
        truncated = False

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )


# ============================================================
# PAYLOAD VALIDATION
# ============================================================

def validate_payload(
    req: ScenarioRequest
) -> np.ndarray:

    # --------------------------------------------------------
    # Check cities
    # --------------------------------------------------------

    if set(req.demands.keys()) != set(CITIES):

        missing = sorted(
            set(CITIES)
            -
            set(req.demands.keys())
        )

        extra = sorted(
            set(req.demands.keys())
            -
            set(CITIES)
        )

        raise HTTPException(
            status_code=400,
            detail={
                "missing_cities":
                    missing,

                "unknown_cities":
                    extra,
            },
        )

    flat = []

    # --------------------------------------------------------
    # Check every city
    # --------------------------------------------------------

    for city in CITIES:

        city_data = (
            req.demands[city]
        )

        if (
            set(city_data.keys())
            !=
            set(CATEGORIES)
        ):

            missing = sorted(
                set(CATEGORIES)
                -
                set(city_data.keys())
            )

            extra = sorted(
                set(city_data.keys())
                -
                set(CATEGORIES)
            )

            raise HTTPException(
                status_code=400,
                detail={
                    "city":
                        city,

                    "missing_categories":
                        missing,

                    "unknown_categories":
                        extra,
                },
            )

        # ----------------------------------------------------
        # Preserve exact category order
        # ----------------------------------------------------

        for category in CATEGORIES:

            value = city_data[
                category
            ]

            try:
                value = float(
                    value
                )

            except (
                TypeError,
                ValueError
            ):

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid demand for "
                        f"{city} / {category}."
                    ),
                )

            if (
                not np.isfinite(value)
                or value < 0
            ):

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Demand for "
                        f"{city} / {category} "
                        f"must be a finite "
                        f"non-negative number."
                    ),
                )

            flat.append(value)

    result = np.asarray(
        flat,
        dtype=np.float64
    )

    if result.shape != (
        N_LOADS,
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {N_LOADS} "
                f"loads, received "
                f"{result.size}."
            ),
        )

    return result


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

def make_env(
    demand: np.ndarray,
    available: float
):

    def _factory():

        return RealtimeSmartGridEnv(
            demands=demand,
            available_mwh=available,
        )

    return _factory


# ============================================================
# BUILD VECNORMALIZE ENVIRONMENT
# ============================================================

def build_vec_env(
    demand: np.ndarray,
    available: float
):

    # --------------------------------------------------------
    # DummyVecEnv wraps the real-time environment because
    # PPO expects a vectorized environment.
    # --------------------------------------------------------

    dummy = DummyVecEnv(
        [
            make_env(
                demand,
                available
            )
        ]
    )

    # --------------------------------------------------------
    # Load the EXACT normalization statistics used during
    # PPO training.
    # --------------------------------------------------------

    vec_env = VecNormalize.load(
        str(NORM_PATH),
        dummy,
    )

    # --------------------------------------------------------
    # Evaluation/inference mode.
    # Do not update normalization statistics.
    # --------------------------------------------------------

    vec_env.training = False

    vec_env.norm_reward = False

    return vec_env


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="El Nino Smart Grid PPO API",
    version="1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# LOAD PPO MODEL
# ============================================================

model = None


try:

    model = PPO.load(
        str(MODEL_PATH)
    )

    print(
        "=========================================="
    )

    print(
        "PPO MODEL LOADED"
    )

    print(
        f"Model : {MODEL_PATH}"
    )

    print(
        f"Norm  : {NORM_PATH}"
    )

    print(
        "=========================================="
    )


except Exception as exc:

    print(
        "=========================================="
    )

    print(
        "FAILED TO LOAD PPO MODEL"
    )

    print(
        repr(exc)
    )

    print(
        "=========================================="
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "ok",

        "model_loaded":
            model is not None,

        "cities":
            N_CITIES,

        "categories":
            N_CATEGORIES,

        "loads":
            N_LOADS,
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    req: ScenarioRequest
):

    global model

    # --------------------------------------------------------
    # Make sure model loaded
    # --------------------------------------------------------

    if model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "PPO model is not loaded. "
                "Check models/ppo_el_nino.zip "
                "and "
                "models/ppo_el_nino_vecnormalize.pkl."
            ),
        )

    # --------------------------------------------------------
    # Validate demand matrix
    # --------------------------------------------------------

    demand = validate_payload(
        req
    )

    available = float(
        req.available_mwh
    )

    total_demand = float(
        demand.sum()
    )

    # --------------------------------------------------------
    # Build environment
    # --------------------------------------------------------

    env = build_vec_env(
        demand,
        available,
    )

    try:

        # ----------------------------------------------------
        # Reset
        # ----------------------------------------------------

        obs = env.reset()

        # ----------------------------------------------------
        # PPO prediction
        # ----------------------------------------------------

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        # ----------------------------------------------------
        # Run one environment step
        # ----------------------------------------------------

        _, _, _, infos = env.step(
            action
        )

        info = infos[0]

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # DO NOT read:
        #
        # raw_env.alloc
        #
        # because DummyVecEnv automatically resets the
        # one-step environment after terminated=True.
        #
        # The allocation is safely preserved in info["alloc"].
        # ----------------------------------------------------

        if (
            "alloc" not in info
            or
            "short" not in info
        ):

            raise RuntimeError(
                "RealtimeSmartGridEnv did not "
                "return alloc/short in info."
            )

        alloc = np.asarray(
            info["alloc"],
            dtype=np.float64
        ).copy()

        short = np.asarray(
            info["short"],
            dtype=np.float64
        ).copy()

    finally:

        env.close()

    # ========================================================
    # SAFETY CHECKS
    # ========================================================

    # Never negative
    alloc = np.maximum(
        alloc,
        0.0
    )

    # Never allocate more than demand
    alloc = np.minimum(
        alloc,
        demand
    )

    # Never allocate more than available electricity
    if (
        alloc.sum()
        >
        available + 1e-6
    ):

        alloc *= (
            available /
            alloc.sum()
        )

    # Recalculate shortage after safety clipping
    short = np.maximum(
        demand -
        alloc,
        0.0
    )

    total_alloc = float(
        alloc.sum()
    )

    total_short = float(
        short.sum()
    )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    demand_met = (
        100.0
        *
        total_alloc
        /
        max(
            total_demand,
            1e-9
        )
    )

    # --------------------------------------------------------
    # Critical load mask
    # --------------------------------------------------------

    critical_mask = (
        np.array(
            [
                PRIORITY[category]
                for _city in CITIES
                for category in CATEGORIES
            ]
        )
        >= 9
    )

    critical_demand = float(
        demand[
            critical_mask
        ].sum()
    )

    critical_alloc = float(
        alloc[
            critical_mask
        ].sum()
    )

    critical_met = (
        100.0
        *
        critical_alloc
        /
        max(
            critical_demand,
            1e-9
        )
    )

    # ========================================================
    # DETAILED 105-LOAD RESULTS
    # ========================================================

    allocations = []

    idx = 0

    for city in CITIES:

        for category in CATEGORIES:

            d = float(
                demand[idx]
            )

            a = float(
                alloc[idx]
            )

            s = float(
                short[idx]
            )

            allocations.append(
                {
                    "city":
                        city,

                    "category":
                        category,

                    "demand_mwh":
                        d,

                    "allocated_mwh":
                        a,

                    "shortage_mwh":
                        s,

                    "served_pct":
                        (
                            100.0 *
                            a /
                            d
                            if d > 0
                            else 100.0
                        ),

                    "priority_weight":
                        PRIORITY[
                            category
                        ],
                }
            )

            idx += 1

    # ========================================================
    # SECTOR SUMMARY
    # ========================================================

    sector_summary = []

    for category in CATEGORIES:

        rows = [
            row
            for row in allocations
            if row["category"]
            == category
        ]

        d = sum(
            row["demand_mwh"]
            for row in rows
        )

        a = sum(
            row["allocated_mwh"]
            for row in rows
        )

        sector_summary.append(
            {
                "category":
                    category,

                "demand_mwh":
                    d,

                "allocated_mwh":
                    a,

                "shortage_mwh":
                    d - a,

                "served_pct":
                    (
                        100.0 *
                        a /
                        max(
                            d,
                            1e-9
                        )
                    ),

                "critical":
                    PRIORITY[
                        category
                    ] >= 9,
            }
        )

    # ========================================================
    # CITY SUMMARY
    # ========================================================

    city_summary = []

    for city in CITIES:

        rows = [
            row
            for row in allocations
            if row["city"]
            == city
        ]

        d = sum(
            row["demand_mwh"]
            for row in rows
        )

        a = sum(
            row["allocated_mwh"]
            for row in rows
        )

        critical_rows = [
            row
            for row in rows
            if row[
                "priority_weight"
            ] >= 9
        ]

        cd = sum(
            row["demand_mwh"]
            for row in critical_rows
        )

        ca = sum(
            row["allocated_mwh"]
            for row in critical_rows
        )

        city_summary.append(
            {
                "city":
                    city,

                "demand_mwh":
                    d,

                "allocated_mwh":
                    a,

                "shortage_mwh":
                    d - a,

                "served_pct":
                    (
                        100.0 *
                        a /
                        max(
                            d,
                            1e-9
                        )
                    ),

                "critical_served_pct":
                    (
                        100.0 *
                        ca /
                        max(
                            cd,
                            1e-9
                        )
                        if cd > 0
                        else 100.0
                    ),
            }
        )

    # ========================================================
    # TRADITIONAL PRIORITY RULE BASELINE
    # ========================================================

    baseline_priority = np.asarray(
        [
            PRIORITY[category]
            for _city in CITIES
            for category in CATEGORIES
        ],
        dtype=np.float64,
    )

    baseline_weights = (
        demand *
        baseline_priority
    )

    base_alloc = waterfill(
        available,
        demand,
        baseline_weights,
    )

    base_alloc = np.clip(
        base_alloc,
        0.0,
        demand,
    )

    base_short = np.maximum(
        demand -
        base_alloc,
        0.0,
    )

    # --------------------------------------------------------
    # Baseline fairness
    # --------------------------------------------------------

    base_city_short = (
        base_short
        .reshape(
            N_CITIES,
            N_CATEGORIES
        )
        .sum(axis=1)
    )

    base_city_dem = (
        demand
        .reshape(
            N_CITIES,
            N_CATEGORIES
        )
        .sum(axis=1)
    )

    base_city_short_pct = (
        100.0
        *
        base_city_short
        /
        np.maximum(
            base_city_dem,
            1e-9
        )
    )

    base_fair = float(
        np.std(
            base_city_short_pct
        )
    )

    # --------------------------------------------------------
    # Baseline critical service
    # --------------------------------------------------------

    base_critical = (
        100.0
        *
        base_alloc[
            critical_mask
        ].sum()
        /
        max(
            critical_demand,
            1e-9
        )
    )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        # ----------------------------------------------------
        # Main metrics
        # ----------------------------------------------------

        "metrics": {

            "total_demand_mwh":
                total_demand,

            "available_mwh":
                available,

            "effective_supply_mwh":
                float(
                    info[
                        "effective_supply_mwh"
                    ]
                ),

            "allocated_mwh":
                total_alloc,

            "total_shortage_mwh":
                total_short,

            "demand_met_pct":
                demand_met,

            "critical_load_met_pct":
                critical_met,

            "priority_weighted_unmet_pct":
                float(
                    info[
                        "weighted_unmet_pct"
                    ]
                ),

            "unfairness":
                float(
                    info[
                        "unfairness"
                    ]
                ),
        },

        # ----------------------------------------------------
        # PPO decision
        # ----------------------------------------------------

        "ppo_action": {

            "battery_action":
                float(
                    info[
                        "battery_action"
                    ]
                ),

            "priority_sharpness":
                float(
                    info[
                        "sharp"
                    ]
                ),

            "rotation_strength":
                float(
                    info[
                        "rotate"
                    ]
                ),

            "battery_soc_after":
                float(
                    info[
                        "battery_soc"
                    ]
                ),
        },

        # ----------------------------------------------------
        # 105 individual loads
        # ----------------------------------------------------

        "allocations":
            allocations,

        # ----------------------------------------------------
        # Sector summary
        # ----------------------------------------------------

        "sector_summary":
            sector_summary,

        # ----------------------------------------------------
        # City summary
        # ----------------------------------------------------

        "city_summary":
            city_summary,

        # ----------------------------------------------------
        # PPO vs traditional rule
        # ----------------------------------------------------

        "comparison": {

            "ppo": {

                "demand_met_pct":
                    demand_met,

                "critical_load_met_pct":
                    critical_met,

                "shortage_pct":
                    (
                        100.0 *
                        total_short
                        /
                        max(
                            total_demand,
                            1e-9
                        )
                    ),

                "unfairness":
                    float(
                        info[
                            "unfairness"
                        ]
                    ),
            },

            "priority_rule": {

                "demand_met_pct":
                    (
                        100.0 *
                        base_alloc.sum()
                        /
                        max(
                            total_demand,
                            1e-9
                        )
                    ),

                "critical_load_met_pct":
                    base_critical,

                "shortage_pct":
                    (
                        100.0 *
                        base_short.sum()
                        /
                        max(
                            total_demand,
                            1e-9
                        )
                    ),

                "unfairness":
                    base_fair,
            },
        },
    }