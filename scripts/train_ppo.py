from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from environment.smart_grid_env import SmartGridEnv


# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
ENSO_PHASE = "El Nino"
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
EPISODE_LEN = 60          # random 60-day windows from 2015-2023 El Nino days

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs" / "ppo_el_nino"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
BEST_MODEL_DIR = MODEL_DIR / "best_el_nino"
for d in (MODEL_DIR, LOG_DIR, CHECKPOINT_DIR, BEST_MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "ppo_el_nino"
STATS_PATH = MODEL_DIR / "ppo_el_nino_vecnormalize.pkl"


# ============================================================
# ENVIRONMENT FACTORIES
# ============================================================
def make_train_env():
    # TRAIN: 2015-2023, random 60-day windows
    return SmartGridEnv(enso_phase=ENSO_PHASE, split="train", episode_len=EPISODE_LEN)


def make_val_env():
    # VALIDATION: 2024-2025, whole sequence in order (used only to pick the best checkpoint)
    # 2026 is NOT touched here - it is kept for the final test in evaluate.py
    return SmartGridEnv(enso_phase=ENSO_PHASE, split="val", eval_mode=True)


def main():
    print("=" * 60)
    print("PPO TRAINING - EL NINO (fixed environment)")
    print("=" * 60)

    train_env = make_vec_env(make_train_env, n_envs=N_ENVS, seed=SEED, vec_env_cls=SubprocVecEnv)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(make_val_env, n_envs=1, seed=SEED + 100)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False            # evaluation must not change the statistics

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(BEST_MODEL_DIR),
        log_path=str(LOG_DIR),
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=1,               # one pass over the whole validation sequence
        deterministic=True,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(25_000 // N_ENVS, 1),
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_el_nino",
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,                    # only 3 actions, no need to force extra exploration
        seed=SEED,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=CallbackList([eval_callback, checkpoint_callback]),
                progress_bar=True)

    model.save(MODEL_PATH)
    train_env.save(STATS_PATH)           # needed at evaluation time!
    train_env.close()
    eval_env.close()

    print("\nPPO TRAINING COMPLETE")
    print("Final model :", MODEL_PATH)
    print("Stats       :", STATS_PATH)
    print("Best model  :", BEST_MODEL_DIR / "best_model.zip")
    print("\nNext: python evaluate.py")


if __name__ == "__main__":
    main()