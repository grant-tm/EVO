from trading_env import TradingEnv

import os
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, ProgressBarCallback
from stable_baselines3.common.utils import get_schedule_fn

# =========================================================
# Prepare Dataset
# =========================================================

# load dataset from .csv
df = pd.read_csv("preprocessed_training_data.csv")
df.dropna(inplace=True)

# split dataset (95% train, 5% eval)
split_idx = int(len(df) * 0.95)
train_df = df.iloc[:split_idx].copy()
eval_df = df.iloc[split_idx:].copy()

# =========================================================
# Prepare Environments
# =========================================================

# create training environment
def make_train_env():
    return Monitor(TradingEnv(train_df))

train_env = DummyVecEnv([make_train_env])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

# create evaluation environment
def make_eval_env():
    return Monitor(TradingEnv(eval_df))

eval_env = DummyVecEnv([make_eval_env])
eval_env = VecNormalize.load("ppo_trading_agent_vecnormalize.pkl", eval_env)
eval_env.training = False
eval_env.norm_reward = False

# check environment compliance
check_env(TradingEnv(train_df), warn=True)

# =========================================================
# Define callbacks
# =========================================================

# evaluate agent every 10000 training steps
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./checkpoints/",
    log_path="./logs/",
    eval_freq=10_000,
    n_eval_episodes=1,
    deterministic=True,
    render=False
)

# log raw rewards
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if "reward" in self.locals:
            self.logger.record("rollout/raw_reward", self.locals["reward"])
        return True

# =========================================================
# Train Model
# =========================================================

# define PPO model
model = PPO(
    "MlpPolicy",
    train_env,
    clip_range_vf=0.2,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
    n_steps=512,
    batch_size=64,
    learning_rate=get_schedule_fn(3e-4),
    ent_coef=0.01,
    gae_lambda=0.90,
    gamma=0.95,
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
)

# train ppo model
model.learn(
    total_timesteps=300_000,
    callback=[eval_callback, RewardLoggerCallback(), ProgressBarCallback()]
)

# save model and environment
model.save("ppo_trading_agent")
train_env.save("ppo_trading_agent_vecnormalize.pkl")
print("Model saved.")
