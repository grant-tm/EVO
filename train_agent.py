from config import DATA_PATH, MODEL_DIR, MODEL_NAME, TRAINING_STEPS, BATCH_SIZE, LEARNING_RATE, CLIP_RANGE, ENTROPY_COEF_INIT, ENTROPY_COEF_FINAL, GAE_LAMBDA, GAMMA
from trading_env import TradingEnv

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, ProgressBarCallback
from stable_baselines3.common.utils import get_schedule_fn

# trains a model with the default training and reward-shaping parameters from config.py
def train_agent():
    train_with_genome(MODEL_NAME, {})

# trains a model with training and reward-shaping parameters defined in a genome
def train_with_genome(genome: dict, model_name: str = None):
    
    # === Load Data ===
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    split_idx = int(len(df) * 0.95)
    train_df = df.iloc[:split_idx].copy()
    eval_df = df.iloc[split_idx:].copy()

    # === Create Environments ===
    train_env = DummyVecEnv([lambda: Monitor(TradingEnv(train_df))])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    eval_env = DummyVecEnv([lambda: Monitor(TradingEnv(eval_df))])
    eval_env = VecNormalize.load("ppo_trading_agent_vecnormalize.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    # === Define & Add Callbacks ===
    class RewardLoggerCallback(BaseCallback):
        def __init__(self, verbose=0): super().__init__(verbose)
        def _on_step(self) -> bool:
            if "reward" in self.locals:
                self.logger.record("rollout/raw_reward", self.locals["reward"])
            return True

    class EntropyAnnealingCallback(BaseCallback):
        def __init__(self, initial_coef, final_coef, total_timesteps, verbose=0):
            super().__init__(verbose)
            self.initial_coef = initial_coef
            self.final_coef = final_coef
            self.total_timesteps = total_timesteps

        def _on_step(self) -> bool:
            progress = self.num_timesteps / self.total_timesteps
            new_coef = self.initial_coef + progress * (self.final_coef - self.initial_coef)
            self.model.ent_coef = new_coef
            return True

    callbacks = [
        #RewardLoggerCallback(),
        ProgressBarCallback(),
        EntropyAnnealingCallback(
            genome.get("entropy_coef_init", ENTROPY_COEF_INIT),
            genome.get("entropy_coef_final", ENTROPY_COEF_FINAL), 
            TRAINING_STEPS
        )
    ]
    
    # === Define PPO Model ===
    model = PPO(
        "MlpPolicy",
        train_env,
        clip_range=genome.get("clip_range", CLIP_RANGE),
        verbose=0,
        tensorboard_log="./ppo_tensorboard/",
        n_steps=512,
        batch_size=genome.get("batch_size", BATCH_SIZE),
        learning_rate=get_schedule_fn(genome.get("learning_rate", LEARNING_RATE)),
        ent_coef=genome.get("entropy_coef_init", ENTROPY_COEF_INIT),
        gae_lambda=genome.get("gae_lambda", GAE_LAMBDA),
        gamma=genome.get("gamma", GAMMA),
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    )

    # === Train & Save the Model ===
    model.learn(
        total_timesteps=TRAINING_STEPS,
        callback=callbacks
    )

    if model_name is None:
        model_name = MODEL_NAME

    model.save(f"{MODEL_DIR}/{model_name}")
