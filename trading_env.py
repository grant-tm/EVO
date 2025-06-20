from config import FEATURES, SEQ_LEN, TP_PCT, SL_PCT, IDLE_PENALTY, SL_PENALTY_COEF, TP_REWARD_COEF, TIMEOUT_DURATION, TIMEOUT_REWARD_COEF, ONGOING_REWARD_COEF, REWARD_CLIP_RANGE, MAX_EPISODE_STEPS

import numpy as np

import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, reward_params=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.features = FEATURES
        self.seq_len = SEQ_LEN
        self.tp_pct = TP_PCT
        self.sl_pct = SL_PCT

        # Reward shaping params (with defaults)
        reward_params = reward_params or {}
        self.idle_penalty = reward_params.get("idle_penalty", IDLE_PENALTY)
        self.sl_penalty_coef = reward_params.get("sl_penalty_coef", SL_PENALTY_COEF)
        self.tp_reward_coef = reward_params.get("tp_reward_coef", TP_REWARD_COEF)
        self.timeout_duration = reward_params.get("timeout_duration", TIMEOUT_DURATION)
        self.timeout_reward_coef = reward_params.get("timeout_reward_coef", TIMEOUT_REWARD_COEF)
        self.ongoing_reward_coef = reward_params.get("ongoing_reward_coef", ONGOING_REWARD_COEF)
        self.reward_clip_range = reward_params.get("reward_clip_range", REWARD_CLIP_RANGE)
        self.max_episode_steps = reward_params.get("max_episode_steps", MAX_EPISODE_STEPS)
        
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell
        self.observation_space = spaces.Box(
            low=-1e10,
            high=1e10,
            shape=(self.seq_len, len(self.df[self.features].columns)),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = self.seq_len
        self.episode_step_count = 0
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = None
        self.entry_index = None
        self.total_reward = 0.0
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        window = self.df.iloc[self.index - self.seq_len: self.index]
        obs = window[self.features].to_numpy(dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e10, neginf=-1e10)
        return obs

    def _get_current_price(self):
        return self.df.loc[self.index, "close"]

    def step(self, action):
        current_price = self._get_current_price()
        reward = 0.0

        # Apply penalty for being idle too long (encourage trading)
        if self.position == 0:
            reward -= self.idle_penalty

        # Reward shaping while holding a position
        if self.position != 0:
            change = (current_price - self.entry_price) / self.entry_price
            step_duration = self.index - self.entry_index
            direction = np.sign(self.position)

            unrealized = change * direction

            # stop loss hit: close trade for a loss and apply penalty
            if unrealized <= -self.sl_pct:
                reward += unrealized * self.sl_penalty_coef
                self._close_position()
            
            # take profit hit: close trade for a profit and apply reward
            elif unrealized >= self.tp_pct:
                reward += unrealized * self.tp_reward_coef
                self._close_position()
            
            # trade timeout hit: close trade and apply scaled reward (can be positive or negative)
            elif step_duration >= self.timeout_duration:
                reward += unrealized * self.timeout_reward_coef
                self._close_position()
            
            # trade held: apply small reward/penalty for unrealized pnl increase/decrease
            else:
                reward += unrealized * self.ongoing_reward_coef

        # Allow agent to open a new position if no position is currently held
        if self.position == 0:
            if action == 1:
                self._open_position(1, current_price)
            elif action == 2:
                self._open_position(-1, current_price)

        # Normalize reward
        reward = np.clip(reward, self.reward_clip_range[0], self.reward_clip_range[1])

        self.index += 1
        self.episode_step_count += 1
        self.done = (
            self.index >= len(self.df)
            or self.episode_step_count >= self.max_episode_steps
        )

        self.total_reward += reward
        return self._get_obs(), reward, self.done, False, {}

    def _reset_position(self):
        self.position = 0
        self.entry_price = None
        self.entry_index = None

    def _open_position(self, direction, price):
        self.position = direction
        self.entry_price = price
        self.entry_index = self.index

    def _close_position(self):
        self.position = 0
        self.entry_price = None
        self.entry_index = None

    def render(self, mode="human"):
        print(f"Step {self.index} | Position: {self.position} | Entry: {self.entry_price} | Total Reward: {self.total_reward}")
