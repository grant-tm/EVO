from config import FEATURES, SEQ_LEN, TP_PCT, SL_PCT, LOOKAHEAD

import numpy as np

import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.features = FEATURES
        self.seq_len = SEQ_LEN
        self.tp_pct = TP_PCT
        self.sl_pct = SL_PCT
        self.lookahead = LOOKAHEAD
        self.max_episode_steps = 5000

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
        window = self.df.iloc[self.index - self.seq_len : self.index]
        obs = window[self.features].to_numpy(dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e10, neginf=-1e10)
        return obs

    def _get_current_price(self):
        return self.df.loc[self.index, "close"]
    
    def step(self, action):
        current_price = self._get_current_price()
        reward = 0.0

        # Penalty for being idle too long
        if self.position == 0:
            reward -= 0.001  # discourage inaction

        # Reward shaping while holding a position
        if self.position != 0:
            change = (current_price - self.entry_price) / self.entry_price
            step_duration = self.index - self.entry_index
            direction = np.sign(self.position)

            unrealized = change * direction

            if unrealized <= -self.sl_pct:
                reward += unrealized * 10.0  # penalty for hitting stop loss
                self._close_position()
            elif unrealized >= self.tp_pct:
                reward += unrealized * 15.0  # reward for hitting take profit
                self._close_position()
            elif step_duration >= 3:
                reward += unrealized * 2.0   # shaped reward on timeout
                self._close_position()
            else:
                reward += unrealized * 0.2   # reward/punish for ongoing trend alignment

        # Handle new action
        if self.position == 0:
            if action == 1:
                self._open_position(1, current_price)
            elif action == 2:
                self._open_position(-1, current_price)

        # Normalize reward
        reward = np.clip(reward, -1.0, 1.0)

        self.index += 1
        self.episode_step_count += 1
        self.done = (
            self.index >= len(self.df) - self.lookahead
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
