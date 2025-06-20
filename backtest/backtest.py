from config import DATA_PATH, FEATURES, SEQ_LEN

from typing import List, Tuple
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import backtrader as bt
from stable_baselines3 import PPO

# =====================================
# Backtrader DataFeed
# =====================================

class FeatureData(bt.feeds.PandasData):
    lines = ("ppo_action",)
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"), 
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
        ("ppo_action", "ppo_action"),
    )

# =====================================
# Strategies
# =====================================

class PPOInferenceStrategy(bt.Strategy):
    
    def __init__(self, track_returns: bool = False):
        self.track_returns = track_returns
        self.initial_value = self.broker.getvalue()
        self.values = [] if track_returns else None

    def next(self):
        action = int(self.datas[0].ppo_action[0])
        if action == 1 and not self.position:
            self.buy()
        elif action == 2 and not self.position:
            self.sell()
        elif action == 0 and self.position:
            self.close()

        if self.track_returns:
            self.values.append(self.broker.getvalue())

    def stop(self):
        if self.track_returns and self.values and len(self.values) > 1:
            self.returns = np.diff(self.values)
        else:
            self.pnl = self.broker.getvalue() - self.initial_value

# =====================================
# Utilities
# =====================================

# load historical data
def load_and_clean_data() -> pd.DataFrame:
    
    # read data csv
    df = pd.read_csv(DATA_PATH)
    
    # format timestamp and drop duplicates
    df["datetime"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("datetime").drop_duplicates(subset="datetime")
    
    # format features and drop nan values
    df = df.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df

# extract slice of data from a dataframe
def sample_data_slice(df_all: pd.DataFrame, slice_length: int) -> pd.DataFrame:
    
    # get maximum slice start index
    max_start = len(df_all) - slice_length - SEQ_LEN
    if max_start <= 0:
        raise ValueError("slice_length too long for dataset")
    
    # get slice start and end indices
    start = np.random.randint(0, max_start)
    end = start + slice_length
    
    # extract data slice
    return df_all.iloc[start:end].copy().reset_index(drop=True)

# modify dataframe to add ppo agent actions for each data step
def add_ppo_predictions(df: pd.DataFrame, model: PPO) -> pd.DataFrame:
    
    # format features
    features = df[FEATURES].to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # construct data sequences 
    X = sliding_window_view(features, window_shape=(SEQ_LEN, features.shape[1]))[:, 0, :, :]
    
    # precompute model predictions
    actions, _ = model.predict(X, deterministic=True)
    
    # add predictions to dataframe
    df = df.iloc[SEQ_LEN - 1:].copy()
    df["ppo_action"] = actions
    df.set_index("datetime", inplace=True)
    
    # return modified dataframe
    return df

# run a backtest
def run_backtest_strategy(df: pd.DataFrame, strategy_cls: bt.Strategy, **strategy_kwargs) -> Tuple[bt.Strategy, bt.Cerebro]:    
    
    # configure backtest engine
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_cls, **strategy_kwargs)
    cerebro.adddata(FeatureData(dataname=df))
    
    # configure broker
    cerebro.broker.set_cash(100_000)
    cerebro.broker.setcommission(commission=0.001)
    
    # run backtest and return result
    results = cerebro.run()
    return results[0], cerebro

# =====================================
# Primary Interfaces
# =====================================

# backtest a model on historical data
def backtest(model_path: str, slice_length: int) -> float:
    
    # load, process, and slice data
    df_all = load_and_clean_data()
    df = sample_data_slice(df_all, slice_length)
    
    # load model and precompute predictions for data slice
    model = PPO.load(model_path)
    df = add_ppo_predictions(df, model)
    
    # run backtest and calculate sharpe ratio
    result, _ = run_backtest_strategy(df, PPOInferenceStrategy, track_returns=True)
    if hasattr(result, "returns") and len(result.returns) > 1:
        mean = np.mean(result.returns)
        std = np.std(result.returns)
        return mean / (std + 1e-8)

    return 0.0

# evaluate a model by performing multiple backtests
def multi_backtest(model_path: str, num_backtests: int, slice_length: int) -> float:
    
    # load and preprocess data once
    df_all = load_and_clean_data()

    # load model from model path
    model = PPO.load(model_path)
    
    # for num_backtests: slice data, precompute predictions, and run backtest
    pnls: List[float] = []
    for _ in range(num_backtests):
        df = sample_data_slice(df_all, slice_length)
        df = add_ppo_predictions(df, model)
        result, _ = run_backtest_strategy(df, PPOInferenceStrategy, track_returns=False)
        if hasattr(result, "pnl"):
            pnls.append(result.pnl)

    # calculate sharpe ratio
    if len(pnls) < 2:
        return 0.0
    mean = np.mean(pnls)
    std = np.std(pnls)
    return mean / (std + 1e-8)
