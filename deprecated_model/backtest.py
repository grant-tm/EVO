import backtrader as bt
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from config import MODEL_NAME, FEATURES, SEQ_LEN, LOOKAHEAD, THRESHOLD

# Feed with custom features
class MLDataFeed(bt.feeds.PandasData):
    lines = tuple(FEATURES)
    params = {feat: -1 for feat in FEATURES}

# Strategy
class MLStrategy(bt.Strategy):
    predicted_labels = None

    def __init__(self):
        self.buffer = []
        self.bar_index = 0
        self.trade_open_bar = None
        self.trade_open_price = None

    def next(self):
        self.bar_index += 1

        # Update buffer
        row = [getattr(self.datas[0], feat)[0] for feat in FEATURES]
        self.buffer.append(row)
        if len(self.buffer) < SEQ_LEN:
            return
        elif len(self.buffer) > SEQ_LEN:
            self.buffer.pop(0)

        pred_index = self.bar_index - SEQ_LEN
        if pred_index >= len(self.predicted_labels):
            return

        current_price = self.data.close[0]

        # ===== Stop-loss logic =====
        if self.position and self.trade_open_price is not None:
            if self.position.size > 0:  # Long position
                if current_price < self.trade_open_price * 0.998:
                    print("Stop-loss hit (Long)")
                    self.close()
                    self.trade_open_bar = None
                    self.trade_open_price = None
                    return
            elif self.position.size < 0:  # Short position
                if current_price > self.trade_open_price * 1.002:
                    print("Stop-loss hit (Short)")
                    self.close()
                    self.trade_open_bar = None
                    self.trade_open_price = None
                    return

        # ===== Time-based exit logic (3-bar timeout) =====
        if self.position and self.trade_open_bar is not None:
            if self.bar_index - self.trade_open_bar >= 3:
                print("Auto-close after 3 bars")
                self.close()
                self.trade_open_bar = None
                self.trade_open_price = None
                return

        # ===== Model prediction logic =====
        prediction = self.predicted_labels[pred_index]

        if not self.position:
            if prediction == 1:
                print("BUY")
                self.buy()
                self.trade_open_bar = self.bar_index
                self.trade_open_price = current_price
            elif prediction == 2:
                print("SELL")
                self.sell()
                self.trade_open_bar = self.bar_index
                self.trade_open_price = current_price
        elif prediction == 0:
            print("HOLD (manual close)")
            self.close()
            self.trade_open_bar = None
            self.trade_open_price = None

# Prediction logic
def compute_predictions(df, model):
    X = []
    for i in range(SEQ_LEN, len(df) - LOOKAHEAD):
        seq = df.iloc[i - SEQ_LEN:i][FEATURES].values.astype(np.float32)
        X.append(seq)
    X = np.array(X)

    predictions = model.predict(X, batch_size=128, verbose=0)
    predicted_labels = []
    predicted_labels = np.argmax(predictions, axis=1)
    '''
    for p in predictions:
        if max(p[1], p[2]) > THRESHOLD:
            if p[1] > p[2]:
                predicted_labels.append(1)
            else:
                predicted_labels.append(2)
        else:
            predicted_labels.append(0)
    '''

    pad_len = len(df) - len(predicted_labels)
    predicted_labels = np.pad(predicted_labels, (pad_len, 0), constant_values=0)

    print(np.unique(predicted_labels, return_counts=True))
    print("Sample probs:", predictions[:5])
    return predicted_labels

# Backtest runner
def run_backtest(csv_path):
    print("Loading csv data...")
    data = pd.read_csv(csv_path, parse_dates=["timestamp"])
    data.set_index("timestamp", inplace=True)

    missing = [feat for feat in FEATURES if feat not in data.columns]
    if missing:
        raise ValueError(f"Missing features in data: {missing}")

    # Random day selection
    unique_days = data.index.normalize().unique()
    chosen_day = random.choice(unique_days)
    print(f"Selected random backtest date: {chosen_day.date()}")

    day_data = data[data.index.normalize() == chosen_day]

    if len(day_data) < SEQ_LEN + LOOKAHEAD:
        raise ValueError("Not enough data on selected day to run backtest.")

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_NAME)

    print("Precomputing predictions...")
    predicted_labels = compute_predictions(day_data, model)
    MLStrategy.predicted_labels = predicted_labels

    print("Feeding data into backtrader")
    data_feed = MLDataFeed(dataname=day_data)

    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.addstrategy(MLStrategy)
    cerebro.broker.set_cash(10000)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print("Starting backtest...")
    results = cerebro.run()
    strat = results[0]

    print("\nTrade Analysis:")
    print(strat.analyzers.trades.get_analysis())

    print("\nSharpe Ratio:")
    print(strat.analyzers.sharpe.get_analysis())

    cerebro.plot()

if __name__ == "__main__":
    run_backtest("preprocessed_training_data.csv")
