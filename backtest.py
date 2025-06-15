import backtrader as bt
import numpy as np
import pandas as pd
import tensorflow as tf
from config import MODEL_NAME, FEATURES, SEQ_LEN, LOOKAHEAD, THRESHOLD # Ensure SEQ_LEN is defined

class MLDataFeed(bt.feeds.PandasData):
    lines = tuple(FEATURES)  # e.g., ('return', 'rsi', 'macd', ...)
    params = {feat: -1 for feat in FEATURES}

class MLStrategy(bt.Strategy):
    predicted_labels = None

    def __init__(self):
        self.buffer = []
        self.bar_index = 0

    def next(self):
        self.bar_index += 1

        # Build buffer
        row = [getattr(self.datas[0], feat)[0] for feat in FEATURES]
        self.buffer.append(row)
        if len(self.buffer) < SEQ_LEN:
            return
        elif len(self.buffer) > SEQ_LEN:
            self.buffer.pop(0)

        pred_index = self.bar_index - SEQ_LEN
        
        if pred_index >= len(self.predicted_labels):
            return  # Out of prediction bounds

        prediction = self.predicted_labels[pred_index]

        if prediction == 1 and not self.position:
            self.buy()
        elif prediction == 2 and not self.position:
            self.sell()
        elif prediction == 0 and self.position:
            self.close()

def compute_predictions(df, model):
    
    X = []
    for i in range(SEQ_LEN, len(df) - LOOKAHEAD):
        seq = df.iloc[i - SEQ_LEN:i][FEATURES].values.astype(np.float32)
        X.append(seq)
    X = np.array(X)
    
    predictions = model.predict(X, batch_size=64, verbose=0)
    predicted_labels = []
    for i, p in enumerate(predictions):
        if p[1] > THRESHOLD:
            predicted_labels.append(1)  # Buy
        elif p[2] > THRESHOLD:
            predicted_labels.append(2)  # Sell
        else:
            predicted_labels.append(0)  # Hold
    pad_len = len(df) - len(predicted_labels)
    predicted_labels = np.pad(predicted_labels, (pad_len, 0), constant_values=0)
    
    print(np.unique(predicted_labels, return_counts=True))
    print("Sample probs:", predictions[:5])
    
    return predicted_labels

def run_backtest(csv_path):
    print("Loading csv data...")
    data = pd.read_csv(csv_path, parse_dates=["timestamp"])
    data.set_index("timestamp", inplace=True)

    missing = [feat for feat in FEATURES if feat not in data.columns]
    if missing:
        raise ValueError(f"Missing features in data: {missing}")

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_NAME)

    print("Precomputing predictions...")
    predicted_labels = compute_predictions(data, model)

    # Store to class-level variable before run
    MLStrategy.predicted_labels = predicted_labels

    print("Feeding data into backtrader")
    data_feed = MLDataFeed(dataname=data)

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
