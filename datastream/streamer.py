import asyncio
import logging

import datetime
import time

from types import SimpleNamespace

import pandas as pd

from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

class LiveDataStream:
    
    # initialize StockDataStream object
    def __init__(self, api_key, api_secret, symbol):
        self.symbol = symbol
        logging.info("[LiveDataStream] Authenticating api keys")
        self.stream = StockDataStream(api_key, api_secret)

    # subscribe to minute bars for the target symbol and run the datastream
    async def start(self, handler):
        logging.info(f"[LiveDataStream] Subscribing to {self.symbol}")
        self.stream.subscribe_bars(handler, self.symbol)
        await self.stream._run_forever()


class SimulatedDataStream:
    
    # initialize simulated datastream
    def __init__(self, api_key, api_secret, symbol, speed=1.0):
        self.symbol = symbol
        self.speed = speed
        self.client = StockHistoricalDataClient(api_key, api_secret)

    # fetch historical data and periodically feed it to the async handler
    async def start(self, handler):
        
        # Pick a guaranteed trading window
        trading_day = datetime.datetime.now() - datetime.timedelta(days=2)
        start = trading_day.replace(hour=10, minute=0, second=0, microsecond=0)
        end = trading_day.replace(hour=12, minute=0, second=0, microsecond=0)

        print(f"[SimulatedDataStream] Fetching {self.symbol} bars from {start} to {end}")

        # Fetch historical bars for stock symbols
        bars = self.client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end
        )).df

        # Unpack multiselection if multiple symbols are selected
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(self.symbol, level="symbol")

        # Exit if no data was received
        if bars.empty:
            print("[SimulatedDataStream] No data returned from Alpaca.")
            return

        # Periodically send minute bars to the handler at simulation speed
        for i, row in bars.iterrows():
            bar = SimpleNamespace(
                timestamp=i,
                symbol=self.symbol,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"]
            )
            await handler(bar)
            await asyncio.sleep(self.speed) # Simulate real-time feed
