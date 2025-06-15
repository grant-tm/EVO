import logging

async def on_minute_bar(bar):
    print(f"[{bar.timestamp}] {bar.symbol} OHLCV = {bar.open}, {bar.high}, {bar.low}, {bar.close}, {bar.volume}")
