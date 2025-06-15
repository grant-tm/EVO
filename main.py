import asyncio
import logging
from config import USE_SIMULATION, API_KEY, API_SECRET, SYMBOL, SIM_SPEED
from datastream.streamer import LiveDataStream, SimulatedDataStream
from datastream.handlers import on_minute_bar

logging.basicConfig(level=logging.INFO)

async def main():
    
    if USE_SIMULATION:
        logging.info("Initalizing simulated datastream")
        stream = SimulatedDataStream(API_KEY, API_SECRET, SYMBOL, speed=SIM_SPEED)
    else:
        logging.info("Initalizing live datastream")
        stream = LiveDataStream(API_KEY, API_SECRET, SYMBOL)

    logging.info("Starting datastream")
    await stream.start(on_minute_bar)

if __name__ == "__main__":
    asyncio.run(main())
