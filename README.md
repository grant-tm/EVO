**EVO** is an experimental neural-network powered trading agent testing the hypothesis that short-term price action in financial markets can be learned and exploited through reinforcement learning.
---

## Overview
EVO uses Proximal Policy Optimization (PPO) to train a scalping-oriented policy that learns to profit from minute-level market fluctuations, typically closing positions 1-3 minutes after opening.

The agent is trained using Stable-Baselines3 in a custom Gymnasium-compatible environment that emits raw price data and technical indicators as observations.

A genetic search engine tunes both PPO hyperparameters and environment-specific reward shaping to find an optimal training configuration (that which yields a maximally profitable agent).

EVO supports both live paper trading via Alpaca and simulated real-time evaluation, enabling flexible deployment and robust testing across conditions.

## Proximal Policy Optimization
EVO uses **Proximal Policy Optimization (PPO)**, a policy gradient algorithm known for balancing exploration and stability. 
PPO updates the policy using a clipped objective function that prevents overly aggressive changes to the policy network, making it well-suited for continuous reinforcement learning in volatile environments like financial markets.

In EVO, PPO is used to train an agent that takes discrete trading actions (`Buy`, `Sell`, or `Hold`) based on a rolling sequence of normalized price features and technical indicators. 
The agent observes a fixed-length window of market data (`SEQ_LEN` timesteps, ~15 minutes in most configurations), extracted from a preprocessed dataset of minute-resolution bars fetched via the Alpaca API. 
Each observation is a matrix of shape `[SEQ_LEN, num_features]`, constructed in the custom `TradingEnv` (see: [Training Environment](##training_environment)).

The PPO model is implemented using **Stable-Baselines3**, with a two-headed MLP architecture for both policy and value networks. 
A number of training hyperparameters are either statically defined in `config.py` or dynamically optimized using a genome-based tuning strategy (see: [Genetic Tuning](##genetic-tuning)).

EVO further customizes PPO behavior by:
- **Normalizing observations** across training episodes with `VecNormalize`, while keeping reward normalization disabled to preserve reward signal magnitude for shaping.
- **Scheduling entropy regularization** using a custom `EntropyAnnealingCallback`, which linearly interpolates the entropy coefficient from an initial to final value during training.
- **Monitoring evaluation performance** with checkpointed environments to assess generalization across train/validation splits.

## Training Environment
The core training environment in EVO is a custom implementation of the OpenAI Gymnasium API designed to simulate realistic intraday trading behavior. 
The environment exposes a discrete action space of three possible decisions at each timestep: `Hold`, `Buy`, or `Sell`. 
Only one open position is allowed at a time, and trades are evaluated using price deltas between the entry and current price.
### Observations
Each observation is a sliding window of normalized price features and standardized technical indicators computed from minute-resolution historical data. 
Observations are shaped as a 2D array of shape `[SEQ_LEN, num_features]`, allowing the PPO agent to incorporate short-term temporal context into its policy decisions.
### Trade Lifecycle and Logic
The environment tracks the agent's open position (`long`, `short`, or `flat`) and the entry price for each trade. It enforces several realistic constraints and behaviors:
- **Stop-loss and take-profit thresholds** (`SL_PCT` and `TP_PCT`) automatically close trades if predefined percentage changes are reached.
- **Timeout exits** forcibly close stale trades after a configurable number of steps (`TIMEOUT_DURATION`), penalizing indecision or poor timing.
- **Unrealized PnL shaping** encourages profitable holding behavior using a small reward/penalty during trade progression (`ONGOING_REWARD_COEF`).
- **Idle penalty** discourages excessive inaction and nudges the agent toward participation (`IDLE_PENALTY`).
### Reward Shaping
Reward shaping is modular and parameterized to support tuning via genetic optimization. 
The agent receives rewards or penalties not just based on realized outcomes, but also for behaviors that indicate poor trade management or excessive risk. 
All rewards are clipped to a configurable range to stabilize training.

This environment is designed to ensure the PPO agent learns trading behavior under realistic constraints with ample opportunity for tuning and shaping behavior through direct manipulation of reward terms.

## Genetic Tuning
To automate hyperparameter optimization and discover high-performing trading behaviors, EVO includes a modular genetic search engine that evolves PPO training and reward-shaping configurations over multiple generations. In the context of genetic tuning, an **individual** is a PPO agent trained under a set of parameters encoded in a **genome**, and a **population** is a group of individuals.
### Genome Structure
The genome includes:
- **Training hyperparameters** such as learning rate, entropy coefficients, clipping range, GAE lambda, and discount factor.
- **Reward shaping parameters** like stop-loss and take-profit coefficients, trade timeout duration, and ongoing reward multipliers.

Parameter domains are defined using `Uniform`, `IntRange`, or `Choice` sampling strategies for continuous values, integer values, and discrete sets, respectively. 
This allows fine control over mutation dynamics and exploration ranges. 
The full set of parameters exposed to genetic tuning can be found in `evolution_engine/genome.py`
### Evolutionary Loop
The `GeneticSearch` class orchestrates the optimization process:
1. **Initialization**: A random population of genomes is generated.
2. **Training**: Each genome is used to train a PPO agent.
3. **Evaluation**: Each trained agent is evaluated via backtesting across multiple randomized data slices, producing a Sharpe ratio-like fitness score based on realized PnL distributions.
4. **Selection & Mutation**: The top-performing genomes are preserved and the next generation's population is generated via mutation from elite genomes and the introduction of  new randomized genomes.
Fitness scores are tracked across generations, and the best genome is returned upon completion.

## Live Trading
EVO is designed to support **real-time paper trading** via integration with the [Alpaca Markets](https://alpaca.markets/) brokerage API. The current live trading pipeline is under active development.
### Streaming Architecture
At the core of EVO's live deployment is an asynchronous streaming interface, which provides minute-level bar updates to a handler function responsible for making decisions:
- `LiveDataStream` subscribes to real-time market data using Alpaca’s `StockDataStream`.
- `SimulatedDataStream` replays historical market data as a stand-in for live conditions. This enables testing policy behavior under time-sensitive constraints without trading real capital.
Both streamers emit data through a common `start()` interface and dispatch each bar to a user-defined handler (e.g., `on_minute_bar`), allowing seamless switching between live and simulated trading modes via the `USE_SIMULATION` flag in `config.py`.
### Usage
To begin streaming and trading:
1. Sign up for an [Alpaca account](https://alpaca.markets/) and generate paper trading API credentials.
2. Create a `.env` file with the following:
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```
3. Configure your `config.py`:
```python
USE_SIMULATION = False  # Use True to run in simulation mode
SYMBOL = "AAPL"         # Your trading symbol
SIM_SPEED = 1.0         # Delay between bars in simulation mode (in seconds)
```
4. Start the agent:
```bash
python main.py
```
### Roadmap
While bar data streaming is operational, the following components are under development:
- **Real-time model inference** and action selection from a trained PPO agent.
- **Order execution logic** to translate agent actions into Alpaca trade orders.
- **Position management** and safety checks (e.g., max drawdown, position size).
- **Live performance monitoring** including win rate, PnL tracking, and latency metrics.
These modules will enable full loop integration—from observation to decision to execution—bringing EVO closer to autonomous live trading.
