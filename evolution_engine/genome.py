import random
from typing import Any, Dict, Union
import copy
import hashlib
import json

# ==========================
# Parameter Range Wrappers
# ==========================

class Uniform:
    
    def __init__(self, low: float, high: float, precision: int = 6):
        self.low = low
        self.high = high
        self.precision = precision  # Number of decimal places to round to

    def sample(self):
        return round(random.uniform(self.low, self.high), self.precision)

    def mutate(self, value, rate=0.2):
        delta = (self.high - self.low) * rate
        mutated = value + random.uniform(-delta, delta)
        return round(min(self.high, max(self.low, mutated)), self.precision)

class IntRange:
    
    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
    
    def sample(self):
        return random.randint(self.low, self.high)
    
    def mutate(self, value):
        choices = list(range(self.low, self.high + 1))
        choices.remove(value)
        return random.choice(choices)

class Choice:
    
    def __init__(self, options):
        self.options = options
    
    def sample(self):
        return random.choice(self.options)
    
    def mutate(self, value):
        options = [o for o in self.options if o != value]
        return random.choice(options)

# ==========================
# Genome Class
# ==========================

class Genome:
    parameter_space: Dict[str, Union[Uniform, IntRange, Choice]] = {
        # Training
        "learning_rate": Uniform(0.0001, 0.01, precision=6),
        "entropy_coef_init": Uniform(0.01, 0.1, precision=5),
        "entropy_coef_final": Uniform(0.001, 0.01, precision=5),
        "gae_lambda": Uniform(0.90, 0.999, precision=4),
        "gamma": Uniform(0.90, 0.999, precision=4),
        "clip_range": Uniform(0.1, 0.3, precision=3),
        "batch_size": Choice([32, 64, 128, 256]),

        # Reward shaping
        "sl_penalty_coef": Uniform(5.0, 20.0, precision=2),
        "tp_reward_coef": Uniform(5.0, 20.0, precision=2),
        "timeout_duration": IntRange(1, 10),
        "ongoing_reward_coef": Uniform(0.1, 1.0, precision=3)
    }

    def __init__(self, values: Dict[str, Any]):
        self.values = values  # Dict[str, Any]

    @classmethod
    def random(cls, seed: int = None) -> "Genome":
        if seed is not None:
            random.seed(seed)
        values = {k: spec.sample() for k, spec in cls.parameter_space.items()}
        return cls(values)

    def mutate(self, mutation_rate=0.1) -> "Genome":
        
        new_values = copy.deepcopy(self.values)
        
        for key, spec in self.parameter_space.items():
            if random.random() < mutation_rate:
                new_values[key] = spec.mutate(new_values[key])
        
        return Genome(new_values)
    
    # Returns a SHA256 hash of the genome's parameters
    def hash(self) -> str:
        serialized = json.dumps(self.values, sort_keys=True)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return self.values

    def __repr__(self):
        return f"Genome({self.values})"
    
    def print_values(self):
        for key in self.values:
            print(f"{key} = {self.values.get(key)}")
