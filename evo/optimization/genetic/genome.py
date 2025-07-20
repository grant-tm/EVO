"""
Genome representation for genetic algorithm optimization.

This module provides the Genome class and parameter space definitions for
hyperparameter optimization in the EVO trading system.
"""

import random
import copy
import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


class ParameterSpec(ABC):
    """Abstract base class for parameter specifications."""
    
    @abstractmethod
    def sample(self) -> Any:
        """Generate a random sample from this parameter space."""
        pass
    
    @abstractmethod
    def mutate(self, value: Any, rate: float = 0.2) -> Any:
        """Mutate a value within this parameter space."""
        pass
    
    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Validate if a value is within this parameter space."""
        pass


class Uniform(ParameterSpec):
    """Uniform continuous parameter specification."""
    
    def __init__(self, low: float, high: float, precision: int = 6):
        self.low = low
        self.high = high
        self.precision = precision
    
    def sample(self) -> float:
        """Generate a random sample from uniform distribution."""
        return round(random.uniform(self.low, self.high), self.precision)
    
    def mutate(self, value: float, rate: float = 0.2) -> float:
        """Mutate value by adding random noise within bounds."""
        delta = (self.high - self.low) * rate
        mutated = value + random.uniform(-delta, delta)
        return round(min(self.high, max(self.low, mutated)), self.precision)
    
    def validate(self, value: Any) -> bool:
        """Check if value is within bounds."""
        return isinstance(value, (int, float)) and self.low <= value <= self.high


class IntRange(ParameterSpec):
    """Integer range parameter specification."""
    
    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
    
    def sample(self) -> int:
        """Generate a random integer sample."""
        return random.randint(self.low, self.high)
    
    def mutate(self, value: int, rate: float = 0.2) -> int:
        """Mutate by selecting a different integer value."""
        choices = list(range(self.low, self.high + 1))
        choices.remove(value)
        return random.choice(choices) if choices else value
    
    def validate(self, value: Any) -> bool:
        """Check if value is a valid integer in range."""
        return isinstance(value, int) and self.low <= value <= self.high


class Choice(ParameterSpec):
    """Discrete choice parameter specification."""
    
    def __init__(self, options: List[Any]):
        self.options = options
    
    def sample(self) -> Any:
        """Generate a random choice from options."""
        return random.choice(self.options)
    
    def mutate(self, value: Any, rate: float = 0.2) -> Any:
        """Mutate by selecting a different option."""
        options = [o for o in self.options if o != value]
        return random.choice(options) if options else value
    
    def validate(self, value: Any) -> bool:
        """Check if value is one of the valid options."""
        return value in self.options


@dataclass
class GenomeConfig:
    """Configuration for genome parameter space."""
    
    # Training hyperparameters
    learning_rate: Uniform = field(default_factory=lambda: Uniform(0.0001, 0.01, 6))
    entropy_coef_init: Uniform = field(default_factory=lambda: Uniform(0.01, 0.1, 5))
    entropy_coef_final: Uniform = field(default_factory=lambda: Uniform(0.001, 0.01, 5))
    gae_lambda: Uniform = field(default_factory=lambda: Uniform(0.90, 0.999, 4))
    gamma: Uniform = field(default_factory=lambda: Uniform(0.90, 0.999, 4))
    clip_range: Uniform = field(default_factory=lambda: Uniform(0.1, 0.3, 3))
    batch_size: Choice = field(default_factory=lambda: Choice([32, 64, 128, 256]))
    
    # Reward shaping parameters
    sl_penalty_coef: Uniform = field(default_factory=lambda: Uniform(5.0, 20.0, 2))
    tp_reward_coef: Uniform = field(default_factory=lambda: Uniform(5.0, 20.0, 2))
    timeout_duration: IntRange = field(default_factory=lambda: IntRange(1, 10))
    ongoing_reward_coef: Uniform = field(default_factory=lambda: Uniform(0.1, 1.0, 3))
    
    def get_parameter_space(self) -> Dict[str, ParameterSpec]:
        """Get the parameter space as a dictionary."""
        return {
            name: getattr(self, name) 
            for name in self.__dataclass_fields__.keys()
        }
    
    def __eq__(self, other: Any) -> bool:
        """Compare configurations by parameter specifications."""
        if not isinstance(other, GenomeConfig):
            return False
        
        # Compare each parameter specification
        for field_name in self.__dataclass_fields__.keys():
            spec1 = getattr(self, field_name)
            spec2 = getattr(other, field_name)
            
            # Compare based on parameter spec type and values
            if type(spec1) != type(spec2):
                return False
            
            if isinstance(spec1, Uniform):
                if (spec1.low != spec2.low or spec1.high != spec2.high or 
                    spec1.precision != spec2.precision):
                    return False
            elif isinstance(spec1, IntRange):
                if spec1.low != spec2.low or spec1.high != spec2.high:
                    return False
            elif isinstance(spec1, Choice):
                if spec1.options != spec2.options:
                    return False
        
        return True


class Genome:
    """
    Genome representation for genetic algorithm optimization.
    
    A genome represents a set of hyperparameters that can be evolved through
    genetic algorithms to optimize trading strategy performance.
    """
    
    def __init__(
        self, 
        values: Dict[str, Any],
        config: Optional[GenomeConfig] = None
    ):
        """
        Initialize genome with parameter values.
        
        Args:
            values: Dictionary of parameter name to value mappings
            config: Genome configuration defining parameter spaces
        """
        self.config = config or GenomeConfig()
        self.parameter_space = self.config.get_parameter_space()
        self.values = self._validate_and_clean_values(values)
    
    def _validate_and_clean_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean input values against parameter space."""
        cleaned = {}
        for name, value in values.items():
            if name not in self.parameter_space:
                raise ValueError(f"Unknown parameter: {name}")
            
            spec = self.parameter_space[name]
            if not spec.validate(value):
                raise ValueError(f"Invalid value for {name}: {value}")
            
            cleaned[name] = value
        
        return cleaned
    
    @classmethod
    def random(cls, config: Optional[GenomeConfig] = None, seed: Optional[int] = None) -> "Genome":
        """
        Create a random genome.
        
        Args:
            config: Genome configuration
            seed: Random seed for reproducibility
            
        Returns:
            Randomly generated genome
        """
        if seed is not None:
            random.seed(seed)
        
        config = config or GenomeConfig()
        parameter_space = config.get_parameter_space()
        values = {name: spec.sample() for name, spec in parameter_space.items()}
        
        return cls(values, config)
    
    def mutate(self, mutation_rate: float = 0.1) -> "Genome":
        """
        Create a mutated copy of this genome.
        
        Args:
            mutation_rate: Probability of mutating each parameter
            
        Returns:
            Mutated genome
        """
        new_values = copy.deepcopy(self.values)
        
        for name, spec in self.parameter_space.items():
            if random.random() < mutation_rate:
                new_values[name] = spec.mutate(new_values[name])
        
        return Genome(new_values, self.config)
    
    def crossover(self, other: "Genome", crossover_rate: float = 0.5) -> "Genome":
        """
        Perform crossover with another genome.
        
        Args:
            other: Genome to crossover with
            crossover_rate: Probability of taking value from other genome
            
        Returns:
            New genome from crossover
        """
        if self.config != other.config:
            raise ValueError("Genomes must have same configuration for crossover")
        
        new_values = {}
        for name in self.values.keys():
            if random.random() < crossover_rate:
                new_values[name] = other.values[name]
            else:
                new_values[name] = self.values[name]
        
        return Genome(new_values, self.config)
    
    def hash(self) -> str:
        """
        Generate a unique hash for this genome.
        
        Returns:
            SHA256 hash string
        """
        serialized = json.dumps(self.values, sort_keys=True)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()
    
    def distance(self, other: "Genome") -> float:
        """
        Calculate distance to another genome.
        
        Args:
            other: Genome to compare with
            
        Returns:
            Euclidean distance between genomes
        """
        if self.config != other.config:
            raise ValueError("Genomes must have same configuration for distance calculation")
        
        distances = []
        for name in self.values.keys():
            val1 = float(self.values[name])
            val2 = float(other.values[name])
            distances.append((val1 - val2) ** 2)
        
        return np.sqrt(sum(distances))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary representation."""
        return copy.deepcopy(self.values)
    
    def to_json(self) -> str:
        """Convert genome to JSON string."""
        return json.dumps(self.values, sort_keys=True)
    
    @classmethod
    def from_json(cls, json_str: str, config: Optional[GenomeConfig] = None) -> "Genome":
        """Create genome from JSON string."""
        values = json.loads(json_str)
        return cls(values, config)
    
    def __repr__(self) -> str:
        return f"Genome({self.values})"
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Genome):
            return False
        return self.values == other.values and self.config == other.config
    
    def __hash__(self) -> int:
        return hash(self.hash())
    
    def print_values(self) -> None:
        """Print genome values in a formatted way."""
        print("Genome Parameters:")
        for name, value in sorted(self.values.items()):
            print(f"  {name} = {value}") 