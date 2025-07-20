"""
Tests for genome representation and parameter specifications.
"""

import pytest
import random
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from evo.optimization.genetic.genome import (
    ParameterSpec,
    Uniform,
    IntRange,
    Choice,
    GenomeConfig,
    Genome
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.optimization,
    pytest.mark.genetic
]


class TestParameterSpec:
    """Test ParameterSpec abstract base class."""
    
    def test_parameter_spec_abstract(self):
        """Test that ParameterSpec cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ParameterSpec()


class TestUniform:
    """Test Uniform parameter specification."""
    
    def test_uniform_creation(self):
        """Test creating a Uniform parameter specification."""
        uniform = Uniform(low=0.0, high=1.0, precision=4)
        
        assert uniform.low == 0.0
        assert uniform.high == 1.0
        assert uniform.precision == 4
    
    def test_uniform_sample(self):
        """Test sampling from uniform distribution."""
        uniform = Uniform(low=0.0, high=1.0, precision=2)
        
        # Test multiple samples
        samples = [uniform.sample() for _ in range(100)]
        
        # All samples should be within bounds
        assert all(0.0 <= sample <= 1.0 for sample in samples)
        
        # All samples should have correct precision
        assert all(len(str(sample).split('.')[-1]) <= 2 for sample in samples)
    
    def test_uniform_mutate(self):
        """Test mutation of uniform parameter."""
        uniform = Uniform(low=0.0, high=1.0, precision=3)
        original_value = 0.5
        
        # Test mutation
        mutated = uniform.mutate(original_value, rate=0.2)
        
        # Should be within bounds
        assert 0.0 <= mutated <= 1.0
        
        # Should have correct precision
        assert len(str(mutated).split('.')[-1]) <= 3
        
        # Should be different from original (with high probability)
        assert mutated != original_value
    
    def test_uniform_mutate_bounds(self):
        """Test that mutation respects bounds."""
        uniform = Uniform(low=0.0, high=1.0, precision=2)
        
        # Test mutation at boundaries
        mutated_low = uniform.mutate(0.0, rate=0.5)
        mutated_high = uniform.mutate(1.0, rate=0.5)
        
        assert 0.0 <= mutated_low <= 1.0
        assert 0.0 <= mutated_high <= 1.0
    
    def test_uniform_validate(self):
        """Test validation of uniform parameter values."""
        uniform = Uniform(low=0.0, high=1.0, precision=2)
        
        # Valid values
        assert uniform.validate(0.0) is True
        assert uniform.validate(0.5) is True
        assert uniform.validate(1.0) is True
        
        # Invalid values
        assert uniform.validate(-0.1) is False
        assert uniform.validate(1.1) is False
        assert uniform.validate("string") is False
        assert uniform.validate(None) is False


class TestIntRange:
    """Test IntRange parameter specification."""
    
    def test_int_range_creation(self):
        """Test creating an IntRange parameter specification."""
        int_range = IntRange(low=1, high=10)
        
        assert int_range.low == 1
        assert int_range.high == 10
    
    def test_int_range_sample(self):
        """Test sampling from integer range."""
        int_range = IntRange(low=1, high=10)
        
        # Test multiple samples
        samples = [int_range.sample() for _ in range(100)]
        
        # All samples should be within bounds and integers
        assert all(1 <= sample <= 10 for sample in samples)
        assert all(isinstance(sample, int) for sample in samples)
    
    def test_int_range_mutate(self):
        """Test mutation of integer range parameter."""
        int_range = IntRange(low=1, high=10)
        original_value = 5
        
        # Test mutation
        mutated = int_range.mutate(original_value, rate=0.5)
        
        # Should be within bounds and integer
        assert 1 <= mutated <= 10
        assert isinstance(mutated, int)
        
        # Should be different from original (with high probability)
        assert mutated != original_value
    
    def test_int_range_mutate_single_value(self):
        """Test mutation when only one value is available."""
        int_range = IntRange(low=5, high=5)
        original_value = 5
        
        # Should return the same value when no other options
        mutated = int_range.mutate(original_value, rate=0.5)
        assert mutated == original_value
    
    def test_int_range_validate(self):
        """Test validation of integer range parameter values."""
        int_range = IntRange(low=1, high=10)
        
        # Valid values
        assert int_range.validate(1) is True
        assert int_range.validate(5) is True
        assert int_range.validate(10) is True
        
        # Invalid values
        assert int_range.validate(0) is False
        assert int_range.validate(11) is False
        assert int_range.validate(5.5) is False
        assert int_range.validate("string") is False
        assert int_range.validate(None) is False


class TestChoice:
    """Test Choice parameter specification."""
    
    def test_choice_creation(self):
        """Test creating a Choice parameter specification."""
        options = ["option1", "option2", "option3"]
        choice = Choice(options)
        
        assert choice.options == options
    
    def test_choice_sample(self):
        """Test sampling from choice options."""
        options = ["option1", "option2", "option3"]
        choice = Choice(options)
        
        # Test multiple samples
        samples = [choice.sample() for _ in range(100)]
        
        # All samples should be from the options
        assert all(sample in options for sample in samples)
    
    def test_choice_mutate(self):
        """Test mutation of choice parameter."""
        options = ["option1", "option2", "option3"]
        choice = Choice(options)
        original_value = "option1"
        
        # Test mutation
        mutated = choice.mutate(original_value, rate=0.5)
        
        # Should be from options
        assert mutated in options
        
        # Should be different from original (with high probability)
        assert mutated != original_value
    
    def test_choice_mutate_single_option(self):
        """Test mutation when only one option is available."""
        options = ["option1"]
        choice = Choice(options)
        original_value = "option1"
        
        # Should return the same value when no other options
        mutated = choice.mutate(original_value, rate=0.5)
        assert mutated == original_value
    
    def test_choice_validate(self):
        """Test validation of choice parameter values."""
        options = ["option1", "option2", "option3"]
        choice = Choice(options)
        
        # Valid values
        assert choice.validate("option1") is True
        assert choice.validate("option2") is True
        assert choice.validate("option3") is True
        
        # Invalid values
        assert choice.validate("option4") is False
        assert choice.validate(123) is False
        assert choice.validate(None) is False


class TestGenomeConfig:
    """Test GenomeConfig class."""
    
    def test_genome_config_creation(self):
        """Test creating a GenomeConfig instance."""
        config = GenomeConfig()
        
        # Check that all parameters are defined
        assert hasattr(config, 'learning_rate')
        assert hasattr(config, 'entropy_coef_init')
        assert hasattr(config, 'entropy_coef_final')
        assert hasattr(config, 'gae_lambda')
        assert hasattr(config, 'gamma')
        assert hasattr(config, 'clip_range')
        assert hasattr(config, 'batch_size')
        assert hasattr(config, 'sl_penalty_coef')
        assert hasattr(config, 'tp_reward_coef')
        assert hasattr(config, 'timeout_duration')
        assert hasattr(config, 'ongoing_reward_coef')
    
    def test_get_parameter_space(self):
        """Test getting parameter space from config."""
        config = GenomeConfig()
        parameter_space = config.get_parameter_space()
        
        # Should return a dictionary
        assert isinstance(parameter_space, dict)
        
        # Should contain all parameters
        expected_params = [
            'learning_rate', 'entropy_coef_init', 'entropy_coef_final',
            'gae_lambda', 'gamma', 'clip_range', 'batch_size',
            'sl_penalty_coef', 'tp_reward_coef', 'timeout_duration',
            'ongoing_reward_coef'
        ]
        
        for param in expected_params:
            assert param in parameter_space
            assert isinstance(parameter_space[param], ParameterSpec)
    
    def test_parameter_types(self):
        """Test that parameters have correct types."""
        config = GenomeConfig()
        parameter_space = config.get_parameter_space()
        
        # Check specific parameter types
        assert isinstance(parameter_space['learning_rate'], Uniform)
        assert isinstance(parameter_space['batch_size'], Choice)
        assert isinstance(parameter_space['timeout_duration'], IntRange)


class TestGenome:
    """Test Genome class."""
    
    def test_genome_creation(self):
        """Test creating a Genome instance."""
        values = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'timeout_duration': 5
        }
        
        genome = Genome(values)
        
        assert genome.values == values
        assert isinstance(genome.config, GenomeConfig)
        assert isinstance(genome.parameter_space, dict)
    
    def test_genome_creation_with_config(self):
        """Test creating a Genome with custom config."""
        config = GenomeConfig()
        values = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'timeout_duration': 5
        }
        
        genome = Genome(values, config)
        
        assert genome.config == config
        assert genome.values == values
    
    def test_genome_validation_valid_values(self):
        """Test genome creation with valid values."""
        values = {
            'learning_rate': 0.001,  # Valid for Uniform(0.0001, 0.01)
            'batch_size': 64,        # Valid for Choice([32, 64, 128, 256])
            'timeout_duration': 5    # Valid for IntRange(1, 10)
        }
        
        genome = Genome(values)
        
        # Should not raise any exceptions
        assert genome.values == values
    
    def test_genome_validation_invalid_values(self):
        """Test genome creation with invalid values."""
        # Invalid learning rate (out of bounds)
        invalid_values = {
            'learning_rate': 0.1,  # Too high for Uniform(0.0001, 0.01)
            'batch_size': 64,
            'timeout_duration': 5
        }
        
        with pytest.raises(ValueError, match="Invalid value for learning_rate"):
            Genome(invalid_values)
    
    def test_genome_validation_unknown_parameter(self):
        """Test genome creation with unknown parameter."""
        values = {
            'learning_rate': 0.001,
            'unknown_param': 123
        }
        
        with pytest.raises(ValueError, match="Unknown parameter"):
            Genome(values)
    
    def test_genome_random_creation(self):
        """Test creating a random genome."""
        genome = Genome.random()
        
        # Should have all parameters
        assert len(genome.values) > 0
        
        # All values should be valid
        for name, value in genome.values.items():
            spec = genome.parameter_space[name]
            assert spec.validate(value)
    
    def test_genome_random_with_config(self):
        """Test creating a random genome with custom config."""
        config = GenomeConfig()
        genome = Genome.random(config=config)
        
        assert genome.config == config
        assert len(genome.values) > 0
    
    def test_genome_random_with_seed(self):
        """Test that random genome creation is reproducible with seed."""
        # Create two genomes with same seed
        genome1 = Genome.random(seed=42)
        genome2 = Genome.random(seed=42)
        
        # Should be identical
        assert genome1.values == genome2.values
    
    def test_genome_mutation(self):
        """Test genome mutation."""

        genome = Genome.random(seed=1)
        
        # Mutate genome
        mutated = genome.mutate(mutation_rate=1.0)
        
        # Should be a different genome
        assert mutated != genome
        
        # Should have same config
        assert mutated.config == genome.config
        
        # All values should still be valid
        for name, value in mutated.values.items():
            spec = mutated.parameter_space[name]
            assert spec.validate(value)
    
    def test_genome_crossover(self):
        """Test genome crossover."""
        values1 = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'timeout_duration': 5
        }
        values2 = {
            'learning_rate': 0.005,
            'batch_size': 128,
            'timeout_duration': 8
        }
        
        genome1 = Genome(values1)
        genome2 = Genome(values2)
        
        # Perform crossover
        offspring = genome1.crossover(genome2, crossover_rate=0.5)
        
        # Should be a valid genome
        assert isinstance(offspring, Genome)
        assert offspring.config == genome1.config
        
        # All values should be valid
        for name, value in offspring.values.items():
            spec = offspring.parameter_space[name]
            assert spec.validate(value)
    
    def test_genome_hash(self):
        """Test genome hashing."""
        values = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'timeout_duration': 5
        }
        genome = Genome(values)
        
        # Should return a string hash
        genome_hash = genome.hash()
        assert isinstance(genome_hash, str)
        assert len(genome_hash) > 0
        
        # Same genome should have same hash
        genome2 = Genome(values)
        assert genome2.hash() == genome_hash
    
    def test_genome_distance(self):
        """Test genome distance calculation."""
        values1 = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'timeout_duration': 5
        }
        values2 = {
            'learning_rate': 0.005,
            'batch_size': 128,
            'timeout_duration': 8
        }
        
        genome1 = Genome(values1)
        genome2 = Genome(values2)
        
        # Calculate distance
        distance = genome1.distance(genome2)
        
        # Should be a positive number
        assert isinstance(distance, float)
        assert distance >= 0
    
    def test_genome_to_dict(self):
        """Test converting genome to dictionary."""
        
        genome = Genome.random(seed=1)
        
        genome_dict = genome.to_dict()
        
        assert isinstance(genome_dict, dict)
        assert genome_dict == genome.values
    
    def test_genome_to_json(self):
        """Test converting genome to JSON."""
        values = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'timeout_duration': 5
        }
        genome = Genome(values)
        
        json_str = genome.to_json()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed == values
    
    def test_genome_from_json(self):
        """Test creating genome from JSON."""
        values = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'timeout_duration': 5
        }
        genome = Genome(values)
        
        json_str = genome.to_json()
        reconstructed = Genome.from_json(json_str)
        
        # Should be equivalent
        assert reconstructed.values == genome.values
    
    def test_genome_equality(self):
        """Test genome equality."""
        values = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'timeout_duration': 5
        }
        
        genome1 = Genome(values)
        genome2 = Genome(values)
        genome3 = Genome({
            'learning_rate': 0.005,
            'batch_size': 64,
            'timeout_duration': 5
        })
        
        # Same values should be equal
        assert genome1 == genome2
        
        # Different values should not be equal
        assert genome1 != genome3
    
    def test_genome_hash_function(self):
        """Test genome hash function for use in sets/dicts."""
        values = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'timeout_duration': 5
        }
        
        genome1 = Genome(values)
        genome2 = Genome(values)
        
        # Same genomes should have same hash
        assert hash(genome1) == hash(genome2)
        
        # Can be used in sets
        genome_set = {genome1, genome2}
        assert len(genome_set) == 1
    
    def test_genome_repr(self):
        """Test genome string representation."""
        values = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'timeout_duration': 5
        }
        genome = Genome(values)
        
        repr_str = repr(genome)
        
        # Should be a string
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0 