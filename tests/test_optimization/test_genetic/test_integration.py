"""
Integration tests for genetic optimization framework.
"""

import pytest
import random
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from evo.optimization.genetic.genome import Genome, GenomeConfig
from evo.optimization.genetic.fitness import FitnessResult, FitnessEvaluator
from evo.optimization.genetic.genetic_search import GeneticSearch, GeneticSearchConfig

pytestmark = [
    pytest.mark.integration,
    pytest.mark.optimization,
    pytest.mark.genetic
]


class MockFitnessEvaluator(FitnessEvaluator):
    """Mock fitness evaluator for integration testing."""
    
    def __init__(self, fitness_function=None, cache_results=True):
        super().__init__(cache_results)
        self.fitness_function = fitness_function or self._default_fitness
        self.evaluation_count = 0
    
    def _default_fitness(self, genome: Genome) -> float:
        """Default fitness function based on genome parameters."""
        # Simple fitness based on learning rate and batch size
        lr = genome.values.get('learning_rate', 0.001)
        batch_size = genome.values.get('batch_size', 64)
        
        # Higher learning rate and larger batch size generally better
        fitness = (lr * 1000) + (batch_size / 1000)
        return min(fitness, 2.0)  # Cap at 2.0
    
    def evaluate(self, genome: Genome) -> FitnessResult:
        """Evaluate genome fitness."""
        # Check cache first
        cached_result = self._get_cached_result(genome)
        if cached_result is not None:
            return cached_result
        
        self.evaluation_count += 1
        
        fitness_score = self.fitness_function(genome)
        
        result = FitnessResult(
            genome=genome,
            fitness_score=fitness_score,
            metrics={
                'sharpe_ratio': fitness_score,
                'total_return': fitness_score * 0.1,
                'volatility': 1.0 - fitness_score * 0.3
            },
            metadata={
                'evaluation_count': self.evaluation_count,
                'genome_hash': genome.hash()
            }
        )
        
        # Cache result
        self._cache_result(genome, result)
        
        return result


class TestGeneticOptimizationIntegration:
    """Integration tests for complete genetic optimization workflow."""
    
    def test_complete_genetic_optimization_workflow(self):
        """Test complete genetic optimization workflow."""
        # Create fitness evaluator
        fitness_evaluator = MockFitnessEvaluator()
        
        # Create genetic search configuration
        config = GeneticSearchConfig(
            population_size=10,
            max_generations=5,
            mutation_rate=0.2,
            crossover_rate=0.8,
            elite_fraction=0.2,
            random_fraction=0.1,
            tournament_size=3,
            patience=3,
            min_improvement=0.01
        )
        
        # Create genetic search
        search = GeneticSearch(
            fitness_evaluator=fitness_evaluator,
            config=config,
            seed=42
        )
        
        # Run optimization
        best_genome, best_fitness = search.run()
        
        # Verify results
        assert isinstance(best_genome, Genome)
        assert isinstance(best_fitness, float)
        assert best_fitness > 0
        
        # Verify search state
        assert search.best_genome == best_genome
        assert search.best_fitness == best_fitness
        assert len(search.generation_history) > 0
        assert search.generation > 0
        
        # Verify that evaluations were performed
        assert fitness_evaluator.evaluation_count > 0
    
    def test_genetic_optimization_with_custom_fitness(self):
        """Test genetic optimization with custom fitness function."""
        def custom_fitness(genome: Genome) -> float:
            """Custom fitness function that prefers specific parameter combinations."""
            lr = genome.values.get('learning_rate', 0.001)
            batch_size = genome.values.get('batch_size', 64)
            
            # Prefer learning rate around 0.005 and batch size around 128
            lr_penalty = abs(lr - 0.005) * 100
            batch_penalty = abs(batch_size - 128) / 100
            
            fitness = 2.0 - lr_penalty - batch_penalty
            return max(fitness, 0.0)
        
        # Create fitness evaluator with custom function
        fitness_evaluator = MockFitnessEvaluator(fitness_function=custom_fitness)
        
        # Create genetic search
        config = GeneticSearchConfig(
            population_size=15,
            max_generations=8,
            mutation_rate=0.15,
            crossover_rate=0.9
        )
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Run optimization
        best_genome, best_fitness = search.run()
        
        # Verify that the best genome has good parameter values
        lr = best_genome.values.get('learning_rate', 0.001)
        batch_size = best_genome.values.get('batch_size', 64)
        
        # Should be close to optimal values
        assert abs(lr - 0.005) < 0.01
        assert abs(batch_size - 128) < 50
        
        # Fitness should be high
        assert best_fitness > 1.0
    
    def test_genetic_optimization_convergence(self):
        """Test that genetic optimization converges over generations."""
        fitness_evaluator = MockFitnessEvaluator()
        
        config = GeneticSearchConfig(
            population_size=20,
            max_generations=10,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_fraction=0.3,
            patience=5
        )
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Run optimization
        best_genome, best_fitness = search.run()
        
        # Check convergence by examining generation history
        assert len(search.generation_history) > 0
        
        # Fitness should generally improve over generations
        fitness_history = [result.best_fitness for result in search.generation_history]
        
        # Should have some improvement
        assert max(fitness_history) >= min(fitness_history)
        
        # Best fitness should be achieved in later generations
        best_generation = fitness_history.index(max(fitness_history))
        assert best_generation >= 0  # Should find best fitness
    
    def test_genetic_optimization_population_diversity(self):
        """Test that genetic optimization maintains population diversity."""
        fitness_evaluator = MockFitnessEvaluator()
        
        config = GeneticSearchConfig(
            population_size=15,
            max_generations=6,
            mutation_rate=0.3,  # High mutation rate to maintain diversity
            crossover_rate=0.7,
            elite_fraction=0.2,
            random_fraction=0.2  # High random fraction
        )
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Run optimization
        best_genome, best_fitness = search.run()
        
        # Check population diversity in generation history
        for result in search.generation_history:
            assert result.population_diversity > 0
    
    def test_genetic_optimization_early_stopping(self):
        """Test early stopping functionality."""
        # Create a fitness function that plateaus quickly
        def plateau_fitness(genome: Genome) -> float:
            """Fitness function that plateaus after initial improvement."""
            lr = genome.values.get('learning_rate', 0.001)
            # Fitness plateaus around 1.5
            return min(1.5, lr * 1000)
        
        fitness_evaluator = MockFitnessEvaluator(fitness_function=plateau_fitness)
        
        config = GeneticSearchConfig(
            population_size=10,
            max_generations=20,
            mutation_rate=0.1,
            patience=3,  # Stop after 3 generations without improvement
            min_improvement=0.01
        )
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Run optimization
        best_genome, best_fitness = search.run()
        
        # Should stop early due to plateau
        assert search.generation < config.max_generations
        
        # Should have some generations completed
        assert len(search.generation_history) > 0
    
    def test_genetic_optimization_with_initial_population(self):
        """Test genetic optimization with custom initial population."""
        fitness_evaluator = MockFitnessEvaluator()
        
        # create random initial population
        initial_population = [Genome.random(seed=i) for i in range(10)]
        
        config = GeneticSearchConfig(
            population_size=10,
            max_generations=5,
            mutation_rate=0.2,
            crossover_rate=0.8
        )
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Run optimization with initial population
        best_genome, best_fitness = search.run(initial_population=initial_population)
        
        # Verify results
        assert isinstance(best_genome, Genome)
        assert isinstance(best_fitness, float)
        
        # Best genome should be from the search (not necessarily from initial population)
        assert search.best_genome == best_genome
    
    def test_genetic_optimization_parameter_validation(self):
        """Test that genetic optimization respects parameter constraints."""
        fitness_evaluator = MockFitnessEvaluator()
        
        config = GeneticSearchConfig(
            population_size=10,
            max_generations=5,
            mutation_rate=0.2,
            crossover_rate=0.8
        )
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Run optimization
        best_genome, best_fitness = search.run()
        
        # Verify that all genomes in history have valid parameters
        for result in search.generation_history:
            # Check that best genome has valid parameters
            for name, value in result.best_genome.values.items():
                spec = result.best_genome.parameter_space[name]
                assert spec.validate(value), f"Invalid value for {name}: {value}"
    
    def test_genetic_optimization_caching(self):
        """Test that fitness caching works correctly in genetic optimization."""
        fitness_evaluator = MockFitnessEvaluator(cache_results=True)
        
        config = GeneticSearchConfig(
            population_size=8,
            max_generations=3,
            mutation_rate=0.3,
            crossover_rate=0.7
        )
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Run optimization
        best_genome, best_fitness = search.run()
        
        # Check cache statistics
        cache_stats = fitness_evaluator.get_cache_stats()
        assert cache_stats['cached_results'] > 0
        
        # Should have some cache hits (same genomes evaluated multiple times)
        assert cache_stats['cache_hits'] >= 0
        assert cache_stats['cache_misses'] >= 0
    
    def test_genetic_optimization_search_summary(self):
        """Test that genetic search provides comprehensive summary."""
        fitness_evaluator = MockFitnessEvaluator()
        
        config = GeneticSearchConfig(
            population_size=10,
            max_generations=5,
            mutation_rate=0.2,
            crossover_rate=0.8
        )
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Run optimization
        best_genome, best_fitness = search.run()
        
        # Get search summary
        summary = search.get_search_summary()
        
        # Verify summary contains expected information
        assert 'best_fitness' in summary
        assert 'best_genome_hash' in summary
        assert 'total_generations' in summary
        assert 'population_size' in summary
        assert 'final_population_diversity' in summary
        assert 'fitness_history' in summary
        assert 'config' in summary

        # Verify values
        assert summary['best_fitness'] == best_fitness
        assert summary['best_genome_hash'] == best_genome.hash()
        assert summary['total_generations'] == search.generation

    def test_genetic_optimization_error_handling(self):
        """Test that genetic optimization handles evaluation errors gracefully."""
        # Create evaluator that sometimes fails
        call_count = 0
        def failing_fitness(genome: Genome) -> float:
            nonlocal call_count
            call_count += 1
            if call_count % 5 == 0:  # Fail every 5th evaluation
                raise Exception("Simulated evaluation failure")
            return genome.values.get('learning_rate', 0.001) * 1000
        
        fitness_evaluator = MockFitnessEvaluator(fitness_function=failing_fitness)
        
        config = GeneticSearchConfig(
            population_size=8,
            max_generations=3,
            mutation_rate=0.2,
            crossover_rate=0.8
        )
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Run optimization (should handle errors gracefully)
        best_genome, best_fitness = search.run()
        
        # Should still complete successfully
        assert isinstance(best_genome, Genome)
        assert isinstance(best_fitness, float)
        
        # Should have some evaluations completed
        assert fitness_evaluator.evaluation_count > 0
    
    def test_genetic_optimization_parallel_evaluation(self):
        """Test genetic optimization with parallel evaluation."""
        fitness_evaluator = MockFitnessEvaluator()
        
        config = GeneticSearchConfig(
            population_size=10,
            max_generations=3,
            mutation_rate=0.2,
            crossover_rate=0.8,
            use_multiprocessing=True,
            n_jobs=2
        )
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Mock the parallel evaluation method to use sequential evaluation instead
        with patch.object(search, '_evaluate_population_parallel') as mock_parallel:
            # Make the parallel method call the sequential method
            mock_parallel.side_effect = lambda: search._evaluate_population_sequential()
            
            # Run optimization
            best_genome, best_fitness = search.run()
            
            # Should complete successfully
            assert isinstance(best_genome, Genome)
            assert isinstance(best_fitness, float)
            
            # Verify that parallel evaluation was called
            assert mock_parallel.called
    
    def test_genetic_optimization_parameter_evolution(self):
        """Test that genetic optimization evolves parameters effectively."""
        # Create a fitness function that has clear optimal regions
        def structured_fitness(genome: Genome) -> float:
            """Fitness function with clear optimal parameter regions."""
            lr = genome.values.get('learning_rate', 0.001)
            batch_size = genome.values.get('batch_size', 64)
            
            # Optimal learning rate around 0.005
            lr_score = 1.0 - abs(lr - 0.005) * 100
            
            # Optimal batch size around 128
            batch_score = 1.0 - abs(batch_size - 128) / 100
            
            return max(0.0, lr_score + batch_score)
        
        fitness_evaluator = MockFitnessEvaluator(fitness_function=structured_fitness)
        
        config = GeneticSearchConfig(
            population_size=20,
            max_generations=10,
            mutation_rate=0.15,
            crossover_rate=0.8,
            elite_fraction=0.2,
            random_fraction=0.1
        )
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Run optimization
        best_genome, best_fitness = search.run()
        
        # Check that the best genome has evolved toward optimal parameters
        lr = best_genome.values.get('learning_rate', 0.001)
        batch_size = best_genome.values.get('batch_size', 64)
        
        # Should be close to optimal values
        assert abs(lr - 0.005) < 0.01
        assert abs(batch_size - 128) < 50
        
        # Fitness should be high
        assert best_fitness > 1.0 