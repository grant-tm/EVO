"""
Tests for genetic search algorithm.
"""

import pytest
import random
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from evo.optimization.genetic.genome import Genome, GenomeConfig
from evo.optimization.genetic.fitness import FitnessResult, FitnessEvaluator
from evo.optimization.genetic.genetic_search import (
    GeneticSearchConfig,
    GenerationResult,
    GeneticSearch
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.optimization,
    pytest.mark.genetic
]


class MockFitnessEvaluator(FitnessEvaluator):
    """Mock fitness evaluator for testing genetic search."""
    
    def __init__(self, fitness_scores=None, cache_results=True):
        super().__init__(cache_results)
        self.fitness_scores = fitness_scores or {}
        self.evaluation_count = 0
    
    def evaluate(self, genome: Genome) -> FitnessResult:
        """Mock evaluation that returns predefined fitness scores."""
        self.evaluation_count += 1
        
        # Get fitness score for this genome
        genome_hash = genome.hash()
        fitness_score = self.fitness_scores.get(genome_hash, 0.5)
        
        return FitnessResult(
            genome=genome,
            fitness_score=fitness_score,
            metrics={'sharpe_ratio': fitness_score, 'total_return': fitness_score * 0.1},
            metadata={'evaluation_count': self.evaluation_count}
        )


class TestGeneticSearchConfig:
    """Test GeneticSearchConfig dataclass."""
    
    def test_genetic_search_config_creation(self):
        """Test creating a GeneticSearchConfig instance."""
        config = GeneticSearchConfig()
        
        # Check default values
        assert config.population_size == 50
        assert config.max_generations == 100
        assert config.mutation_rate == 0.1
        assert config.crossover_rate == 0.8
        assert config.elite_fraction == 0.2
        assert config.random_fraction == 0.1
        assert config.tournament_size == 3
        assert config.selection_pressure == 1.5
        assert config.num_backtests == 10
        assert config.backtest_length == 1000
        assert config.fitness_metric == "sharpe_ratio"
        assert config.n_jobs == 1
        assert config.use_multiprocessing is False
        assert config.patience == 20
        assert config.min_improvement == 0.001
        assert config.log_interval == 5
        assert config.save_best_genome is True
        assert config.save_history is True
    
    def test_genetic_search_config_custom_values(self):
        """Test creating a GeneticSearchConfig with custom values."""
        config = GeneticSearchConfig(
            population_size=100,
            max_generations=50,
            mutation_rate=0.2,
            crossover_rate=0.9,
            elite_fraction=0.3,
            random_fraction=0.2,
            tournament_size=5,
            selection_pressure=2.0,
            num_backtests=5,
            backtest_length=500,
            fitness_metric="total_return",
            n_jobs=4,
            use_multiprocessing=True,
            patience=10,
            min_improvement=0.01,
            log_interval=10,
            save_best_genome=False,
            save_history=False
        )
        
        # Check custom values
        assert config.population_size == 100
        assert config.max_generations == 50
        assert config.mutation_rate == 0.2
        assert config.crossover_rate == 0.9
        assert config.elite_fraction == 0.3
        assert config.random_fraction == 0.2
        assert config.tournament_size == 5
        assert config.selection_pressure == 2.0
        assert config.num_backtests == 5
        assert config.backtest_length == 500
        assert config.fitness_metric == "total_return"
        assert config.n_jobs == 4
        assert config.use_multiprocessing is True
        assert config.patience == 10
        assert config.min_improvement == 0.01
        assert config.log_interval == 10
        assert config.save_best_genome is False
        assert config.save_history is False


class TestGenerationResult:
    """Test GenerationResult dataclass."""
    
    def test_generation_result_creation(self):
        """Test creating a GenerationResult instance."""
        genome = Genome.random(seed=1)
        
        result = GenerationResult(
            generation=5,
            best_fitness=0.85,
            avg_fitness=0.65,
            worst_fitness=0.45,
            best_genome=genome,
            population_diversity=0.75,
            metadata={'test': True}
        )
        
        assert result.generation == 5
        assert result.best_fitness == 0.85
        assert result.avg_fitness == 0.65
        assert result.worst_fitness == 0.45
        assert result.best_genome == genome
        assert result.population_diversity == 0.75
        assert result.metadata == {'test': True}


class TestGeneticSearch:
    """Test GeneticSearch class."""
    
    def test_genetic_search_creation(self):
        """Test creating a GeneticSearch instance."""
        fitness_evaluator = MockFitnessEvaluator()
        
        search = GeneticSearch(
            fitness_evaluator=fitness_evaluator,
            seed=42
        )
        
        assert search.fitness_evaluator == fitness_evaluator
        assert isinstance(search.config, GeneticSearchConfig)
        assert isinstance(search.genome_config, GenomeConfig)
        assert search.population == []
        assert search.generation_history == []
        assert search.best_genome is None
        assert search.best_fitness == float('-inf')
        assert search.generation == 0
    
    def test_genetic_search_creation_with_configs(self):
        """Test creating a GeneticSearch with custom configs."""
        fitness_evaluator = MockFitnessEvaluator()
        config = GeneticSearchConfig(population_size=100, max_generations=50)
        genome_config = GenomeConfig()
        
        search = GeneticSearch(
            fitness_evaluator=fitness_evaluator,
            config=config,
            genome_config=genome_config,
            seed=42
        )
        
        assert search.config == config
        assert search.genome_config == genome_config
        assert search.config.population_size == 100
        assert search.config.max_generations == 50
    
    def test_initialize_population(self):
        """Test population initialization."""
        fitness_evaluator = MockFitnessEvaluator()
        config = GeneticSearchConfig(population_size=10)
        
        search = GeneticSearch(
            fitness_evaluator=fitness_evaluator,
            config=config,
            seed=42
        )
        
        search.initialize_population()
        
        assert len(search.population) == 10
        assert all(isinstance(genome, Genome) for genome in search.population)
        
        # All genomes should be valid
        for genome in search.population:
            for name, value in genome.values.items():
                spec = genome.parameter_space[name]
                assert spec.validate(value)
    
    def test_initialize_population_with_seed(self):
        """Test that population initialization is reproducible with seed."""
        fitness_evaluator = MockFitnessEvaluator()
        config = GeneticSearchConfig(population_size=5)
        
        # Create two searches with same seed
        search1 = GeneticSearch(fitness_evaluator, config=config, seed=42)
        search2 = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        search1.initialize_population()
        search2.initialize_population()
        
        # Populations should be identical
        assert len(search1.population) == len(search2.population)
        for g1, g2 in zip(search1.population, search2.population):
            assert g1.values == g2.values
    
    def test_run_with_initial_population(self):
        """Test running genetic search with initial population."""
        fitness_evaluator = MockFitnessEvaluator()
        config = GeneticSearchConfig(population_size=5, max_generations=2)
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Create initial population with random genomes
        initial_population = [
            Genome.random(seed=1),
            Genome.random(seed=2),
            Genome.random(seed=3),
            Genome.random(seed=4),
            Genome.random(seed=5)
        ]
        
        # Run search
        best_genome, best_fitness = search.run(initial_population=initial_population)
        
        assert isinstance(best_genome, Genome)
        assert isinstance(best_fitness, float)
        assert best_fitness > float('-inf')
    
    def test_run_with_random_population(self):
        """Test running genetic search with randomly initialized population."""
        fitness_evaluator = MockFitnessEvaluator()
        config = GeneticSearchConfig(population_size=5, max_generations=2)
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Run search
        best_genome, best_fitness = search.run()
        
        assert isinstance(best_genome, Genome)
        assert isinstance(best_fitness, float)
        assert best_fitness > float('-inf')
        assert len(search.population) == 5
    
    def test_evaluate_population_sequential(self):
        """Test sequential population evaluation."""
        fitness_evaluator = MockFitnessEvaluator()
        config = GeneticSearchConfig(population_size=3, use_multiprocessing=False)
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        search.initialize_population()
        
        # Evaluate population
        results = search._evaluate_population_sequential()
        
        assert len(results) == 3
        assert all(isinstance(result, FitnessResult) for result in results)
        assert all(result.genome in search.population for result in results)
    
    @patch('evo.optimization.genetic.genetic_search.as_completed')
    @patch('evo.optimization.genetic.genetic_search.ProcessPoolExecutor')
    def test_evaluate_population_parallel(self, mock_executor, mock_as_completed):
        """Test parallel population evaluation."""
        fitness_evaluator = MockFitnessEvaluator()
        config = GeneticSearchConfig(population_size=3, use_multiprocessing=True, n_jobs=2)
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        search.initialize_population()
        
        # Mock parallel execution
        mock_context = Mock()
        mock_executor.return_value.__enter__.return_value = mock_context
        
        # Mock results
        mock_future1 = Mock()
        mock_future1.result.return_value = FitnessResult(
            genome=search.population[0],
            fitness_score=0.8,
            metrics={},
            metadata={}
        )
        mock_future2 = Mock()
        mock_future2.result.return_value = FitnessResult(
            genome=search.population[1],
            fitness_score=0.7,
            metrics={},
            metadata={}
        )
        mock_future3 = Mock()
        mock_future3.result.return_value = FitnessResult(
            genome=search.population[2],
            fitness_score=0.9,
            metrics={},
            metadata={}
        )
        
        # Mock the submit method to return futures
        mock_context.submit.side_effect = [mock_future1, mock_future2, mock_future3]
        
        # Mock as_completed to return futures in order
        mock_as_completed.return_value = [mock_future1, mock_future2, mock_future3]
        
        # Evaluate population
        results = search._evaluate_population_parallel()
        
        assert len(results) == 3
        assert all(isinstance(result, FitnessResult) for result in results)
    
    def test_update_best_solution(self):
        """Test updating best solution."""
        fitness_evaluator = MockFitnessEvaluator()
        search = GeneticSearch(fitness_evaluator, seed=42)
        
        # Create test genomes and results
        genome1 = Genome.random(seed=1)
        genome2 = Genome.random(seed=2)
        
        result1 = FitnessResult(genome1, 0.7, {}, {})
        result2 = FitnessResult(genome2, 0.9, {}, {})
        
        # Update with first result
        search._update_best_solution([result1])
        assert search.best_genome == genome1
        assert search.best_fitness == 0.7
        
        # Update with better result
        search._update_best_solution([result2])
        assert search.best_genome == genome2
        assert search.best_fitness == 0.9
        
        # Update with worse result (should not change)
        result3 = FitnessResult(genome1, 0.5, {}, {})
        search._update_best_solution([result3])
        assert search.best_genome == genome2
        assert search.best_fitness == 0.9
    
    def test_create_generation_result(self):
        """Test creating generation result."""
        fitness_evaluator = MockFitnessEvaluator()
        search = GeneticSearch(fitness_evaluator, seed=42)
        
        # Create test results
        genome1 = Genome.random(seed=1)
        genome2 = Genome.random(seed=2)
        genome3 = Genome.random(seed=3)
        
        # Set the population for diversity calculation
        search.population = [genome1, genome2, genome3]
        
        results = [
            FitnessResult(genome1, 0.9, {}, {}),
            FitnessResult(genome2, 0.7, {}, {}),
            FitnessResult(genome3, 0.5, {}, {})
        ]
        
        # Create generation result
        generation_result = search._create_generation_result(results)
        
        assert isinstance(generation_result, GenerationResult)
        assert generation_result.generation == 0
        assert generation_result.best_fitness == 0.9
        assert np.isclose(generation_result.avg_fitness, 0.7)
        assert generation_result.worst_fitness == 0.5
        assert generation_result.best_genome == genome1
        assert generation_result.population_diversity > 0
    
    def test_calculate_population_diversity(self):
        """Test population diversity calculation."""
        fitness_evaluator = MockFitnessEvaluator()
        search = GeneticSearch(fitness_evaluator, seed=42)
        
        # Create diverse population
        search.population = [
            Genome.random(seed=1),
            Genome.random(seed=2),
            Genome.random(seed=3)
        ]
        
        diversity = search._calculate_population_diversity()
        
        assert isinstance(diversity, float)
        assert diversity > 0
    
    def test_should_stop_early_no_improvement(self):
        """Test early stopping when no improvement."""
        fitness_evaluator = MockFitnessEvaluator()
        config = GeneticSearchConfig(patience=3, min_improvement=0.01)
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Simulate no improvement for several generations
        search.best_fitness_history = [0.8, 0.8, 0.8, 0.8, 0.8]
        search.generations_without_improvement = 4
        
        should_stop = search._should_stop_early()
        assert should_stop is True
    
    def test_should_stop_early_with_improvement(self):
        """Test early stopping when there is improvement."""
        fitness_evaluator = MockFitnessEvaluator()
        config = GeneticSearchConfig(patience=3, min_improvement=0.01)
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Simulate recent improvement
        search.best_fitness_history = [0.8, 0.8, 0.8, 0.85, 0.9]
        search.generations_without_improvement = 0
        
        should_stop = search._should_stop_early()
        assert should_stop is False
    
    def test_create_next_generation(self):
        """Test creating next generation."""
        fitness_evaluator = MockFitnessEvaluator()
        config = GeneticSearchConfig(
            population_size=6,
            elite_fraction=0.33,  # 2 elite
            random_fraction=0.17,  # 1 random
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Create test population and results using Genome.random() to ensure all parameters are present
        genomes = [
            Genome.random(seed=1),
            Genome.random(seed=2),
            Genome.random(seed=3),
            Genome.random(seed=4),
            Genome.random(seed=5),
            Genome.random(seed=6)
        ]
        
        results = [
            FitnessResult(genomes[0], 0.9, {}, {}),  # Best
            FitnessResult(genomes[1], 0.8, {}, {}),  # Second best
            FitnessResult(genomes[2], 0.7, {}, {}),
            FitnessResult(genomes[3], 0.6, {}, {}),
            FitnessResult(genomes[4], 0.5, {}, {}),
            FitnessResult(genomes[5], 0.4, {}, {})   # Worst
        ]
        
        search.population = genomes
        
        # Create next generation
        next_generation = search._create_next_generation(results)
        
        assert len(next_generation) == 6
        assert all(isinstance(genome, Genome) for genome in next_generation)
    
    def test_select_parent(self):
        """Test parent selection."""
        fitness_evaluator = MockFitnessEvaluator()
        config = GeneticSearchConfig(tournament_size=3)
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        
        # Create test results
        genomes = [
            Genome.random(seed=1),
            Genome.random(seed=2),
            Genome.random(seed=3),
            Genome.random(seed=4),
            Genome.random(seed=5)
        ]
        
        results = [
            FitnessResult(genomes[0], 0.9, {}, {}),
            FitnessResult(genomes[1], 0.8, {}, {}),
            FitnessResult(genomes[2], 0.7, {}, {}),
            FitnessResult(genomes[3], 0.6, {}, {}),
            FitnessResult(genomes[4], 0.5, {}, {})
        ]
        
        # Select parent
        parent = search._select_parent(results)
        
        assert parent in genomes
    
    def test_finalize_search(self):
        """Test search finalization."""
        fitness_evaluator = MockFitnessEvaluator()
        search = GeneticSearch(fitness_evaluator, seed=42)
        
        # Set up some state
        search.best_genome = Genome.random(seed=1)
        search.best_fitness = 0.9
        search.generation_history = [Mock(), Mock()]
        
        # Finalize search
        search._finalize_search()
        
        # Should not raise any exceptions
        assert search.best_genome is not None
        assert search.best_fitness == 0.9
    
    @patch('evo.optimization.genetic.genetic_search.json.dump')
    @patch('evo.optimization.genetic.genetic_search.open')
    def test_save_best_genome(self, mock_open, mock_json_dump):
        """Test saving best genome."""
        fitness_evaluator = MockFitnessEvaluator()
        config = GeneticSearchConfig(save_best_genome=True)
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        search.best_genome = Genome.random(seed=1)
        
        # Save best genome
        search._save_best_genome()
        
        # Should attempt to save
        mock_open.assert_called()
        mock_json_dump.assert_called()
    
    @patch('evo.optimization.genetic.genetic_search.json.dump')
    @patch('evo.optimization.genetic.genetic_search.open')
    def test_save_search_history(self, mock_open, mock_json_dump):
        """Test saving search history."""
        fitness_evaluator = MockFitnessEvaluator()
        config = GeneticSearchConfig(save_history=True)
        
        search = GeneticSearch(fitness_evaluator, config=config, seed=42)
        search.generation_history = [Mock(), Mock()]
        
        # Save search history
        search._save_search_history()
        
        # Should attempt to save
        mock_open.assert_called()
        mock_json_dump.assert_called()
    
    def test_get_search_summary(self):
        """Test getting search summary."""
        fitness_evaluator = MockFitnessEvaluator()
        search = GeneticSearch(fitness_evaluator, seed=42)
        
        # Set up some state
        search.best_genome = Genome.random(seed=1)
        search.best_fitness = 0.9
        search.generation = 5
        search.generation_history = [Mock(), Mock(), Mock()]
        
        # Get summary
        summary = search.get_search_summary()
        
        assert isinstance(summary, dict)
        assert 'best_fitness' in summary
        assert 'best_genome_hash' in summary
        assert 'total_generations' in summary
        assert 'population_size' in summary
        assert 'final_population_diversity' in summary
        assert 'fitness_history' in summary
        assert 'config' in summary
        
        assert summary['best_fitness'] == 0.9
        assert summary['total_generations'] == 5 