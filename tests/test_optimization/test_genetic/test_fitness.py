"""
Tests for fitness evaluation in genetic optimization.
"""

import numpy as np
import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from evo.optimization.genetic.genome import Genome, GenomeConfig
from evo.optimization.genetic.fitness import (
    FitnessResult,
    FitnessEvaluator,
    BacktestFitnessEvaluator,
    MultiObjectiveFitnessEvaluator,
    RobustFitnessEvaluator
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.optimization,
    pytest.mark.genetic
]


class MockFitnessEvaluator(FitnessEvaluator):
    """Mock fitness evaluator for testing."""
    
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
        
        result = FitnessResult(
            genome=genome,
            fitness_score=fitness_score,
            metrics={'sharpe_ratio': fitness_score, 'total_return': fitness_score * 0.1},
            metadata={'evaluation_count': self.evaluation_count}
        )

        if self.cache_results:
            self._cache_result(genome, result)

        return result


class TestFitnessResult:
    """Test FitnessResult dataclass."""
    
    def test_fitness_result_creation(self):
        """Test creating a FitnessResult instance."""
        genome = Genome.random(seed=1)
        
        result = FitnessResult(
            genome=genome,
            fitness_score=0.75,
            metrics={'sharpe_ratio': 1.2, 'total_return': 0.15},
            metadata={'test': True}
        )
        
        assert result.genome == genome
        assert result.fitness_score == 0.75
        assert result.metrics == {'sharpe_ratio': 1.2, 'total_return': 0.15}
        assert result.metadata == {'test': True}
    
    def test_fitness_result_validation(self):
        """Test FitnessResult validation."""
        genome = Genome.random(seed=1)
        
        # Valid result
        result = FitnessResult(
            genome=genome,
            fitness_score=0.75,
            metrics={'sharpe_ratio': 1.2},
            metadata={}
        )
        
        # Should not raise any exceptions
        assert result.fitness_score == 0.75
    
    def test_fitness_result_invalid_fitness_score(self):
        """Test FitnessResult with invalid fitness score."""
        genome = Genome.random(seed=1)
        
        with pytest.raises(ValueError, match="Fitness score must be numeric"):
            FitnessResult(
                genome=genome,
                fitness_score="invalid",
                metrics={'sharpe_ratio': 1.2},
                metadata={}
            )
    
    def test_fitness_result_invalid_metrics(self):
        """Test FitnessResult with invalid metrics."""
        genome = Genome.random(seed=1)
        
        with pytest.raises(ValueError, match="Metrics must be a dictionary"):
            FitnessResult(
                genome=genome,
                fitness_score=0.75,
                metrics="invalid",
                metadata={}
            )


class TestFitnessEvaluator:
    """Test FitnessEvaluator abstract base class."""
    
    def test_fitness_evaluator_abstract(self):
        """Test that FitnessEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FitnessEvaluator()
    
    def test_concrete_evaluator_creation(self):
        """Test creating a concrete evaluator implementation."""
        evaluator = MockFitnessEvaluator()
        
        assert evaluator.cache_results is True
        assert evaluator._cache == {}
    
    def test_evaluate_batch(self):
        """Test batch evaluation of genomes."""
        evaluator = MockFitnessEvaluator()
        
        genomes = [
            Genome.random(seed=1),
            Genome.random(seed=2),
            Genome.random(seed=3)
        ]
        
        results = evaluator.evaluate_batch(genomes)
        
        assert len(results) == 3
        assert all(isinstance(result, FitnessResult) for result in results)
        assert all(result.genome in genomes for result in results)
    
    def test_cache_functionality(self):
        """Test caching functionality."""
        evaluator = MockFitnessEvaluator(cache_results=True)
        genome = Genome({'learning_rate': 0.001, 'batch_size': 64})
        
        # First evaluation
        result1 = evaluator.evaluate(genome)
        
        # Second evaluation should use cache
        result2 = evaluator.evaluate(genome)
        
        # Results should be the same
        assert result1.fitness_score == result2.fitness_score
        assert result1.genome == result2.genome
    
    def test_cache_disabled(self):
        """Test evaluator with caching disabled."""
        evaluator = MockFitnessEvaluator(cache_results=False)
        genome = Genome({'learning_rate': 0.001, 'batch_size': 64})
        
        # Both evaluations should run
        result1 = evaluator.evaluate(genome)
        result2 = evaluator.evaluate(genome)
        
        # Should have evaluated twice
        assert evaluator.evaluation_count == 2
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        evaluator = MockFitnessEvaluator(cache_results=True)
        genome = Genome({'learning_rate': 0.001, 'batch_size': 64})
        
        # Evaluate to populate cache
        evaluator.evaluate(genome)
        assert len(evaluator._cache) == 1
        
        # Clear cache
        evaluator.clear_cache()
        assert len(evaluator._cache) == 0
    
    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        evaluator = MockFitnessEvaluator(cache_results=True)
        
        stats = evaluator.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert 'cached_results' in stats
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats


class TestBacktestFitnessEvaluator:
    """Test BacktestFitnessEvaluator class."""
    
    def test_backtest_evaluator_creation(self):
        """Test creating a BacktestFitnessEvaluator instance."""
        mock_backtest_engine = Mock()
        mock_model_trainer = Mock()
        
        evaluator = BacktestFitnessEvaluator(
            backtest_engine=mock_backtest_engine,
            model_trainer=mock_model_trainer,
            model_dir="/tmp/models",
            num_backtests=5,
            backtest_length=500,
            fitness_metric="sharpe_ratio"
        )
        
        assert evaluator.backtest_engine == mock_backtest_engine
        assert evaluator.model_trainer == mock_model_trainer
        assert evaluator.model_dir == "/tmp/models"
        assert evaluator.num_backtests == 5
        assert evaluator.backtest_length == 500
        assert evaluator.fitness_metric == "sharpe_ratio"
    
    def test_backtest_evaluator_creation_creates_model_dir(self, temp_dir):
        """Test that model directory is created if it doesn't exist."""
        mock_backtest_engine = Mock()
        mock_model_trainer = Mock()
        
        model_dir = temp_dir / "models"
        
        evaluator = BacktestFitnessEvaluator(
            backtest_engine=mock_backtest_engine,
            model_trainer=mock_model_trainer,
            model_dir=str(model_dir)
        )
        
        assert model_dir.exists()
    
    @patch('evo.optimization.genetic.fitness.os.makedirs')
    def test_backtest_evaluator_creation_existing_dir(self, mock_makedirs):
        """Test evaluator creation with existing model directory."""
        mock_backtest_engine = Mock()
        mock_model_trainer = Mock()
        
        BacktestFitnessEvaluator(
            backtest_engine=mock_backtest_engine,
            model_trainer=mock_model_trainer,
            model_dir="/tmp/existing_models"
        )
        
        mock_makedirs.assert_called_once_with("/tmp/existing_models", exist_ok=True)
    
    def test_evaluate_with_cached_result(self):
        """Test evaluation with cached result."""
        mock_backtest_engine = Mock()
        mock_model_trainer = Mock()
        
        evaluator = BacktestFitnessEvaluator(
            backtest_engine=mock_backtest_engine,
            model_trainer=mock_model_trainer,
            model_dir="/tmp/models"
        )
        
        genome = Genome({'learning_rate': 0.001, 'batch_size': 64})
        
        # Create a cached result
        cached_result = FitnessResult(
            genome=genome,
            fitness_score=0.75,
            metrics={'sharpe_ratio': 1.2},
            metadata={}
        )
        evaluator._cache[evaluator._get_cache_key(genome)] = cached_result
        
        # Evaluate should return cached result
        result = evaluator.evaluate(genome)
        
        assert result == cached_result
        # Model trainer should not be called
        mock_model_trainer.assert_not_called()
    
    @patch('evo.optimization.genetic.fitness.BacktestFitnessEvaluator._get_or_train_model')
    @patch('evo.optimization.genetic.fitness.BacktestFitnessEvaluator._run_backtests')
    @patch('evo.optimization.genetic.fitness.BacktestFitnessEvaluator._calculate_fitness_score')
    @patch('evo.optimization.genetic.fitness.BacktestFitnessEvaluator._aggregate_metrics')
    def test_evaluate_successful(self, mock_aggregate, mock_calculate, mock_backtests, mock_train):
        """Test successful evaluation."""
        mock_backtest_engine = Mock()
        mock_model_trainer = Mock()
        
        evaluator = BacktestFitnessEvaluator(
            backtest_engine=mock_backtest_engine,
            model_trainer=mock_model_trainer,
            model_dir="/tmp/models"
        )
        
        genome = Genome({'learning_rate': 0.001, 'batch_size': 64})
        
        # Mock the internal methods
        mock_train.return_value = "/tmp/models/model.zip"
        mock_backtests.return_value = [{'sharpe_ratio': 1.2}, {'sharpe_ratio': 1.3}]
        mock_calculate.return_value = 1.25
        mock_aggregate.return_value = {'sharpe_ratio': 1.25, 'total_return': 0.15}
        
        # Evaluate
        result = evaluator.evaluate(genome)
        
        assert isinstance(result, FitnessResult)
        assert result.genome == genome
        assert result.fitness_score == 1.25
        assert result.metrics == {'sharpe_ratio': 1.25, 'total_return': 0.15}
        
        # Verify methods were called
        mock_train.assert_called_once_with(genome)
        mock_backtests.assert_called_once_with("/tmp/models/model.zip")
        mock_calculate.assert_called_once()
        mock_aggregate.assert_called_once()
    
    def test_evaluate_failure_handling(self):
        """Test evaluation failure handling."""
        mock_backtest_engine = Mock()
        mock_model_trainer = Mock()
        
        evaluator = BacktestFitnessEvaluator(
            backtest_engine=mock_backtest_engine,
            model_trainer=mock_model_trainer,
            model_dir="/tmp/models"
        )
        
        genome = Genome({'learning_rate': 0.001, 'batch_size': 64})
        
        # Mock model trainer to raise exception
        mock_model_trainer.side_effect = Exception("Training failed")
        
        # Evaluate should handle exception gracefully
        result = evaluator.evaluate(genome)
        
        assert isinstance(result, FitnessResult)
        assert result.genome == genome
        assert result.fitness_score == -float("inf")  # Default failure penalty
        assert 'error' in result.metadata
    
    def test_get_or_train_model(self):
        """Test model training and caching."""
        mock_backtest_engine = Mock()
        mock_model_trainer = Mock()
        
        evaluator = BacktestFitnessEvaluator(
            backtest_engine=mock_backtest_engine,
            model_trainer=mock_model_trainer,
            model_dir="/tmp/models"
        )
        
        genome = Genome({'learning_rate': 0.001, 'batch_size': 64})
        genome_hash = genome.hash()
        model_path = os.path.normpath(f"/tmp/models/{genome_hash}.zip")

        # Ensure the model file does not exist before the test
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Mock model trainer
        mock_model_trainer.return_value = model_path
        
        # First call should train model
        model_path1 = evaluator._get_or_train_model(genome)
        assert os.path.normpath(model_path1) == model_path
        mock_model_trainer.assert_called_once_with(genome)

        # Simulate the model file being created
        os.makedirs(os.path.dirname(model_path1), exist_ok=True)
        with open(model_path1, "w") as f:
            f.write("")

        # Second call should use cached path
        model_path2 = evaluator._get_or_train_model(genome)
        assert os.path.normpath(model_path2) == os.path.normpath(model_path1)
        # Should not call trainer again
        assert mock_model_trainer.call_count == 1

        # Cleanup: remove the temp model file
        if os.path.exists(model_path1):
            os.remove(model_path1)
    
    def test_run_backtests(self):
        """Test running backtests."""
        mock_backtest_engine = Mock()
        mock_model_trainer = Mock()
        
        evaluator = BacktestFitnessEvaluator(
            backtest_engine=mock_backtest_engine,
            model_trainer=mock_model_trainer,
            model_dir="/tmp/models"
        )
        
        model_path = "/tmp/models/model.zip"
        
        # Mock backtest result
        mock_result = Mock()
        mock_result.metrics = {'sharpe_ratio': 1.2, 'total_return': 0.15}
        mock_backtest_engine.run_backtest.return_value = mock_result

        # Run backtests
        results = evaluator._run_backtests(model_path)

        # Verify backtest engine was called the correct number of times
        assert mock_backtest_engine.run_backtest.call_count == evaluator.num_backtests

        # Verify results format
        assert len(results) == evaluator.num_backtests
        assert all(isinstance(result, dict) for result in results)
        assert all('sharpe_ratio' in result for result in results)
        assert all('total_return' in result for result in results)
    
    def test_calculate_fitness_score(self):
        """Test fitness score calculation."""
        mock_backtest_engine = Mock()
        mock_model_trainer = Mock()
        
        evaluator = BacktestFitnessEvaluator(
            backtest_engine=mock_backtest_engine,
            model_trainer=mock_model_trainer,
            model_dir="/tmp/models",
            fitness_metric="sharpe_ratio"
        )
        
        # Mock backtest results
        results = [
            {'sharpe_ratio': 1.2, 'total_return': 0.15},
            {'sharpe_ratio': 1.3, 'total_return': 0.18},
            {'sharpe_ratio': 1.1, 'total_return': 0.12}
        ]
        
        # Calculate fitness score
        fitness_score = evaluator._calculate_fitness_score(results)
        
        # Should be the mean of sharpe ratios
        expected_score = (1.2 + 1.3 + 1.1) / 3
        assert np.isclose(fitness_score, expected_score, rtol=1e-3)
    
    def test_aggregate_metrics(self):
        """Test metrics aggregation."""
        mock_backtest_engine = Mock()
        mock_model_trainer = Mock()
        
        evaluator = BacktestFitnessEvaluator(
            backtest_engine=mock_backtest_engine,
            model_trainer=mock_model_trainer,
            model_dir="/tmp/models"
        )
        
        # Mock backtest results
        results = [
            {'sharpe_ratio': 1.2, 'total_return': 0.15},
            {'sharpe_ratio': 1.3, 'total_return': 0.18},
            {'sharpe_ratio': 1.1, 'total_return': 0.12}
        ]
        
        # Aggregate metrics
        aggregated = evaluator._aggregate_metrics(results)
        
        # Should contain mean values
        assert 'sharpe_ratio_median' in aggregated
        assert 'total_return_median' in aggregated
        assert aggregated['sharpe_ratio_median'] == (1.2 + 1.3 + 1.1) / 3
        assert aggregated['total_return_median'] == (0.15 + 0.18 + 0.12) / 3


class TestMultiObjectiveFitnessEvaluator:
    """Test MultiObjectiveFitnessEvaluator class."""
    
    def test_multi_objective_evaluator_creation(self):
        """Test creating a MultiObjectiveFitnessEvaluator instance."""
        evaluator1 = MockFitnessEvaluator()
        evaluator2 = MockFitnessEvaluator()
        
        multi_evaluator = MultiObjectiveFitnessEvaluator(
            evaluators=[evaluator1, evaluator2],
            weights=[0.6, 0.4]
        )
        
        assert multi_evaluator.evaluators == [evaluator1, evaluator2]
        assert multi_evaluator.weights == [0.6, 0.4]
    
    def test_multi_objective_evaluator_default_weights(self):
        """Test MultiObjectiveFitnessEvaluator with default weights."""
        evaluator1 = MockFitnessEvaluator()
        evaluator2 = MockFitnessEvaluator()
        
        multi_evaluator = MultiObjectiveFitnessEvaluator(
            evaluators=[evaluator1, evaluator2]
        )
        
        # Should have equal weights
        assert multi_evaluator.weights == [0.5, 0.5]
    
    def test_multi_objective_evaluate(self):
        """Test multi-objective evaluation."""
        # Create evaluators with different fitness scores
        evaluator1 = MockFitnessEvaluator({'genome1': 0.8, 'genome2': 0.6})
        evaluator2 = MockFitnessEvaluator({'genome1': 0.9, 'genome2': 0.7})
        
        multi_evaluator = MultiObjectiveFitnessEvaluator(
            evaluators=[evaluator1, evaluator2],
            weights=[0.6, 0.4]
        )
        
        genome = Genome({'learning_rate': 0.001, 'batch_size': 64})
        
        # Evaluate
        result = multi_evaluator.evaluate(genome)
        
        assert isinstance(result, FitnessResult)
        assert result.genome == genome
        
        # Fitness score should be weighted combination
        expected_score = 0.6 * 0.5 + 0.4 * 0.5  # Default scores from MockFitnessEvaluator
        assert result.fitness_score == expected_score
        
        # Should contain metrics from both evaluators
        assert 'obj_0_sharpe_ratio' in result.metrics
        assert 'obj_1_sharpe_ratio' in result.metrics
    
    def test_multi_objective_evaluate_with_errors(self):
        """Test multi-objective evaluation with evaluation errors."""
        # Create evaluators where one fails
        evaluator1 = MockFitnessEvaluator()
        evaluator2 = MockFitnessEvaluator()
        
        # Make evaluator2 raise an exception
        def failing_evaluate(genome):
            raise Exception("Evaluation failed")
        
        evaluator2.evaluate = failing_evaluate
        
        multi_evaluator = MultiObjectiveFitnessEvaluator(
            evaluators=[evaluator1, evaluator2],
            weights=[0.6, 0.4]
        )
        
        genome = Genome({'learning_rate': 0.001, 'batch_size': 64})
        
        # Evaluate should handle the error
        result = multi_evaluator.evaluate(genome)
        
        assert isinstance(result, FitnessResult)
        assert result.genome == genome
        # Should have poor fitness due to failed evaluation
        assert result.fitness_score < 0


class TestRobustFitnessEvaluator:
    """Test RobustFitnessEvaluator class."""
    
    def test_robust_evaluator_creation(self):
        """Test creating a RobustFitnessEvaluator instance."""
        base_evaluator = MockFitnessEvaluator()
        
        robust_evaluator = RobustFitnessEvaluator(
            base_evaluator=base_evaluator,
            max_retries=3,
            failure_penalty=-1.0
        )
        
        assert robust_evaluator.base_evaluator == base_evaluator
        assert robust_evaluator.max_retries == 3
        assert robust_evaluator.failure_penalty == -1.0
    
    def test_robust_evaluate_success(self):
        """Test robust evaluation with successful evaluation."""
        base_evaluator = MockFitnessEvaluator()
        
        robust_evaluator = RobustFitnessEvaluator(
            base_evaluator=base_evaluator,
            max_retries=3
        )
        
        genome = Genome({'learning_rate': 0.001, 'batch_size': 64})
        
        # Evaluate
        result = robust_evaluator.evaluate(genome)
        
        assert isinstance(result, FitnessResult)
        assert result.genome == genome
        assert result.fitness_score == 0.5  # Default from MockFitnessEvaluator
        
        # Should have succeeded on first try
        assert result.metadata.get('retries', 0) == 0
    
    def test_robust_evaluate_with_retries(self):
        """Test robust evaluation with retries."""
        base_evaluator = MockFitnessEvaluator()
        
        # Make evaluator fail twice, then succeed
        call_count = 0
        def failing_evaluate(genome):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception(f"Evaluation failed {call_count}")
            return FitnessResult(
                genome=genome,
                fitness_score=0.8,
                metrics={'sharpe_ratio': 1.2},
                metadata={}
            )
        
        base_evaluator.evaluate = failing_evaluate
        
        robust_evaluator = RobustFitnessEvaluator(
            base_evaluator=base_evaluator,
            max_retries=3
        )
        
        genome = Genome({'learning_rate': 0.001, 'batch_size': 64})
        
        # Evaluate
        result = robust_evaluator.evaluate(genome)
        
        assert isinstance(result, FitnessResult)
        assert result.genome == genome
        assert result.fitness_score == 0.8
        
        # Should have retried twice
        assert result.metadata.get('retries', 0) == 2
    
    def test_robust_evaluate_max_retries_exceeded(self):
        """Test robust evaluation when max retries are exceeded."""
        base_evaluator = MockFitnessEvaluator()
        
        # Make evaluator always fail
        def always_failing_evaluate(genome):
            raise Exception("Always fails")
        
        base_evaluator.evaluate = always_failing_evaluate
        
        robust_evaluator = RobustFitnessEvaluator(
            base_evaluator=base_evaluator,
            max_retries=2,
            failure_penalty=-2.0
        )
        
        genome = Genome({'learning_rate': 0.001, 'batch_size': 64})
        
        # Evaluate
        result = robust_evaluator.evaluate(genome)
        
        assert isinstance(result, FitnessResult)
        assert result.genome == genome
        assert result.fitness_score == -2.0  # Failure penalty
        
        # Should have retried max times
        assert result.metadata.get('retries', 0) == 1
        assert 'error' in result.metadata 