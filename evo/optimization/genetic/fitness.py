"""
Fitness evaluation for genetic algorithm optimization.

This module provides interfaces and implementations for evaluating the fitness
of genomes in the genetic optimization process.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np

from ..genetic.genome import Genome
from ..backtesting.engine import BacktestEngine
from ..backtesting.metrics import PerformanceMetrics
from evo.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FitnessResult:
    """Result of a fitness evaluation."""
    
    genome: Genome
    fitness_score: float
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate fitness result."""
        if not isinstance(self.fitness_score, (int, float)):
            raise ValueError("Fitness score must be numeric")
        if not isinstance(self.metrics, dict):
            raise ValueError("Metrics must be a dictionary")


class FitnessEvaluator(ABC):
    """Abstract base class for fitness evaluation."""
    
    def __init__(self, cache_results: bool = True):
        """
        Initialize fitness evaluator.
        
        Args:
            cache_results: Whether to cache evaluation results
        """
        self.cache_results = cache_results
        self._cache: Dict[str, FitnessResult] = {}
    
    @abstractmethod
    def evaluate(self, genome: Genome) -> FitnessResult:
        """
        Evaluate the fitness of a genome.
        
        Args:
            genome: Genome to evaluate
            
        Returns:
            Fitness evaluation result
        """
        pass
    
    def evaluate_batch(self, genomes: List[Genome]) -> List[FitnessResult]:
        """
        Evaluate multiple genomes.
        
        Args:
            genomes: List of genomes to evaluate
            
        Returns:
            List of fitness results
        """
        results = []
        for genome in genomes:
            results.append(self.evaluate(genome))
        return results
    
    def _get_cache_key(self, genome: Genome) -> str:
        """Generate cache key for genome."""
        return genome.hash()
    
    def _get_cached_result(self, genome: Genome) -> Optional[FitnessResult]:
        """Get cached result if available."""
        if not self.cache_results:
            return None
        
        cache_key = self._get_cache_key(genome)
        return self._cache.get(cache_key)
    
    def _cache_result(self, genome: Genome, result: FitnessResult) -> None:
        """Cache evaluation result."""
        if not self.cache_results:
            return
        
        cache_key = self._get_cache_key(genome)
        self._cache[cache_key] = result
    
    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cached_results": len(self._cache),
            "cache_hits": getattr(self, '_cache_hits', 0),
            "cache_misses": getattr(self, '_cache_misses', 0)
        }


class BacktestFitnessEvaluator(FitnessEvaluator):
    """
    Fitness evaluator that uses backtesting to assess genome performance.
    """
    
    def __init__(
        self,
        backtest_engine: BacktestEngine,
        model_trainer: Callable[[Genome], str],
        model_dir: str,
        num_backtests: int = 10,
        backtest_length: int = 1000,
        fitness_metric: str = "sharpe_ratio",
        cache_results: bool = True
    ):
        """
        Initialize backtest fitness evaluator.
        
        Args:
            backtest_engine: Backtesting engine for evaluation
            model_trainer: Function to train model from genome
            model_dir: Directory to store trained models
            num_backtests: Number of backtests to run per evaluation
            backtest_length: Length of each backtest
            fitness_metric: Metric to use as fitness score
            cache_results: Whether to cache results
        """
        super().__init__(cache_results)
        self.backtest_engine = backtest_engine
        self.model_trainer = model_trainer
        self.model_dir = model_dir
        self.num_backtests = num_backtests
        self.backtest_length = backtest_length
        self.fitness_metric = fitness_metric
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
    
    def evaluate(self, genome: Genome) -> FitnessResult:
        """
        Evaluate genome fitness using backtesting.
        
        Args:
            genome: Genome to evaluate
            
        Returns:
            Fitness evaluation result
        """
        # Check cache first
        cached_result = self._get_cached_result(genome)
        if cached_result is not None:
            return cached_result
        
        logger.info(f"Evaluating genome {genome.hash()}")
        
        try:
            # Train model if not already trained
            model_path = self._get_or_train_model(genome)
            
            # Run backtests
            results = self._run_backtests(model_path)
            
            # Calculate fitness score
            fitness_score = self._calculate_fitness_score(results)
            
            # Create result
            result = FitnessResult(
                genome=genome,
                fitness_score=fitness_score,
                metrics=self._aggregate_metrics(results),
                metadata={
                    "model_path": model_path,
                    "num_backtests": self.num_backtests,
                    "backtest_length": self.backtest_length
                }
            )
            
            # Cache result
            if self.cache_results:
                self._cache_result(genome, result)
            
            logger.info(f"Genome {genome.hash()} fitness: {fitness_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating genome {genome.hash()}: {e}")
            # Return poor fitness for failed evaluations
            return FitnessResult(
                genome=genome,
                fitness_score=float('-inf'),
                metrics={},
                metadata={"error": str(e)}
            )
    
    def _get_or_train_model(self, genome: Genome) -> str:
        """Get existing model path or train new model."""
        genome_id = genome.hash()
        model_path = os.path.join(self.model_dir, f"{genome_id}.zip")
        
        if os.path.exists(model_path):
            logger.debug(f"Using existing model for genome {genome_id}")
            return model_path
        
        logger.info(f"Training new model for genome {genome_id}")
        return self.model_trainer(genome)
    
    def _run_backtests(self, model_path: str) -> List[Dict[str, float]]:
        """Run multiple backtests and return results."""
        results = []
        
        for i in range(self.num_backtests):
            try:
                # Run single backtest
                backtest_result = self.backtest_engine.run_backtest(
                    model_path=model_path,
                    length=self.backtest_length
                )
                results.append(backtest_result.metrics)
                
            except Exception as e:
                logger.warning(f"Backtest {i} failed: {e}")
                # Add poor metrics for failed backtest
                results.append({
                    "sharpe_ratio": -1.0,
                    "total_return": -0.1,
                    "max_drawdown": -0.5,
                    "win_rate": 0.0
                })
        
        return results
    
    def _calculate_fitness_score(self, results: List[Dict[str, float]]) -> float:
        """Calculate fitness score from backtest results."""
        if not results:
            return float('-inf')
        
        # Extract metric values
        metric_values = [r.get(self.fitness_metric, 0.0) for r in results]
        
        # Calculate robust fitness score (median to handle outliers)
        fitness_score = np.median(metric_values)
        
        # Apply penalty for high variance (unstable performance)
        variance = np.var(metric_values)
        stability_penalty = min(variance * 0.1, 0.5)  # Cap penalty at 0.5
        
        return fitness_score - stability_penalty
    
    def _aggregate_metrics(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across all backtests."""
        if not results:
            return {}
        
        aggregated = {}
        for metric in results[0].keys():
            values = [r.get(metric, 0.0) for r in results]
            aggregated[f"{metric}_mean"] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)
            aggregated[f"{metric}_median"] = np.median(values)
            aggregated[f"{metric}_min"] = np.min(values)
            aggregated[f"{metric}_max"] = np.max(values)
        
        return aggregated


class MultiObjectiveFitnessEvaluator(FitnessEvaluator):
    """
    Multi-objective fitness evaluator for Pareto-optimal solutions.
    """
    
    def __init__(
        self,
        evaluators: List[FitnessEvaluator],
        weights: Optional[List[float]] = None,
        cache_results: bool = True
    ):
        """
        Initialize multi-objective fitness evaluator.
        
        Args:
            evaluators: List of fitness evaluators
            weights: Weights for each objective (default: equal weights)
            cache_results: Whether to cache results
        """
        super().__init__(cache_results)
        self.evaluators = evaluators
        self.weights = weights or [1.0 / len(evaluators)] * len(evaluators)
        
        if len(self.weights) != len(evaluators):
            raise ValueError("Number of weights must match number of evaluators")
    
    def evaluate(self, genome: Genome) -> FitnessResult:
        """
        Evaluate genome using multiple objectives.
        
        Args:
            genome: Genome to evaluate
            
        Returns:
            Combined fitness result
        """
        # Check cache first
        cached_result = self._get_cached_result(genome)
        if cached_result is not None:
            return cached_result
        
        # Evaluate with each evaluator, handling errors
        individual_results = []
        errors = []
        for idx, evaluator in enumerate(self.evaluators):
            try:
                result = evaluator.evaluate(genome)
            except Exception as e:
                # Penalize failed evaluation
                result = FitnessResult(
                    genome=genome,
                    fitness_score=-1.0,
                    metrics={},
                    metadata={
                        'error': str(e),
                        'evaluator_index': idx
                    }
                )
                errors.append({'index': idx, 'error': str(e)})
            individual_results.append(result)
        
        # Combine fitness scores
        combined_score = sum(
            result.fitness_score * weight 
            for result, weight in zip(individual_results, self.weights)
        )
        
        # Combine metrics
        combined_metrics = {}
        for i, result in enumerate(individual_results):
            for key, value in result.metrics.items():
                combined_metrics[f"obj_{i}_{key}"] = value
        
        # Create combined result
        combined_result = FitnessResult(
            genome=genome,
            fitness_score=combined_score,
            metrics=combined_metrics,
            metadata={
                "individual_results": individual_results,
                "weights": self.weights,
                "errors": errors if errors else None
            }
        )
        
        # Cache result
        if self.cache_results:
            self._cache_result(genome, combined_result)
        
        return combined_result


class RobustFitnessEvaluator(FitnessEvaluator):
    """
    Robust fitness evaluator that handles evaluation failures gracefully.
    """
    
    def __init__(
        self,
        base_evaluator: FitnessEvaluator,
        max_retries: int = 3,
        failure_penalty: float = -1.0,
        cache_results: bool = True
    ):
        """
        Initialize robust fitness evaluator.
        
        Args:
            base_evaluator: Base fitness evaluator
            max_retries: Maximum number of retry attempts
            failure_penalty: Fitness score for failed evaluations
            cache_results: Whether to cache results
        """
        super().__init__(cache_results)
        self.base_evaluator = base_evaluator
        self.max_retries = max_retries
        self.failure_penalty = failure_penalty
    
    def evaluate(self, genome: Genome) -> FitnessResult:
        """
        Evaluate genome with retry logic.
        
        Args:
            genome: Genome to evaluate
            
        Returns:
            Fitness evaluation result
        """
        # Check cache first
        cached_result = self._get_cached_result(genome)
        if cached_result is not None:
            return cached_result
        
        # Try evaluation with retries
        for attempt in range(self.max_retries):
            try:
                result = self.base_evaluator.evaluate(genome)
                # Add retry information to metadata (attempt represents retries since first attempt is not a retry)
                if hasattr(result, 'metadata') and result.metadata is not None:
                    result.metadata['retries'] = attempt
                else:
                    # Create new result with retry metadata if original has no metadata
                    result = FitnessResult(
                        genome=result.genome,
                        fitness_score=result.fitness_score,
                        metrics=result.metrics,
                        metadata={'retries': attempt}
                    )
                # Cache result
                if self.cache_results:
                    self._cache_result(genome, result)
                return result
                
            except Exception as e:
                logger.warning(
                    f"Evaluation attempt {attempt + 1} failed for genome "
                    f"{genome.hash()}: {e}"
                )
                
                if attempt == self.max_retries - 1:
                    # Final attempt failed, return failure result
                    failure_result = FitnessResult(
                        genome=genome,
                        fitness_score=self.failure_penalty,
                        metrics={},
                        metadata={"error": str(e), "attempts": self.max_retries, "retries": attempt}
                    )
                    self._cache_result(genome, failure_result)
                    return failure_result
        
        # Should not reach here
        raise RuntimeError("Unexpected error in robust evaluation") 