"""
Genetic algorithm for hyperparameter optimization.

This module provides a genetic algorithm implementation for optimizing
trading strategy hyperparameters through evolutionary search.
"""

import os
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

from evo.core.logging import get_logger
from .genome import Genome, GenomeConfig
from .fitness import FitnessEvaluator, FitnessResult
from evo.optimization.backtesting.engine import BacktestEngine

logger = get_logger(__name__)


@dataclass
class GeneticSearchConfig:
    """Configuration for genetic search algorithm."""
    
    # Population parameters
    population_size: int = 50
    max_generations: int = 100
    
    # Genetic operators
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_fraction: float = 0.2
    random_fraction: float = 0.1
    
    # Selection parameters
    tournament_size: int = 3
    selection_pressure: float = 1.5
    
    # Fitness evaluation
    num_backtests: int = 10
    backtest_length: int = 1000
    fitness_metric: str = "sharpe_ratio"
    
    # Parallel processing
    n_jobs: int = 1
    use_multiprocessing: bool = False
    
    # Early stopping
    patience: int = 20
    min_improvement: float = 0.001
    
    # Logging
    log_interval: int = 5
    save_best_genome: bool = True
    save_history: bool = True


@dataclass
class GenerationResult:
    """Result of a single generation."""
    
    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    best_genome: Genome
    population_diversity: float
    metadata: Dict[str, Any]


class GeneticSearch:
    """
    Genetic algorithm for hyperparameter optimization.
    
    This class implements a genetic algorithm to optimize trading strategy
    hyperparameters through evolutionary search.
    """
    
    def __init__(
        self,
        fitness_evaluator: FitnessEvaluator,
        config: Optional[GeneticSearchConfig] = None,
        genome_config: Optional[GenomeConfig] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize genetic search.
        
        Args:
            fitness_evaluator: Evaluator for genome fitness
            config: Genetic search configuration
            genome_config: Genome configuration
            seed: Random seed for reproducibility
        """
        self.fitness_evaluator = fitness_evaluator
        self.config = config or GeneticSearchConfig()
        self.genome_config = genome_config or GenomeConfig()
        
        # Store seed for reproducibility
        self.seed = seed
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize state
        self.population: List[Genome] = []
        self.generation_history: List[GenerationResult] = []
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = float('-inf')
        self.generation: int = 0
        
        # Early stopping
        self.best_fitness_history: List[float] = []
        self.generations_without_improvement: int = 0
    
    def initialize_population(self) -> None:
        """Initialize the population with random genomes."""
        logger.info(f"Initializing population of size {self.config.population_size}")
        
        # Set seed for reproducible population generation
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        self.population = []
        for _ in range(self.config.population_size):
            genome = Genome.random(config=self.genome_config)
            self.population.append(genome)
        
        logger.info("Population initialized successfully")
    
    def run(self, initial_population: Optional[List[Genome]] = None) -> Tuple[Genome, float]:
        """
        Run the genetic algorithm.
        
        Args:
            initial_population: Optional initial population
            
        Returns:
            Tuple of (best_genome, best_fitness)
        """
        logger.info("Starting genetic search")
        
        # Initialize population
        if initial_population is not None:
            self.population = initial_population
            if len(self.population) != self.config.population_size:
                logger.warning(f"Initial population size {len(self.population)} "
                             f"doesn't match config size {self.config.population_size}")
        else:
            self.initialize_population()
        
        # Main evolution loop
        for generation in range(self.config.max_generations):
            self.generation = generation
            
            logger.info(f"\n=== Generation {generation + 1}/{self.config.max_generations} ===")
            
            # Evaluate population
            fitness_results = self._evaluate_population()
            
            # Update best solution
            self._update_best_solution(fitness_results)
            
            # Log generation results
            generation_result = self._create_generation_result(fitness_results)
            self.generation_history.append(generation_result)
            
            # Log progress
            if (generation + 1) % self.config.log_interval == 0:
                self._log_generation_progress(generation_result)
            
            # Check early stopping
            if self._should_stop_early():
                logger.info(f"Early stopping at generation {generation + 1}")
                break
            
            # Create next generation
            self.population = self._create_next_generation(fitness_results)
        
        # Final evaluation and logging
        self._finalize_search()
        
        return self.best_genome, self.best_fitness
    
    def _evaluate_population(self) -> List[FitnessResult]:
        """Evaluate fitness of entire population."""
        logger.info(f"Evaluating {len(self.population)} genomes")
        
        if self.config.use_multiprocessing and self.config.n_jobs > 1:
            return self._evaluate_population_parallel()
        else:
            return self._evaluate_population_sequential()
    
    def _evaluate_population_sequential(self) -> List[FitnessResult]:
        """Evaluate population sequentially."""
        results = []
        for i, genome in enumerate(self.population):
            logger.debug(f"Evaluating genome {i + 1}/{len(self.population)}")
            try:
                result = self.fitness_evaluator.evaluate(genome)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating genome {i}: {e}")
                # Create poor fitness result for failed evaluation
                result = FitnessResult(
                    genome=genome,
                    fitness_score=float('-inf'),
                    metrics={},
                    metadata={"error": str(e)}
                )
                results.append(result)
        
        return results
    
    def _evaluate_population_parallel(self) -> List[FitnessResult]:
        """Evaluate population using parallel processing."""
        results = [None] * len(self.population)
        
        with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
            # Submit evaluation jobs
            future_to_index = {
                executor.submit(self.fitness_evaluator.evaluate, genome): i
                for i, genome in enumerate(self.population)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.error(f"Error evaluating genome {index}: {e}")
                    # Create poor fitness result for failed evaluation
                    results[index] = FitnessResult(
                        genome=self.population[index],
                        fitness_score=float('-inf'),
                        metrics={},
                        metadata={"error": str(e)}
                    )
        
        return results
    
    def _update_best_solution(self, fitness_results: List[FitnessResult]) -> None:
        """Update the best solution found so far."""
        for result in fitness_results:
            if result.fitness_score > self.best_fitness:
                self.best_fitness = result.fitness_score
                self.best_genome = result.genome
                self.generations_without_improvement = 0
                logger.info(f"New best fitness: {self.best_fitness:.4f}")
            else:
                self.generations_without_improvement += 1
        
        self.best_fitness_history.append(self.best_fitness)
    
    def _create_generation_result(self, fitness_results: List[FitnessResult]) -> GenerationResult:
        """Create result summary for current generation."""
        fitness_scores = [r.fitness_score for r in fitness_results]
        
        # Find best genome
        best_result = max(fitness_results, key=lambda x: x.fitness_score)
        
        # Calculate population diversity
        diversity = self._calculate_population_diversity()
        
        return GenerationResult(
            generation=self.generation,
            best_fitness=max(fitness_scores),
            avg_fitness=np.mean(fitness_scores),
            worst_fitness=min(fitness_scores),
            best_genome=best_result.genome,
            population_diversity=diversity,
            metadata={
                "fitness_std": np.std(fitness_scores),
                "num_evaluations": len(fitness_results)
            }
        )
    
    def _calculate_population_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        # Calculate average pairwise distance between genomes
        total_distance = 0.0
        num_pairs = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self.population[i].distance(self.population[j])
                total_distance += distance
                num_pairs += 1
        
        return total_distance / num_pairs if num_pairs > 0 else 0.0
    
    def _log_generation_progress(self, result: GenerationResult) -> None:
        """Log progress for current generation."""
        logger.info(
            f"Generation {result.generation + 1}: "
            f"Best={result.best_fitness:.4f}, "
            f"Avg={result.avg_fitness:.4f}, "
            f"Worst={result.worst_fitness:.4f}, "
            f"Diversity={result.population_diversity:.4f}"
        )
    
    def _should_stop_early(self) -> bool:
        """Check if early stopping conditions are met."""
        if len(self.best_fitness_history) < self.config.patience:
            return False
        
        # Check if improvement is below threshold
        recent_improvement = (
            self.best_fitness_history[-1] - 
            self.best_fitness_history[-self.config.patience]
        )
        
        return recent_improvement < self.config.min_improvement
    
    def _create_next_generation(self, fitness_results: List[FitnessResult]) -> List[Genome]:
        """Create the next generation using genetic operators."""
        new_population = []
        
        # Sort by fitness
        sorted_results = sorted(fitness_results, key=lambda x: x.fitness_score, reverse=True)
        
        # Elitism: keep best individuals
        n_elite = max(1, int(self.config.elite_fraction * self.config.population_size))
        elites = [result.genome for result in sorted_results[:n_elite]]
        new_population.extend(elites)
        
        # Random individuals: add some random genomes
        n_random = int(self.config.random_fraction * self.config.population_size)
        for _ in range(n_random):
            random_genome = Genome.random(config=self.genome_config)
            new_population.append(random_genome)
        
        # Fill remaining slots with offspring
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self._select_parent(fitness_results)
            parent2 = self._select_parent(fitness_results)
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                offspring = parent1.crossover(parent2)
            else:
                offspring = parent1
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                offspring = offspring.mutate(self.config.mutation_rate)
            
            new_population.append(offspring)
        
        # Ensure population size is correct
        return new_population[:self.config.population_size]
    
    def _select_parent(self, fitness_results: List[FitnessResult]) -> Genome:
        """Select a parent using tournament selection."""
        tournament_size = min(self.config.tournament_size, len(fitness_results))
        
        # Select random individuals for tournament
        tournament_indices = random.sample(range(len(fitness_results)), tournament_size)
        tournament_results = [fitness_results[i] for i in tournament_indices]
        
        # Select winner based on fitness
        winner = max(tournament_results, key=lambda x: x.fitness_score)
        return winner.genome
    
    def _finalize_search(self) -> None:
        """Finalize the search and save results."""
        logger.info(f"\n=== Genetic Search Completed ===")
        logger.info(f"Best fitness: {self.best_fitness:.4f}")
        logger.info(f"Best genome: {self.best_genome.hash()}")
        
        if self.best_genome:
            logger.info("Best genome parameters:")
            self.best_genome.print_values()
        
        # Save results if configured
        if self.config.save_best_genome and self.best_genome:
            self._save_best_genome()
        
        if self.config.save_history:
            self._save_search_history()
    
    def _save_best_genome(self) -> None:
        """Save the best genome to file."""
        if not self.best_genome:
            return
        
        try:
            os.makedirs("checkpoints", exist_ok=True)
            filename = f"checkpoints/best_genome_{self.best_genome.hash()}.json"
            
            with open(filename, 'w') as f:
                json.dump({
                    "genome": self.best_genome.to_dict(),
                    "fitness": self.best_fitness,
                    "generation": self.generation,
                    "metadata": {
                        "config": self.config.__dict__,
                        "timestamp": str(pd.Timestamp.now())
                    }
                }, f, indent=2)
            
            logger.info(f"Best genome saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving best genome: {e}")
    
    def _save_search_history(self) -> None:
        """Save search history to file."""
        try:
            os.makedirs("checkpoints", exist_ok=True)
            filename = f"checkpoints/search_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            history_data = []
            for result in self.generation_history:
                history_data.append({
                    "generation": result.generation,
                    "best_fitness": result.best_fitness,
                    "avg_fitness": result.avg_fitness,
                    "worst_fitness": result.worst_fitness,
                    "population_diversity": result.population_diversity,
                    "best_genome_hash": result.best_genome.hash(),
                    "metadata": result.metadata
                })
            
            with open(filename, 'w') as f:
                json.dump({
                    "history": history_data,
                    "config": self.config.__dict__,
                    "final_best_fitness": self.best_fitness,
                    "final_best_genome_hash": self.best_genome.hash() if self.best_genome else None
                }, f, indent=2)
            
            logger.info(f"Search history saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving search history: {e}")
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get a summary of the search results."""
        return {
            "best_fitness": self.best_fitness,
            "best_genome_hash": self.best_genome.hash() if self.best_genome else None,
            "total_generations": self.generation,
            "population_size": self.config.population_size,
            "final_population_diversity": self._calculate_population_diversity(),
            "fitness_history": self.best_fitness_history,
            "config": self.config.__dict__
        } 