from config import MODEL_DIR
from config import MUTATION_RATE, ELITE_PROPORTION, NUM_BACKTESTS, BACKTEST_LENGTH

from evolution_engine.genome import Genome
from train_agent import train_with_genome
from backtest.backtest import multi_backtest

import os
import random
from typing import List, Tuple, Optional

from stable_baselines3 import PPO

class GeneticSearch:
    def __init__(
        self,
        population_size: int,
        max_generations: int,
        mutation_rate: float = MUTATION_RATE,
        elite_fraction: float = ELITE_PROPORTION,
        seed: Optional[int] = None
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction
        self.random_fraction = 0.25
        self.seed = seed
        self.history = []  # stores best fitness per generation

        if seed is not None:
            random.seed(seed)

    # trains PPO models with training and reward-shaping parameters specified by a genome
    def train_models(self, genomes: List[Genome]) -> None:
        os.makedirs(MODEL_DIR, exist_ok=True)

        print(f"[TRAIN]: Preparing to train {len(genomes)} models")

        # for each genome, train a model if no model with the same genome already exists
        for genome in genomes:
            
            # get genome id
            genome_id = genome.hash()
            
            # if model with genome has already been trained, skip training
            model_path = os.path.join(MODEL_DIR, f"{genome_id}.zip")
            if os.path.exists(model_path):
                print(f"[TRAIN]: Skipping training model for existing genome {genome_id}")
                continue

            # train model with genome
            print(f"[TRAIN]: Training model with genome {genome_id}")
            train_with_genome(genome.values, model_name=genome_id)

    # evaluates performance for PPO models trained with parameters defined by a genome
    def evaluate_models(self, genomes: List[Genome]) -> List[Tuple[Genome, float]]:
        print(f"[EVAL]: Evaluating {len(genomes)} models over {NUM_BACKTESTS} backtests of length {BACKTEST_LENGTH}")
        
        evaluation_results = []

        for genome in genomes:
            
            # load model from genome id
            genome_id = genome.hash()
            model_path = os.path.join(MODEL_DIR, f"{genome_id}.zip")

            # assert model has already been trained
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"[EVAL]: Model for genome {genome_id} not found.")

            # load and backtest model
            score = multi_backtest(model_path, NUM_BACKTESTS, BACKTEST_LENGTH)  # Assume this returns Sharpe, PnL, etc.
            print(f"[EVAL]: Fitness score {score} for genome {genome_id}")

            evaluation_results.append((genome, score))

        return evaluation_results

    def run(self) -> Tuple[Genome, float]:
        
        # Initialize random population
        population = [Genome.random() for _ in range(self.population_size)]
        best_genome = None
        best_fitness = float("-inf")

        # Run for N generations
        for gen in range(self.max_generations):
            
            print(f"\n=== Generation {gen + 1} ===")
            
            # train and evaluate models
            self.train_models(population)
            eval_results = self.evaluate_models(population)

            # Rank models
            eval_results.sort(key=lambda x: x[1], reverse=True)

            # Logging
            top_score = eval_results[0][1]
            avg_score = sum(f for _, f in eval_results) / len(eval_results)
            
            print(f"=== Best fitness: {top_score:.4f} | Avg fitness: {avg_score:.4f} ===")
            
            self.history.append(top_score)

            if top_score > best_fitness:
                best_genome, best_fitness = eval_results[0]

            # Selection + Elitism
            n_elite = max(1, int(self.elite_fraction * self.population_size))
            elites = [g for g, _ in eval_results[:n_elite]]

            # Fill new population
            new_population = elites.copy()
        
            # Create mutated offspring
            n_mutants = self.population_size - len(elites) - n_random
            for _ in range(n_mutants):
                parent = random.choice(elites)
                child = parent.mutate(mutation_rate=self.mutation_rate)
                new_population.append(child)

            # Create random newcomers
            n_random = int(self.random_fraction * self.population_size)
            for _ in range(n_random):
                new_population.append(Genome.random())

            population = new_population

        return best_genome, best_fitness