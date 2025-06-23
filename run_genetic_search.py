from config import MUTATION_RATE, ELITE_PROPORTION
from evolution_engine.genetic_search import GeneticSearch

ga = GeneticSearch(
    population_size = 10,
    max_generations = 25,
    mutation_rate = MUTATION_RATE,
    elite_fraction = ELITE_PROPORTION
)

best_genome, best_fitness = ga.run()

print(f"Best Genome: {best_genome.hash()}")
print(f"Fitness: {best_fitness}")