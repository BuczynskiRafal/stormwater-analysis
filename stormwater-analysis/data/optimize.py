"""
Trying to build genetic algorithm to optimize conduits network.
"""


import numpy as np
import pandas as pd
from random import choices, randint, random, uniform
from typing import List, Tuple
from feature_engineering import conduits_data


def crossover(
    parent1: List[float], parent2: List[float]
) -> Tuple[List[float], List[float]]:
    """
    Performs one-point crossover between two parents.

    Args:
        parent1 (List[float]): The first parent.
        parent2 (List[float]): The second parent.

    Returns:
        Tuple[List[float], List[float]]: The offspring of the parents.
    """
    crossover_point = randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(individual: List[float], mutation_rate: float) -> List[float]:
    """
    Mutates an individual with a given mutation rate.

    Args:
        individual (List[float]): The individual to mutate.
        mutation_rate (float): The mutation rate.

    Returns:
        List[float]: The mutated individual.
    """
    mutated_individual = []
    for gene in individual:
        if random() < mutation_rate:
            mutated_individual.append(gene + random() * 2 - 1)
        else:
            mutated_individual.append(gene)
    return mutated_individual


def create_initial_population(
    conduits_data: pd.DataFrame, population_size: int
) -> List[List[float]]:
    """
    Creates the initial population for the genetic algorithm.

    Args:
        conduits_data (pd.DataFrame): The conduits dataframe.
        population_size (int): The size of the population.

    Returns:
        List[List[float]]: The initial population.
    """
    num_conduits = len(conduits_data)
    return [uniform(-1, 1) for _ in range(num_conduits)]


class GeneticAlgorithm:
    def __init__(
        self,
        conduits_data: pd.DataFrame,
        population_size: int,
        mutation_rate: float,
        num_generations: int,
        elitism: int,
    ) -> None:
        self.conduits_data = conduits_data
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.elitism = elitism

        self.population = self.initialize_population()

    def initialize_population(self) -> List[List[float]]:
        return [
            [random() for _ in range(len(self.conduits_data))]
            for _ in range(self.population_size)
        ]

    def fitness(self, individual: List[float]) -> float:
        total_slope = np.sum(np.array(individual) * self.conduits_data["SlopePerMile"])
        return abs(total_slope)  # Ensure positive fitness values

    def selection(self) -> List[List[float]]:
        """
        Selects individuals from the current population to create the mating pool.

        Returns:
            List[List[float]]: The mating pool.
        """
        fitness_scores = [self.fitness(individual) for individual in self.population]
        return choices(self.population, weights=fitness_scores, k=len(self.population))

    def run(self) -> Tuple[List[float], float]:
        """
        Runs the genetic algorithm for the specified number of generations.

        Returns:
            Tuple[List[float], float]: The best individual and its fitness score.
        """
        for generation in range(self.num_generations):
            # Selection
            mating_pool = self.selection()

            # Crossover and mutation
            new_population = []
            for i in range(0, len(mating_pool) - 1, 2):
                parent1 = mating_pool[i]
                parent2 = mating_pool[i + 1]

                # Crossover
                child1, child2 = crossover(parent1, parent2)

                # Mutation
                child1 = mutate(child1, self.mutation_rate)
                child2 = mutate(child2, self.mutation_rate)

                new_population.append(child1)
                new_population.append(child2)

            # Elitism
            for i in range(self.elitism):
                new_population[i] = self.population[i]

            self.population = new_population

        # Get the best individual and its fitness score
        best_individual = max(self.population, key=self.fitness)
        best_fitness = self.fitness(best_individual)

        return best_individual, best_fitness


cd = conduits_data.conduits.copy()
population_size = 100
mutation_rate = 0.1
num_generations = 100
elitism = 1

ga = GeneticAlgorithm(
    conduits_data=cd,
    population_size=population_size,
    mutation_rate=mutation_rate,
    num_generations=num_generations,
    elitism=elitism,
)

best_individual, best_fitness = ga.run()
cd["SlopePerMile"] += np.array(best_individual) * cd["SlopePerMile"]


print("\n")
print(conduits_data.conduits)
print(cd)
# print(best_individual)
# print(best_fitness)
