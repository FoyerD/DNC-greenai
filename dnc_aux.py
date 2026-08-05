import numpy as np

from eckity.evaluators import SimpleIndividualEvaluator


class BinPackingEvaluator(SimpleIndividualEvaluator):

    def __init__(self, n_items, item_weights, bin_capacity, fitness_dict):
        super().__init__()
        self.n_items = n_items
        self.item_weights = item_weights
        self.bin_capacity = bin_capacity
        self.fitness_dict = fitness_dict

    def evaluate_individual(self, individual):
        """
            Compute the fitness value of a given individual.

            Parameters
            ----------
            individual: Vector
                The individual to compute the fitness value for.

            Returns
            -------
            float
                The evaluated fitness value of the given individual.
        """
        return self.get_bin_packing_fitness(np.array(individual.vector))

    def get_bin_packing_fitness(self, individual, penalty=100):
        fitness_dict = self.fitness_dict

        if tuple(individual) in fitness_dict:
            return fitness_dict[tuple(individual)]

        fitness = 0
        bin_capacities = np.zeros(self.n_items)
        legal_solution = True

        for item_index, bin_index in enumerate(individual):
            bin_capacities[bin_index] += self.item_weights[item_index]

            if bin_capacities[bin_index] > self.bin_capacity:
                legal_solution = False
                fitness -= penalty

        if legal_solution:
            utilized_bins = bin_capacities[bin_capacities > 0]
            fitness = ((bin_capacities / self.bin_capacity) ** 2).sum() / len(utilized_bins)

        fitness_dict[tuple(individual)] = fitness
        return fitness
