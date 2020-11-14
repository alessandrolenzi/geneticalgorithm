import time

import numpy as np
from geneticalgorithm.geneticalgorithm import geneticalgorithm as ga

TESTS = [
    np.random.rand(8) for i in range(0, 20)
]

target_individual = [5.42, 64.12, 22.04, -88.11, 4, 1, 1, 0]

def calculated(individual, x):
    return sum(
        individual[i] * x[i]
        for i, _ in enumerate(individual)
    )


def cost_function(individual):
    # time.sleep(0.001)
    return sum(
        abs(
            calculated(individual, i) - calculated(target_individual, i)
        ) for i in TESTS
    )

if __name__ == '__main__':
    varbound=np.array([[-30,+30], [-100,+100], [-30, +30], [-100,+100], [0, 4], [0, 2], [0, 1], [0, 3]])
    vartype=np.array([['real'],['real'],['real'], ['real'], ['int'], ['int'], ['int'], ['int']])
    model = ga(function=cost_function,
               dimension=8,
               variable_type_mixed=vartype,
               variable_boundaries=varbound,
               algorithm_parameters={'max_num_iteration': None,
                       'population_size': 100,
                       'mutation_probability':0.1,
                       'elit_ratio': 0.03,
                       'crossover_probability': 0.5,
                       'surviving_parents_portion': 0.3,
                       'crossover_type':'uniform',
                       'max_iteration_without_improv': None
            }

    )

    model.run()
