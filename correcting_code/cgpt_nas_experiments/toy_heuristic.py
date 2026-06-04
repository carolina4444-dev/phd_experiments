# heuristic.py

import numpy as np

class ToyHeuristic:

    def __call__(self, arch_id):

        np.random.seed(arch_id)

        return np.random.random()