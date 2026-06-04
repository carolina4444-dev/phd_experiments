# benchmark.py

import numpy as np

class NASBenchmark:

    def __init__(self, api):
        self.api = api

    def sample_architecture(self):
        return np.random.randint(len(self.api))

    def get_accuracy(self, arch_id):

        info = self.api.get_more_info(
            arch_id,
            "cifar10"
        )

        return info["test-accuracy"]

    def get_neighbors(self, arch_id):

        total = len(self.api)

        return np.random.choice(
            total,
            size=20,
            replace=False
        ).tolist()