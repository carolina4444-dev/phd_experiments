# search_random.py

class RandomSearch:

    def __init__(self, benchmark):

        self.benchmark = benchmark

    def run(self, budget):

        history = []

        best_acc = 0

        for t in range(budget):

            arch = self.benchmark.sample_architecture()

            acc = self.benchmark.get_accuracy(arch)

            best_acc = max(best_acc, acc)

            history.append(best_acc)

        return history