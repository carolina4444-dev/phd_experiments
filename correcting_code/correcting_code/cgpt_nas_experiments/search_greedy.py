# search_greedy.py

class GreedySearch:

    def __init__(self, benchmark):

        self.benchmark = benchmark

    def run(self, budget):

        current = self.benchmark.sample_architecture()

        best_acc = self.benchmark.get_accuracy(current)

        history = [best_acc]

        evaluations = 1

        while evaluations < budget:

            neighbors = self.benchmark.get_neighbors(current)

            scores = []

            for n in neighbors:

                acc = self.benchmark.get_accuracy(n)

                scores.append((acc, n))

                evaluations += 1

                best_acc = max(best_acc, acc)

                history.append(best_acc)

                if evaluations >= budget:
                    break

            if evaluations >= budget:
                break

            current = max(scores)[1]

        return history[:budget]