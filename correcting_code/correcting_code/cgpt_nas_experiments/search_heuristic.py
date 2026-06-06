# search_heuristic.py

import heapq

class HeuristicSearch:

    def __init__(self,
                 benchmark,
                 heuristic):

        self.benchmark = benchmark
        self.heuristic = heuristic

    def run(self, budget):

        start = self.benchmark.sample_architecture()

        frontier = []

        heapq.heappush(
            frontier,
            (-self.heuristic(start), start)
        )

        visited = set()

        best_acc = 0

        history = []

        evaluations = 0

        while frontier and evaluations < budget:

            _, node = heapq.heappop(frontier)

            if node in visited:
                continue

            visited.add(node)

            acc = self.benchmark.get_accuracy(node)

            best_acc = max(best_acc, acc)

            history.append(best_acc)

            evaluations += 1

            for nbr in self.benchmark.get_neighbors(node):

                if nbr not in visited:

                    score = self.heuristic(nbr)

                    heapq.heappush(
                        frontier,
                        (-score, nbr)
                    )

        return history