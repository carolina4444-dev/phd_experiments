# run_experiment.py

import argparse
import json
import numpy as np
from pathlib import Path

from benchmark import NASBenchmark
from search_random import RandomSearch
from search_greedy import GreedySearch
from search_heuristic import HeuristicSearch
from heuristic import ToyHeuristic

parser = argparse.ArgumentParser()

parser.add_argument("--method", required=True)
parser.add_argument("--seed", required=True, type=int)
parser.add_argument("--budget", required=True, type=int)

args = parser.parse_args()

METHOD = args.method
SEED = args.seed
BUDGET = args.budget

print(f"Method: {METHOD}")
print(f"Seed: {SEED}")
print(f"Budget: {BUDGET}")

np.random.seed(SEED)

# -------------------------------------------------
# initialize benchmark
# -------------------------------------------------

from nats_bench import create
from benchmark import NASBenchmark

api = create(
    "/data/ccarvalho/phd_working/cgpt_nas_experiemnts/benchmarks/NATS-tss-v1_0-3ffb9-simple",
    "tss",
    fast_mode=True,
    verbose=False
)

benchmark = NASBenchmark(api)

# -------------------------------------------------
# choose search method
# -------------------------------------------------

if METHOD == "random":

    search = RandomSearch(
        benchmark
    )

elif METHOD == "greedy":

    search = GreedySearch(
        benchmark
    )

elif METHOD == "heuristic":

    heuristic = ToyHeuristic()

    search = HeuristicSearch(
        benchmark,
        heuristic
    )

else:

    raise ValueError(
        f"Unknown method {METHOD}"
    )

# -------------------------------------------------
# run search
# -------------------------------------------------

history = search.run(BUDGET)

best_accuracy = history[-1]

# -------------------------------------------------
# save results
# -------------------------------------------------

results_dir = Path("results")

results_dir.mkdir(
    exist_ok=True
)

output = {

    "method": METHOD,
    "seed": SEED,
    "budget": BUDGET,
    "best_accuracy": float(best_accuracy),
    "history": [float(x) for x in history]

}

outfile = (
    results_dir /
    f"{METHOD}_seed{SEED}_budget{BUDGET}.json"
)

with open(outfile, "w") as f:

    json.dump(
        output,
        f,
        indent=2
    )

print(
    f"Saved {outfile}"
)