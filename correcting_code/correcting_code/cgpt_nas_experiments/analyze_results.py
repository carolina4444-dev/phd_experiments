from pathlib import Path
import json
import numpy as np

files = list(
    Path("results").glob("*.json")
)

results = []

for f in files:

    with open(f) as fp:

        results.append(
            json.load(fp)
        )

print(
    f"Loaded {len(results)} runs"
)