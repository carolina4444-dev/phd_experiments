# ============================================================
# LATEX TABLE GENERATOR
# ============================================================

import numpy as np
from collections import defaultdict

# ------------------------------------------------------------
# FORMAT HELPERS
# ------------------------------------------------------------

def fmt_mean_std(mean, std):

    return f"{mean:.3f} $\\pm$ {std:.3f}"

# ------------------------------------------------------------
# SINGLE TABLE GENERATOR
# ------------------------------------------------------------

def generate_latex_table(results, group_key="algorithm"):

    """
    Generates a LaTeX table grouped by a config field.

    Example:
        group_key = "algorithm"
        group rows by random / a2c / etc.
    """

    grouped = defaultdict(list)

    for r in results:

        cfg = r["config"]

        key = cfg[group_key]

        grouped[key].append(r)

    rows = []

    for key, runs in grouped.items():

        means = [r["mean_accuracy"] for r in runs]
        stds = [r["std_accuracy"] for r in runs]

        mean = np.mean(means)
        std = np.mean(stds)

        rows.append((key, mean, std))

    # sort by performance
    rows.sort(key=lambda x: x[1], reverse=True)

    # --------------------------------------------------------
    # BUILD LATEX
    # --------------------------------------------------------

    latex = []

    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{l c}")
    latex.append("\\toprule")
    latex.append(f"{group_key} & Accuracy \\\\")
    latex.append("\\midrule")

    for key, mean, std in rows:

        latex.append(
            f"{key} & {fmt_mean_std(mean, std)} \\\\"
        )

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Ablation study over " + group_key + "}")
    latex.append("\\end{table}")

    return "\n".join(latex)

# ------------------------------------------------------------
# MULTI-FACTOR TABLE (ADVANCED ABLATION)
# ------------------------------------------------------------

def generate_multi_factor_table(results):

    """
    Creates a compact paper-style table:

    algorithm | policy | reward | output | accuracy
    """

    def key(r, fields):

        cfg = r["config"]

        return tuple(cfg[f] for f in fields)

    groups = defaultdict(list)

    fields = [
        "algorithm",
        "policy_type",
        "reward_type",
        "output_mode"
    ]

    for r in results:

        groups[key(r, fields)].append(r)

    rows = []

    for k, runs in groups.items():

        mean = np.mean(
            [r["mean_accuracy"] for r in runs]
        )

        std = np.mean(
            [r["std_accuracy"] for r in runs]
        )

        rows.append((*k, mean, std))

    # sort best first
    rows.sort(key=lambda x: x[-2], reverse=True)

    # --------------------------------------------------------
    # LATEX
    # --------------------------------------------------------

    latex = []

    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{l l l l c}")
    latex.append("\\toprule")
    latex.append("Alg & Policy & Reward & Output & Acc \\\\")
    latex.append("\\midrule")

    for row in rows:

        alg, pol, rew, out, mean, std = row

        latex.append(
            f"{alg} & {pol} & {rew} & {out} & "
            f"{fmt_mean_std(mean, std)} \\\\"
        )

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Full NAS ablation comparison}")
    latex.append("\\end{table}")

    return "\n".join(latex)

# ------------------------------------------------------------
# SAVE FUNCTIONS
# ------------------------------------------------------------

def save_latex_tables(results, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # Table 1: by algorithm
    table_alg = generate_latex_table(
        results,
        group_key="algorithm"
    )

    with open(
        os.path.join(
            output_dir,
            "table_algorithm.tex"
        ),
        "w"
    ) as f:

        f.write(table_alg)

    # Table 2: by reward
    table_reward = generate_latex_table(
        results,
        group_key="reward_type"
    )

    with open(
        os.path.join(
            output_dir,
            "table_reward.tex"
        ),
        "w"
    ) as f:

        f.write(table_reward)

    # Table 3: full comparison
    table_full = generate_multi_factor_table(
        results
    )

    with open(
        os.path.join(
            output_dir,
            "table_full.tex"
        ),
        "w"
    ) as f:

        f.write(table_full)

    print(
        f"[LaTeX] Tables saved to {output_dir}"
    )