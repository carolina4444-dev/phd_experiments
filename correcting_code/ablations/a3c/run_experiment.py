import os
import json
import numpy as np
import tensorflow as tf
from nats_bench import create
from .experiments_config import ExperimentConfig


def run_experiment(cfg):

    print("="*80)
    print(cfg)
    print("="*80)

    experiment_name = (
        f"{cfg.algorithm}_"
        f"{cfg.encoder}_"
        f"{cfg.dataset}"
    )

    output_dir = os.path.join(
        "results",
        experiment_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    api = create(
        cfg.nats_path,
        "tss",
        fast_mode=True,
        verbose=False
    )

    env = NATSRefinementEnv(
        api=api,
        dataset=cfg.dataset,
        cfg=cfg
    )

    network_builder = (
        NETWORK_FACTORY[
            cfg.encoder
        ]
    )

    exploration_strategy = (
        EXPLORATION_FACTORY[
            cfg.exploration
        ](cfg)
    )

    reward_function = (
        REWARD_FACTORY[
            cfg.reward_type
        ](cfg)
    )

    replay_buffer = build_replay(
        cfg
    )

    search_strategy = build_search(
        cfg
    )

    init_arch = (
        get_initial_architecture(
            cfg
        )
    )

    AgentClass = (
        AGENT_FACTORY[
            cfg.algorithm
        ]
    )

    kwargs = {

        "env":env,
        "network_builder":
            network_builder
    }

    if cfg.algorithm == "a3c":

        kwargs[
            "num_workers"
        ] = cfg.num_workers

    if cfg.algorithm == "dqn":

        kwargs[
            "replay_buffer"
        ] = replay_buffer

    agent = AgentClass(
        **kwargs
    )

    all_runs = []

    for seed in cfg.seeds:

        np.random.seed(seed)
        tf.random.set_seed(seed)

        result = agent.train(

            episodes=
                cfg.episodes,

            exploration_strategy=
                exploration_strategy,

            reward_function=
                reward_function,

            search_strategy=
                search_strategy,

            output_layer=
                cfg.output_mode,

            action_mask=
                cfg.action_masking,

            init_arch=
                init_arch,
        )

        all_runs.append(
            result
        )

    best_accs = [

        r["best_accuracy"]

        for r in all_runs
    ]

    summary = {

        "experiment":
            experiment_name,

        "config":
            vars(cfg),

        "mean_accuracy":
            float(
                np.mean(
                    best_accs
                )
            ),

        "std_accuracy":
            float(
                np.std(
                    best_accs
                )
            ),

        "runs":
            all_runs,
    }

    with open(

        os.path.join(
            output_dir,
            "summary.json"
        ),

        "w"

    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    return summary


if __name__ == "__main__":

    for cfg in build_experiment_configs():

        run_experiment(cfg)