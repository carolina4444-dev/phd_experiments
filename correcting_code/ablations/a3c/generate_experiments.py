from itertools import product
from grid_search_driver import GRID
from experiments_config import ExperimentConfig
import os
import json
import numpy as np
import tensorflow as tf
from nats_api import create
from env import NATSRefinementEnv
from agents import (
    RandomSearchAgent,
    ReinforceAgent,
    A2CAgent,
    A3CAgent,
    DQNAgent
)


def build_experiment_configs():

    keys = list(GRID.keys())

    values = list(GRID.values())

    for combo in product(*values):

        cfg = ExperimentConfig()

        for k, v in zip(keys, combo):

            setattr(cfg, k, v)

        yield cfg





def run_experiment(cfg):

    print("=" * 80)
    print("RUNNING EXPERIMENT")
    print(cfg)
    print("=" * 80)

    experiment_name = (
        f"{cfg.algorithm}_"
        f"{cfg.encoder}_"
        f"{cfg.dataset}_"
        f"{cfg.reward_type}"
    )

    output_dir = os.path.join(
        "results",
        experiment_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # =====================================================
    # DATASET ABLATION
    # =====================================================

    dataset = cfg.dataset

    if dataset == "cifar10":

        env_dataset = "cifar10"

    elif dataset == "cifar100":

        env_dataset = "cifar100"

    elif dataset == "ImageNet16-120":

        env_dataset = "ImageNet16-120"

    else:

        raise ValueError(
            f"Unknown dataset {dataset}"
        )

    # =====================================================
    # ENVIRONMENT
    # =====================================================

    api = create(
        cfg.nats_path,
        "tss",
        fast_mode=True,
        verbose=False
    )

    env = NATSRefinementEnv(
        api=api,
        dataset=env_dataset,
        cfg=cfg
    )

    # =====================================================
    # STATE ENCODING ABLATION
    # =====================================================

    if cfg.state_encoding == "full":

        env.use_accuracy_bucket = True
        env.use_target_improvement = True
        env.use_positional_encoding = False

    elif cfg.state_encoding == "reduced":

        env.use_accuracy_bucket = True
        env.use_target_improvement = False
        env.use_positional_encoding = False

    elif cfg.state_encoding == "positional":

        env.use_accuracy_bucket = True
        env.use_target_improvement = True
        env.use_positional_encoding = True

    else:

        raise ValueError(
            "Unknown state encoding"
        )

    # =====================================================
    # ENCODER ABLATION
    # =====================================================

    if cfg.encoder == "mlp":

        network_builder = build_mlp_network

    elif cfg.encoder == "cnn":

        network_builder = build_cnn_network

    elif cfg.encoder == "transformer":

        network_builder = build_transformer_network

    elif cfg.encoder == "vqvae":

        network_builder = build_vqvae_network

    elif cfg.encoder == "transformer_vqvae":

        network_builder = build_transformer_vqvae

    else:

        raise ValueError(
            "Unknown encoder"
        )

    # =====================================================
    # POLICY ABLATION
    # =====================================================

    if cfg.policy_type == "single":

        policy_mode = "single"

    elif cfg.policy_type == "dual":

        policy_mode = "dual"

    else:

        raise ValueError(
            "Unknown policy type"
        )

    # =====================================================
    # OUTPUT LAYER ABLATION
    # =====================================================

    if cfg.output_mode == "softmax":

        output_layer = "softmax"

    elif cfg.output_mode == "gumbel":

        output_layer = "gumbel"

    else:

        raise ValueError(
            "Unknown output layer"
        )

    # =====================================================
    # ACTION MASKING ABLATION
    # =====================================================

    if cfg.action_masking:

        action_mask = True

    else:

        action_mask = False

    # =====================================================
    # EXPLORATION ABLATION
    # =====================================================

    if cfg.exploration == "softmax":

        exploration_strategy = SoftmaxSampling()

    elif cfg.exploration == "epsilon_greedy":

        exploration_strategy = EpsilonGreedy(
            epsilon=cfg.epsilon
        )

    elif cfg.exploration == "entropy":

        exploration_strategy = EntropySampling(
            beta=cfg.entropy_beta
        )

    else:

        raise ValueError(
            "Unknown exploration strategy"
        )

    # =====================================================
    # REPLAY BUFFER ABLATION
    # =====================================================

    replay_buffer = None

    if cfg.replay:

        if cfg.replay_type == "uniform":

            replay_buffer = ReplayBuffer(
                cfg.replay_size
            )

        elif cfg.replay_type == "prioritized":

            replay_buffer = PrioritizedReplayBuffer(
                cfg.replay_size
            )

        else:

            raise ValueError(
                "Unknown replay type"
            )

    # =====================================================
    # REWARD ABLATION
    # =====================================================

    if cfg.reward_type == "dense":

        reward_function = DenseReward()

    elif cfg.reward_type == "sparse":

        reward_function = SparseReward()

    elif cfg.reward_type == "flops":

        reward_function = FLOPSReward(
            lambda_flops=cfg.flops_lambda
        )

    elif cfg.reward_type == "complexity":

        reward_function = ComplexityReward(
            lambda_complexity=
            cfg.complexity_lambda
        )

    else:

        raise ValueError(
            "Unknown reward type"
        )

    # =====================================================
    # SEARCH STRATEGY ABLATION
    # =====================================================

    if cfg.refinement:

        search_strategy = SequentialRefinement()

    else:

        search_strategy = SingleShotSearch()

    # =====================================================
    # INITIALIZATION ABLATION
    # =====================================================

    if cfg.initialization == "random":

        init_arch = random_architecture()

    elif cfg.initialization == "conv3x3":

        init_arch = np.array(
            [3,3,3,3,3,3]
        )

    elif cfg.initialization == "skip":

        init_arch = np.array(
            [1,1,1,1,1,1]
        )

    elif cfg.initialization == "best_known":

        init_arch = load_best_architecture()

    else:

        raise ValueError(
            "Unknown initialization"
        )

    # =====================================================
    # RL ALGORITHM ABLATION
    # =====================================================

    if cfg.algorithm == "random":

        agent = RandomSearchAgent(
            env=env
        )

    elif cfg.algorithm == "reinforce":

        agent = ReinforceAgent(
            env=env,
            network_builder=network_builder
        )

    elif cfg.algorithm == "a2c":

        agent = A2CAgent(
            env=env,
            network_builder=network_builder
        )

    elif cfg.algorithm == "a3c":

        agent = A3CAgent(
            env=env,
            network_builder=network_builder,
            num_workers=cfg.num_workers
        )

    elif cfg.algorithm == "dqn":

        agent = DQNAgent(
            env=env,
            network_builder=network_builder,
            replay_buffer=replay_buffer
        )

    else:

        raise ValueError(
            f"Unknown algorithm {cfg.algorithm}"
        )

    # =====================================================
    # SEED REPETITIONS
    # =====================================================

    all_runs = []

    for seed in cfg.seeds:

        print(
            f"Running seed {seed}"
        )

        np.random.seed(seed)
        tf.random.set_seed(seed)

        results = agent.train(
            episodes=cfg.episodes,
            exploration_strategy=
            exploration_strategy,
            reward_function=
            reward_function,
            search_strategy=
            search_strategy,
            output_layer=
            output_layer,
            action_mask=
            action_mask,
            init_arch=
            init_arch
        )

        all_runs.append(results)

    # =====================================================
    # AGGREGATION
    # =====================================================

    best_accs = [
        r["best_accuracy"]
        for r in all_runs
    ]

    mean_acc = np.mean(best_accs)
    std_acc = np.std(best_accs)

    summary = {

        "experiment_name":
            experiment_name,

        "config":
            vars(cfg),

        "mean_accuracy":
            float(mean_acc),

        "std_accuracy":
            float(std_acc),

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

    print(
        f"Mean accuracy = "
        f"{mean_acc:.4f}"
    )

    print(
        f"Std accuracy = "
        f"{std_acc:.4f}"
    )

    return summary


for cfg in build_experiment_configs():

    run_experiment(cfg)