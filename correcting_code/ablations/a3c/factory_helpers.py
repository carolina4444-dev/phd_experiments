NETWORK_FACTORY = {

    "mlp":
        build_mlp_network,

    "cnn":
        build_cnn_network,

    "transformer":
        build_transformer_network,

    "vqvae":
        build_vqvae_network,

    "transformer_vqvae":
        build_transformer_vqvae,
}

EXPLORATION_FACTORY = {

    "softmax":
        lambda cfg:
            SoftmaxSampling(),

    "epsilon_greedy":
        lambda cfg:
            EpsilonGreedy(
                cfg.epsilon
            ),

    "entropy":
        lambda cfg:
            EntropySampling(
                cfg.entropy_beta
            ),
}

REWARD_FACTORY = {

    "dense":
        lambda cfg:
            DenseReward(),

    "sparse":
        lambda cfg:
            SparseReward(),

    "flops":
        lambda cfg:
            FLOPSReward(
                cfg.flops_lambda
            ),

    "complexity":
        lambda cfg:
            ComplexityReward(
                cfg.complexity_lambda
            ),
}

AGENT_FACTORY = {

    "random":
        RandomSearchAgent,

    "reinforce":
        ReinforceAgent,

    "a2c":
        A2CAgent,

    "a3c":
        A3CAgent,

    "dqn":
        DQNAgent,
}


