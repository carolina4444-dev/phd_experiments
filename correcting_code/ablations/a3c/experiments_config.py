from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:

    dataset:str = "cifar10"

    algorithm:str = "a2c"

    encoder:str = "transformer_vqvae"

    policy_type:str = "dual"

    refinement:bool = True

    output_mode:str = "softmax"

    action_masking:bool = False

    exploration:str = "softmax"

    entropy_beta:float = 0.0

    epsilon:float = 0.1

    replay:bool = False

    replay_type:str = "uniform"

    replay_size:int = 10000

    reward_type:str = "dense"

    state_encoding:str = "full"

    initialization:str = "conv3x3"

    episodes:int = 100

    seeds:list = field(
        default_factory=lambda:[42]
    )

    num_workers:int = 4

    flops_lambda:float = 0.01

    complexity_lambda:float = 0.01

    nats_path:str = "/data/ccarvalho/phd_working/cgpt_nas_experiemnts/benchmarks/NATS-tss-v1_0-3ffb9-simple"

