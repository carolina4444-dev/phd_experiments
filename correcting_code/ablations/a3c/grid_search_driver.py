from itertools import product

GRID = {

    "dataset":[
        "cifar10",
        "cifar100"
    ],

    "algorithm":[
        "random",
        "reinforce",
        "a2c"
    ],

    "encoder":[
        "mlp",
        "transformer_vqvae"
    ],

    "output_mode":[
        "softmax",
        "gumbel"
    ],

    "reward_type":[
        "dense",
        "flops"
    ]
}