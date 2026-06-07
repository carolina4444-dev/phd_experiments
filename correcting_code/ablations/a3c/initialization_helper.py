def get_initial_architecture(cfg):

    if cfg.initialization == "random":

        return np.random.randint(
            0,
            4,
            size=6
        )

    if cfg.initialization == "conv3x3":

        return np.array(
            [3,3,3,3,3,3]
        )

    if cfg.initialization == "skip":

        return np.array(
            [1,1,1,1,1,1]
        )

    if cfg.initialization == "best_known":

        return np.array(
            [3,3,3,1,3,3]
        )

    raise ValueError