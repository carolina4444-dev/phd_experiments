def build_replay(cfg):

    if not cfg.replay:

        return None

    if cfg.replay_type == "uniform":

        return ReplayBuffer(
            cfg.replay_size
        )

    if cfg.replay_type == "prioritized":

        return PrioritizedReplayBuffer(
            cfg.replay_size
        )

    raise ValueError


def build_search(cfg):

    if cfg.refinement:

        return SequentialRefinement()

    return SingleShotSearch()