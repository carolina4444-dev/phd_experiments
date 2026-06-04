class DARTSNASProblem(NASProblem):

    def __init__(
        self,
        train_dataset,
        valid_dataset,
        num_value_actions,
        num_position_actions,
    ):

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.model = DARTSSearchNetwork(
            num_value_actions=num_value_actions,
            num_position_actions=num_position_actions,
        )

        self.weight_optimizer = tf.keras.optimizers.Adam(
            1e-3
        )

        self.arch_optimizer = tf.keras.optimizers.Adam(
            3e-4
        )

        self.loss_fn = tf.keras.losses.MeanSquaredError()

    # =====================================================
    # SEARCH
    # =====================================================

    def search(self, epochs=10):

        valid_iter = iter(
            self.valid_dataset.repeat()
        )

        for epoch in range(epochs):

            for train_batch in self.train_dataset:

                valid_batch = next(valid_iter)

                self._train_step(
                    train_batch,
                    valid_batch
                )

            print(f"Epoch {epoch} complete")

    # =====================================================
    # DARTS STEP
    # =====================================================

    @tf.function
    def _train_step(
        self,
        train_batch,
        valid_batch,
    ):

        train_x, train_y = train_batch
        valid_x, valid_y = valid_batch

        # ============================================
        # UPDATE WEIGHTS
        # ============================================

        with tf.GradientTape() as tape:

            value_q, position_q = self.model(
                train_x,
                training=True
            )

            loss = (
                tf.reduce_mean(value_q)
                + tf.reduce_mean(position_q)
            )

        grads = tape.gradient(
            loss,
            self.model.weight_parameters
        )

        self.weight_optimizer.apply_gradients(
            zip(
                grads,
                self.model.weight_parameters
            )
        )

        # ============================================
        # UPDATE ALPHAS
        # ============================================

        with tf.GradientTape() as tape:

            value_q, position_q = self.model(
                valid_x,
                training=True
            )

            arch_loss = (
                tf.reduce_mean(value_q)
                + tf.reduce_mean(position_q)
            )

        arch_grads = tape.gradient(
            arch_loss,
            self.model.arch_parameters
        )

        self.arch_optimizer.apply_gradients(
            zip(
                arch_grads,
                self.model.arch_parameters
            )
        )

    # =====================================================
    # EXPORT DISCRETE ARCHITECTURE
    # =====================================================

    def export_architecture(self):

        op_names = [
            "conv3x3",
            "conv5x5",
            "skip",
            "maxpool",
        ]

        architecture = []

        for alpha in self.model.arch_parameters:

            probs = tf.nn.softmax(alpha)

            best_idx = tf.argmax(probs)

            architecture.append(
                op_names[int(best_idx)]
            )

        return architecture

    # =====================================================
    # EVALUATE
    # =====================================================

    def evaluate(self):

        arch = self.export_architecture()

        print("Selected architecture:")
        print(arch)

        return arch