from torch import FloatTensor


class TrainingModule:
    """Collection of Model, Loss Function, and Optimizer with built-in training loops."""

    def training_step(self, model_input, correct_output) -> FloatTensor:  # noqa: ANN001
        """Single Batch of Training Loop (must be overridden by a subclass).

        Args:
            model_input: Input batch to model
            correct_output: Ground truth output of model

        Returns:
            FloatTensor: Loss of model
        """
        raise NotImplementedError("Must be implemented by subclass")
