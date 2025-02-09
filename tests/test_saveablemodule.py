import os

import torch

from miexp.models.btransformer import SaveableModule


def test_saveable_module_init():
    """Test the initialization of SaveableModule."""
    module = SaveableModule()
    assert isinstance(module.hyperparameters, dict)
    assert module.hyperparameters == {}


def test_save_to_checkpoint():
    """Test the save_to_checkpoint method of SaveableModule."""
    module = SaveableModule()
    checkpoint_path = os.path.join("checkpoint.pt")

    # Save the module to a checkpoint
    module.save_to_checkpoint(checkpoint_path)

    # Check if the checkpoint file is created
    assert os.path.exists(checkpoint_path)

    # Load the checkpoint and verify its contents
    checkpoint = torch.load(checkpoint_path)
    os.remove(checkpoint_path)
    assert "hyperparameters" in checkpoint
    assert "state_dict" in checkpoint
    assert checkpoint["hyperparameters"] == module.hyperparameters
    assert checkpoint["state_dict"] == module.state_dict()
