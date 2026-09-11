"""
loss.py – Training objective for the displacement/position UNet.
"""

import torch
import torch.nn as nn


def l1_regularisation(model: nn.Module) -> torch.Tensor:
    """Sum of absolute values of all Conv2d weight tensors."""
    reg = torch.tensor(0.0)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            reg = reg + torch.sum(torch.abs(module.weight))
    return reg


def combined_loss(outputs: torch.Tensor, targets: torch.Tensor,
                   model: nn.Module, lambda_l1: float) -> torch.Tensor:
    """MSE reconstruction loss plus an L1 penalty on convolutional weights."""
    return nn.functional.mse_loss(outputs, targets) + lambda_l1 * l1_regularisation(model)