"""
loss.py – Training objective for the displacement/position UNet.
"""

import torch
import torch.nn as nn


def l1_regularisation(model: nn.Module) -> torch.Tensor:
    """Sum of absolute values of every weight tensor with ndim > 1
    (Conv/ConvTranspose/Linear-style weight matrices), excluding
    biases and normalization-layer parameters (which are always 1D)."""
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        if param.dim() > 1:
            reg = reg + torch.sum(torch.abs(param))
    return reg

def combined_loss(outputs, targets, model=None, lambda_l1: float = 0.0):
    """MSE reconstruction loss plus an L1 penalty."""
    loss = nn.functional.mse_loss(outputs, targets)
    if lambda_l1 > 0:
        loss = loss + lambda_l1 * l1_regularisation(model)
    return loss