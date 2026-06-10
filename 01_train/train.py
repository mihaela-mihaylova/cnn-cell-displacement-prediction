"""
train.py  –  Train the UNet to predict cell positions and XY displacement.

Usage:
    python train.py                          # uses config.yaml in the same folder
    python train.py --config path/to/config.yaml
    python train.py --config config.yaml --lr 0.001 --batch_size 8   # CLI overrides

The script reads all hyperparameters from a YAML config file.
Any value can be overridden from the command line (see --help).
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import yaml
from matplotlib import pyplot as plt
from skimage.io import imread
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from unet import UNet


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device (MPS → CUDA → CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("Warning: no GPU found, falling back to CPU.")
    return torch.device("cpu")


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Override config values with any non-None CLI arguments."""
    overrides = {
        "train_images":  args.train_images,
        "train_masks":   args.train_masks,
        "output_dir":    args.output_dir,
        "random_seed":   args.random_seed,
        "perc_data":     args.perc_data,
        "test_prop":     args.test_prop,
        "batch_size":    args.batch_size,
        "num_epochs":    args.num_epochs,
        "lr":            args.lr,
        "weight_decay":  args.weight_decay,
        "lambda_l1":     args.lambda_l1,
        "patience":      args.patience,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(cfg: dict):
    """
    Load patch images and masks, apply train/val split.

    Returns (X_train, X_val, Y_train, Y_val).
    Each X has shape (N, H, W, 6): first + second image stacked on channels.
    Each Y has shape (N, H, W, 3): [displ_x, displ_y, position_mask].
    """
    train_images = cfg["train_images"]
    train_masks  = cfg["train_masks"]
    img_channels = cfg["img_channels"]
    patch_size   = cfg["patch_size"]
    perc_data    = cfg["perc_data"]
    test_prop    = cfg["test_prop"]

    first_folder = os.path.join(train_images, "first")
    file_count   = len([
        f for f in os.listdir(first_folder)
        if os.path.isfile(os.path.join(first_folder, f))
    ])

    ids = random.sample(range(file_count), round(perc_data * file_count))
    ids.sort()
    n   = len(ids)

    print(f"Loading {n} patches from {file_count} available ({perc_data:.0%} selected)...")
    start = time.time()

    X = np.zeros((n, patch_size, patch_size, 2 * img_channels), dtype=np.float32)
    Y = np.zeros((n, patch_size, patch_size, 3),                 dtype=np.float32)

    for i, idx in enumerate(ids):
        img      = imread(os.path.join(train_images, "first",  f"{idx}.jpg"))[:, :, :img_channels]
        next_img = imread(os.path.join(train_images, "second", f"{idx}.jpg"))[:, :, :img_channels]
        X[i, :, :, :img_channels]   = img      / 255.0
        X[i, :, :, img_channels:]   = next_img / 255.0

        x_img = imread(os.path.join(train_masks, "displ_masks", "displ_x", f"{idx}.png"))
        y_img = imread(os.path.join(train_masks, "displ_masks", "displ_y", f"{idx}.png"))
        pos   = imread(os.path.join(train_masks, "pos_masks",               f"{idx}.png"))

        # positive channel minus negative channel → signed displacement
        Y[i, :, :, 0] = np.float32(x_img[:, :, 1]) - np.float32(x_img[:, :, 0])
        Y[i, :, :, 1] = np.float32(y_img[:, :, 1]) - np.float32(y_img[:, :, 0])
        # position mask normalised to [0, ~10] range (stored as 0–255 in uint8)
        Y[i, :, :, 2] = np.float32(pos) / 25.0

    print(f"Loaded {n} patches in {time.time() - start:.1f}s")

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_prop)
    return X_train, X_val, Y_train, Y_val


def make_loaders(X_train, X_val, Y_train, Y_val, batch_size: int):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(Y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader


# ─── Augmentation ─────────────────────────────────────────────────────────────

def data_augmentation(img: torch.Tensor, mask: torch.Tensor):
    """
    Apply random flips, brightness jitter, and Gaussian noise.

    img  shape: (6, H, W) – first image on channels 0:3, second on 3:6
    mask shape: (3, H, W) – [displ_x, displ_y, position]

    Displacement channels are sign-corrected after spatial flips so that
    the predicted direction stays consistent with the transformed image.
    """
    img  = img.clone()
    mask = mask.clone()

    # Vertical flip
    if random.random() > 0.5:
        img[0:3]  = TF.vflip(img[0:3])
        img[3:6]  = TF.vflip(img[3:6])
        mask[0]   = TF.vflip(mask[0])
        mask[1]   = TF.vflip(mask[1])
        mask[2]   = TF.vflip(mask[2])
        mask[1]   = torch.mul(mask[1], -1)   # y-displacement flips sign

    # Horizontal flip
    if random.random() > 0.5:
        img[0:3]  = TF.hflip(img[0:3])
        img[3:6]  = TF.hflip(img[3:6])
        mask[0]   = TF.hflip(mask[0])
        mask[1]   = TF.hflip(mask[1])
        mask[2]   = TF.hflip(mask[2])
        mask[0]   = torch.mul(mask[0], -1)   # x-displacement flips sign

    # Brightness jitter
    if random.random() > 0.5:
        brightness = random.uniform(1, 3)
        img[0:3]   = TF.adjust_brightness(img[0:3], brightness_factor=brightness)
        img[3:6]   = TF.adjust_brightness(img[3:6], brightness_factor=brightness)

    # Gaussian noise
    if random.random() > 0.5:
        img = img + torch.normal(0, 0.05, size=img.shape)

    return img, mask


# ─── Loss utilities ───────────────────────────────────────────────────────────

def l1_regularisation(model: nn.Module) -> torch.Tensor:
    """Sum of absolute values of all Conv2d weight tensors."""
    reg = torch.tensor(0.0)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            reg = reg + torch.sum(torch.abs(module.weight))
    return reg


# ─── Logging helpers ──────────────────────────────────────────────────────────

def make_run_name(cfg: dict, num_patches: int) -> str:
    p = cfg["patch_size"]
    return (
        f"{p}x{p}_{num_patches}"
        f"_mse_l1_{cfg['lambda_l1']}"
        f"_l2_{cfg['weight_decay']}"
        f"_{cfg['perc_data']}data"
        f"_test_prop_{cfg['test_prop']}"
        f"_pat{cfg['patience']}"
        f"_bat{cfg['batch_size']}"
        f"_adam_lr_{cfg['lr']}"
    )


def plot_loss(train_losses, val_losses, path: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses,   label="Validation Loss", color="magenta")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train and Validation Loss")
    ax.legend()
    plt.savefig(path)
    plt.close(fig)


def save_loss(train_losses, val_losses, path: str) -> None:
    if len(train_losses) != len(val_losses):
        raise ValueError("train_losses and val_losses must have the same length.")
    with open(path, "w") as f:
        f.write("epoch,train_loss,validation_loss\n")
        for epoch, (tl, vl) in enumerate(zip(train_losses, val_losses), start=1):
            f.write(f"{epoch},{tl},{vl}\n")


# ─── Training loop ────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    device = get_device()
    print(f"Using device: {device}")

    # Reproducibility
    random.seed(cfg["random_seed"])
    np.random.seed(cfg["random_seed"])
    torch.manual_seed(cfg["random_seed"])

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    X_train, X_val, Y_train, Y_val = load_data(cfg)
    train_loader, val_loader = make_loaders(
        X_train, X_val, Y_train, Y_val, cfg["batch_size"]
    )
    num_patches = X_train.shape[0] + X_val.shape[0]
    run_name    = make_run_name(cfg, num_patches)
    print(f"Run name: {run_name}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet().to(device)

    # ── Optimiser & loss ──────────────────────────────────────────────────────
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    # ── Training state ────────────────────────────────────────────────────────
    num_epochs   = cfg["num_epochs"]
    patience     = cfg["patience"]
    lambda_l1    = cfg["lambda_l1"]
    batch_size   = cfg["batch_size"]
    snap_start   = cfg.get("snapshot_start_epoch", 40)
    snap_interval= cfg.get("snapshot_interval", 10)

    best_loss    = float("inf")
    counter      = 0
    train_losses = []
    val_losses   = []

    output       = cfg["output_dir"]

    # ── Epoch loop ────────────────────────────────────────────────────────────
    total_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()

        for inputs, targets in train_loader:
            # (N, H, W, C) → (N, C, H, W)
            inputs  = inputs.permute(0, 3, 1, 2)
            targets = targets.permute(0, 3, 1, 2)

            for j in range(batch_size):
                inputs[j], targets[j] = data_augmentation(inputs[j], targets[j])

            inputs  = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets) + lambda_l1 * l1_regularisation(model)
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs  = val_inputs.permute(0, 3, 1, 2).to(device)
                val_targets = val_targets.permute(0, 3, 1, 2).to(device)
                val_outputs = model(val_inputs)
                val_loss   += criterion(val_outputs, val_targets) + lambda_l1 * l1_regularisation(model)

        val_loss /= len(val_loader)
        val_losses.append(val_loss.item())

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {loss.item():.6f} | "
            f"Val Loss: {val_loss.item():.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # ── Save loss curve and log every epoch ───────────────────────────────
        plot_loss(train_losses, val_losses,
                  os.path.join(output, f"displ_xy_segm_train_loss_plot_model_{run_name}.jpg"))
        save_loss(train_losses, val_losses,
                  os.path.join(output, f"model_displ_xy_segm_{run_name}.txt"))

        # ── Save latest checkpoint ────────────────────────────────────────────
        torch.save(model.state_dict(),
                   os.path.join(output, f"latest_model_displ_xy_segm_{run_name}.pt"))

        # ── Periodic snapshots ────────────────────────────────────────────────
        if epoch >= snap_start and epoch % snap_interval == 0:
            torch.save(model.state_dict(),
                       os.path.join(output, f"model_epoch_{epoch}_{run_name}.pt"))

        # ── Best model & early stopping ───────────────────────────────────────
        if val_loss < best_loss:
            best_loss = val_loss
            counter   = 0
            torch.save(model.state_dict(),
                       os.path.join(output, f"displ_xy_segm_best_model_{run_name}.pt"))
            print(f"  ✓ New best model saved (val_loss={best_loss.item():.6f})")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete in {time.time() - total_start:.1f}s")
    print(f"Best validation loss: {best_loss:.6f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train UNet for cell tracking (position + XY displacement)."
    )
    p.add_argument("--config", default="config.yaml",
                   help="Path to YAML config file (default: config.yaml)")

    # Optional overrides for every config key
    p.add_argument("--train_images",  type=str,   default=None)
    p.add_argument("--train_masks",   type=str,   default=None)
    p.add_argument("--output_dir",    type=str,   default=None)
    p.add_argument("--random_seed",   type=int,   default=None)
    p.add_argument("--perc_data",     type=float, default=None)
    p.add_argument("--test_prop",     type=float, default=None)
    p.add_argument("--batch_size",    type=int,   default=None)
    p.add_argument("--num_epochs",    type=int,   default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--weight_decay",  type=float, default=None)
    p.add_argument("--lambda_l1",     type=float, default=None)
    p.add_argument("--patience",      type=int,   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = apply_cli_overrides(cfg, args)
    train(cfg)


if __name__ == "__main__":
    main()
