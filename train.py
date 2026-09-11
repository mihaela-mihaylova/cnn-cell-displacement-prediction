"""
train.py  –  Train the UNet to predict cell positions and XY displacement.

Usage:
    python train.py                          # uses config.yaml in the same folder
    python train.py --config path/to/config.yaml
    python train.py --config config.yaml --lr 0.001 --batch_size 8   # CLI overrides
    python train.py --resume output/latest_model_displ_xy_segm_<run_name>.pt

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
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from unet import UNet
from utils import set_seed, seed_worker
from dataset import CellDisplDataset, make_dataloader, load_data, split_data
from loss import combined_loss

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
        "val_prop":     args.val_prop,
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


# ─── Logging helpers ──────────────────────────────────────────────────────────

def make_run_name(cfg: dict) -> str:
    run_name = cfg["output_dir"].split('/')[-1]
    return run_name

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

def train(cfg: dict, resume_path: str = None) -> None:
    device = get_device()

    seed = cfg["random_seed"]
    set_seed(seed)                 # seeds python/numpy/torch RNGs globally, BEFORE
                                   # anything random happens -- model init, dropout, etc.
                                   # seed_worker (used inside make_dataloader) is a
                                   # separate, narrower thing: it only reseeds each
                                   # DataLoader worker process's RNG for reproducible
                                   # shuffling/augmentation, and doesn't cover this.

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # load data
    X, Y = load_data(cfg, seed)
    X_train, X_val, Y_train, Y_val = split_data(X, Y, cfg['val_prop'], seed=seed)

    train_loader, train_dataset = make_dataloader(
        X_train, Y_train, batch_size=cfg["batch_size"], shuffle=True, seed=seed,
        augment=cfg.get("augment", False),
    )
    val_loader, val_dataset = make_dataloader(
        X_val, Y_val, batch_size=cfg["batch_size"], shuffle=False, seed=seed,
        augment=False  # for reproducability augment should be false in val
    )

    num_patches = len(train_dataset) + len(val_dataset)
    run_name    = make_run_name(cfg)
    print(f"Run name: {run_name}")

    # save config parameters inside run's folder
    cfg_to_save = {**cfg, "num_patches": num_patches}
    with open(f"{cfg['output_dir']}/config.yaml", "w") as f:
        yaml.dump(cfg_to_save, f)  # save a copy alongside the run's outputs, for reproducibility


    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet().to(device)

    # ── Optimiser ──────────────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    # ── Training state (overwritten below if resuming) ───────────────────────
    start_epoch  = 0
    best_loss    = float("inf")
    counter      = 0
    train_losses = []
    val_losses   = []

    if resume_path:
        print(f"resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        best_loss    = ckpt.get("best_loss", float("inf"))
        counter      = ckpt.get("counter", 0)
        train_losses = ckpt.get("train_losses", [])
        val_losses   = ckpt.get("val_losses", [])
        print(f"resuming at epoch {start_epoch}, best_loss so far: {best_loss:.6f}")

    num_epochs    = cfg["num_epochs"]
    patience      = cfg["patience"]
    lambda_l1     = cfg["lambda_l1"]
    snap_start    = cfg.get("snapshot_start_epoch", 40)
    snap_interval = cfg.get("snapshot_interval", 10)

    output = cfg["output_dir"]

    # ── Per-epoch pass over one loader ────────────────────────────────────────
    def run_epoch(loader, train: bool) -> float:
        """
        Run one full pass over `loader`, either training (train=True: computes
        gradients and steps the optimizer) or evaluating (train=False: forward
        pass only). Returns the loss averaged over all batches in the pass.
        """
        model.train() if train else model.eval()
        total_loss, n_batches = 0.0, 0

        for inputs, targets in loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            if train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                outputs = model(inputs)
                loss = combined_loss(outputs, targets, model, lambda_l1)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    # ── Epoch loop ────────────────────────────────────────────────────────────
    total_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        train_loss = run_epoch(train_loader, train=True)
        train_losses.append(train_loss)

        val_loss = run_epoch(val_loader, train=False)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # ── Save loss curve and log every epoch ───────────────────────────────
        plot_loss(train_losses, val_losses,
                  os.path.join(output, f"displ_xy_segm_train_loss_plot_model_{run_name}.jpg"))
        save_loss(train_losses, val_losses,
                  os.path.join(output, f"model_displ_xy_segm_{run_name}.txt"))

        # ── Best-model / early-stopping bookkeeping ───────────────────────────
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            counter   = 0
        else:
            counter += 1

        # ── One resumable checkpoint, reused for latest/periodic/best ────────
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "counter": counter,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

        # ── Save latest checkpoint ────────────────────────────────────────────
        torch.save(checkpoint,
                   os.path.join(output, f"latest_model_displ_xy_segm_{run_name}.pt"))

        # ── Periodic snapshots ────────────────────────────────────────────────
        if epoch >= snap_start and epoch % snap_interval == 0:
            torch.save(checkpoint,
                       os.path.join(output, f"model_epoch_{epoch}_{run_name}.pt"))

        # ── Best model & early stopping ───────────────────────────────────────
        if is_best:
            torch.save(checkpoint,
                       os.path.join(output, f"displ_xy_segm_best_model_{run_name}.pt"))
            print(f"  ✓ New best model saved (val_loss={best_loss:.6f})")
        elif counter >= patience:
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
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint (e.g. latest_model_..._{run_name}.pt) to resume from")

    # Optional overrides for every config key
    p.add_argument("--train_images",  type=str,   default=None)
    p.add_argument("--train_masks",   type=str,   default=None)
    p.add_argument("--output_dir",    type=str,   default=None)
    p.add_argument("--random_seed",   type=int,   default=None)
    p.add_argument("--perc_data",     type=float, default=None)
    p.add_argument("--val_prop",     type=float, default=None)
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
    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()