"""
dataset.py – PyTorch Dataset for paired (first, second) image patches and
their displacement / position masks.

Wraps the numpy arrays produced by train.load_data() (X: image patches,
Y: displacement/position masks) and applies the same augmentations as
train.data_augmentation, performed per-sample in __getitem__ rather than
on the assembled batch.
"""
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from utils import seed_worker
import os
from skimage.io import imread
from sklearn.model_selection import train_test_split
import time

# ─── Augmentation ─────────────────────────────────────────────────────────────
def data_augmentation(img: torch.Tensor,
                       mask: torch.Tensor,
                       p_v_flip: float = 0.5,
                       p_h_flip: float = 0.5,
                       p_brightness: float = 0.5,
                       p_noise: float = 0.5,
                       rng: random.Random = random):
    """
    Apply random vertical/horizontal flips, brightness jitter, and additive
    Gaussian noise.

    img  shape: (6, H, W) – first frame on channels 0:3, second on 3:6
    mask shape: (3, H, W) – [displ_x, displ_y, position]

    Each p_* argument is the probability with which that transform is
    applied. `rng` is the random generator to draw decisions from -- pass
    a dedicated random.Random(seed) instance to make this call's outcomes
    independent of anything else in the program that uses the `random`
    module; defaults to that shared global module if not given.
    Displacement channels are sign-corrected after spatial flips so that
    the predicted direction remains consistent with the transformed image
    pair.
    """
    img  = img.clone()
    mask = mask.clone()

    # Vertical flip
    if rng.random() < p_v_flip:
        img[0:3] = TF.vflip(img[0:3])
        img[3:6] = TF.vflip(img[3:6])
        mask[0]  = TF.vflip(mask[0])
        mask[1]  = TF.vflip(mask[1])
        mask[2]  = TF.vflip(mask[2])
        mask[1]  = torch.mul(mask[1], -1)   # y-displacement flips sign

    # Horizontal flip
    if rng.random() < p_h_flip:
        img[0:3] = TF.hflip(img[0:3])
        img[3:6] = TF.hflip(img[3:6])
        mask[0]  = TF.hflip(mask[0])
        mask[1]  = TF.hflip(mask[1])
        mask[2]  = TF.hflip(mask[2])
        mask[0]  = torch.mul(mask[0], -1)   # x-displacement flips sign

    # Brightness jitter
    if rng.random() < p_brightness:
        brightness = rng.uniform(1, 3)
        img[0:3] = TF.adjust_brightness(img[0:3], brightness_factor=brightness)
        img[3:6] = TF.adjust_brightness(img[3:6], brightness_factor=brightness)

    # Additive Gaussian noise
    if rng.random() < p_noise:
        img = img + torch.normal(0, 0.05, size=img.shape)

    return img, mask

# ─── Load data ─────────────────────────────────────────────────────────────

def load_data(cfg: dict, seed: int):
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

    first_folder = os.path.join(train_images, "first")
    file_count   = len([
        f for f in os.listdir(first_folder)
        if os.path.isfile(os.path.join(first_folder, f))
    ])

    # A local, dedicated generator -- so which patch ids get selected depends
    # only on `seed`, regardless of anything else in the program that uses
    # the shared `random` module.
    rng = random.Random(seed)
    ids = rng.sample(range(file_count), round(perc_data * file_count))
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

    return X, Y

def split_data(X, Y, val_prop: float, seed: int):
    """Partition (X, Y) into train/val, explicitly and reproducibly seeded."""
    return train_test_split(X, Y, test_size=val_prop, random_state=seed)

# ─── Dataset ──────────────────────────────────────────────────────────────────

class CellDisplDataset(Dataset):
    """
    X : (N, H, W, 6) float32 - first + second image stacked on channels
    Y : (N, H, W, 3) float32 - [displ_x, displ_y, position_mask]

    The channel-last to channel-first permute is performed per-sample in
    __getitem__
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, augment: bool = False,
                 seed: int = None,
                 p_v_flip: float = 0.5, p_h_flip: float = 0.5,
                 p_brightness: float = 0.5, p_noise: float = 0.5,
                 debug: bool = False):
        """
        Parameters
        ----------
        X, Y         : input arrays
        augment      : whether to apply data_augmentation() in __getitem__.
        seed         : if given, augmentation draws from a private generator
                       seeded with this value, independent of anything else
                       in the program that uses the `random` module. If not
                       given, augmentation falls back to that shared module.
        p_v_flip, p_h_flip, p_brightness, p_noise :
                       per-transform probabilities, forwarded to
                       data_augmentation().
        debug        : if True, print the shape of each sample as it is
                       retrieved.
        """
        self.X = X
        self.Y = Y
        self.augment = augment
        self.rng = random.Random(seed) if seed is not None else random
        self.p_v_flip = p_v_flip
        self.p_h_flip = p_h_flip
        self.p_brightness = p_brightness
        self.p_noise = p_noise
        self.debug = debug

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        """Return the (image, mask) tensor pair at index idx, augmented if enabled."""
        img  = torch.from_numpy(self.X[idx]).permute(2, 0, 1)  # (H,W,6) -> (6,H,W)
        mask = torch.from_numpy(self.Y[idx]).permute(2, 0, 1)  # (H,W,3) -> (3,H,W)

        if self.augment:
            img, mask = data_augmentation(
                img, mask,
                p_v_flip=self.p_v_flip,
                p_h_flip=self.p_h_flip,
                p_brightness=self.p_brightness,
                p_noise=self.p_noise,
                rng=self.rng,
            )

        if self.debug:
            print(f"[{idx}] img={tuple(img.shape)} mask={tuple(mask.shape)}")

        return img, mask

def make_dataloader(X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool,
                     seed: int, augment: bool = False, p_v_flip: float = 0.5,
                     p_h_flip: float = 0.5, p_brightness: float = 0.5,
                     p_noise: float = 0.5, debug: bool = False):
    """
    Construct a CellDisplDataset over (X, Y) and wrap it in a DataLoader.

    Shuffling is driven by a dedicated generator, seeded independently of
    the global RNGs, so batch order is reproducible without being coupled
    to any other source of randomness in the run (e.g. augmentation).
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    dataset = CellDisplDataset(
        X, Y,
        augment=augment,
        seed=seed,
        p_v_flip=p_v_flip,
        p_h_flip=p_h_flip,
        p_brightness=p_brightness,
        p_noise=p_noise,
        debug=debug,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return loader, dataset