"""
dataset.py - Data pipeline for cell-tracking patches: paired (first, second)
image frames and their displacement/position targets.

Patches are identified by integer ids assigned during data preparation
(0, 1, ..., N-1). extract_idx() selects which ids are used for a given
config, optionally subsampling via perc_data; split_data() partitions
those ids into train and validation sets. CellDisplDataset reads and
normalizes each patch's image and mask files from disk lazily, one
sample at a time, in __getitem__, rather than loading the full dataset
into memory upfront, and applies data_augmentation() to the training
split. make_dataloader() constructs a CellDisplDataset and wraps it in
a PyTorch DataLoader with reproducible shuffling.
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

def extract_idx(cfg: dict, seed: int):
    """
    Extract the ids of the files that we use for training, in case perc_data < 1
    """
    train_images = cfg["train_images"]
    perc_data    = cfg["perc_data"]

    first_folder = os.path.join(train_images, "first")
    file_count   = len([
        f for f in os.listdir(first_folder)
        if (os.path.isfile(os.path.join(first_folder, f)) and f.endswith('.jpg'))
    ])

    # A local, dedicated generator -- so which patch ids get selected depends
    # only on `seed`, regardless of anything else in the program that uses
    # the shared `random` module.
    rng = random.Random(seed)
    # This sampling is without replacement.
    ids = rng.sample(range(file_count), round(perc_data * file_count))
    ids.sort()

    return ids


def load_patch(cfg:dict, idx):
    """Loads a patch img, corresponding to idx x. 
    Only works because our image ids start from 0 through N-1.
    Returns img, next img, pos, x_img, y_img.
    """
    train_images = cfg["train_images"]
    train_masks  = cfg["train_masks"]
    img_channels = cfg["img_channels"]

    # img patch
    img = imread(os.path.join(train_images, "first",  f"{idx}.jpg"))[:, :, :img_channels]
    next_img = imread(os.path.join(train_images, "second", f"{idx}.jpg"))[:, :, :img_channels]

    # mask patch
    x_img = imread(os.path.join(train_masks, "displ_masks", "displ_x", f"{idx}.png"))
    y_img = imread(os.path.join(train_masks, "displ_masks", "displ_y", f"{idx}.png"))
    pos   = imread(os.path.join(train_masks, "pos_masks",               f"{idx}.png"))

    return img, next_img, pos, x_img, y_img

def normalize_patch(cfg:dict,
                    img, next_img,
                    pos, x_img, y_img):
    """ 
    Takes in patches - img, next_img (first and second patch), and pos, x_img, y_img (mask patches).
    Returns X and Y - normalized patches,
    Each X has shape (N, H, W, 6): first + second image stacked on channels.
    Each Y has shape (N, H, W, 3): [displ_x, displ_y, position_mask]."""
    
    img_channels = cfg["img_channels"]
    patch_size   = cfg["patch_size"]
    
    X = np.zeros((patch_size, patch_size, 2 * img_channels), dtype=np.float32)
    Y = np.zeros((patch_size, patch_size, 3),                 dtype=np.float32)
    
    X[:, :, :img_channels]   = img      / 255.0
    X[:, :, img_channels:]   = next_img / 255.0

    # positive channel minus negative channel → signed displacement
    Y[:, :, 0] = np.float32(x_img[:, :, 1]) - np.float32(x_img[:, :, 0])
    Y[:, :, 1] = np.float32(y_img[:, :, 1]) - np.float32(y_img[:, :, 0])
    # position mask normalised to [0, ~10] range (stored as 0–255 in uint8)
    Y[:, :, 2] = np.float32(pos) / 25.0

    return X, Y


def split_data(ids, val_prop: float, seed: int):
    """ Takes in ids drawn in extract_idx.
    Partitions them into ids for training and val. Explicitly and reproducibly seeded."""
    train_ids,val_ids = train_test_split(ids, test_size=val_prop, random_state=seed)
    return train_ids, val_ids

# ─── Dataset ──────────────────────────────────────────────────────────────────

class CellDisplDataset(Dataset):
    """
    X : (N, H, W, 6) float32 - first + second image stacked on channels
    Y : (N, H, W, 3) float32 - [displ_x, displ_y, position_mask]

    The channel-last to channel-first permute is performed per-sample in
    __getitem__
    """

    def __init__(self, list_ids:list, cfg:dict, augment: bool = False,
                 seed: int = None,
                 p_v_flip: float = 0.5, p_h_flip: float = 0.5,
                 p_brightness: float = 0.5, p_noise: float = 0.5,
                 debug: bool = False):
        """
        Parameters
        ----------
        list_ids     : patch ids belonging to this split (train or val), as
                    produced by extract_idx() / split_data().
        cfg          : the run's config dict (paths, img_channels, patch_size, ...).
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
        self.list_ids = list_ids
        self.cfg = cfg
        self.augment = augment
        self.rng = random.Random(seed) if seed is not None else random
        self.p_v_flip = p_v_flip
        self.p_h_flip = p_h_flip
        self.p_brightness = p_brightness
        self.p_noise = p_noise
        self.debug = debug

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.list_ids)

    def __getitem__(self, idx: int):
        """Return the (image, mask) tensor pair at index idx, augmented if enabled."""
        patch_id = self.list_ids[idx]
        first_img, sec_img, pos, x_img, y_img = load_patch(self.cfg, patch_id)
        X, Y = normalize_patch(self.cfg, first_img, sec_img, pos, x_img, y_img)
        img  = torch.from_numpy(X).permute(2, 0, 1)  # (H,W,6) -> (6,H,W)
        mask = torch.from_numpy(Y).permute(2, 0, 1)  # (H,W,3) -> (3,H,W)

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

def make_dataloader(list_ids:list, cfg:dict, batch_size: int, shuffle: bool,
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
        list_ids = list_ids,
        augment=augment,
        cfg=cfg,
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