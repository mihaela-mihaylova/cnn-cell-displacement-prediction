"""
split_images_to_patches.py

Splits padded image pairs and their corresponding masks into fixed-size patches.
Only patches that contain at least one nucleus (i.e. where the position mask is
non-zero) are saved — empty background patches are discarded.

A CSV log (patch_record.csv) is saved to output_dir recording which original
image each patch came from, useful for debugging.

Usage:
    python split_images_to_patches.py \\
        --input_dir        /path/to/padded_train/padded_input \\
        --output_dir       /path/to/patches_train \\
        --img_height       1104 \\
        --img_width        1376 \\
        --patch_size       128 \\
        --img_channels     3

Expected input structure:
    input_dir/
        padded_first/    0.jpg, 1.jpg, ...
        padded_second/   0.jpg, 1.jpg, ...
        padded_pos_masks/          0.png, ...
        padded_displ_masks/
            padded_displ_x/        0.png, ...
            padded_displ_y/        0.png, ...

Output structure:
    output_dir/
        input/
            first/     0.jpg, 1.jpg, ...
            second/    0.jpg, 1.jpg, ...
        output/
            pos_masks/             0.png, ...
            displ_masks/
                displ_x/           0.png, ...
                displ_y/           0.png, ...
        patch_record.csv
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
from skimage.io import imread, imsave


def compute_padding(img_height, img_width, patch_size):
    padding_top   = (math.ceil(img_height / patch_size) * patch_size) - img_height
    padding_right = (math.ceil(img_width  / patch_size) * patch_size) - img_width
    return padding_top, padding_right


def make_output_dirs(output_dir):
    dirs = {
        "first":   os.path.join(output_dir, "input",  "first"),
        "second":  os.path.join(output_dir, "input",  "second"),
        "pos":     os.path.join(output_dir, "output", "pos_masks"),
        "displ_x": os.path.join(output_dir, "output", "displ_masks", "displ_x"),
        "displ_y": os.path.join(output_dir, "output", "displ_masks", "displ_y"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def split_to_patches(input_dir, output_dir, img_height, img_width, patch_size, img_channels):

    padding_top, padding_right = compute_padding(img_height, img_width, patch_size)
    padded_height = img_height + padding_top
    padded_width  = img_width  + padding_right

    print(
        f"Image size: {img_height}x{img_width} | "
        f"Padded: {padded_height}x{padded_width} | "
        f"Patch size: {patch_size}"
    )

    dirs = make_output_dirs(output_dir)

    first_dir = os.path.join(input_dir, "padded_first")
    file_count = len([
        f for f in os.listdir(first_dir)
        if os.path.isfile(os.path.join(first_dir, f))
    ])
    print(f"Found {file_count} image pairs to patch...")

    patch_record = []
    existing = [int(f.split('.')[0]) for f in os.listdir(dirs["first"]) if f.endswith('.jpg')]
    patch_id = max(existing) + 1 if existing else 0

    for i in range(file_count):
        img      = imread(os.path.join(input_dir, "padded_first",  f"{i}.jpg"))[:, :, :img_channels]
        next_img = imread(os.path.join(input_dir, "padded_second", f"{i}.jpg"))[:, :, :img_channels]
        x_img    = imread(os.path.join(input_dir, "padded_displ_masks", "padded_displ_x", f"{i}.png"))
        y_img    = imread(os.path.join(input_dir, "padded_displ_masks", "padded_displ_y", f"{i}.png"))
        pos_img  = imread(os.path.join(input_dir, "padded_pos_masks", f"{i}.png"))

        patch_within_image = 0
        h = 0
        while h < padded_height:
            w = 0
            while w < padded_width:
                pos_patch = pos_img[h:h + patch_size, w:w + patch_size]

                # Skip patches with no nucleus
                if np.any(pos_patch):
                    img_patch      = img[h:h + patch_size,     w:w + patch_size, :]
                    next_img_patch = next_img[h:h + patch_size, w:w + patch_size, :]
                    x_patch        = x_img[h:h + patch_size,   w:w + patch_size, :]
                    y_patch        = y_img[h:h + patch_size,   w:w + patch_size, :]

                    imsave(os.path.join(dirs["first"],   f"{patch_id}.jpg"), img_patch.astype(np.uint8))
                    imsave(os.path.join(dirs["second"],  f"{patch_id}.jpg"), next_img_patch.astype(np.uint8))
                    imsave(os.path.join(dirs["pos"],     f"{patch_id}.png"), pos_patch.astype(np.uint8))
                    imsave(os.path.join(dirs["displ_x"], f"{patch_id}.png"), x_patch.astype(np.uint8))
                    imsave(os.path.join(dirs["displ_y"], f"{patch_id}.png"), y_patch.astype(np.uint8))

                    patch_record.append((patch_id, patch_within_image, i))
                    patch_id += 1

                patch_within_image += 1
                w += patch_size
            h += patch_size

    df = pd.DataFrame(patch_record, columns=["id", "id_within_image", "orig_image_id"])
    record_path = os.path.join(output_dir, "patch_record.csv")
    df.to_csv(record_path, index=False)

    print(f"Done. Saved {patch_id} patches to {output_dir}")
    print(f"Patch record saved to {record_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split padded image pairs and masks into fixed-size patches."
    )
    parser.add_argument("--input_dir",    required=True,
                        help="Folder containing padded_first/, padded_second/, "
                             "padded_pos_masks/, and padded_displ_masks/.")
    parser.add_argument("--output_dir",   required=True,
                        help="Folder where input/ and output/ patch folders will be created.")
    parser.add_argument("--img_height",   type=int, default=1104,
                        help="Original (unpadded) image height in pixels (default: 1104).")
    parser.add_argument("--img_width",    type=int, default=1376,
                        help="Original (unpadded) image width in pixels (default: 1376).")
    parser.add_argument("--patch_size",   type=int, default=128,
                        help="Patch size in pixels; patches are square (default: 128).")
    parser.add_argument("--img_channels", type=int, default=3,
                        help="Number of image channels (default: 3).")
    args = parser.parse_args()

    split_to_patches(
        input_dir    = args.input_dir,
        output_dir   = args.output_dir,
        img_height   = args.img_height,
        img_width    = args.img_width,
        patch_size   = args.patch_size,
        img_channels = args.img_channels,
    )


if __name__ == "__main__":
    main()
