"""
pad_images.py

Pads a folder of images with zeros so that their dimensions are divisible by
the patch size used during training. Padding is added to the top and right
edges only, to keep the origin (top-left corner) consistent.

The required padding is calculated automatically from img_height, img_width,
and patch_size — you do not need to specify padding_top or padding_right
manually.

Usage:
    python pad_images.py \\
        --input_dir   /path/to/train/input/second \\
        --output_dir  /path/to/padded_train/padded_input/padded_second \\
        --img_height  1104 \\
        --img_width   1376 \\
        --patch_size  128 \\
        --img_channels 3 \\
        --format      jpg
"""

import argparse
import math
import os

import numpy as np
from skimage.io import imread, imsave


def compute_padding(img_height, img_width, patch_size):
    """Return (padding_top, padding_right) needed to make dimensions divisible by patch_size."""
    padding_top   = (math.ceil(img_height / patch_size) * patch_size) - img_height
    padding_right = (math.ceil(img_width  / patch_size) * patch_size) - img_width
    return padding_top, padding_right


def pad_images(input_dir, output_dir, img_height, img_width, patch_size, img_channels, fmt):
    os.makedirs(output_dir, exist_ok=True)

    padding_top, padding_right = compute_padding(img_height, img_width, patch_size)
    print(
        f"Image size: {img_height}x{img_width} | "
        f"Patch size: {patch_size} | "
        f"Padding — top: {padding_top}, right: {padding_right}"
    )

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(f".{fmt}")
    ])

    if not image_files:
        print(f"No .{fmt} files found in {input_dir}")
        return

    print(f"Padding {len(image_files)} images...")

    for fname in image_files:
        img = imread(os.path.join(input_dir, fname))

        if img_channels == 1:
            # Grayscale: pad the 2D array directly
            padded = np.pad(img, ((padding_top, 0), (0, padding_right)),
                            mode="constant", constant_values=0)
        else:
            # Colour: pad each channel independently
            padded = np.zeros(
                (img_height + padding_top, img_width + padding_right, img_channels),
                dtype="uint8"
            )
            for c in range(img_channels):
                padded[:, :, c] = np.pad(
                    img[:, :, c],
                    ((padding_top, 0), (0, padding_right)),
                    mode="constant", constant_values=0
                )

        imsave(os.path.join(output_dir, fname), padded)

    print(f"Done. Padded images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Pad images so their dimensions are divisible by patch_size."
    )
    parser.add_argument("--input_dir",    required=True,
                        help="Folder containing images to pad.")
    parser.add_argument("--output_dir",   required=True,
                        help="Folder to save padded images.")
    parser.add_argument("--img_height",   type=int, default=1104,
                        help="Original image height in pixels (default: 1104).")
    parser.add_argument("--img_width",    type=int, default=1376,
                        help="Original image width in pixels (default: 1376).")
    parser.add_argument("--patch_size",   type=int, default=128,
                        help="Patch size used during training (default: 128). "
                             "Padding is calculated automatically from this.")
    parser.add_argument("--img_channels", type=int, default=3,
                        help="Number of image channels: 3 for RGB, 1 for grayscale (default: 3).")
    parser.add_argument("--format",       type=str, default="jpg",
                        help="Image file extension without dot, e.g. jpg or png (default: jpg).")
    args = parser.parse_args()

    pad_images(
        input_dir    = args.input_dir,
        output_dir   = args.output_dir,
        img_height   = args.img_height,
        img_width    = args.img_width,
        patch_size   = args.patch_size,
        img_channels = args.img_channels,
        fmt          = args.format,
    )


if __name__ == "__main__":
    main()
