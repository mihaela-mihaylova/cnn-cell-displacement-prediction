"""
segment_with_our_nn_and_predict_xy_displ.py

Runs the trained UNet on a video to predict nucleus positions and XY displacement.

Usage:
    python segment_with_our_nn_and_predict_xy_displ.py \\
        /path/to/video_folder \\
        /path/to/output_folder \\
        --model_path /path/to/best_model.pt \\
        --img_height 1104 \\
        --img_width  1376 \\
        --patch_size 128

The video folder must contain first/ and second/ subfolders.
Run generate_first_second_folder.py first if you only have raw frames.
Padding is calculated automatically from img_height, img_width, and patch_size.
"""

import math
import os

import numpy as np
import pandas as pd
import torch
from argparse import ArgumentParser
from skimage import feature
from skimage.io import imread

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from unet import UNet
from helper_functions import average_pixel_value


def get_device() -> torch.device:
    """Return the best available device (MPS → CUDA → CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("Warning: no GPU found, falling back to CPU.")
    return torch.device("cpu")

# Function to find the smallest file id number
def find_smallest_k(folder):
    files = os.listdir(folder)
    numbers = []
    for file in files:
        if file.endswith(('.jpg', '.png')):
            try:
                k = int(file.split('.')[0])
                numbers.append(k)
            except ValueError:
                continue
    return min(numbers) if numbers else None


def compute_padding(img_height, img_width, patch_size):
    padding_top   = (math.ceil(img_height / patch_size) * patch_size) - img_height
    padding_right = (math.ceil(img_width  / patch_size) * patch_size) - img_width
    return padding_top, padding_right


def predict_displacement(test_images,
                         output_path,
                         model_path,
                         img_height=1104,
                         img_width=1376,
                         patch_size=128,
                         radius=2):

    device = get_device()
    print(f"Using device: {device}")
    print(f"Processing video {test_images}")

    IMG_CHANNELS  = 3
    padding_height, padding_width = compute_padding(img_height, img_width, patch_size)
    print(f"Image size: {img_height}x{img_width} | Patch size: {patch_size} | "
          f"Padding — top: {padding_height}, right: {padding_width}")

    first_frames_path  = os.path.join(test_images, "first")
    second_frames_path = os.path.join(test_images, "second")

    if not os.path.exists(first_frames_path):
        print(f"Error: Directory {first_frames_path} does not exist.")
        return

    img_format       = [f for f in os.listdir(first_frames_path)
                        if f.endswith((".jpg", ".png"))][0].split(".")[-1]
    smallest_k_first = find_smallest_k(first_frames_path)
    num_images       = len([f for f in os.listdir(first_frames_path)
                            if f.endswith((".jpg", ".png"))])

    if num_images == 0:
        print(f"No images found in {first_frames_path}")
        return

    #-------- Load model ------------
    model = UNet()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")

    padded_height = img_height + padding_height
    padded_width  = img_width + padding_width

    print("Running inference...")
    all_points = []

    for n in range(num_images):
        frame_id = n + smallest_k_first
        print(f"Processing image {n}")

        img      = imread(os.path.join(first_frames_path,  f"{frame_id}.{img_format}"))[:, :, :IMG_CHANNELS]
        next_img = imread(os.path.join(second_frames_path, f"{frame_id}.{img_format}"))[:, :, :IMG_CHANNELS]

        x_frame = np.zeros((img_height, img_width, 2 * IMG_CHANNELS), dtype=np.float32)
        x_frame[:, :, :IMG_CHANNELS] = img      / 255.0
        x_frame[:, :, IMG_CHANNELS:] = next_img / 255.0

        padded_image = np.zeros((padded_height, padded_width, 2 * IMG_CHANNELS), dtype=np.float32)
        for c in range(2 * IMG_CHANNELS):
            padded_image[:, :, c] = np.pad(
                x_frame[:, :, c],
                ((padding_height, 0), (0, padding_width)),
                mode="constant", constant_values=0
            )

        patches = []
        h = 0
        while h < padded_height:
            w = 0
            while w < padded_width:
                patches.append(padded_image[h:h + patch_size, w:w + patch_size, :])
                w += patch_size
            h += patch_size

        patches_tensor = torch.from_numpy(
            np.array(patches).astype("float32")
        ).permute(0, 3, 1, 2).to(device)

        with torch.no_grad():
            patch_preds = model(patches_tensor)

        patch_preds = patch_preds.permute(0, 2, 3, 1).cpu().numpy().astype("float32")

        whole_image = np.zeros((padded_height, padded_width, 3), dtype=np.float32)
        i = 0
        h = 0
        while h < padded_height:
            w = 0
            while w < padded_width:
                whole_image[h:h + patch_size, w:w + patch_size, :] = patch_preds[i]
                w += patch_size
                i += 1
            h += patch_size

        cropped = whole_image[padding_height:img_height + padding_height, 0:img_width, :]

        segm_image    = cropped[:, :, 2] / np.amax(cropped[:, :, 2])
        displ_x_image = cropped[:, :, 0]
        displ_y_image = cropped[:, :, 1]

        blobs = feature.blob_log(segm_image, threshold=0.3,
                                 min_sigma=4, max_sigma=6, overlap=0.6)
        for c in blobs:
            y       = int(c[0])
            x       = int(c[1])
            displ_x = average_pixel_value(y, x, radius, displ_x_image)
            displ_y = average_pixel_value(y, x, radius, displ_y_image)
            all_points.append((n, y, x, displ_y, displ_x))

    df = pd.DataFrame(all_points, columns=["tframe", "y", "x", "displ_y", "displ_x"])
    if smallest_k_first > 0:
        df["tframe"] = df["tframe"] + smallest_k_first

    output_csv = os.path.join(output_path, f"nn_pred_{os.path.basename(test_images)}.csv")
    df.to_csv(output_csv, index=False)
    print(f"Detections saved to {output_csv}")

def main():
    parser = ArgumentParser(description="Run UNet prediction for cell positions and XY displacement.")
    parser.add_argument("test_images",  type=str,
                        help="Folder containing first/ and second/ subfolders.")
    parser.add_argument("output_path",  type=str,
                        help="Folder where the output CSV will be saved.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model .pt file.")
    parser.add_argument("--img_height", type=int, default=1104,
                        help="Image height in pixels (default: 1104).")
    parser.add_argument("--img_width",  type=int, default=1376,
                        help="Image width in pixels (default: 1376).")
    parser.add_argument("--patch_size", type=int, default=128,
                        help="Patch size used during training (default: 128). "
                             "Padding is calculated automatically from this.")
    parser.add_argument("--radius", type=int, default=2,
                            help="Radius for the blob used to calculate the displacement.")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    predict_displacement(
        test_images = args.test_images,
        output_path = args.output_path,
        model_path  = args.model_path,
        img_height  = args.img_height,
        img_width   = args.img_width,
        patch_size  = args.patch_size,
        radius = args.radius
    )


if __name__ == "__main__":
    main()


