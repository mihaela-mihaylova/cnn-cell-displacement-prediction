"""
predict.py

Runs the trained UNet on a video to predict nucleus positions and XY displacement.

Usage:
    python predict.py \\
        /path/to/video_folder \\
        /path/to/output_folder \\
        --model_path /path/to/best_model.pt \\
        --img_height 1104 \\
        --img_width  1376 \\
        --patch_size 128

    # using existing ground-truth detections instead of blob detection:
    python predict.py \\
        /path/to/video_folder \\
        /path/to/output_folder \\
        --model_path /path/to/best_model.pt \\
        --detections_csv /path/to/positions.csv

    # also save per-frame QC images (frame, position, displ_x, displ_y panels):
    python predict.py \\
        /path/to/video_folder \\
        /path/to/output_folder \\
        --model_path /path/to/best_model.pt \\
        --save_qc

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
from matplotlib import pyplot as plt
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


def save_qc_image(img, cropped, output_path):
    """
    Save the raw frame alongside its raw predicted channels (position mask,
    displ_x, displ_y). displ_x/displ_y are signed (positive vs. negative
    displacement), so they use a diverging blue-white-red colormap,
    symmetric around zero -- blue = negative, red = positive, white = ~0 --
    with a colorbar showing the displacement scale each color corresponds to.
    """
    displ_x = cropped[:, :, 0]
    displ_y = cropped[:, :, 1]
    vmax = max(np.abs(displ_x).max(), np.abs(displ_y).max())

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(img)
    axes[0].set_title("frame")
    axes[1].imshow(cropped[:, :, 2])
    axes[1].set_title("position")

    im_x = axes[2].imshow(displ_x, cmap="bwr", vmin=-vmax, vmax=vmax)
    axes[2].set_title("displ_x")
    fig.colorbar(im_x, ax=axes[2], fraction=0.046, pad=0.04)

    im_y = axes[3].imshow(displ_y, cmap="bwr", vmin=-vmax, vmax=vmax)
    axes[3].set_title("displ_y")
    fig.colorbar(im_y, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def predict_displacement(test_images,
                         output_path,
                         model_path,
                         img_height=1104,
                         img_width=1376,
                         patch_size=128,
                         radius=2,
                         detections_csv=None,
                         save_qc=False,
                         qc_output_dir=None):

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

    # -------- Use existing detections instead of blob detection, if given ------
    available_coords = None
    if detections_csv is not None:
        available_coords = pd.read_csv(detections_csv)
        print(f"Using existing detections from {detections_csv} "
              f"({len(available_coords)} rows) instead of blob detection.")

    # -------- QC output setup ---------------------------------------------------
    if save_qc:
        if qc_output_dir is None:
            qc_output_dir = os.path.join(output_path, f"qc_{os.path.basename(test_images)}")
        os.makedirs(qc_output_dir, exist_ok=True)
        print(f"QC images will be saved to {qc_output_dir}")

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

        # -------- Extract points: from existing detections, or blob detection ---
        if available_coords is not None:
            frame_rows = available_coords.loc[available_coords["tframe"] == frame_id]
            for _, row in frame_rows.iterrows():
                y = int(row["y"])
                x = int(row["x"])
                displ_x = average_pixel_value(y, x, radius, displ_x_image)
                displ_y = average_pixel_value(y, x, radius, displ_y_image)
                all_points.append((n, y, x, displ_y, displ_x))
        else:
            blobs = feature.blob_log(segm_image, threshold=0.3,
                                     min_sigma=4, max_sigma=6, overlap=0.6)
            for c in blobs:
                y       = int(c[0])
                x       = int(c[1])
                displ_x = average_pixel_value(y, x, radius, displ_x_image)
                displ_y = average_pixel_value(y, x, radius, displ_y_image)
                all_points.append((n, y, x, displ_y, displ_x))

        if save_qc:
            save_qc_image(img, cropped, os.path.join(qc_output_dir, f"qc_{frame_id}.jpg"))

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
    parser.add_argument("--detections_csv", type=str, default=None,
                        help="Path to a CSV of existing ground-truth detections "
                             "(columns: tframe, y, x). If given, displacement is "
                             "computed at these positions instead of running blob "
                             "detection on the predicted position mask.")
    parser.add_argument("--save_qc", action="store_true",
                        help="Also save one QC image per frame: the raw frame plus "
                             "the predicted position/displ_x/displ_y channels, for "
                             "visual inspection.")
    parser.add_argument("--qc_output_dir", type=str, default=None,
                        help="Where to save QC images (default: <output_path>/qc_<video_name>/).")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    predict_displacement(
        test_images     = args.test_images,
        output_path     = args.output_path,
        model_path      = args.model_path,
        img_height      = args.img_height,
        img_width       = args.img_width,
        patch_size      = args.patch_size,
        radius          = args.radius,
        detections_csv  = args.detections_csv,
        save_qc         = args.save_qc,
        qc_output_dir   = args.qc_output_dir,
    )


if __name__ == "__main__":
    main()