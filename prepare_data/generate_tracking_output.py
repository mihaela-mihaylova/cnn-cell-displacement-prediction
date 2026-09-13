"""
generate_tracking_output.py

Converts a tracking CSV into three sets of image masks used as training targets:
  - Position masks      : a disk of radius 10px drawn at each nucleus centre
  - X-displacement masks: signed x-displacement encoded into two colour channels
  - Y-displacement masks: signed y-displacement encoded into two colour channels

Displacement encoding (same convention used everywhere in this pipeline):
  Positive displacement → stored in channel 1 (green)
  Negative displacement → stored in channel 0 (red) as its absolute value

Usage:
    python generate_tracking_output.py \\
        --path_to_csv            /path/to/tracking.csv \\
        --output_dir     /path/to/output \\
        --img_height     1104 \\
        --img_width      1376 \\
        --for_training

Arguments:
    --path_to_csv    Path to tracking CSV.
                     Required columns: tframe, y, x, track, displ_y, displ_x
    --output_dir     Root folder where pos_masks/, displ_masks/displ_x/,
                     and displ_masks/displ_y/ will be created.
    --img_height     Height of the training images in pixels (default: 1104).
    --img_width      Width of the training images in pixels (default: 1376).
    --for_training   If given, skip the last frame of each video when
                     generating masks (there's no future frame to measure
                     displacement to). Omit for inference-style output that
                     keeps every frame.
"""

import argparse
import os

import numpy as np
import pandas as pd
from skimage.draw import disk
from skimage.io import imsave
from merge_csv import get_prefix_number
from datetime import datetime
import yaml

def generate_masks(df, output_dir, img_height, img_width, first_frame_id=0, for_training=True):

    # Sanity check: do coordinates fit inside the image after scaling?
    max_x = max(df["x"].values)
    max_y = max(df["y"].values)
    if max_x >= img_width or max_y >= img_height:
        print(
            f"Warning: some coordinates exceed image dimensions. "
            f"Check that --img_height/--img_width are correct."
        )

    # ── Create output directories ─────────────────────────────────────────────
    pos_dir     = os.path.join(output_dir, "pos_masks")
    displ_x_dir = os.path.join(output_dir, "displ_masks", "displ_x")
    displ_y_dir = os.path.join(output_dir, "displ_masks", "displ_y")
    for d in [pos_dir, displ_x_dir, displ_y_dir]:
        os.makedirs(d, exist_ok=True)

    if for_training:
        # don't generate output frames for the very last frame of a video,
        # as we don't by definition have displacement info about it
        num_frames = len(df["tframe"].unique()) - 1
        print("Creating output for training...")
    else:
        num_frames = len(df["tframe"].unique())

    print(f"Generating masks for {num_frames} frames ({img_height}x{img_width})...")

    for i in range(num_frames):
        image_pos     = np.zeros((img_height, img_width),     dtype="uint8")
        image_displ_x = np.zeros((img_height, img_width, 3),  dtype="uint8")
        image_displ_y = np.zeros((img_height, img_width, 3),  dtype="uint8")

        for _, row in df[df["tframe"] == i].iterrows():
            displ_x  = row["displ_x"]
            displ_y  = row["displ_y"]
            centre_y = round(row["y"])
            centre_x = round(row["x"])

            rr, cc = disk((centre_y, centre_x), radius=10)

            # Clip to image bounds
            valid  = (rr >= 0) & (rr < img_height) & (cc >= 0) & (cc < img_width)
            rr, cc = rr[valid], cc[valid]

            image_pos[rr, cc] = 255

            # Positive → channel 1, negative → channel 0 (absolute value)
            if displ_x < 0:
                image_displ_x[rr, cc, 0] = abs(displ_x)
            else:
                image_displ_x[rr, cc, 1] = displ_x

            if displ_y < 0:
                image_displ_y[rr, cc, 0] = abs(displ_y)
            else:
                image_displ_y[rr, cc, 1] = displ_y

        file_id = i + first_frame_id
        imsave(os.path.join(pos_dir,     f"{file_id}.png"), image_pos)
        imsave(os.path.join(displ_x_dir, f"{file_id}.png"), image_displ_x)
        imsave(os.path.join(displ_y_dir, f"{file_id}.png"), image_displ_y)

    print(f"Done. Masks saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a tracking CSV into position and displacement image masks."
    )
    parser.add_argument("--path_to_csv", required=True,
                        help="Path to tracking CSV file.")
    parser.add_argument("--output_dir", required=True,
                        help="Root folder for output masks.")
    parser.add_argument("--img_height", type=int, default=1104,
                        help="Image height in pixels (default: 1104).")
    parser.add_argument("--img_width", type=int, default=1376,
                        help="Image width in pixels (default: 1376).")
    parser.add_argument("--for_training", action="store_true",
                        help="Skip the last frame of each video (default: True).")

    args = parser.parse_args()

    path_to_csv = args.path_to_csv
    output_dir = args.output_dir
    img_height = args.img_height
    img_width = args.img_width
    for_training = args.for_training

    df = pd.read_csv(path_to_csv)

    # extract the name of the file to use in the folder path for all data
    all_tracks_filename = path_to_csv.split('/')[-1].split('.')[0]

    files_list = sorted(df['video_id'].unique(), key=get_prefix_number)
    print(files_list)

    init_frame_id_video = 0
    video_files_metadata = {}

    for f in files_list:
        # add metadata to know which images came from which video
        # the very first image id corresponding to video f
        first_id = init_frame_id_video

        temp = df[df['video_id'] == f]
        temp = temp.sort_values(by=['track_no', 'tframe'], ignore_index=True)    

        generate_masks(temp, output_dir,
                       img_height=img_height,
                       img_width=img_width,
                       first_frame_id=init_frame_id_video,
                       for_training=for_training)

        # allocate the smallest frame id for the following video
        # allocate the smallest frame id for the following video
        num_unique_frames = len(temp['tframe'].unique())
        frames_written = num_unique_frames - 1 if for_training else num_unique_frames
        init_frame_id_video = init_frame_id_video + frames_written

        # subtracting 1 to land at the exact img id (#frames - 1, as they start from 0)
        last_id = init_frame_id_video - 1

        video_files_metadata[f] = {
            'first_img_id_video': first_id,
            'last_img_id_video': last_id,
        }

    # ── Write run metadata (YAML) and per-video frame ranges (CSV) ─────────────
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "path_to_csv": os.path.abspath(path_to_csv),
        "output_dir": os.path.abspath(output_dir),
        "img_height": img_height,
        "img_width": img_width,
        "for_training": for_training,
        "num_videos": len(files_list),
    }
    metadata_path = os.path.join(output_dir, f"{all_tracks_filename}_metadata.yaml")
    with open(metadata_path, "w") as fp:
        yaml.dump(metadata, fp, sort_keys=False, default_flow_style=False)
    print(f"Run metadata saved to {metadata_path}")
 
    video_ranges_df = pd.DataFrame(
        [
            {"video_id": vid, "first_img_id": bounds["first_img_id_video"], "last_img_id": bounds["last_img_id_video"]}
            for vid, bounds in video_files_metadata.items()
        ]
    )
    video_ranges_path = os.path.join(output_dir, f"{all_tracks_filename}_video_frame_ranges.csv")
    video_ranges_df.to_csv(video_ranges_path, index=False)
    print(f"Video frame ranges saved to {video_ranges_path}")


if __name__ == "__main__":
    main()
