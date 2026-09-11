"""
flatten_frames_folders.py

One-off utility: given a folder of folders (e.g. one subfolder per video),
where each subfolder has its images nested one level deeper inside a
"frames" subfolder, move those images up into the subfolder itself and
remove the now-empty "frames" folder.

Before:
    parent_folder/
        video_1/
            frames/
                0.jpg
                1.jpg
                ...
        video_2/
            frames/
                0.jpg
                ...

After:
    parent_folder/
        video_1/
            0.jpg
            1.jpg
            ...
        video_2/
            0.jpg
            ...

Usage:
    python flatten_frames_folders.py /path/to/parent_folder
    python flatten_frames_folders.py /path/to/parent_folder --inner-name frames --dry-run
"""

import argparse
import os
import shutil


def flatten(parent_folder, inner_name="frames", dry_run=False):
    entries = sorted(os.listdir(parent_folder))

    for entry in entries:
        video_dir = os.path.join(parent_folder, entry)
        if not os.path.isdir(video_dir):
            continue

        inner_dir = os.path.join(video_dir, inner_name)
        if not os.path.isdir(inner_dir):
            print(f"Skipping '{entry}': no '{inner_name}' subfolder found.")
            continue

        files = os.listdir(inner_dir)
        print(f"'{entry}': moving {len(files)} file(s) out of '{inner_name}/'...")

        for fname in files:
            src = os.path.join(inner_dir, fname)
            dst = os.path.join(video_dir, fname)
            if os.path.exists(dst):
                print(f"  WARNING: '{dst}' already exists — skipping '{fname}'.")
                continue
            if not dry_run:
                shutil.move(src, dst)

        if not dry_run:
            os.rmdir(inner_dir)

    print("Done." if not dry_run else "Dry run complete — nothing was moved.")


def main():
    parser = argparse.ArgumentParser(
        description="Move images out of a nested per-video 'frames' subfolder and remove it."
    )
    parser.add_argument("parent_folder", type=str,
                        help="Folder containing one subfolder per video.")
    parser.add_argument("--inner-name", type=str, default="frames",
                        help="Name of the nested subfolder to flatten (default: frames).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without moving/deleting anything.")
    args = parser.parse_args()

    flatten(args.parent_folder, inner_name=args.inner_name, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
