import csv
import os
import shutil
import argparse
from PIL import Image
import numpy as np
from merge_csv import get_prefix_number
from datetime import datetime
import json

'TO RUN: python generate_first_second_folder.py /path/to/input_folder'
def find_smallest_k(folder):
    files = os.listdir(folder)
    numbers = []
    for file in files:
        if file.endswith(('.jpg', '.png')):
            try:
                k = int(file.split('.')[0])  # Extract the numeric part of the filename
                numbers.append(k)
            except ValueError:
                print(f"Skipping file {file}: Not a valid number.")
                continue
    print(f"Extracted numbers: {numbers}")  # Debugging line
    return min(numbers) if numbers else None


def _prepare_folders(output_folder):
    first_folder = os.path.join(output_folder, "first")
    second_folder = os.path.join(output_folder, "second")
    os.makedirs(first_folder, exist_ok=True)
    os.makedirs(second_folder, exist_ok=True)

    return first_folder, second_folder

def _extract_files_info(input_folder):
    # Get a sorted list of image files in the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))], key=lambda x: int(x.split('.')[0]))
    # extract the file format
    file_format = image_files[0].split('.')[1]
    # extract the smallest numerical id, required in case the images dont start from 0.
    smallest_id = find_smallest_k(input_folder)
    num_images = len(image_files)

    return num_images, smallest_id, file_format, image_files

def copy_images_for_training(input_folder, output_folder, start_id):
    if output_folder is None:
        raise ValueError("output_folder is required for training.")
    
    first_folder, second_folder = _prepare_folders(output_folder)
    num_images, smallest_id, format, _ = _extract_files_info(input_folder)

    # Copy images to the "first" folder
    # drop the last frame, as we dont have displacement info for it
    for i in range(num_images - 1):
        src_path = os.path.join(input_folder, f'{smallest_id + i}.{format}')
        # in case smallest_id > 0,
        # we want the images in the first_folder to start from 0,
        # so it doesnt mess up the order when multiple files are available
        dest_path = os.path.join(first_folder, f'{start_id+i}.{format}')
        shutil.copy(src_path, dest_path)

     # Create the "second" folder images
    for i in range(1, num_images):  # Start from 1 to n
        src_path = os.path.join(input_folder, f'{smallest_id+i}.{format}')  # Use "first" folder images
        dest_path = os.path.join(second_folder, f"{start_id + i - 1}.{format}")  # Save as 0, 1, 2, ...
        shutil.copy(src_path, dest_path)

    # Run image comparison checks
    # smallest_id = start_id, so that only the files generated in this run are checked.
    # Otherwise all files starting from 0 will be checked, even if that's the n-th file and its start id > 0
    if verify_image_copies(first_folder, second_folder, num_images, format, smallest_id=start_id):
        print(f"Images successfully copied to '{first_folder}' and '{second_folder}'.")
    else:
        ValueError('Check for image matching did not pass. Images in first and second folder are not compatible.')

def copy_images_for_inference(input_folder, output_folder):

    video_folder_name = input_folder.split('/')[-1]
    # so that the images are easily associated with the video they came from
    output_folder = os.path.join(output_folder, video_folder_name)

    first_folder, second_folder = _prepare_folders(output_folder)
    num_images, smallest_id, format, image_files = _extract_files_info(input_folder)

    for i in range(num_images):
        src_path = os.path.join(input_folder, f'{smallest_id + i}.{format}')
        dest_path = os.path.join(first_folder, f'{smallest_id + i}.{format}')
        shutil.copy(src_path, dest_path)
    
    # Create the "second" folder images
    for i in range(1, num_images):  # Start from 1 to n
        src_path = os.path.join(first_folder, f'{smallest_id + i}.{format}')  # Use "first" folder images
        dest_path = os.path.join(second_folder, f"{smallest_id + i - 1}.{format}")  # Save as 0, 1, 2, ...
        shutil.copy(src_path, dest_path)

    # Duplicate the last image
    last_image_src = os.path.join(first_folder, image_files[-1])  # Use the last image in "first"
    last_image_dest = os.path.join(second_folder, f"{smallest_id + num_images - 1}.{format}")  # Save as image n in "second"
    shutil.copy(last_image_src, last_image_dest)

    print(f"Images successfully copied to '{first_folder}' and '{second_folder}'.")

    # Run image comparison checks
    if verify_image_copies(first_folder, second_folder, len(image_files), format, smallest_id):
        print(f"Images successfully copied to '{first_folder}' and '{second_folder}'.")
    else:
        ValueError('Check for image matching did not pass. Images in first and second folder are not compatible.')


def verify_image_copies(first_folder, second_folder, num_images, format, smallest_id):
    """
    Function to verify whether images in 'second' are correctly copied and match the expected images in 'first'.
    """
    check_passed=False
    # Compare 1.{format} in first with 0.{format} in second
    first_image_path = os.path.join(first_folder, f"{smallest_id+1}.{format}")
    second_image_path = os.path.join(second_folder, f"{smallest_id}.{format}")
    if compare_images(first_image_path, second_image_path):
        check_passed=True
    else:
        check_passed=False

    # Compare n.{format} in first with n-1.{format} in second
    first_last_image_path = os.path.join(first_folder, f"{smallest_id + num_images - 1}.{format}")
    second_last_image_path = os.path.join(second_folder, f"{smallest_id + num_images - 2}.{format}")
    if compare_images(first_last_image_path, second_last_image_path):
        check_passed=True
    else:
        check_passed=False

    # Compare n-1.{format} in second with n.jpg in second
    second_last_image_dest_path = os.path.join(second_folder, f"{smallest_id + num_images - 2}.{format}")
    second_last_dup_image_path = os.path.join(second_folder, f"{smallest_id + num_images - 1}.{format}")
    if compare_images(second_last_image_dest_path, second_last_dup_image_path):
        check_passed=True
    else:
        check_passed=False

    return check_passed


def compare_images(image_path_1, image_path_2):
    """
    Compares two images and returns True if they are identical, otherwise False.
    """
    try:
        # Open and convert images to grayscale for comparison
        img1 = Image.open(image_path_1).convert("RGB")
        img2 = Image.open(image_path_2).convert("RGB")

        # Convert images to NumPy arrays
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        # Compare the arrays
        return np.array_equal(arr1, arr2)

    except Exception as e:
        print(f"Error comparing {image_path_1} and {image_path_2}: {e}")
        return False

def save_metadata(input_folder, output_folder, subfolders, final_frame_count, metadata, csv_rows):
    run_metadata = {
            'completed_at': datetime.now().isoformat(timespec='seconds'),
            'input_folder': str(input_folder),
            'output_folder': str(output_folder),
            'num_videos_processed': len(subfolders),
            'videos_processed': [f.name for f in subfolders],
            'final_frame_count': final_frame_count,
            'videos': metadata,
        }
     
    metadata_path = output_folder / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(run_metadata, f, indent=2)
    
    csv_path = output_folder / 'video_frame_ranges.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'first_id_video', 'last_id_video'])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f'Metadata saved at {metadata_path}')
    print(f'CSV saved at {csv_path}')


"""Used to iterate over folder of folders and create first and second folder.
   Last frame is dropped as this is required for training."""
def iterate_over_folder_for_training(input_folder, output_folder, debug=True):
    if debug:
        print('For training')
    has_subdirs = any(item.is_dir() for item in input_folder.iterdir())

    if not has_subdirs:
        ValueError('This is not a nested folder.')
    else:
        # sort subfolders
        subfolders = sorted(
            [f for f in input_folder.iterdir() if f.is_dir()],
            key=get_prefix_number
        )
        start_id = 0
        metadata = {}
        csv_rows = []

        for child_folder in subfolders:
            first_id_video = start_id
            copy_images_for_training(os.path.join(input_folder,child_folder), output_folder=output_folder, start_id=start_id)
            num_img_files,_,_,_ = _extract_files_info(child_folder)
            # subtract 1 because the last frame is dropped
            num_frames_used = num_img_files - 1
            # last frame id for this video (0-indexed, cumulative across videos)
            last_id_video = first_id_video + num_frames_used - 1

            metadata[child_folder.name] = {
            'first_id_video': first_id_video,
            'last_id_video': last_id_video,
            }

            csv_rows.append({
                'video_id': child_folder.name,
                'first_id_video': first_id_video,
                'last_id_video': last_id_video,
            })

            start_id = last_id_video + 1

    final_frame_count = start_id  # total number of frames written across all videos
 
    run_metadata = {
        'completed_at': datetime.now().isoformat(timespec='seconds'),
        'input_folder': str(input_folder),
        'output_folder': str(output_folder),
        'num_videos_processed': len(subfolders),
        'videos_processed': [f.name for f in subfolders],
        'final_frame_count': final_frame_count,
        'videos': metadata,
    }
 
    metadata_path = output_folder / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(run_metadata, f, indent=2)
 
    csv_path = output_folder / 'video_frame_ranges.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'first_id_video', 'last_id_video'])
        writer.writeheader()
        writer.writerows(csv_rows)

    save_metadata(input_folder, output_folder, subfolders, final_frame_count, metadata,csv_rows)
    print(f'Creation of first and second folder for {input_folder} is done!')
    print(f'First and second folder saved at {output_folder}')
    
def iterate_over_folder_for_inference(input_folder, output_folder, debug=True):
    if debug:
        print('For inference...')

    has_subdirs = any(item.is_dir() for item in input_folder.iterdir())
    
    if not has_subdirs:
        copy_images_for_inference(input_folder, output_folder)
    else:
        # sort subfolders
        subfolders = sorted(
            [f for f in input_folder.iterdir() if f.is_dir()],
            key=get_prefix_number
        )
        for child_folder in subfolders:
            copy_images_for_inference(child_folder, output_folder)

    print(f'Creating of first and second folder for {input_folder} is done!')
    print(f'First and second folder saved at {output_folder}')


def main():
    # Set up argument parser for command-line usage
    parser = argparse.ArgumentParser(description="Copy images to 'first' and 'second' folders with custom logic and verify image consistency.")
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing images.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder containing images.')

    parser.add_argument('for_training', type=str, help='Declare whether the files in the folder are used for training. If true,last frame in the folder is skipped.')

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    # create if it doesnt exist
    os.makedirs(output_folder,exist_ok=True)
    for_training = args.for_training

    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: The input folder '{input_folder}' does not exist.")
        return

    if for_training:
        iterate_over_folder_for_training(input_folder, output_folder)
    else:
        iterate_over_folder_for_inference(input_folder, output_folder)


if __name__ == "__main__":
    main()