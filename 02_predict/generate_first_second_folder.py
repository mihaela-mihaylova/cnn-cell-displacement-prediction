import os
import shutil
import argparse
from PIL import Image
import numpy as np

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


def copy_images_to_folders(input_folder):
    # Create destination folders
    first_folder = os.path.join(input_folder, "first")
    second_folder = os.path.join(input_folder, "second")

    os.makedirs(first_folder, exist_ok=True)
    os.makedirs(second_folder, exist_ok=True)

    # Get a sorted list of image files in the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))], key=lambda x: int(x.split('.')[0]))
    # extract the file format
    format = image_files[0].split('.')[1]
    # extract the smallest numerical id, required in case the images dont start from 0.
    smallest_id = find_smallest_k(input_folder)

    # Copy images to the "first" folder
    for image_file in image_files:
        src_path = os.path.join(input_folder, image_file)
        dest_path = os.path.join(first_folder, image_file)
        shutil.copy(src_path, dest_path)

    # Create the "second" folder images
    for i in range(1, len(image_files)):  # Start from 1 to n
        src_path = os.path.join(first_folder, f'{smallest_id+i}.{format}')  # Use "first" folder images
        dest_path = os.path.join(second_folder, f"{smallest_id + i - 1}.{format}")  # Save as 0, 1, 2, ...
        shutil.copy(src_path, dest_path)

    # Duplicate the last image
    last_image_src = os.path.join(first_folder, image_files[-1])  # Use the last image in "first"
    last_image_dest = os.path.join(second_folder, f"{smallest_id + len(image_files) - 1}.{format}")  # Save as image n in "second"
    shutil.copy(last_image_src, last_image_dest)

    print(f"Images successfully copied to '{first_folder}' and '{second_folder}'.")

    # Run image comparison checks
    if verify_image_copies(first_folder, second_folder, len(image_files), format, smallest_id):
        print(f"Images successfully copied to '{first_folder}' and '{second_folder}'.")



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


def main():
    # Set up argument parser for command-line usage
    parser = argparse.ArgumentParser(description="Copy images to 'first' and 'second' folders with custom logic and verify image consistency.")
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing images.')

    args = parser.parse_args()
    input_folder = args.input_folder

    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: The input folder '{input_folder}' does not exist.")
        return

    # Run the image copying function
    copy_images_to_folders(input_folder)


if __name__ == "__main__":
    main()
