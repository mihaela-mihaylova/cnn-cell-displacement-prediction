import numpy as np
from skimage import transform
import random
from skimage.util import random_noise
from skimage import exposure
import numpy as np
import random


def average_pixel_value(x, y, r, image):
    min_x = int(max(x - r, 0))
    max_x = int(min(x + r + 1, image.shape[0]))
    min_y = int(max(y - r, 0))
    max_y = int(min(y + r + 1, image.shape[1]))
    
    disk_pixels = []
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            if (i - x) ** 2 + (j - y) ** 2 <= r ** 2:
                pixel_value = image[i, j]
                if not np.isnan(pixel_value):
                    disk_pixels.append(pixel_value)

    if disk_pixels:
        average = np.mean(disk_pixels)
    else:
        average = 0

    #print("Disk Pixels:", disk_pixels)
    #print("Average:", average)

    return average

# translates the last three channels on the y-axis
'''def image_y_translation(img, mask):
    #if random.random() > 0.5:
    translation_distance = np.random.normal(0, 4, 1)[0]  # Adjust the range as per your requirement
    print(translation_distance)
    # Perform translation
    translated_image = transform.warp(img[:,:,3:6], transform.SimilarityTransform(translation=(0, translation_distance)), mode='constant', cval=0)
    img[:,:,3:6] = translated_image
    # change y_displ channel of mask
    segm_mask = mask[:,:,2]
    translated_mask = mask[:,:,1]
    translated_mask[segm_mask!=0] += translation_distance 
    mask[:,:,1] = translated_mask
    return img, mask'''

def data_augmentation_bf_training(image, mask):
    first_image = np.copy(image[:,:,0:3])
    second_image = np.copy(image[:,:,3:6])
    mask = np.copy(mask)

    # rotation at 90 degrees counterclockwise
    if random.random() > 0.5:
        first_image = np.rot90(first_image, k=-1)
        second_image = np.rot90(second_image, k=-1)
        mask = np.rot90(mask, k=-1)
        # swapping x- and y-displacement channels due to the rotation (as directions would change)
        swap_dummy = mask[:,:,0]
        mask[:,:,0] = mask[:,:,1]
        mask[:,:,1] = (-1)*swap_dummy


    # vertical flip
    if random.random() > 0.5:
        first_image = np.flipud(first_image)
        second_image = np.flipud(second_image)
        mask = np.flipud(mask)
        # multiply by -1 as the y-displacement direction is changing
        mask[:,:,1] = (-1)*mask[:,:,1]


    # horizontal flip
    if random.random() > 0.5:
        first_image = np.fliplr(first_image)
        second_image = np.fliplr(second_image)
        mask = np.fliplr(mask)
        # multiply by -1 as x-displacement direction is changing
        mask[:,:,0] = (-1)*mask[:,:,0]

    #gaussian noise (only added to training images)
    if random.random() > 0.5:
        first_image = random_noise(first_image, mode='gaussian', seed=None, clip=True, mean=0, var=0.01)
        second_image = random_noise(second_image, mode='gaussian', seed=None, clip=True, mean=0, var=0.01)

    # adjust brightness with gamma filter
    if random.random() > 0.5:
        first_image = exposure.adjust_gamma(first_image, gamma=random.uniform(0.9, 1.1),gain=1)
        second_image = exposure.adjust_gamma(second_image, gamma=random.uniform(0.9, 1.1),gain=1)

    height = first_image.shape[0]
    width = first_image.shape[1]
    chan = first_image.shape[2]
    aug_image = np.zeros((height, width, 2*chan))
    aug_image[:,:,0:chan] = first_image
    aug_image[:,:,chan:2*chan] = second_image
  
    return aug_image, mask
