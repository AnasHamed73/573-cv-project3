#!/usr/bin/env python3
"""
Denoise Problem
(Due date: Nov. 25, 11:59 P.M., 2019)
The goal of this task is to denoise image using median filter.

Do NOT modify the code provided to you.
Do NOT import ANY library or API besides what has been listed.
Hint: 
Please complete all the functions that are labeled with '#to do'. 
You are suggested to use utils.zero_pad.
"""


import utils
import numpy as np
import json


def find_median(patch):
    flat = patch.flatten()
    sort = sorted(flat)
    length = len(sort)
    if length % 2 == 0:
        return (sort[length//2] + sort[(length//2) - 1]) / 2
    else:
        return int(sort[length//2])


def median_filter(img):
    """
    Implement median filter on the given image.
    Steps:
    (1) Pad the image with zero to ensure that the output is of the same size as the input image.
    (2) Calculate the filtered image.
    Arg: Input image. 
    Return: Filtered image.
    """
    # TODO: implement this function.
    filter_size = 3
    result = np.zeros(np.shape(img), dtype=np.uint8)
    img_height, img_width = np.shape(img)
    pad_width = filter_size // 2
    padded_img = utils.zero_pad(img, pad_width, pad_width)
    padded_img_height, padded_img_width = np.shape(padded_img)
    for i in range(img_height):
        rows = padded_img[i:i+filter_size, :]
        for j in range(img_width):
            patch = rows[:, j:j+filter_size]
            median = find_median(patch)
            result[i, j] = median
    return result


def mse(img1, img2):
    """
    Calculate mean square error of two images.
    Arg: Two images to be compared.
    Return: Mean square error.
    """    
    # TODO: implement this function.
    s = 0
    height, width = np.shape(img1)
    for i in range(height):
        for j in range(width):
            diff = int(img1[i, j]) - int(img2[i, j])
            s += (diff) ** 2
    mse = s / (width * height)
    return mse 
    

if __name__ == "__main__":
    img = utils.read_image('lenna-noise.png')
    gt = utils.read_image('lenna-denoise.png')

    result = median_filter(img)
    error = mse(gt, result)

    with open('results/task2.json', "w") as file:
        json.dump(error, file)
    utils.write_image(result,'results/task2_result.jpg')


