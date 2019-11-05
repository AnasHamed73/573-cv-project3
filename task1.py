#!/usr/bin/env python3
"""
K-Means Segmentation Problem
(Due date: Nov. 25, 11:59 P.M., 2019)
The goal of this task is to segment image using k-means clustering.

Do NOT modify the code provided to you.
Do NOT import ANY library or API besides what has been listed.
Hint: 
Please complete all the functions that are labeled with '#to do'. 
You are allowed to add your own functions if needed.
You should design you algorithm as fast as possible. To avoid repetitve calculation, you are suggested to depict clustering based on statistic histogram [0,255]. 
You will be graded based on the total distortion, e.g., sum of distances, the less the better your clustering is.
"""


import utils
import numpy as np
import json
import time


class Center():


    def __init__(self, value):
        self.value = value
        self.sum = 0
        self.points_count = 0

    def add(self, val, quantity):
        self.sum += val * quantity
        self.points_count += quantity


    def recalc_center(self):
        if self.points_count == 0:
            return
        self.value = int(self.sum / self.points_count)


    def reset_values(self):
        self.sum = 0
        self.points_count = 0


def get_histogram(img):
    int_map = {}
    for row in img:
        for pixel in row:
            val = int_map.get(pixel)
            if val is None:
                int_map[pixel] = 0
            int_map[pixel] += 1
    return int_map


def random_int(lo, hi):
    return np.random.randint(lo, hi)


def init_centers(k, min_intensity, max_intensity):
    centers = []
    partitions = (max_intensity - min_intensity) // k
    for j in range(k):
        lo = min_intensity + (j * partitions)
        hi = lo + partitions
        c = random_int(lo, hi + 1)
        centers.append(Center(c))
    return centers


def closest_center(pixel, centers):
    closest = centers[0]
    index = 0
    for i in range(1, len(centers)):
        if abs(pixel - centers[i].value) < abs(pixel - closest.value):
            closest = centers[i]
            index = i
    return closest, index


def centers_equal(old, new):
    for oc, nc in zip(old, new):
        if oc != nc:
            return False
    return True


def update_centers_(centers, histo_map):
    while True:
        old_vals = [c.value for c in centers]
        for intensity, count in histo_map.items():
            closest, _ = closest_center(intensity, centers)
            closest.add(intensity, count)
        for center in centers:
            center.recalc_center()
            center.reset_values()  # Reset sum and point count for next iteration
        new_vals = [c.value for c in centers]
        if centers_equal(old_vals, new_vals):
            break


def labels_and_distance(img, centers):
    img_height, img_width = np.shape(img)
    distance_sum = 0
    labels = np.zeros((img_height, img_width), dtype=np.uint8)
    for i in range(img_height):
        for j in range(img_width):
            closest, index = closest_center(img[i, j], centers)
            labels[i, j] = index
            distance_sum += int(abs(img[i, j] - closest.value))
    return labels, distance_sum


def kmeans(img, k):
    """
    Implement kmeans clustering on the given image.
    Steps:
    (1) Random initialize the centers.
    (2) Calculate distances and update centers, stop when centers do not change.
    (3) Iterate all initializations and return the best result.
    Arg: Input image;
         Number of K. 
    Return: Clustering center values;
            Clustering labels of all pixels;
            Minimum summation of distance between each pixel and its center.  
    """
    # TODO: implement this function.
    img_height, img_height = np.shape(img)
    min_intensity = np.amin(img)
    max_intensity = np.amax(img)
    histo_map = get_histogram(img)

    # Randomly initialize centers
    centers = init_centers(k, min_intensity, max_intensity)

    # Update center locations according to the K-means algorithm
    update_centers_(centers, histo_map)

    # Find closest center for each pixel and calculate total intra-cluster distance
    labels, distance_sum = labels_and_distance(img, centers)

    center_vals = [c.value for c in centers]
    return center_vals, labels, distance_sum


def visualize(centers, labels):
    """
    Convert the image to segmentation map replacing each pixel value with its center.
    Arg: Clustering center values;
         Clustering labels of all pixels. 
    Return: Segmentation map.
    """
    # TODO: implement this function.
    h, w = np.shape(labels)
    seg_map = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            seg_map[i, j] = int(centers[labels[i, j]])
    return seg_map 

     
if __name__ == "__main__":
    img = utils.read_image('lenna.png')
    k = 2

    start_time = time.time()
    centers, labels, sumdistance = kmeans(img,k)
    result = visualize(centers, labels)
    end_time = time.time()

    running_time = end_time - start_time
    print(running_time)

    centers = list(centers)
    with open('results/task1.json', "w") as jsonFile:
        jsonFile.write(json.dumps({"centers":centers, "distance":sumdistance, "time":running_time}))
    utils.write_image(result, 'results/task1_result.jpg')
