import cv2 as cv
import numpy as np
from utils_cv import load_images, show, show2, read_img


images_path = load_images()


def calculate_skip_image(sum_of_edges_prim, x, y, w):
    x1 = x - w // 2
    x2 = x + w // 2

    skip_image = sum_of_edges_prim[y, x2] - sum_of_edges_prim[y, x1]
    return skip_image

def calculate_sum_of_edges_prim(BE):
    height, width = BE.shape
    sum_of_edges_prim = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if x > 0:
                sum_of_edges_prim[y, x] += sum_of_edges_prim[y, x-1]
                if BE[y, x-1] != BE[y, x]:
                    sum_of_edges_prim[y, x] += 1
    return sum_of_edges_prim
                


def calculate_skip_quantity(BE, x, y):
    if x > 0:
        skip_quantity = int(BE[y, x-1] != BE[y, x])
    else:
        skip_quantity = 0
    return skip_quantity


def compute_integral_image(image):
    height, width = image.shape

    integral_image = np.zeros((height, width), dtype=np.uint32)

    for i in range(height):
        for j in range(width):
            if i == 0 and j == 0:
                integral_image[i, j] = image[i, j]
            elif i == 0:
                integral_image[i, j] = integral_image[i, j-1] + image[i, j]
            elif j == 0:
                integral_image[i, j] = integral_image[i-1, j] + image[i, j]
            else:
                integral_image[i, j] = integral_image[i-1, j] + \
                    integral_image[i, j-1] - \
                    integral_image[i-1, j-1] + image[i, j]
    return integral_image

def edge_extraction(image, coefficient = 4, percentage = 0.75):
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    sobel_y = np.abs(sobel_y)
    mean_gradient = np.mean(sobel_y)
    coefficient = 4
    threshold_mean = coefficient * mean_gradient

    hist, bins = np.histogram(sobel_y, bins=256, range=[0, 256])
    cum_hist = np.cumsum(hist)

    threshold_hist_idx = np.argmax(cum_hist >= cum_hist[-1] * percentage)
    threshold_hist = bins[threshold_hist_idx]

    threshold = max(threshold_mean, threshold_hist)

    threshold_image = np.zeros_like(image)
    threshold_image[sobel_y >= threshold] = 255
    return threshold_image

for image_path in images_path:
    image, gray = read_img(image_path, scale=1/4)

    threshold_image = edge_extraction(gray)
    print(threshold_image.shape)
    print(gray.shape)
    print(gray[750-8, 1000-1])
    integral_mat = compute_integral_image(gray)
    # integral_img = cv.cvtColor(integral_img, cv.COLOR_GRAY2BGR)
    sum_of_edges = calculate_sum_of_edges_prim(threshold_image)
    # skip_quantity = calculate_skip_quantity(threshold_image, x, y)
    show2(sum_of_edges, threshold_image)