import cv2 as cv
import numpy as np
from processing.utils_cv import load_images, find_four_corners
from processing.utils_cv import show
# from utils_cv import load_images, find_four_corners, show


def license_plate_roi(mask, image):
    contours, _ = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv.contourArea(
        contour) > 100]  # Remove some sort of mess
    for count, contour in enumerate(contours):
        black_box = np.zeros_like(image)

        # Contour Approximation
        epsilon = 0.04 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        if len(np.squeeze(approx)) >= 4:  # Take only with 4 corners, (to change)
            cv.drawContours(black_box, contours, count, (255, 255, 255), -1)

        new_image = cv.bitwise_and(image, black_box)
        # Convert to gray color
        gray_image = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)

        _, thresh = cv.threshold(
            gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        opening = cv.morphologyEx(
            thresh, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        erosion = cv.erode(opening, np.ones((3, 3), np.uint8), iterations=1)
        erosion = cv.bitwise_not(erosion)
        cnts, h = cv.findContours(
            erosion, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # check what is in ROI
        letters = False
        if h is not None and cnts is not None:
            _, counts = np.unique(h[:, :, 3], return_counts=True)
            letters = len(counts) > 1 and np.any((counts > 6) & (counts < 9))
        # return countour of license plate mask
        if letters:
            erosion = cv.dilate(erosion, None, iterations=10)
            cnts, h_ = cv.findContours(
                erosion, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            if len(cnts) >= 1:
                cnts = cnts[0]
                pts = find_four_corners(cnts)
            return pts


def detection_license_plate(image):
    image = cv.resize(image, None, fx=0.2, fy=0.2)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)  # Convert to hsv
    mask = cv.inRange(hsv, np.array([0, 0, 138]), np.array(
        [180, 75, 255]))  # Find the mask of white color
    show(mask)
    license_plate_pts = license_plate_roi(mask, image)
    if license_plate_pts is None:
        # add grayscale mask
        mask2 = cv.inRange(hsv, np.array([0, 0, 88]), np.array([180, 75, 141]))
        mask = mask + mask2
        show(mask)
        license_plate_pts = license_plate_roi(mask, image)

    if license_plate_pts is None:
        return None, image
    else:
        return license_plate_pts, image


def main():
    images_path = load_images()
    images_path = images_path
    for image_path in images_path:
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        pts, image = detection_license_plate(image)
        # show(image)
        # print(pts)


if __name__ == '__main__':
    main()
