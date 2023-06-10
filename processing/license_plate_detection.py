import math
import cv2 as cv
import numpy as np
# from utils_cv import load_images, find_four_corners, show, sort_points_by_corners
from processing.utils_cv import load_images, find_four_corners, show, sort_points_by_corners


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


def preprocess_image(image):
    image = cv.resize(image, None, fx=0.25, fy=0.25)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 11, 35, 100)
    thresholded = cv.adaptiveThreshold(
        gray, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv.THRESH_BINARY, blockSize=11, C=3)
    thresholded = cv.morphologyEx(
        thresholded, op=cv.MORPH_OPEN, kernel=cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
    return thresholded, image


def check_ratio(ratio):
    if 3.0 <= ratio <= 6.0:
        return True
    else:
        return False


def check_license_plate_width(width, image_width):
    if width >= image_width / 3.0:
        return True
    else:
        return False


def check_size_distortion(diff_height, diff_width):
    # of license plate ( if is rotated ), in percentage
    if diff_height < 31.0 and diff_width < 12.0:
        return True
    else:
        return False


def detect_license_plate(image):
    thresholded, image = preprocess_image(image)
    contours, _ = cv.findContours(
        thresholded, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    roi_contours_list = []
    for contour in contours:
        epsilon = 0.015 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4 and cv.contourArea(contour) > 20000:
            apx = sort_points_by_corners(np.squeeze(approx))
            # Calculate size of sorted pts [left_top, left_bottom, right_top, right_bottom]
            width = np.array(
                [np.linalg.norm(apx[3, :] - apx[1, :]), np.linalg.norm(apx[2, :]-apx[0, :])])
            width_max = width[np.argmax(width)]
            diff_width = np.abs(np.diff(width) / float(width_max)) * 100.0

            height = np.array(
                [np.linalg.norm(apx[1, :] - apx[0, :]), np.linalg.norm(apx[3, :]-apx[2, :])])
            height_max = height[np.argmax(height)]
            diff_height = np.abs(np.diff(height) / float(height_max)) * 100.0

            ratio = width_max / height_max

            if check_ratio(ratio) and check_license_plate_width(width_max, image.shape[1]) and check_size_distortion(diff_height, diff_width):
                roi_contours_list.append(approx)

    roi_contour = None
    if len(roi_contours_list) > 0:
        roi_areas = [cv.contourArea(roi) for roi in roi_contours_list]
        roi_contour = np.squeeze(roi_contours_list[np.argmin(roi_areas)])

    return roi_contour, image


def main():
    images_path = load_images()
    print(len(images_path))
    # images_path = images_path[:19]
    for image_path in images_path:
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        # pts, image = detection_license_plate(image)
        print(image_path)
        pts, image = detect_license_plate(image)
        if pts is not None:
            print("show pts", pts)
            for p in pts[:]:
                cv.putText(image, ".", (p[0], p[1]),  cv.FONT_HERSHEY_SIMPLEX, 1, color=(
                    0, 0, 255), thickness=10)
            # print(pts)
        show(image)
        print("#!"*30)
        print("\n"*5)

        # print(pts)


if __name__ == '__main__':
    main()
