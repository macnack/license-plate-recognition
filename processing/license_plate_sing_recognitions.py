import cv2 as cv
from check_ground_truth import load_ground_thruth, get_ground_truth
from utils_cv import load_images, show, show2
import numpy as np

ground_truth = load_ground_thruth()
images_path = load_images()
images_path = images_path[:19]


def sort_points_by_corners(pts):
    corner_sums = np.sum(pts, axis=1)
    sorted_idx = np.argsort(corner_sums)
    return pts[sorted_idx].astype(np.float32)


license_plate_shape = (512, 114)
dstPts = np.array([[0, 0], [0, license_plate_shape[1]], [license_plate_shape[0], 0], [
                  license_plate_shape[0], license_plate_shape[1]]], np.float32)
dstPts = sort_points_by_corners(dstPts)


def create_license_plate_image(image, pts):
    transform = cv.getPerspectiveTransform(pts, dstPts)
    license_plate = cv.warpPerspective(image, transform, license_plate_shape)
    return license_plate


char_paths = load_images("dane/font")
char_images = [cv.imread(char_path, cv.IMREAD_GRAYSCALE)
               for char_path in char_paths]
chars_cnts = []
for char_path in char_paths:
    img_color = cv.imread(char_path, cv.IMREAD_COLOR)
    img = cv.imread(char_path, cv.IMREAD_GRAYSCALE)
    contours, hierarchy = cv.findContours(
        img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img_color, contours, 0, color=(255,0,0), thickness=2)
    chars_cnts.append((char_path.rsplit("/")[-1][:-4], contours[0]))

for image_path in images_path:
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    image = cv.resize(image, None, fx=0.25, fy=0.25)

    pts, text = get_ground_truth(
        image_path, ground_truth, image.shape[0], image.shape[1])
    pts = sort_points_by_corners(pts)

    license_plate_org = create_license_plate_image(image, pts)
    hsv = cv.cvtColor(license_plate_org, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    ret, thg = cv.threshold(v, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    license_plate_closed = cv.morphologyEx(
        thg, cv.MORPH_CLOSE, (25, 15), iterations=7)
    license_plate_closed = cv.bitwise_not(license_plate_closed)
    show(license_plate_org)

    contours, hierarchy = cv.findContours(
        license_plate_closed, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv.contourArea(
        contour) > 100.0]
    # contours = [contour for contour in contours if cv.contourArea(
    #     contour) > 500.0]
    for idx_cnt, cnt in enumerate(contours):
        cv.fillPoly(license_plate_closed, [cnt], 255)
        x, y, w, h = cv.boundingRect(cnt)
    contours, hierarchy = cv.findContours(
        license_plate_closed, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if cv.contourArea(
        contour) > 100.0]
    letters = []
    for idx_cnt, cnt in enumerate(contours):
        x, y, w, h = cv.boundingRect(cnt)
        # letters.append( )
        if 0.2 < w/h < .8:
            # cv.rectangle(license_plate_org, (x,y),(x+w, y+h), color=(0,0,255), thickness=2)
            cv.drawContours(license_plate_org, contours,
                            idx_cnt, color=(255, 0, 0), thickness=3)
            license_plate_char = license_plate_closed[y:y+h, x:x+w]
            output = []
            for key, char_cnt in chars_cnts:
                distance = cv.matchShapes(
                    char_cnt, cnt, cv.CONTOURS_MATCH_I1, 0)
                output.append((key, distance))
            output = sorted(output, key=lambda x: x[1])
            print(output)
            show(license_plate_org)
    # print(contours)
    show(license_plate_org)
