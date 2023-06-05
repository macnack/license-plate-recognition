import cv2 as cv
from check_ground_truth import load_ground_thruth, get_ground_truth
from utils_cv import load_images, show, show2
import numpy as np

ground_truth = load_ground_thruth()
images_path = load_images()
# images_path = images_path[:19]


def empty_callback(value):
    pass


cv.namedWindow('Trackbar')
cv.createTrackbar("ON/OFF tresh", 'Trackbar', 0, 1, empty_callback)
cv.createTrackbar("thresh", 'Trackbar', 0, 255, empty_callback)
cv.createTrackbar("maxval", 'Trackbar', 0, 255, empty_callback)
cv.createTrackbar("kernel", 'Trackbar', 0, 21, empty_callback)
cv.createTrackbar("sigma", 'Trackbar', 0, 256, empty_callback)


def get_track_bar():
    th = cv.getTrackbarPos("ON/OFF tresh", 'Trackbar')
    thresh = cv.getTrackbarPos("thresh", 'Trackbar')
    maxval = cv.getTrackbarPos("maxval", 'Trackbar')
    kernel = cv.getTrackbarPos("kernel", 'Trackbar')
    if int(kernel) < 3 or not int(kernel) % 2:
        kernel = None

    sigma = cv.getTrackbarPos("sigma", 'Trackbar')
    return th, thresh, maxval, kernel, sigma


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


idx = 0
license_plate = None
nibla = np.zeros((114,512))
while True:
    # get trackbars
    th, thresh, maxval, kernel, sigma =  get_track_bar()
    # create image of license plated
    if license_plate is None:
        # image read
        image = cv.imread(images_path[idx], cv.IMREAD_COLOR)
        image = cv.resize(image, None, fx=0.25, fy=0.25)
        

        # load pts
        pts, text = get_ground_truth(
        images_path[idx], ground_truth, image.shape[0], image.shape[1])
        pts = sort_points_by_corners(pts)

        license_plate_org = create_license_plate_image(image, pts)
        license_plate_gray = cv.cvtColor(license_plate_org, cv.COLOR_BGR2GRAY)
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        hsv = cv.cvtColor(license_plate_org, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        license_plate = license_plate_org
        ret, thg = cv.threshold(v, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        ret2, thg2 = cv.threshold(image_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        closing = cv.morphologyEx(thg, cv.MORPH_CLOSE, (25, 15), iterations=7)
        closing = cv.bitwise_not(closing)
        contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # area = [cv.contourArea(contour) for contour in contours]
        contours = [contour for contour in contours if cv.contourArea(
        contour) < 4000.0]
        cv.drawContours(license_plate, contours, -1, color=(255,0,0), thickness=2)
    else:
        cv.imshow('License_plate', license_plate)
        cv.imshow("closing", closing)
        cv.imshow("th", thg)
        cv.imshow("image", thg2)
        # cv.imshow('RetVal', ret)


    q = cv.waitKey(10)
    if q == ord('q'):
        break
    if q == ord('d'):
        license_plate = None
        idx = (idx + 1) % len(images_path)
    if q == ord('a'):
        license_plate = None
        idx = (idx - 1) % len(images_path)

cv.destroyAllWindows()
