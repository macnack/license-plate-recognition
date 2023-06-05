from utils_cv import load_images, show, show2
import numpy as np
import cv2 as cv
import time
import math 
images_path = load_images()
# images_path = images_path[:19]

def hsv_threshold(hsv, img):
    blue_mask_wk = np.array([100, 80, 50])
    blue_mask_st = np.array([130, 255, 255])
    bl_mask = cv.inRange(hsv, blue_mask_wk, blue_mask_st)

    # wh_mask_wk = np.array([0, 0, 100])
    # wh_mask_st = np.array([180, 255, 255])
    # wh_mask = cv.inRange(hsv, wh_mask_wk, wh_mask_st)

    mask = cv.bitwise_not(bl_mask) # wh_mask

    return mask, cv.bitwise_and(img, img, mask=mask)
    

for image_path in images_path:
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    image = cv.resize(image, None, fx=0.25, fy=0.25)
    start = time.time()
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask, result = hsv_threshold(hsv, image)

    contours, hierarchy = cv.findContours(
        cv.bitwise_not(mask), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    

    ## calc ROI area of license plate and euroband
    width = image.shape[1]
    width = 1/3.0 * width
    license_plate_area = width * width / 3.0
    euroband_area = 0.06 * license_plate_area

    #remove not used countours
    contours = [contour for contour in contours if cv.contourArea(
        contour) > euroband_area]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edge = cv.GaussianBlur(gray, (3,3), 6)
    edge = cv.Canny(edge, 130, 255)
    cnt = image.copy()
    min_x = np.inf
    black_box = np.zeros_like(image)
    for count, contour in enumerate(contours):
        color = (0, 0, 255)
        epsilon = 0.04 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        if 6 > len(np.squeeze(approx)) >= 4:
            cv.drawContours(cnt, contours, count, color, 2)
            cv.drawContours(black_box, contours, count, (255, 255, 255), -1)
            if min_x > np.min(np.squeeze(contour)[:, 0]):
                min_x = np.min(np.squeeze(contour)[:, 0])
                cnt_top_y = np.min(np.squeeze(contour)[:, 1])
                cnt_bottom_y = np.max(np.squeeze(contour)[:, 1])
                top_y = -image.shape[1] + ( cnt_top_y + min_x )
                top_y = int(top_y)
                if top_y < 0:
                    top_y = 0
                bottom_y = image.shape[1] + ( cnt_bottom_y - min_x )
                bottom_y = int(bottom_y)
                if bottom_y > image.shape[0]:
                    bottom_y = image.shape[0]
                print(time.time() - start)
                show(cnt[top_y: bottom_y, min_x:])
    edge = edge[top_y: bottom_y, min_x:]
    cnt = cnt[top_y: bottom_y, min_x:]
    cnts, h = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for idx_c, c in enumerate(cnts):
        ret, interR = cv.rotatedRectangleIntersection(approx, c)
    # cv.drawContours(cnt, cnts, -1, color=(255, 0,0),thickness=3)
    print(interR)

    # result = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    show2(cnt, black_box)
    q = cv.waitKey(10)
    if q == ord('q'):
        break
    if q == ord('d'):
        i = (i + 1) % len(images_path)
    if q == ord('a'):
        i = (i - 1) % len(images_path)