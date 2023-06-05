from utils_cv import load_images, show, show2, pts_to_corners, get_iou
from check_ground_truth import load_ground_thruth, get_ground_truth, get_intersection_area
from utils_plate import read_img
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
import cv2 as cv
import numpy as np
import imutils
import json
ground_truth = load_ground_thruth()
images_path = load_images()


def show_ground_truth():
    for image_path in images_path:
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        image = cv.resize(image, None, fx=0.25, fy=0.25)

        ground_pts, text = get_ground_truth(
            image_path, ground_truth, image.shape[0], image.shape[1])
        left, top, right, bottom = pts_to_corners(ground_pts)
        box = cv.boundingRect(ground_pts)
        print(box)
        print(left, top, right, bottom)
        cv.polylines(image, [ground_pts], True, (0, 255, 0), thickness=4)
        cv.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), 10)
        cv.putText(image, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                   2, (0, 255, 0), 2, cv.LINE_AA)
        show(image, image_path)
        cv.waitKey(500)

def app4(path):
    scale = 1/4.0
    img, gray = read_img(path, scale)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    weaker = np.array( [100, 70, 50])
    stronger = np.array( [150, 255, 255])
    mask = cv.inRange(hsv, weaker, stronger)
    mask = cv.erode(mask, None, iterations=1)
    contours, hierarchy = cv.findContours(
        mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv.contourArea(
        contour) > 100]
    for cnt in contours:
        
        epsilon = 0.05 * cv.arcLength(cnt, True)
        # Contour Approximation
        approx = cv.approxPolyDP(cnt, epsilon, True)
        box = cv.boundingRect(approx)
        cv.drawContours(img, [cnt], None,  color=(0,0,255), thickness=3)
        # cv.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color=(0,0,255), thickness=2)
        # roi_w = box[0]+4*box[3]
        # if roi_w < img.shape[0]:
        #     cv.rectangle(img, (box[0], box[1]), (roi_w, box[1]+box[3]), color=(0,255,0), thickness=2)

    show2(img, mask)

def app3(path):
    scale = 1/4.0
    img, gray = read_img(path, scale)
    rectKern = cv.getStructuringElement(
        cv.MORPH_RECT, (int(img.shape[0]/3.0), int(img.shape[1]/15.0)))
    blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, rectKern)
    squareKern = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    light = cv.morphologyEx(gray, cv.MORPH_CLOSE, squareKern)
    light = cv.threshold(light, 0, 255,
                         cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    gradX = cv.Sobel(blackhat, ddepth=cv.CV_32F,
                     dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")
    gradX = cv.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv.morphologyEx(gradX, cv.MORPH_CLOSE, rectKern)
    thresh = cv.threshold(gradX, 0, 255,
                          cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    # show(thresh, "th")
    thresh = cv.erode(thresh, None, iterations=3)
    # show(thresh, "erode")
    thresh = cv.dilate(thresh, None, iterations=3)
    # show(thresh, "dilate")
    thresh = cv.bitwise_and(thresh, thresh, mask=light)
    thresh = cv.dilate(thresh, None, iterations=9)
    # thresh = cv.erode(thresh, None, iterations=3)

    blackhat = cv.threshold(blackhat, 0, 255,
                            cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    img_ = cv.bitwise_and(blackhat, blackhat, mask=thresh)
    img_ = cv.blur(img_, (3, 3))
    squareKern = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    img_ = cv.morphologyEx(img_, cv.MORPH_CLOSE, squareKern)
    img_ = cv.threshold(img_, 0, 255,
                        cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    # img_ = cv.dilate(img_, None,iterations=1)
    # img_ = cv.erode(img_, None, iterations=1)
    # edge = cv.Canny(img_, 20, 100)
    cnts = cv.findContours(img_.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    img_copy = img.copy()
    # referencja
    ground_pts, text = get_ground_truth(
        path, ground_truth, img.shape[0], img.shape[1])
    x_g, y_g, w_g, h_g = cv.boundingRect(ground_pts)
    img_copy = cv.rectangle(img_copy.copy(), (x_g, y_g),
                            (x_g+w_g, y_g+h_g), (0, 255, 0), 2)
    for c in cnts:
        # img_copy = cv.drawContours(img_copy.copy(), [c], -1, (0, 255, 0), 2)
        # epsilon = 0.01 * cv.arcLength(c, True)
        # approx = cv.approxPolyDP(c, epsilon, True)
        # img_copy = cv.drawContours(img_copy.copy(), [approx], -1, (0, 0, 255), 5)
        x, y, w, h = cv.boundingRect(c)
        M = cv.moments(c)
        area = float(M['m00'])
        if area == 0:
            area = 1e-8
        cx = int(M['m10']/area)
        cy = int(M['m01']/area)

        perimeter = cv.arcLength(c, True)

        color = (0, 255, 255)
        if 0.18 < w/h < 1.0:

            if area > 20.0:
                color = (0, 0, 255)
                # img_copy = cv.putText(img_copy.copy(), f"{area}", (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv.LINE_4)
                print(perimeter, area)
        img_copy = cv.rectangle(img_copy.copy(), (x, y),
                                (x+w, y+h), color, 2)

        # show(img_copy)

        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        # img_copy = cv.drawContours(
        #     img_copy.copy(), [box], 0, (0, 0, 255), 2)
        # get_iou(ground_pts, )

    show2(img_copy, img_)
    # show2(light, blackhat, path)


def main(path, ground_truth):
    image = cv.imread(path)
    image = cv.resize(image, None, fx=0.2, fy=0.2)

    correct = 0.0
    ground_pts, text = get_ground_truth(
        path, ground_truth, image.shape[0], image.shape[1])
    ground_truth_box = cv.boundingRect(ground_pts)
    ground_truth_box = [ground_truth_box[0], ground_truth_box[1], ground_truth_box[0]+ground_truth_box[2], ground_truth_box[1]+ground_truth_box[3]]
    print("box:::: ", ground_truth_box)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)  # Convert to hsv
    mask = cv.inRange(hsv, np.array([0, 0, 110]), np.array(
        [180, 75, 255]))  # Find the mask of white color
    mask = cv.inRange(hsv, np.array([0, 0, 0]), np.array(
        [180, 255, 110]))  # Find the mask of white color
    # show(mask)
    contours, hierarchy = cv.findContours(
        mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    print("Patrz ile było", len(contours))
    contours = [contour for contour in contours if cv.contourArea(
        contour) > 100]  # Remove some sort of mess
    print("A ile zostało - bardziej wydajne", len(contours))
    vis = image.copy()
    for count, contour in enumerate(contours):
        epsilon = 0.05 * cv.arcLength(contour, True)
        # Contour Approximation
        approx = cv.approxPolyDP(contour, epsilon, True)
        box = cv.boundingRect(approx)
        box = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
        iou = get_iou(ground_truth_box, box)
        if iou > 0.4:
            correct = 1.0
            print("iou: ", iou)
        cv.drawContours(vis, contours, count, (0, 0, 255), 2)
        # if len(np.squeeze(approx)) == 4:  # Take only with 4 corners
        #     # Draw each contour only for visualisation purposes
        #     # x, y, w, h = cv2.boundingRect(contour)
        #     black_box = np.zeros_like(image)
        #     cv.drawContours(vis, contours, count, (0, 0, 255), 2)
        #     cv.drawContours(black_box, contours, count, (255, 255, 255), -1)
        #     cnt = cv.boundingRect(contour)
            
        #     new_image = cv.bitwise_and(image, black_box)

        #     # cv.imshow("Window2", new_image)
        #     # ROI = new_image[y:y + h, x:x + w]  # crop
        #     # cv2.imshow("Window3", ROI)
        #     # Convert to gray color
        #     gray_image = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)
        #     ret, thresh = cv.threshold(
        #         gray_image, 127, 255, cv.THRESH_BINARY_INV)
        #     opening = cv.morphologyEx(
        #         thresh, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
        #     cnts, h = cv.findContours(
        #         opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        #     # sprawdzenie hierarchii na szybko
        #     kupa = 0
        #     for x in h[0]:
        #         if x[3] == 1:
        #             kupa += 1
        #     # if kupa == 7 or kupa == 8:  # patrzenie czy kontur ma 7 lub 8 znaków w srodku
        #         # tablica nasza i z niej wyciagasz znaki
        #         # cv.imshow("TABLICA", new_image)
        #         # cv.waitKey()
    cv.imshow(path, vis)
    cv.waitKey()
    cv.destroyAllWindows()
    return correct, 0


if __name__ == "__main__":
    # app()
    images_path = load_images()
    ground_truth = load_ground_thruth()
    # show_ground_truth()
    correct = 0
    letters = 0
    for image_path in images_path:
        # app3(image_path)
        find, letter = main(image_path, ground_truth)
        # correct += find
        # letters += letter
    print(correct / len(images_path))

