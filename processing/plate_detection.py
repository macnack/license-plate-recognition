from utils_cv import load_images, show, show2, pts_to_corners
from check_ground_truth import load_ground_thruth, get_ground_truth, get_iou, get_intersection_area
from utils_plate import read_img
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
        cv.polylines(image, [ground_pts], True, (0, 255, 0), thickness=4)
        cv.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv.putText(image, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                   2, (0, 255, 0), 2, cv.LINE_AA)
        show(image, image_path)
        cv.waitKey(500)

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
    edge = cv.Canny(img_, 20, 100)
    cnts = cv.findContours(edge.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print(len(cnts))
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    img_copy = img.copy()
    ground_pts, text = get_ground_truth(
        path, ground_truth, img.shape[0], img.shape[1])
    x_g, y_g, w_g, h_g = cv.boundingRect(ground_pts)
    img_copy = cv.rectangle(img_copy.copy(), (x_g, y_g),
                            (x_g+w_g, y_g+h_g), (0, 255, 0), 2)
    intersect = []
    no_intersect = []
    for c in cnts:
        # img_copy = cv.drawContours(img_copy.copy(), [c], -1, (0, 255, 0), 2)
        # epsilon = 0.01 * cv.arcLength(c, True)
        # approx = cv.approxPolyDP(c, epsilon, True)
        # img_copy = cv.drawContours(img_copy.copy(), [approx], -1, (0, 0, 255), 5)
        x, y, w, h = cv.boundingRect(c)
        M = cv.moments(c)
        area = float(M['m00'])
        if area == 0:
            area  = 1e-8
        cx = int(M['m10']/area)
        cy = int(M['m01']/area)

        perimeter = cv.arcLength(c, True)
        k = cv.isContourConvex(c)
        color = (255, 0, 0)
        desc = {'pts': [x, y, x+w, y+h],
                'centroid': (cx, cy), 'area': area, 'perimeter': perimeter, 'convex': k}
        if True:#get_intersection_area([x_g, y_g, x_g+w_g, y_g+h_g], [x, y, x+w, y+h]) > 0:
            intersect.append(desc)
            color = (0, 255, 255)
            if 0.18 < w/h < 1.0:
                
                if area > 20.0:
                    color = (0, 0, 255)
                    img_copy = cv.putText(img_copy.copy(), f"{area}", (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv.LINE_4)
                    print(area)
        else:
            no_intersect.append(desc)
        img_copy = cv.rectangle(img_copy.copy(), (x, y),
                                (x+w, y+h), color, 2)

        # show(img_copy)

        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        # img_copy = cv.drawContours(
        #     img_copy.copy(), [box], 0, (0, 0, 255), 2)
        # get_iou(ground_pts, )

    # img_copy = cv.drawContours(img_copy.copy(), [ground_pts], 0, (0, 255, 0), 2)
    show2(img_copy, edge, path)
    return intersect, no_intersect


if __name__ == "__main__":
    # app()
    images_path = load_images()
    ground_truth = load_ground_thruth()
    # show_ground_truth()
    all_data = []
    for image_path in images_path:
        inter, no_inter = app3(image_path)
        # process_data = {
        #     'image': image_path,
        #     'intersect': inter,
        #     'no_intersect': no_inter
        # }
        # all_data.append(process_data)
    
    # with open("data.json", "w") as outfile:
    #     json.dump(all_data, outfile, indent=4)
