import cv2
import numpy as np
from os.path import isfile, join
from os import listdir

def empty_callback(value):
    pass


def load_images(images_dir = "dane/data"):
    images = [join(images_dir, f) for f in listdir(
        images_dir) if isfile(join(images_dir, f))]
    return sorted(images)

cv2.namedWindow('Trackbar')
cv2.createTrackbar("lower_h", 'Trackbar', 0, 180, empty_callback)
cv2.createTrackbar("upper_h", 'Trackbar', 0, 180, empty_callback)
cv2.createTrackbar("lower_s", 'Trackbar', 0, 256, empty_callback)
cv2.createTrackbar("upper_s", 'Trackbar', 0, 256, empty_callback)
cv2.createTrackbar("lower_v", 'Trackbar', 0, 256, empty_callback)
cv2.createTrackbar("upper_v", 'Trackbar', 0, 256, empty_callback)
cv2.createTrackbar("th1_canny", 'Trackbar', 0, 1000, empty_callback)
cv2.createTrackbar("th2_canny", 'Trackbar', 0, 1000, empty_callback)


images= load_images()
images= images[:19]
i = 0

# add_mask = cv2.inRange(hsv, weaker, stronger)

kernel = np.ones((5,5), np.uint8)
# blue_mask_wk = np.array([90, 50, 50])
blue_mask_wk = np.array([100, 80, 50])
blue_mask_st = np.array([130, 255, 255])
yl_mask_wk = np.array([4, 60, 120])
yl_mask_st = np.array([52, 209, 240])

while True:

    lower_h = cv2.getTrackbarPos("lower_h", 'Trackbar')
    upper_h = cv2.getTrackbarPos("upper_h", 'Trackbar')
    lower_s = cv2.getTrackbarPos("lower_s", 'Trackbar')
    upper_s = cv2.getTrackbarPos("upper_s", 'Trackbar')
    lower_v = cv2.getTrackbarPos("lower_v", 'Trackbar')
    upper_v = cv2.getTrackbarPos("upper_v", 'Trackbar')
    th1_e = cv2.getTrackbarPos("th1_canny", 'Trackbar')
    th2_e = cv2.getTrackbarPos("th2_canny", 'Trackbar')


    image = cv2.imread(images[i])
    mask = np.zeros_like(image)
    image = cv2.resize(image, None, fx=0.2, fy=0.2)
    # image = cv2.blur(image, (7,7))
    cv2.imshow(images[i], image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, th1_e, th2_e, L2gradient=True)
    gray_image = cv2.GaussianBlur(gray_image, (5,5), 3)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    edgesH = cv2.Canny(h, th1_e, th2_e, L2gradient=True)
    edgesS = cv2.Canny(s, th1_e, th2_e, L2gradient=True)
    edgesV = cv2.Canny(v, th1_e, th2_e, L2gradient=True)
    blue_mask = cv2.inRange(hsv, blue_mask_wk, blue_mask_st)


    image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(blue_mask))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)



    weaker = np.array([lower_h, lower_s, lower_v])
    stronger = np.array([upper_h, upper_s, upper_v])


    mask = cv2.inRange(hsv, weaker, stronger)
    cv2.imshow('without blue image', image)
    # cv2.imshow('mask_gray', thresh)
    # cv2.imshow('mask_opening', opening)
    cv2.imshow('mask', mask)
    contours, hierarchy = cv2.findContours(
        cv2.bitwise_not(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    width = image.shape[1]
    width = 1/3.0 * width
    license_plate_area = width * width / 3.0
    euroband_area = 0.06 * license_plate_area
    contours = [contour for contour in contours if cv2.contourArea(
        contour) > euroband_area]  # Remove some sort of mess
    cnt = image.copy()
    for count, contour in enumerate(contours):
        color = (0, 0, 255)
        epsilon = 0.03 * cv2.arcLength(contour, True)
        # Contour Approximation
        approx = cv2.approxPolyDP(contour, epsilon, True)
        box = cv2.boundingRect(approx)
        
        if len(np.squeeze(approx)) == 4:
            coord_of_corners = np.squeeze(approx)
            for y, x in coord_of_corners:
                cnt = cv2.putText(cnt, ".", (y, x), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 10, cv2.LINE_AA)
            color = (255, 0, 0)
        cnt = cv2.drawContours(cnt, contours, count, color, 2)
        # cnt = cv2.rectangle(cnt, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255,255, 0), 3)
    # contours = [contour for contour in contours if cv2.contourArea(
    # contour) > 100]  # Remove some sort of mess
    img_ = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # th = 
    cv2.imshow('cnt', cnt)
    cv2.imshow("edges", edges)
    cv2.imshow("edge S", edgesS)
    cv2.imshow("edge H", edgesH)
    cv2.imshow("edge V", edgesV)
    q = cv2.waitKey(10)
    if q == ord('q'):
        break
    if q == ord('d'):
        i = (i + 1) % len(images)
    if q == ord('a'):
        i = (i - 1) % len(images)

cv2.destroyAllWindows()