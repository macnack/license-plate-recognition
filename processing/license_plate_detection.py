import cv2 as cv
import numpy as np
from utils_cv import load_images, show

images_path = load_images()
images_path = images_path[:19]

def check(mask, image):
    width = image.shape[1]
    width = 1 / 3 * width
    license_plate_area_min = width * width / 6.0
    license_plate_area_max = width * width / 3.0

    contours, hierarchy = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv.contourArea(contour) > 100]  # Remove some sort of mess
    vis = image.copy()
    for count, contour in enumerate(contours):
        black_box = np.zeros_like(image)
        area = cv.contourArea(contour)

        epsilon = 0.04 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)  # Contour Approximation
        if len(np.squeeze(approx)) >= 4:  # Take only with 4 corners
            cv.drawContours(vis, contours, count, (255, 0, 0), 3)
            cv.drawContours(black_box, contours, count, (255, 255, 255), -1)
        # Draw each contour only for visualisation purposes
        # x, y, w, h = cv.boundingRect(contour)
        
        cv.drawContours(vis, contours, count, (0, 0, 255), 2)
        

        # cv.imshow("Window", vis)
        new_image = cv.bitwise_and(image, black_box)
        # cv.imshow("new_image", new_image)


        # cv.waitKey()
        # ROI = new_image[y:y + h, x:x + w]  # crop
        # cv.imshow("Window3", ROI)
        gray_image = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)  # Convert to gray color

        ret, thresh = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        # show(thresh, "th")
        opening = cv.morphologyEx(thresh, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # show(opening, "open")
        erosion = cv.erode(opening, np.ones((3, 3), np.uint8), iterations=1)
        # show(erosion, "erode")
        # h, s, v = cv.split(hsv)
        # ret, thg = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # license_plate_closed = cv.morphologyEx(thg, cv.MORPH_CLOSE, (25, 15), iterations=7)
        # license_plate_closed = cv.bitwise_not(license_plate_closed)

        cnts, h = cv.findContours(erosion, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(new_image, cnts, -1, (0, 0, 255), 1)
        # cv.imshow("new_image_ads", opening)
        # cv.imshow("new_image_cnts", new_image)
        # cv.waitKey()
        # sprawdzenie hierarchii na szybko
        print(h)
        values, counts = np.unique(h[:, :, 3], return_counts=True)
        letters = len(counts) > 1 and np.any((counts > 6) & (counts < 9))
        print("btw" ,letters)
        # if len(counts) > 1 and np.any((counts > 6) and (counts < 9)):
        #     return new_image
        if letters:
            return new_image
        # else:
        #     show(new_image, "idk")


def main():
    for image_path in images_path:
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        image = cv.resize(image, None, fx=0.2, fy=0.2)
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)  # Convert to hsv
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Convert to gray color
        mask = cv.inRange(hsv, np.array([0, 0, 138]), np.array([180, 75, 255]))  # Find the mask of white color
        cv.imshow(image_path, image)
        cv.waitKey()
        tablica = check(mask, image)
        if tablica is None:
            # ret2, th = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            # mask = cv.inRange(hsv, np.array([0, 0, 110]), np.array([180, 75, 255]))  # Find the mask of white color
            mask2 = cv.inRange(hsv, np.array([0, 0, 88]), np.array([180, 75, 141]))  # Find the mask of gray color
            mask = mask + mask2
            # erosion = cv.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
            # cv.imshow("mask", mask)  # tablica nasza i z niej wyciagasz znaki
            # cv.waitKey()
            tablica = check(mask, image)
            try:
                cv.imshow(image_path, tablica)  # tablica nasza i z niej wyciagasz znaki
                cv.waitKey()
            except:
                pass
        else:
            cv.imshow(image_path, tablica)  # tablica nasza i z niej wyciagasz znaki
            cv.waitKey()

        cv.destroyAllWindows()


if __name__ == '__main__':
    main()