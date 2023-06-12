import cv2 as cv
import numpy as np
from processing.license_plate_detection import detect_license_plate
from processing.utils_cv import load_images, sort_points_by_corners
license_plate_shape = (512, 114)
from os.path import basename

def convert_letters(string):
    letters_changer = {'B': '8', 'D': '0', 'I': '1', 'O': '0', 'Z': '2'}
    if len(string) < 3:
        return string
    else:
        converted_string = string[:3] + \
            ''.join(letters_changer.get(char, char) for char in string[3:])
        return converted_string


def create_license_plate_image(image, pts):
    # warp license plate contour to rectangle
    dstPts = np.array([[0, 0], [0, license_plate_shape[1]], [license_plate_shape[0], 0], [
        license_plate_shape[0], license_plate_shape[1]]], np.float32)
    transform = cv.getPerspectiveTransform(pts, dstPts)
    license_plate = cv.warpPerspective(image, transform, license_plate_shape)
    return license_plate


def sign_recognitions(image, points, char_images):
    # sorting points to top_left, top_right, bottom_left, bottom_right
    pts = sort_points_by_corners(points)

    license_plate_org = create_license_plate_image(image, pts)
    hsv = cv.cvtColor(license_plate_org, cv.COLOR_BGR2HSV)
    h, _, v = cv.split(hsv)
    _, thg = cv.threshold(v, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    license_plate_closed = cv.morphologyEx(
        thg, cv.MORPH_CLOSE, (25, 15), iterations=7)
    license_plate_closed = cv.bitwise_not(license_plate_closed)
    license_plate_closed_ = license_plate_closed.copy()

    contours, _ = cv.findContours(
        license_plate_closed_, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv.contourArea(
        contour) > 100.0]
    for idx_cnt, cnt in enumerate(contours):
        cv.fillPoly(license_plate_closed, [cnt], 255)
        x, y, w, h = cv.boundingRect(cnt)
    contours, _ = cv.findContours(
        license_plate_closed, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if cv.contourArea(
        contour) > 100.0]

    letters = []
    letters_bbox = [cv.boundingRect(cnt) for cnt in contours if 0.2 <= cv.boundingRect(
        cnt)[2]/float(cv.boundingRect(cnt)[3]) <= 0.8]

    letters_bbox = sorted(letters_bbox, key=lambda pose: pose[0])
    for idx_lbbox, letter_bbox in enumerate(letters_bbox):
        x, y, w, h = letter_bbox
        license_plate_char = license_plate_closed_[y:y+h, x:x+w]
        score = []
        char_names = []
        if idx_lbbox < 2:
            offset = 10
        else:
            offset = 0
        for char_image, char_name in char_images[offset:]:
            char_image = cv.resize(char_image, (50, 50), cv.INTER_CUBIC)
            license_plate_char = cv.resize(
                license_plate_char, (50, 50), cv.INTER_CUBIC)
            res = cv.matchTemplate(
                char_image, license_plate_char, cv.TM_CCOEFF_NORMED)
            score.append(res[0][0])
            filename_path = os.path.basename(char_name)
            char_recognize = str(filename_path[:-4])            if len(char_recognize) > 1:
                char_recognize = char_recognize[0]
            char_names.append(char_recognize)
        if len(score) > 0:
            idx = np.argmax(score)
            letters.append([x, char_names[idx]])
    # show(license_plate_org)

    if len(letters) > 0:
        result = ''.join(item[1] for item in letters)
        result = convert_letters(result)
        return result


def main():
    images_path = load_images()
    images_path = images_path
    char_paths = load_images("dane/fonts")
    char_images = [cv.imread(char_path, cv.IMREAD_GRAYSCALE)
                   for char_path in char_paths]
    for image_path in images_path:
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        pts, image = detect_license_plate(image)
        if pts is not None:
            text = sign_recognitions(image, pts, char_images, char_paths)
            # show(image)
            # print(text)


if __name__ == '__main__':
    main()
