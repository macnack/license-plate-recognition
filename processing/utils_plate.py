import numpy as np
import cv2 as cv
import imutils

MIN_RATIO = 3.0
MAX_RATIO = 6.5


def check_ratio_corner(pts):
    [left, top, right, bottom] = pts
    ratio = (right - left) / (bottom - top)
    if MIN_RATIO < ratio < MAX_RATIO:
        return True
    else:
        return False
    

def check_ratio_box(pts):
    x, y, w, h = cv.boundingRect(pts)
    ratio = 0
    if w > h:
        ratio = w / h
    else:
        ratio = h / w
    if MIN_RATIO < ratio < MAX_RATIO:
        return True
    else:
        return False

def check_ratio_pts(pts):
    distances = [np.linalg.norm(pts[(i+1) % len(pts)] - pts[i])
                 for i in range(len(pts))]
    diagonal = max(distances)
    perpendicular = min(distances)
    height = np.sqrt(np.power(diagonal, 2) - np.power(perpendicular, 2))
    if perpendicular >= height:
        ratio = perpendicular / height
    else:
        ratio = height / perpendicular

    if MIN_RATIO < ratio < MAX_RATIO:
        return True
    else:
        return False
    
def read_img(path, scale):
    img = cv.imread(path, cv.IMREAD_COLOR)
    img = cv.resize(img, None, fx=scale, fy=scale)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img, gray

def approx_rectangle(cnts, alpha=0.01):
    epsilon = alpha * cv.arcLength(cnts, True)
    approx = cv.approxPolyDP(cnts, epsilon, True)

    if len(approx) == 4:
        return approx
    else:
        return None
    
def find_countours(img, keep=0):
    cnts = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return None
    if keep > 0:
        return sorted(cnts, key=cv.contourArea, reverse=True)[:keep]
    else:
        return cnts