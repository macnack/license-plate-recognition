import json
from os.path import isfile
import numpy as np
from utils_cv import pts_to_corners
import cv2 as cv


path = "dane/ground_thruth.json"
result = "dane/results.json"


def load_ground_thruth():
    if isfile(path):
        with open(path, 'r') as f:
            data = json.load(f)
        ground_truth = {}
        for d in data:
            ground_truth_plate_loc = np.array(
                d['annotations'][0]['result'][-1]['value']['points'])
            ground_truth_plate_recognize = d['annotations'][0]['result'][-1]['value']['text'][0]
            image_path = d['data']['ocr'].split("/")[-1].split('-', 1)[1]
            ground_truth[image_path] = (
                ground_truth_plate_loc, ground_truth_plate_recognize)
        return ground_truth
    else:
        print(f"Can't load {path}")
        return None


def score_recognize(filename, text, truth):
    score = 0
    if filename in truth:
        _, text_ground_truth = truth[filename]
        points = len(text_ground_truth) + 3
        if len(text) == 0:
            return [score, points]
        if text == text_ground_truth:
            score = points
        else:
            for char1, char2 in zip(text, text_ground_truth):
                if char1 == char2:
                    score += 1
        return [score, points]
    print("File: ", filename, " is not in", path)
    return 0, 0


def achieved_pts(result, ground_thruth):
    sum_of_scores = 0
    sum_of_points = 0
    for filename in result:
        score, total = score_recognize(
            filename, result[filename], ground_thruth)
        sum_of_scores += score
        sum_of_points += total
    print("Achieved:", sum_of_scores, " from:", sum_of_points, "points.")
    print("Score:", sum_of_scores / sum_of_points * 100, "%")


def get_truth(target, ground):
    target_key = target.split('/')[-1]
    if target_key in ground:
        return ground[target_key]
    else:
        print(f"No key in Dictonary, {target_key}")
        return (None, None)


def get_ground_truth(target, ground, width, height):
    ground_pts, text = get_truth(target, ground)
    ground_pts[:, 0] = ground_pts[:, 0] * height / 100.0
    ground_pts[:, 1] = ground_pts[:, 1] * width / 100.0
    ground_pts = ground_pts.astype(np.int32)
    return (ground_pts, text)


def save_ground(images_path, ground_truth, scale):
    name = "ground-truth.txt"
    with open(name, 'w') as f:
        for image_path in images_path:
            image = cv.imread(image_path, cv.IMREAD_COLOR)
            image = cv.resize(image, None, fx=scale, fy=scale)

            ground_pts, text = get_ground_truth(
                image_path, ground_truth, image.shape[0], image.shape[1])
            left, top, right, bottom = pts_to_corners(ground_pts)

            f.write(
                f"{image_path.split('/')[-1]} {left} {top} {right} {bottom}\n")
    print(f"SAVED {name}")


def area_of_intersection(ground_truth, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])

    # Intersection height and width.
    i_height = np.maximum(np.abs(iy2 - iy1)+1, np.array(0.))
    i_width = np.maximum(np.abs(ix2 - ix1)+1, np.array(0.))

    return i_height * i_width


def coverage_area(ground_truth, pred):
    x = area_of_intersection(ground_truth, ground_truth)
    y = area_of_intersection(ground_truth, pred)
    return 100.0 * y / x


def get_iou(ground_truth, pred):

    area = area_of_intersection(ground_truth, pred)
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1

    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area

    iou = area / area_of_union

    return iou

def get_intersection_area(box1, box2):
    """
    Calculates the intersection area of two bounding boxes where (x1,y1) indicates the top left corner and (x2,y2)
    indicates the bottom right corner
    :param box1: List of coordinates(x1,y1,x2,y2) of box1
    :param box2: List of coordinates(x1,y1,x2,y2) of box2
    :return: float: area of intersection of the two boxes
    """
    x1 = max(box1[0], box2[0])
    x2 = min(box1[2], box2[2])
    y1 = max(box1[1], box2[1])
    y2 = min(box1[3], box2[3])
    # Check for the condition if there is no overlap between the bounding boxes (either height or width
    # of intersection box are negative)
    if (x2 - x1 < 0) or (y2 - y1 < 0):
        return 0.0
    else:
        return (x2 - x1 + 1) * (y2 - y1 + 1)

if __name__ == "__main__":
    ground_thruth = load_ground_thruth()
    with open(result, 'r') as f:
        result = json.load(f)

    achieved_pts(result, ground_thruth)
